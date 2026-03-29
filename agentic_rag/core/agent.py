
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

from langgraph.graph import StateGraph, END

import agentic_rag.config as config
from agentic_rag.core import (
    ElasticsearchVectorStore,
    CustomRetriever,
    Reranker,
    LLMWrapper,
    SearchResult,
    RAGResult
)


# ==================== Agent状态定义 ====================

class AgentState(TypedDict):
    """
    Agent状态定义 - 定义工作流中传递的状态
    """
    original_query: str
    mode: str

    # 新增：意图识别结果
    intent: str  # 意图类型："chat"(闲聊) / "simple"(简单问答) / "complex"(复杂多跳推理)
    intent_reasoning: str  # 意图识别的推理说明

    sub_questions: List[str]
    sub_answers: Dict[str, str]
    retrieved_docs: Dict[str, List[SearchResult]]
    current_sub_question_index: int

    conversation_history: str

    final_answer: str
    sources: List[SearchResult]
    fact_check_result: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str]

    # 新增：重试和冲突相关状态
    retry_count: int  # 当前重试次数
    max_retries: int  # 最大重试次数
    retry_target: Optional[str]  # 重试目标："retrieve" 或 "synthesize"
    retry_reason: Optional[str]  # 重试原因
    retry_queries: Optional[List[str]]  # Critic节点发现信息缺失时的新搜索词
    conflicts_detected: Optional[List[Dict[str, Any]]]  # 检测到的冲突列表
    user_choice: Optional[str]  # 用户选择的冲突解决方案

    # 新增：知识库ID
    knowledge_base_id: Optional[str]  # ES 知识库过滤ID


# ==================== 文档去重工具 ====================

def deduplicate_documents(
    retrieved_docs: Dict[str, List[SearchResult]]
) -> List[SearchResult]:
    """
    合并并去重来自不同子问题的检索结果
    使用 Chunk ID 进行去重，保留所有唯一文档（不做切片）

    Args:
        retrieved_docs: 子问题到文档列表的映射

    Returns:
        去重后的完整文档列表，按相关性排序
    """
    all_docs = []
    seen_chunk_ids = set()

    # 收集所有文档
    for sub_q, docs in retrieved_docs.items():
        for doc in docs:
            # 使用 Chunk ID 进行去重（优先取 chunk_id 或 id）
            chunk_id = doc.metadata.get("chunk_id") if hasattr(doc, 'metadata') and doc.metadata else None
            if not chunk_id and hasattr(doc, 'id'):
                chunk_id = doc.id
            if not chunk_id:
                # 降级：使用内容哈希作为备选
                import hashlib
                chunk_id = hashlib.md5(doc.content.strip().encode()).hexdigest()

            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                all_docs.append(doc)

    # 按相关性分数排序（如果可用）
    all_docs.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)

    print(f"[文档去重] 原始文档数: {sum(len(docs) for docs in retrieved_docs.values())}, "
          f"去重后: {len(all_docs)} (保留全量)")

    return all_docs


def calculate_content_similarity(text1: str, text2: str) -> float:
    """
    计算两段文本的相似度（用于更精确的去重）
    """
    # 简单的Jaccard相似度
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


# ==================== Agent节点函数 ====================

class QuestionDecomposer:
    """
    问题分解器 - 使用LLM进行意图识别和问题分解

    【新增】意图识别功能：
    - chat: 简单闲聊，直接走纯LLM生成
    - simple: 普通问答，走传统单次RAG
    - complex: 多跳复杂推理，启动全套Agentic流程
    """

    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.decompose_model = config.DECOMPOSE_MODEL

    def decompose(self, query: str, history_context: str = "") -> tuple[str, List[str], str]:
        """
        意图识别 + 问题分解
        返回: (意图, 子问题列表, 推理说明)
        """
        history_section = f"\n\n【历史对话摘要】：\n{history_context}\n\n" if history_context else ""

        system_prompt = f"""{history_section}你是一个智能查询分析专家。你的任务是分析用户输入的意图，并决定如何处理该查询。

【意图分类标准】

1. chat（闲聊/寒暄）
   - 特征：问候、感谢、日常闲聊、个人情感表达、无实质知识需求
   - 示例："你好"、"谢谢"、"今天天气不错"、"你真聪明"、"帮我写首诗"
   - 处理：直接走纯LLM生成，无需检索

2. simple（简单问答/单跳检索）
   - 特征：单一知识点、事实性查询、概念定义、无需多步推理
   - 示例："什么是RAG？"、"Python的列表怎么排序？"、"谁是爱因斯坦？"
   - 处理：走传统单次RAG，一次检索即可回答

3. complex（复杂多跳推理）
   - 特征：需要多步推理、涉及多个概念关联、需要比较分析、需要综合多个信息源
   - 示例：
     * "对比RAG和Fine-tuning的优缺点及适用场景"
     * "Agentic-RAG相比传统RAG有哪些优势，如何实现？"
     * "某公司过去三年的营收趋势如何，受哪些因素影响？"
     * "解释XX技术的原理、应用场景和局限性"
   - 处理：启动全套Agentic流程（分解→并行检索→整合→核查）

【输出格式】
请严格按照以下JSON格式输出：
{{
    "intent": "chat" | "simple" | "complex",
    "reasoning": "意图判断的简要说明",
    "sub_questions": ["子问题1", "子问题2", "子问题3"]  // 仅当intent=complex时提供
}}

注意事项：
- intent只能是chat/simple/complex三者之一
- 如果是chat或simple，sub_questions可以为空或包含原问题
- 如果是complex，建议分解为3-5个有逻辑递进关系的子问题"""

        user_message = f"【用户输入】：{query}\n\n请分析该输入的意图，并按需分解为子问题（仅当复杂查询时）。"

        result = self.llm.generate(
            prompt=user_message,
            model=self.decompose_model,
            system_prompt=system_prompt
        )

        response_text = result.get("response", "")

        # 解析JSON
        try:
            import json
            # 提取JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                intent = parsed.get("intent", "simple")
                reasoning = parsed.get("reasoning", "")
                sub_questions = parsed.get("sub_questions", [query])

                # 验证intent有效性
                if intent not in ["chat", "simple", "complex"]:
                    intent = "simple"

                # chat/simple情况下，如果没有子问题，使用原问题
                if intent in ["chat", "simple"] and not sub_questions:
                    sub_questions = [query]

                return intent, sub_questions, reasoning
        except Exception as e:
            print(f"解析意图识别结果失败: {e}")

        # 解析失败，默认使用simple模式
        return "simple", [query], "解析失败，使用默认简单问答模式"


class SubQuestionAnswerer:
    """子问题回答器 - 检索相关文档并生成答案"""

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        vectorstore: ElasticsearchVectorStore,
        reranker: Optional[Reranker] = None
    ):
        self.llm = llm_wrapper
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.answer_model = config.ANSWER_MODEL

    def answer(self, sub_question: str, knowledge_base_id: str = None, history_context: str = "") -> tuple[str, List[SearchResult]]:
        """
        回答子问题
        返回: (答案, 检索到的文档)
        """
        retrieved_docs = self._retrieve(sub_question, knowledge_base_id)
        answer = self._generate_answer(sub_question, retrieved_docs, history_context)
        return answer, retrieved_docs

    def _retrieve(self, query: str, knowledge_base_id: str = None) -> List[SearchResult]:
        """检索文档"""
        retriever = CustomRetriever(
            vectorstore=self.vectorstore,
            reranker_model=self.reranker.model if self.reranker else None,
            top_k=config.TOP_K,
            rerank_top_k=config.RERANK_CANDIDATES
        )

        results = retriever.get_relevant_documents(
            query=query,
            use_reranker=self.reranker is not None,
            knowledge_base_id=knowledge_base_id
        )

        return results

    def _generate_answer(self, question: str, docs: List[SearchResult], history_context: str = "") -> str:
        """生成答案"""
        if docs:
            doc_content = "\n\n".join([
                f"文档片段{i+1}：\n{doc.content}"
                for i, doc in enumerate(docs)
            ])
            context = f"参考文档：\n{doc_content}\n\n"
        else:
            context = ""

        history_section = f"【历史对话摘要】：\n{history_context}\n\n" if history_context else ""

        system_prompt = """你是一个专业的知识助手，基于提供的参考文档回答具体问题。

要求：
1. 基于提供的参考文档内容进行回答
2. 回答要准确、具体、有逻辑性
3. 如果参考文档不足以回答问题，使用你的专业知识补充
4. 回答长度控制在300-600字"""

        user_message = f"{context}{history_section}问题：{question}\n\n请基于参考文档结合专业知识，提供详尽完整的回答。"

        result = self.llm.generate(
            prompt=user_message,
            model=self.answer_model,
            system_prompt=system_prompt
        )

        return result.get("response", "")


class AnswerSynthesizer:
    """
    答案综合器 - 将子问题的答案综合为最终答案，支持冲突检测

    【职责界定】
    - 视角：聚合视角。输入只能是 original_query 和 sub_qa_pairs（子问题及其答案）。
    - 严禁：接收检索到的参考文档 (sources)。
    - 核心任务：拼图、润色、按照逻辑重构最终答案。
    - 冲突处理：检测"信息源层面的矛盾"（子问题答案之间的矛盾），标记并输出，不主动发起重试。
    """

    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.synthesis_model = config.SYNTHESIS_MODEL

    def synthesize(
        self,
        original_query: str,
        sub_qa_pairs: List[tuple[str, str]]
    ) -> tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        综合答案，检测子问题答案之间的冲突
        返回: (最终答案, 冲突列表或None)
        """
        # 构建子问题-答案内容
        sub_qa_content = "\n".join([
            f"子问题{i+1}: {q}\n答案: {a}\n"
            for i, (q, a) in enumerate(sub_qa_pairs)
        ])

        system_prompt = f"""你是一个专业的知识整合专家，专门负责将多个子问题的答案整合成针对原始问题的精准回答。

【原始问题】：{original_query}

核心原则：
1. 严格对标原始问题，最终回答必须直接、完整地回答原始问题
2. 信息筛选与聚焦，只提取与原始问题直接相关的信息
3. 逻辑重构，按照原始问题的内在逻辑重新组织内容
4. 深度整合，体现出对原始问题的深层理解
5. 冲突检测：如果发现子问题答案之间存在事实冲突，必须在输出中标记

【重要】你只能基于子问题的答案进行整合，不能参考其他外部文档。

质量标准：
- 回答的每一句话都必须服务于原始问题
- 确保回答的完整性，涵盖原始问题的所有关键方面
- 确保回答的准确性，不歪曲或过度延伸子问题的答案

冲突检测（信息源层面矛盾）：
如果发现不同子问题答案对同一事实有不同说法，请标记为冲突并列出所有说法。
这表示信息源本身存在矛盾，需要用户裁决。

输出格式（JSON）：
{{
    "has_conflicts": true/false,
    "conflicts": [
        {{
            "topic": "冲突的主题",
            "conflicting_statements": ["说法1", "说法2"],
            "sources": ["子问题1", "子问题2"]
        }}
    ],
    "answer": "整合后的答案（如果有冲突，给出最合理的版本或说明不确定性）"
}}"""

        user_message = f"""【任务目标】
请基于提供的子问题答案，生成一个完全针对原始问题的综合性回答。

【原始问题（请牢记）】
{original_query}

【子问题与答案】
{sub_qa_content}

请生成最终回答（建议1000字以上，要有深度和实用性），并以JSON格式输出："""

        result = self.llm.generate(
            prompt=user_message,
            model=self.synthesis_model,
            system_prompt=system_prompt
        )

        response_text = result.get("response", "")

        try:
            # 提取并解析 JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                parsed = json.loads(response_text[start:end], strict=False)
                has_conflicts = parsed.get("has_conflicts", False)
                conflicts = parsed.get("conflicts", []) if has_conflicts else None
                answer = parsed.get("answer", response_text)
                return answer, conflicts
        except Exception as e:
            print(f"综合答案解析失败: {e}")

        # 解析失败或没有冲突，返回原始响应
        return response_text, None

    def synthesize_with_correction(
        self,
        original_query: str,
        sub_qa_pairs: List[tuple[str, str]],
        previous_answer: str,
        correction_feedback: str
    ) -> str:
        """
        根据Critic的修正意见重新综合答案（用于Retry Synthesize）

        Args:
            original_query: 原始查询
            sub_qa_pairs: 子问题-答案对
            previous_answer: 之前的答案（包含错误）
            correction_feedback: Critic提供的修正意见
        """
        sub_qa_content = "\n".join([
            f"子问题{i+1}: {q}\n答案: {a}\n"
            for i, (q, a) in enumerate(sub_qa_pairs)
        ])

        system_prompt = f"""你是一个专业的知识整合专家。

【原始问题】：{original_query}

【Critic的修正意见】：
{correction_feedback}

请基于修正意见，重新整合最终答案。确保消除之前答案中的事实错误和幻觉内容。"""

        user_message = f"""【子问题与答案】
{sub_qa_content}

【之前的问题答案（供参考）】
{previous_answer}

请根据Critic的修正意见，生成修正后的最终回答："""

        result = self.llm.generate(
            prompt=user_message,
            model=self.synthesis_model,
            system_prompt=system_prompt
        )

        return result.get("response", "")

    def synthesize_with_user_choice(
        self,
        original_query: str,
        sub_qa_pairs: List[tuple[str, str]],
        user_choice: str
    ) -> str:
        """
        根据用户选择的冲突解决方案重新综合答案
        """
        sub_qa_content = "\n".join([
            f"子问题{i+1}: {q}\n答案: {a}\n"
            for i, (q, a) in enumerate(sub_qa_pairs)
        ])

        system_prompt = f"""你是一个专业的知识整合专家。

【原始问题】：{original_query}

【用户选择的冲突解决方案】：
{user_choice}

请基于用户的冲突解决方案，重新整合最终答案。"""

        user_message = f"""【子问题与答案】
{sub_qa_content}

请生成最终回答："""

        result = self.llm.generate(
            prompt=user_message,
            model=self.synthesis_model,
            system_prompt=system_prompt
        )

        return result.get("response", "")



class FactChecker:
    """
    事实核查器 - 验证生成答案与参考文档的一致性

    【职责界定】
    - 视角：验证视角。输入是 Synthesizer 生成的 final_answer 以及所有去重后的原始参考文档 (sources)。
    - 严禁：修改答案，只能标记问题并发起重试。
    - 核心任务：找茬、查杀幻觉、评估信息完整度。
    - 冲突处理 & 重试导向：整个系统的纠偏反馈回路发起者。
        * 发现"生成层面的矛盾/事实错误"（大模型瞎编，与sources不符）：强制打回给Synthesizer重新整合（Retry Synthesize），附带明确的修改意见。
        * 发现"信息缺失"（无法完整回答原问题）：强制打回给检索节点（Retry Retrieve），必须输出新的搜索关键词（Query Rewriting）。
    """

    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.model = config.SYNTHESIS_MODEL

    def check_and_correct(self, query: str, answer: str, sources: List[SearchResult], retry_count: int = 0) -> tuple[str, Dict[str, Any]]:
        """
        核查答案，返回重试决策
        返回: (原始答案, 核查元数据包含retry决策)
        【注意】Critic不修改答案，只返回重试决策
        """
        # 准备参考上下文（使用所有来源，不做切片）
        context = "\n\n".join([f"[参考资料{i+1}]: {doc.content}" for i, doc in enumerate(sources)])

        system_prompt = """你是一个严谨的事实核查专家，负责验证答案与参考文档的一致性。

【核心职责】
1. 验证答案中的每个事实性陈述是否能在参考文档中找到依据
2. 检测"生成层面的矛盾"：答案内容与参考文档不符（幻觉、编造）
3. 检测"信息缺失"：答案未能完整回答原始问题，缺少关键信息

【处理规则】
1. 事实错误：如果答案中的某点与参考资料明确矛盾，标记为"fact_error"
2. 信息缺失：如果答案缺少关键信息来完整回答原始问题，标记为"information_missing"
3. 容忍合理推断：普适性描述（如"提高效率"）即使没有明确提到但符合逻辑，不视为错误

【重试决策】
- fact_error → retry_type: "synthesize"，附带详细的修正意见
- information_missing → retry_type: "retrieve"，必须输出新的搜索关键词

输出格式（JSON）：
{
    "status": "passed" 或 "needs_retry",
    "retry_type": "none" 或 "retrieve" 或 "synthesize",
    "retry_reason": "具体说明为什么需要重试",
    "correction_feedback": "给Synthesizer的修正意见（如果是fact_error）",
    "retry_queries": ["新搜索词1", "新搜索词2"]  // 如果是information_missing，必须提供
}"""

        user_message = f"""【原始问题】：{query}

【参考文档（全量）】：
{context}

【待核查答案】：
{answer}

请开始核查，输出JSON格式结果："""

        result = self.llm.generate(
            prompt=user_message,
            model=self.model,
            system_prompt=system_prompt
        )

        response_text = result.get("response", "")

        try:
            # 提取并解析 JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            parsed = json.loads(response_text[start:end], strict=False)

            status = parsed.get("status", "unknown")
            retry_type = parsed.get("retry_type", "none")
            retry_reason = parsed.get("retry_reason", "")

            # 构建返回元数据
            metadata = {
                "fact_check_status": status,
                "retry_type": retry_type,
                "retry_reason": retry_reason,
                "correction_feedback": parsed.get("correction_feedback", ""),
                "retry_queries": parsed.get("retry_queries", [])
            }

            # Critic不修改答案，只返回重试决策
            return answer, metadata
        except Exception as e:
            print(f"事实核查解析失败: {e}")
            return answer, {"fact_check_status": "error", "error": str(e), "retry_type": "none", "retry_queries": []}


class TraditionalRAG:
    """传统RAG - 简单检索增强生成"""

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        vectorstore: ElasticsearchVectorStore,
        reranker: Optional[Reranker] = None
    ):
        self.llm = llm_wrapper
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.answer_model = config.SYNTHESIS_MODEL

    def query(self, query: str, knowledge_base_id: str = None) -> tuple[str, List[SearchResult]]:
        """执行传统RAG"""
        # 检索
        retriever = CustomRetriever(
            vectorstore=self.vectorstore,
            reranker_model=self.reranker.model if self.reranker else None,
            top_k=config.TOP_K,
            rerank_top_k=config.RERANK_CANDIDATES
        )

        retrieved_docs = retriever.get_relevant_documents(
            query=query,
            use_reranker=self.reranker is not None,
            knowledge_base_id=knowledge_base_id
        )

        # 生成答案
        answer = self._generate_answer(query, retrieved_docs)

        return answer, retrieved_docs

    def _generate_answer(self, query: str, docs: List[SearchResult]) -> str:
        """生成答案"""
        if not docs:
            # 没有检索到文档，使用纯生成
            system_prompt = """你是一个专业的AI助手，擅长提供详尽的知识解答。
请按照以下要求回答用户问题：
1. 内容全面：提供完整、深入且覆盖问题各方面的回答
2. 逻辑清晰：使用分层结构
3. 详细解释：对关键概念给予透彻解释
4. 长度充分：至少800字"""

            user_message = f"问题: {query}"
        else:
            # 使用RAG
            doc_content = "\n\n".join([
                f"参考文档{i+1}：\n{doc.content}"
                for i, doc in enumerate(docs)
            ])

            system_prompt = """你是一个专业的知识助手，擅长提供极其全面、详尽且精确的回答。
请严格遵循以下规则：
1. 深入分析所有参考文本块，识别与问题高度相关的关键信息
2. 在参考文本块基础上，大幅扩展和丰富内容
3. 补充必要的背景信息、定义、理论框架等
4. 回答长度应充分，至少800字"""

            user_message = f"参考信息:\n{doc_content}\n\n问题: {query}\n\n请结合参考信息和你自己的知识，提供详尽回答。"

        result = self.llm.generate(
            prompt=user_message,
            model=self.answer_model,
            system_prompt=system_prompt
        )

        return result.get("response", "")


class PureLLMGenerator:
    """纯LLM生成器 - 用于简单闲聊，无需检索"""

    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        self.model = config.SYNTHESIS_MODEL

    def generate(self, query: str, history_context: str = "") -> str:
        """
        纯LLM生成回答（无检索）
        适用于：闲聊、问候、创造性写作等无需知识检索的场景
        """
        history_section = f"【历史对话】\n{history_context}\n\n" if history_context else ""

        system_prompt = """你是一个友善、乐于助人的AI助手。

回答风格：
1. 自然亲切，像朋友一样交流
2. 简洁明了，不冗长
3. 对于闲聊问候，热情回应
4. 对于创造性请求（如写诗、编故事），发挥创意
5. 如果涉及知识性问题但你不确定，坦诚告知

注意：无需检索外部文档，直接基于你的知识回答即可。"""

        user_message = f"{history_section}用户说：{query}\n\n请自然地回应用户。"

        result = self.llm.generate(
            prompt=user_message,
            model=self.model,
            system_prompt=system_prompt
        )

        return result.get("response", "")


# ==================== LangGraph工作流 ====================

class AgenticRAGWorkflow:
    """
    Agentic-RAG工作流 - 使用LangGraph编排
    """

    def __init__(
        self,
        vectorstore: ElasticsearchVectorStore,
        reranker: Optional[Reranker] = None
    ):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.llm = LLMWrapper()

        # 初始化组件
        self.decomposer = QuestionDecomposer(self.llm)
        self.answerer = SubQuestionAnswerer(
            self.llm,
            vectorstore,
            reranker
        )
        self.synthesizer = AnswerSynthesizer(self.llm)
        self.traditional_rag = TraditionalRAG(
            self.llm,
            vectorstore,
            reranker
        )
        self.pure_llm_generator = PureLLMGenerator(self.llm)
        self.fact_checker = FactChecker(self.llm)

        # 构建图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流 - 支持意图识别和重试循环"""
        graph = StateGraph(AgentState)

        # 添加节点
        graph.add_node("decompose", self._decompose_node)
        graph.add_node("pure_llm", self._pure_llm_node)
        graph.add_node("retrieve_and_answer", self._retrieve_and_answer_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("traditional_rag", self._traditional_rag_node)
        graph.add_node("fact_check", self._fact_check_node)

        # 设置入口
        graph.set_entry_point("decompose")

        # 添加条件边 - 根据意图识别结果选择路径
        graph.add_conditional_edges(
            "decompose",
            self._route_by_intent,
            {
                "chat": "pure_llm",
                "simple": "traditional_rag",
                "complex": "retrieve_and_answer"
            }
        )

        # 添加边 - retrieve_and_answer -> synthesize
        graph.add_edge("retrieve_and_answer", "synthesize")

        # 添加边 - synthesize -> fact_check
        graph.add_edge("synthesize", "fact_check")

        # 添加条件边 - fact_check后根据重试目标决定走向
        graph.add_conditional_edges(
            "fact_check",
            self._should_retry_or_end,
            {
                "retry_retrieve": "retrieve_and_answer",
                "retry_synthesize": "synthesize",
                "end": END
            }
        )

        # 纯LLM和传统RAG直接结束
        graph.add_edge("pure_llm", END)
        graph.add_edge("traditional_rag", END)

        return graph.compile()

    def _should_retry_or_end(self, state: AgentState) -> str:
        """
        判断是否需要重试或结束
        根据 retry_target 决定走向
        """
        retry_target = state.get("retry_target")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if retry_target == "retrieve" and retry_count < max_retries:
            return "retry_retrieve"
        elif retry_target == "synthesize" and retry_count < max_retries:
            return "retry_synthesize"
        else:
            return "end"


    def _fact_check_node(self, state: AgentState) -> AgentState:
        """
        事实核查节点 - 验证答案与参考文档的一致性
        【职责】纠偏反馈回路发起者，发现错误时返回重试决策
        """
        print("\n" + "="*30 + " 启动事实核查与修正 " + "="*30)

        # 1. 获取核查前状态
        original_query = state["original_query"]
        pre_check_answer = state["final_answer"]
        sources = state.get("sources", [])  # 使用 get 避免 KeyError，默认为空列表
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        # 2. 调用Critic核查器（使用所有sources）
        try:
            # 【关键修改】传入所有去重后的sources，不做切片
            checked_answer, check_meta = self.fact_checker.check_and_correct(
                original_query,
                pre_check_answer,
                sources,  # 全量文档
                retry_count
            )
        except Exception as e:
            print(f"❌ 事实核查调用发生异常: {e}")
            check_meta = {
                "fact_check_status": "exception",
                "error": str(e),
                "retry_type": "none",
                "retry_queries": []
            }
            checked_answer = pre_check_answer

        # 3. 处理重试决策
        status = check_meta.get('fact_check_status', 'unknown')
        retry_type = check_meta.get('retry_type', 'none')
        retry_reason = check_meta.get('retry_reason', '')
        retry_queries = check_meta.get('retry_queries', [])

        print(f"DEBUG: Critic核查完成。状态: {status}, 重试类型: {retry_type}")

        # 【Retry Retrieve】信息缺失，需要重新检索
        if retry_type == "retrieve" and retry_count < max_retries:
            print(f"⚠️ 信息缺失，需要重检索: {retry_reason}")
            if retry_queries:
                print(f"DEBUG: Critic提供的新的搜索词: {retry_queries}")
            state["retry_count"] = retry_count + 1
            state["retry_target"] = retry_type
            state["retry_reason"] = retry_reason
            state["retry_queries"] = retry_queries  # 【新增】存储新的搜索词
            state["fact_check_result"] = check_meta
            print(f"DEBUG: 将使用新的查询词重试检索 (重试 {state['retry_count']}/{max_retries})")
            print("="*80 + "\n")
            return state

        # 【Retry Synthesize】事实错误，需要重新整合
        if retry_type == "synthesize" and retry_count < max_retries:
            correction_feedback = check_meta.get('correction_feedback', '')
            print(f"⚠️ 发现事实错误，需要重整合: {retry_reason}")
            if correction_feedback:
                print(f"DEBUG: Critic的修正意见: {correction_feedback[:200]}...")
            state["retry_count"] = retry_count + 1
            state["retry_target"] = retry_type
            state["retry_reason"] = retry_reason
            state["fact_check_result"] = check_meta  # 包含correction_feedback
            print(f"DEBUG: 将重试到 'synthesize' 节点 (重试 {state['retry_count']}/{max_retries})")
            print("="*80 + "\n")
            return state

        # 【通过核查】
        if status == "passed":
            print("✅ DEBUG: 答案通过Critic核查，未发现事实性错误。")
        elif status == "error":
            print(f"⚠️ DEBUG: 核查由于解析失败降级。错误原因: {check_meta.get('error')}")
        else:
            print(f"DEBUG: 核查状态: {status}")

        # 4. 更新状态（通过核查或达到最大重试次数）
        state["final_answer"] = checked_answer
        state["fact_check_result"] = check_meta
        state["metadata"]["fact_check_executed"] = True
        state["metadata"]["fact_check_status"] = status
        state["retry_target"] = None  # 清除重试目标
        state["retry_reason"] = None
        state["retry_queries"] = None  # 清除重试查询词

        print("="*80 + "\n")
        return state

    def _decompose_node(self, state: AgentState) -> AgentState:
        """
        分解问题节点 - 【新增】意图识别
        根据用户查询意图决定后续流程：chat/simple/complex
        """
        query = state["original_query"]
        history_context = state.get("conversation_history", "")

        # 【修改】调用意图识别 + 问题分解
        intent, sub_questions, reasoning = self.decomposer.decompose(query, history_context)

        # 设置意图和推理
        state["intent"] = intent
        state["intent_reasoning"] = reasoning
        state["sub_questions"] = sub_questions

        # 根据意图设置状态
        if intent == "chat":
            state["metadata"]["status"] = f"意图识别: 闲聊模式 - {reasoning}"
        elif intent == "simple":
            state["metadata"]["status"] = f"意图识别: 简单问答 - {reasoning}"
        elif intent == "complex":
            state["metadata"]["decomposition_reasoning"] = reasoning
            state["metadata"]["status"] = f"意图识别: 复杂推理 - 分解为 {len(sub_questions)} 个子问题"

        state["current_sub_question_index"] = 0
        state["sub_answers"] = {}
        state["retrieved_docs"] = {}

        return state

    def _route_by_intent(self, state: AgentState) -> str:
        """
        【修改】根据意图识别结果路由到不同节点
        - chat: 纯LLM生成
        - simple: 传统单次RAG
        - complex: 全套Agentic流程
        """
        intent = state.get("intent", "simple")
        # 验证 intent 有效性，如果不是三个有效值之一，默认使用 simple
        if intent not in ["chat", "simple", "complex"]:
            print(f"[WARNING] 未知的意图类型 '{intent}'，使用默认 simple 模式")
            intent = "simple"
        return intent

    def _pure_llm_node(self, state: AgentState) -> AgentState:
        """
        【新增】纯LLM生成节点 - 用于闲聊场景
        无需检索，直接生成回答
        """
        query = state["original_query"]
        history_context = state.get("conversation_history", "")

        print(f"\n[纯LLM生成] 处理闲聊请求: {query[:50]}...")

        # 调用纯LLM生成
        answer = self.pure_llm_generator.generate(query, history_context)

        state["final_answer"] = answer
        state["sources"] = []  # 闲聊无来源
        state["metadata"]["status"] = "纯LLM生成完成"
        state["metadata"]["answer_source"] = "pure_llm"

        return state

    def _retrieve_and_answer_node(self, state: AgentState) -> AgentState:
        """
        检索并回答节点 - 支持并行处理和文档去重
        """
        # 检查是否是重试检索流程
        retry_count = state.get("retry_count", 0)
        retry_target = state.get("retry_target")
        retry_queries = state.get("retry_queries")
        knowledge_base_id = state.get("knowledge_base_id")

        # 决定使用哪种查询：正常流程用sub_questions，重试流程用retry_queries
        if retry_count > 0 and retry_target == "retrieve" and retry_queries:
            print(f"\n[重试检索 {retry_count}/3] 使用Critic提供的查询词: {retry_queries}")
            queries_to_process = retry_queries
            # 清空之前的状态
            state["sub_answers"] = {}
            state["retrieved_docs"] = {}
        else:
            queries_to_process = state["sub_questions"]

        # 回答子问题时不传入历史上下文，避免干扰文档检索
        history_context = ""

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_sub_question(sub_q: str) -> tuple:
            """处理单个子问题"""
            print(f"\n处理子问题: {sub_q}")
            answer, docs = self.answerer.answer(sub_q, knowledge_base_id=knowledge_base_id, history_context=history_context)
            return sub_q, answer, docs

        if len(queries_to_process) > 1:
            print(f"并行处理 {len(queries_to_process)} 个查询...")
            with ThreadPoolExecutor(max_workers=min(len(queries_to_process), 3)) as executor:
                futures = {executor.submit(process_sub_question, sq): sq for sq in queries_to_process}
                for future in as_completed(futures):
                    sub_q, answer, docs = future.result()
                    state["sub_answers"][sub_q] = answer
                    state["retrieved_docs"][sub_q] = docs
        else:
            for current_idx, sub_q in enumerate(queries_to_process):
                if sub_q not in state["sub_answers"]:
                    print(f"\n处理查询 {current_idx + 1}/{len(queries_to_process)}: {sub_q}")
                    answer, docs = self.answerer.answer(sub_q, knowledge_base_id=knowledge_base_id, history_context=history_context)
                    state["sub_answers"][sub_q] = answer
                    state["retrieved_docs"][sub_q] = docs

        # 文档池合并与去重 - 保留全量文档，不做切片
        print("\n[文档去重] 合并各查询的检索结果...")
        deduplicated_sources = deduplicate_documents(state["retrieved_docs"])
        state["sources"] = deduplicated_sources

        state["metadata"]["status"] = f"已完成 {len(queries_to_process)} 个查询，去重后文档数: {len(deduplicated_sources)}"
        state["metadata"]["deduplication_info"] = {
            "original_count": sum(len(docs) for docs in state["retrieved_docs"].values()),
            "after_dedup": len(deduplicated_sources)
        }

        return state

    def _synthesize_node(self, state: AgentState) -> AgentState:
        """
        综合答案节点 - 仅基于子问题答案进行整合，不接收sources
        冲突检测：仅检测子问题答案之间的信息源层面矛盾
        """
        original_query = state["original_query"]
        sub_qa_pairs = list(state["sub_answers"].items())

        # 检查是否是重试整合流程（带修正意见）
        retry_count = state.get("retry_count", 0)
        retry_target = state.get("retry_target")
        user_choice = state.get("user_choice")
        fact_check_result = state.get("fact_check_result", {})

        # 【Retry Synthesize】根据Critic的修正意见重新整合
        if retry_count > 0 and retry_target == "synthesize":
            correction_feedback = fact_check_result.get("correction_feedback", "")
            if correction_feedback:
                print(f"\n[重试整合 {retry_count}/3] 根据Critic的修正意见重新生成答案...")
                final_answer = self.synthesizer.synthesize_with_correction(
                    original_query=original_query,
                    sub_qa_pairs=sub_qa_pairs,
                    previous_answer=state.get("final_answer", ""),
                    correction_feedback=correction_feedback
                )
                state["final_answer"] = final_answer
                state["conflicts_detected"] = None
                return state

        # 【User Choice】根据用户选择的冲突解决方案重新整合
        if user_choice and state.get("conflicts_detected"):
            print(f"\n根据用户选择重新综合答案...")
            final_answer = self.synthesizer.synthesize_with_user_choice(
                original_query, sub_qa_pairs, user_choice
            )
            state["final_answer"] = final_answer
            state["conflicts_detected"] = None
            return state

        print("\n综合最终答案...")
        print("\n" + "="*30 + " DEBUG: 子问题执行结果 " + "="*30)
        for q, a in state["sub_answers"].items():
            print(f"【子问题】: {q}")
            # 打印答案前150个字，方便观察是否偏离主题
            answer_preview = a[:150].replace('\n', ' ') + "..." if len(a) > 150 else a
            print(f"【子答案】: {answer_preview}")

            # 可选：如果你想看这个子问题关联了多少文档
            docs_count = len(state["retrieved_docs"].get(q, []))
            print(f"【参考文档数】: {docs_count}")
            print("-" * 70)
        print("="*80 + "\n")

        # 【关键修改】调用合成器时不再传入sources，严格解耦
        # Synthesizer只基于sub_qa_pairs进行整合
        final_answer, conflicts = self.synthesizer.synthesize(
            original_query=original_query,
            sub_qa_pairs=sub_qa_pairs
        )

        state["final_answer"] = final_answer
        state["conflicts_detected"] = conflicts

        if conflicts:
            print(f"\n⚠️ 检测到 {len(conflicts)} 处信息源层面冲突:")
            for i, conflict in enumerate(conflicts, 1):
                print(f"  冲突 {i}: {conflict.get('topic', 'Unknown')}")
                for stmt in conflict.get('conflicting_statements', []):
                    print(f"    - {stmt}")
            print("提示：这些冲突将返回给前端，等待用户裁决（Human-in-the-loop）")

        state["metadata"]["status"] = "完成"
        state["metadata"]["conflicts_detected"] = len(conflicts) if conflicts else 0

        return state

    def _traditional_rag_node(self, state: AgentState) -> AgentState:
        """传统RAG节点"""
        query = state["original_query"]
        knowledge_base_id = state.get("knowledge_base_id")

        print(f"\n执行传统RAG检索: {query}")
        answer, docs = self.traditional_rag.query(query, knowledge_base_id=knowledge_base_id)

        state["final_answer"] = answer
        state["sources"] = docs
        state["metadata"]["status"] = "传统RAG完成"

        return state

    def run(self, query: str, mode: str = "agentic", history_context: str = "", user_choice: str = None, knowledge_base_id: str = None) -> RAGResult:
        """
        运行Agentic-RAG工作流
        支持用户选择的冲突解决方案和知识库过滤
        返回: RAGResult对象
        """
        initial_state: AgentState = {
            "original_query": query,
            "mode": mode,
            "intent": "",
            "intent_reasoning": "",
            "sub_questions": [],
            "sub_answers": {},
            "retrieved_docs": {},
            "current_sub_question_index": 0,
            "conversation_history": history_context,
            "final_answer": "",
            "sources": [],
            "fact_check_result": None,
            "metadata": {"status": "初始化"},
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
            "retry_target": None,
            "retry_reason": None,
            "retry_queries": None,
            "conflicts_detected": None,
            "user_choice": user_choice,
            "knowledge_base_id": knowledge_base_id
        }

        try:
            result_state = self.graph.invoke(initial_state)

            return RAGResult(
                original_query=result_state["original_query"],
                sub_questions=result_state["sub_questions"],
                sub_answers=result_state["sub_answers"],
                final_answer=result_state["final_answer"],
                sources=result_state["sources"],
                metadata=result_state["metadata"]
            )
        except Exception as e:
            print(f"工作流执行失败: {e}")
            return RAGResult(
                original_query=query,
                error=str(e)
            )

    def run_streaming(self, query: str, mode: str = "agentic", history_context: str = "", user_choice: str = None, knowledge_base_id: str = None):
        """
        流式运行工作流 - 逐步返回结果
        """
        initial_state: AgentState = {
            "original_query": query,
            "mode": mode,
            "intent": "",
            "intent_reasoning": "",
            "sub_questions": [],
            "sub_answers": {},
            "retrieved_docs": {},
            "current_sub_question_index": 0,
            "conversation_history": history_context,
            "final_answer": "",
            "sources": [],
            "fact_check_result": None,
            "metadata": {"status": "初始化"},
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
            "retry_target": None,
            "retry_reason": None,
            "retry_queries": None,
            "conflicts_detected": None,
            "user_choice": user_choice,
            "knowledge_base_id": knowledge_base_id
        }

        for event in self.graph.stream(initial_state):
            for node_name, node_state in event.items():
                yield node_name, node_state


# ==================== 便捷函数 ====================

def create_agentic_rag(vectorstore: ElasticsearchVectorStore, reranker: Reranker = None):
    """创建Agentic-RAG实例"""
    return AgenticRAGWorkflow(vectorstore, reranker)
