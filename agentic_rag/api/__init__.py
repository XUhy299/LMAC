"""
FastAPI后端服务 - 用户认证、对话API、知识库管理
"""
import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from sqlalchemy.orm import Session

import config
from agentic_rag.db import (
    get_db, init_db,
    User, KnowledgeBase, Conversation, Message,
    UserCreate, UserResponse, Token,
    KnowledgeBaseCreate, KnowledgeBaseResponse,
    ConversationCreate, ConversationResponse, MessageCreate, MessageResponse,
    create_user, get_user_by_username, get_user_by_id,
    create_knowledge_base, get_user_knowledge_bases, get_knowledge_base_by_id,
    update_knowledge_base, update_knowledge_base_index_status, delete_knowledge_base,
    create_conversation, get_conversation, get_user_conversations,
    create_message, get_conversation_messages,
    verify_password, get_password_hash, create_usage_stats, get_user_stats
)
from agentic_rag.core import (
    DocumentLoader, TextSplitter, ElasticsearchVectorStore,
    Reranker, LLMWrapper, SearchResult
)
from agentic_rag.core.agent import AgenticRAGWorkflow
from agentic_rag.core.cache import conv_cache

app = FastAPI(
    title="Agentic-RAG API",
    description="智能RAG系统后端API - 支持本地知识库和ES混合检索",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# 全局变量
vectorstore = None
reranker = None
agent_workflow = None
document_loader = None
text_splitter = None

class MessageRequest(BaseModel):
    content: str
    mode: str = "agentic"
    knowledge_base_id: Optional[int] = None  # 可选：指定知识库

class QueryRequest(BaseModel):
    query: str
    mode: str = "agentic"
    top_k: int = 5
    knowledge_base_id: Optional[int] = None

class ScanKnowledgeBaseRequest(BaseModel):
    kb_id: int

class MockAgentWorkflow:
    """MOCK_LLM=true 时使用，跳过向量存储直接返回 mock 响应"""
    def __init__(self):
        self.llm = LLMWrapper()

    def run(self, query: str, mode: str = "agentic", history_context: str = "", **kwargs):
        from core import RAGResult
        result = self.llm._mock_generate(query, config.ANSWER_MODEL)
        return RAGResult(
            original_query=query,
            sub_questions=["mock子问题1", "mock子问题2"],
            final_answer=result["response"]
        )

    def run_streaming(self, query: str, mode: str = "agentic", history_context: str = "", **kwargs):
        from core import RAGResult
        result = self.llm._mock_generate(query, config.ANSWER_MODEL)
        yield ("final", {"final_answer": result["response"]})

def create_access_token(data: dict, expires_delta: timedelta = None):
    """创建JWT访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """获取当前登录用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global vectorstore, reranker, agent_workflow, document_loader, text_splitter

    if agent_workflow is not None:
        return

    init_db()

    if os.getenv("MOCK_LLM", "false").lower() == "true":
        print("[MOCK_LLM] 跳过向量存储初始化，使用 MockAgentWorkflow")
        agent_workflow = MockAgentWorkflow()
        return

    document_loader = DocumentLoader()
    text_splitter = TextSplitter()

    try:
        # 初始化 Elasticsearch 向量存储
        vectorstore = ElasticsearchVectorStore()
        stats = vectorstore.get_stats()
        print(f"[ES] 连接成功，索引: {stats['index_name']}, 文档数: {stats['total_docs']}")

        # 初始化重排序模型
        reranker = Reranker()

        # 初始化 Agent 工作流
        agent_workflow = AgenticRAGWorkflow(vectorstore, reranker)

    except Exception as e:
        print(f"[WARNING] 系统初始化失败，自动降级到 Mock 模式: {type(e).__name__}")
        agent_workflow = MockAgentWorkflow()

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Agentic-RAG API", "version": "2.0.0", "features": ["elasticsearch", "hybrid_search", "knowledge_base"]}

@app.get("/health")
async def health_check():
    """健康检查"""
    es_status = "connected"
    try:
        if vectorstore is not None:
            vectorstore.es.ping()
        else:
            es_status = "mock_mode"
    except Exception:
        es_status = "disconnected"

    return {"status": "healthy", "elasticsearch": es_status}

# ==================== 认证路由 ====================

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """用户注册"""
    from sqlalchemy.exc import IntegrityError
    existing_user = get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="用户名已被注册")

    try:
        user = create_user(
            db,
            username=user_data.username,
            email=user_data.email or "",
            password=user_data.password
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    return user

@app.post("/api/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """用户登录"""
    user = get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user

# ==================== 知识库路由 ====================

@app.post("/api/knowledge-bases", response_model=KnowledgeBaseResponse)
async def create_new_knowledge_base(
    kb_data: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """创建新知识库"""
    # 验证路径是否存在
    if not os.path.exists(kb_data.path):
        raise HTTPException(status_code=400, detail="指定的知识库路径不存在")

    kb = create_knowledge_base(
        db,
        user_id=current_user.id,
        name=kb_data.name,
        path=kb_data.path,
        description=kb_data.description,
        file_types=kb_data.file_types
    )
    return kb

@app.get("/api/knowledge-bases", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户所有知识库"""
    kbs = get_user_knowledge_bases(db, current_user.id)
    return kbs

@app.get("/api/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取知识库详情"""
    kb = get_knowledge_base_by_id(db, kb_id, current_user.id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb

@app.post("/api/knowledge-bases/{kb_id}/scan")
async def scan_knowledge_base(
    kb_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """扫描知识库目录并建立索引"""
    kb = get_knowledge_base_by_id(db, kb_id, current_user.id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if not os.path.exists(kb.path):
        raise HTTPException(status_code=400, detail="知识库路径不存在")

    # 在后台执行扫描
    background_tasks.add_task(_scan_and_index_kb, kb_id, current_user.id)

    return {"message": "扫描任务已启动", "kb_id": kb_id}

def _scan_and_index_kb(kb_id: int, user_id: int):
    """后台扫描并索引知识库"""
    from database import SessionLocal
    global vectorstore

    # Mock 模式下 vectorstore 为 None，无法执行扫描
    if vectorstore is None:
        print(f"[WARNING] 扫描知识库 {kb_id} 失败: 当前处于 Mock 模式，无法访问向量存储")
        return

    db = SessionLocal()
    try:
        kb = get_knowledge_base_by_id(db, kb_id, user_id)
        if not kb:
            return

        # 加载文档
        file_types = kb.file_types.split(",") if kb.file_types else [".pdf", ".txt", ".md"]
        loader = DocumentLoader()
        documents = loader.load_directory(kb.path, file_types)

        if not documents:
            update_knowledge_base_index_status(
                db, kb_id, user_id,
                is_indexed=True,
                file_count=0,
                chunk_count=0,
                es_index_name=config.ES_INDEX_NAME
            )
            return

        # 分块
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        # 删除旧索引
        kb_es_id = f"kb_{user_id}_{kb_id}"
        vectorstore.delete_by_knowledge_base(kb_es_id)

        # 添加新索引
        vectorstore.add_documents(chunks, knowledge_base_id=kb_es_id)

        # 更新状态
        update_knowledge_base_index_status(
            db, kb_id, user_id,
            is_indexed=True,
            file_count=len(documents),
            chunk_count=len(chunks),
            es_index_name=config.ES_INDEX_NAME
        )

        print(f"知识库 {kb.name} 扫描完成: {len(documents)} 文件, {len(chunks)} 块")

    except Exception as e:
        print(f"扫描知识库失败: {e}")
    finally:
        db.close()

@app.put("/api/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_kb(
    kb_id: int,
    kb_data: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新知识库"""
    kb = update_knowledge_base(
        db, kb_id, current_user.id,
        name=kb_data.name,
        description=kb_data.description,
        file_types=kb_data.file_types
    )
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb

@app.delete("/api/knowledge-bases/{kb_id}")
async def delete_kb(
    kb_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除知识库"""
    kb = get_knowledge_base_by_id(db, kb_id, current_user.id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 删除 ES 中的索引（仅在非 Mock 模式下）
    if vectorstore is not None:
        try:
            kb_es_id = f"kb_{current_user.id}_{kb_id}"
            vectorstore.delete_by_knowledge_base(kb_es_id)
        except Exception as e:
            print(f"删除 ES 索引失败: {e}")

    # 删除数据库记录
    delete_knowledge_base(db, kb_id, current_user.id)

    return {"message": "知识库已删除"}

# ==================== 对话路由 ====================

@app.post("/api/conversations", response_model=ConversationResponse)
async def create_new_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """创建新对话"""
    conversation = create_conversation(
        db,
        user_id=current_user.id,
        title=conversation_data.title or "新对话",
        mode=conversation_data.mode
    )
    return conversation

@app.get("/api/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户所有对话"""
    conversations = get_user_conversations(db, current_user.id, limit)
    return conversations

@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_details(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取对话详情"""
    conversation = get_conversation(db, conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除对话"""
    conversation = get_conversation(db, conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    conv_cache.evict(conversation_id)
    return {"message": "Conversation deleted"}

# ==================== 消息路由 ====================

@app.get("/api/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取对话的所有消息"""
    conversation = get_conversation(db, conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = get_conversation_messages(db, conversation_id)
    return messages

@app.post("/api/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: int,
    request_data: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """发送消息并获取回复"""
    conversation = get_conversation(db, conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    start_time = time.time()
    content = request_data.content
    mode = request_data.mode

    # 冷启动：缓存为空时从 DB 加载该会话的历史消息
    if not conv_cache.is_loaded(conversation_id):
        db_messages = get_conversation_messages(db, conversation_id)
        conv_cache.load_from_db(conversation_id, db_messages)

    # 写用户消息：DB + 缓存同时写
    user_message = create_message(
        db,
        conversation_id=conversation_id,
        role="user",
        content=content
    )
    conv_cache.append(conversation_id, "user", content)

    # 从缓存取历史上下文
    history_context = conv_cache.get_context_string(conversation_id, exclude_last=1)

    # 构建知识库过滤参数
    kb_filter = None
    if request_data.knowledge_base_id:
        kb = get_knowledge_base_by_id(db, request_data.knowledge_base_id, current_user.id)
        if kb and kb.is_indexed:
            kb_filter = f"kb_{current_user.id}_{kb.id}"

    try:
        result = agent_workflow.run(
            content,
            mode=mode,
            history_context=history_context,
            knowledge_base_id=kb_filter
        )

        # 计算延迟和token数（中文按1.5字符/token估算）
        latency = time.time() - start_time
        token_count = int(len(result.final_answer) / 1.5) if result.final_answer else 0

        # 写助手回复
        assistant_message = create_message(
            db,
            conversation_id=conversation_id,
            role="assistant",
            content=result.final_answer,
            model_used=config.ANSWER_MODEL,
            token_count=token_count,
            latency=latency
        )
        conv_cache.append(conversation_id, "assistant", result.final_answer)

        create_usage_stats(
            db,
            user_id=current_user.id,
            api_call_type="chat",
            model_name=config.ANSWER_MODEL,
            tokens_used=token_count,
            latency=latency
        )

        return assistant_message

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(
    message: str = None,
    conversation_id: Optional[int] = None,
    mode: str = "agentic",
    knowledge_base_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """流式聊天接口"""

    async def event_generator():
        if conversation_id:
            conversation = get_conversation(db, conversation_id, current_user.id)
            if not conversation:
                conversation = create_conversation(db, current_user.id, "新对话", mode)
        else:
            conversation = create_conversation(db, current_user.id, "新对话", mode)

        # 获取历史消息构建上下文
        history_context = ""
        if conversation_id:
            messages = get_conversation_messages(db, conversation_id)
            if messages:
                history_msgs = messages[-20:] if len(messages) > 20 else messages
                history_parts = []
                for msg in history_msgs:
                    role_label = "用户" if msg.role == "user" else "助手"
                    history_parts.append(f"{role_label}: {msg.content}")
                history_context = "\n".join(history_parts)

        create_message(db, conversation.id, "user", message)

        full_response = ""

        # 构建知识库过滤参数
        kb_filter = None
        if knowledge_base_id:
            kb = get_knowledge_base_by_id(db, knowledge_base_id, current_user.id)
            if kb and kb.is_indexed:
                kb_filter = f"kb_{current_user.id}_{kb.id}"

        try:
            last_sent_length = 0
            for node_name, node_state in agent_workflow.run_streaming(
                message, mode=mode, history_context=history_context, knowledge_base_id=kb_filter
            ):
                if "final_answer" in node_state:
                    full_response = node_state.get("final_answer", "")
                    # 只发送新增的内容（增量）
                    if len(full_response) > last_sent_length:
                        new_content = full_response[last_sent_length:]
                        last_sent_length = len(full_response)
                        yield f"data: {json.dumps({'content': new_content, 'done': False})}\n\n"

            # 计算token数（中文按1.5字符/token估算）
            token_count = int(len(full_response) / 1.5) if full_response else 0
            create_message(
                db, conversation.id, "assistant", full_response,
                model_used=config.ANSWER_MODEL,
                token_count=token_count
            )

            yield f"data: {json.dumps({'content': '', 'done': True, 'conversation_id': conversation.id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==================== 统计路由 ====================

@app.get("/api/stats")
async def get_usage_stats(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户使用统计"""
    stats = get_user_stats(db, current_user.id, days)
    return stats

# ==================== RAG路由 ====================

@app.post("/api/rag/query")
async def rag_query(
    request_data: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """直接RAG查询"""

    # 构建知识库过滤参数
    kb_filter = None
    if request_data.knowledge_base_id:
        kb = get_knowledge_base_by_id(db, request_data.knowledge_base_id, current_user.id)
        if kb and kb.is_indexed:
            kb_filter = f"kb_{current_user.id}_{kb.id}"

    # 运行 Agent 工作流
    result = agent_workflow.run(
        request_data.query,
        mode=request_data.mode,
        knowledge_base_id=kb_filter
    )

    return {
        "query": result.original_query,
        "answer": result.final_answer,
        "sub_questions": result.sub_questions,
        "sources": [
            {
                "content": s.content[:200],
                "score": s.score,
                "metadata": s.metadata
            }
            for s in result.sources[:request_data.top_k]
        ]
    }

# ==================== 文档上传路由（已废弃，保留兼容） ====================

@app.post("/api/documents/upload")
async def upload_document_deprecated():
    """文档上传接口已废弃，请使用知识库管理"""
    raise HTTPException(
        status_code=410,
        detail="文档上传已废弃，请使用 /api/knowledge-bases 管理知识库"
    )

@app.get("/api/documents")
async def list_documents_deprecated():
    """文档列表接口已废弃，请使用知识库管理"""
    raise HTTPException(
        status_code=410,
        detail="文档列表已废弃，请使用 /api/knowledge-bases 管理知识库"
    )
