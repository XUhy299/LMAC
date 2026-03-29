"""
对话历史内存缓存 - Write-Through Cache 模式

设计：
  - 写消息时同时写 DB 和缓存
  - 读历史时直接从缓存取，不查 DB
  - 每个会话最多保留 MAX_ROUNDS 轮（=MAX_ROUNDS*2 条消息）
  - 超出时自动从左侧剔除最旧的消息
  - 冷启动时（缓存为空）从 DB 加载后写入缓存
"""

from collections import deque
from typing import Dict, List, Optional
import agentic_rag.config as config


class ConversationCache:
    def __init__(self, max_rounds: int = None):
        # 最大轮数，每轮 = 1条user + 1条assistant
        self.max_rounds: int = max_rounds or getattr(config, "MAX_HISTORY_ROUNDS", 10)
        # {conversation_id: deque([{"role": ..., "content": ...}, ...])}
        self._store: Dict[int, deque] = {}
        # 标记哪些 conversation_id 已完成冷启动加载
        self._loaded: set = set()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def append(self, conversation_id: int, role: str, content: str) -> None:
        """追加一条消息，超过 max_rounds 轮自动剔除最旧的"""
        if conversation_id not in self._store:
            self._store[conversation_id] = deque()

        self._store[conversation_id].append({"role": role, "content": content})

        # 维护最大条数 = max_rounds * 2
        max_msgs = self.max_rounds * 2
        while len(self._store[conversation_id]) > max_msgs:
            self._store[conversation_id].popleft()

    def load_from_db(self, conversation_id: int, db_messages: list) -> None:
        """
        冷启动：将 DB 消息加载进缓存（只在该 conv 第一次被访问时调用）。
        db_messages 是 SQLAlchemy Message 对象列表，按时间升序。
        """
        if conversation_id in self._loaded:
            return

        self._store[conversation_id] = deque()
        for msg in db_messages:
            self._store[conversation_id].append({
                "role": msg.role,
                "content": msg.content
            })

        # 初始也裁剪到 max_rounds 范围
        max_msgs = self.max_rounds * 2
        while len(self._store[conversation_id]) > max_msgs:
            self._store[conversation_id].popleft()

        self._loaded.add(conversation_id)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def is_loaded(self, conversation_id: int) -> bool:
        """判断该会话是否已完成冷启动加载"""
        return conversation_id in self._loaded

    def get_messages(self, conversation_id: int) -> List[dict]:
        """返回该会话缓存中的所有消息列表"""
        return list(self._store.get(conversation_id, []))

    def get_context_string(self, conversation_id: int, exclude_last: int = 1) -> str:
        """
        构建传给 LLM 的历史上下文字符串。

        exclude_last: 排除末尾 N 条（默认排除最后 1 条，
                      即刚写入的当前用户消息，避免重复出现在 context 里）

        返回格式：
            用户: ...
            助手: ...
            用户: ...
        """
        msgs = list(self._store.get(conversation_id, []))

        if exclude_last:
            msgs = msgs[:-exclude_last]

        if not msgs:
            return ""

        parts = []
        for msg in msgs:
            role_label = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role_label}: {msg['content']}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def evict(self, conversation_id: int) -> None:
        """删除某个会话的缓存（如对话被删除时调用）"""
        self._store.pop(conversation_id, None)
        self._loaded.discard(conversation_id)

    def clear(self) -> None:
        """清空所有缓存"""
        self._store.clear()
        self._loaded.clear()


# 全局单例，在 api.py 中 import 使用
conv_cache = ConversationCache()
