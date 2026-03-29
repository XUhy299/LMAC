
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from passlib.context import CryptContext
from pydantic import BaseModel
from agentic_rag.config import DATABASE_URL


# 创建数据库引擎（支持自动降级到 SQLite）
def _create_engine_with_fallback():
    """创建数据库引擎，如果 MySQL 失败则回退到 SQLite"""
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # MySQL 连接保活
            pool_recycle=3600,   # 1小时后回收连接
            echo=False
        )
        # 测试连接
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, False  # False = 不是 SQLite 回退
    except Exception as e:
        # 回退到 SQLite
        import os
        sqlite_path = os.path.join(os.path.dirname(__file__), "data", "agentic_rag.db")
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        sqlite_url = f"sqlite:///{sqlite_path}"
        print(f"[DB] MySQL 连接失败，回退到 SQLite: {sqlite_url}")
        engine = create_engine(sqlite_url, echo=False)
        return engine, True  # True = 是 SQLite 回退

engine, _using_sqlite_fallback = _create_engine_with_fallback()

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ==================== SQLAlchemy ORM模型 ====================

class User(Base):
    """用户模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # 关系
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", back_populates="user", cascade="all, delete-orphan")


class KnowledgeBase(Base):
    """知识库模型 - 存储本地知识库路径和配置"""
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)  # 知识库名称
    path = Column(String(500), nullable=False)  # 本地知识库路径
    description = Column(Text)  # 知识库描述
    file_types = Column(String(100), default=".pdf,.txt,.md")  # 扫描的文件类型
    file_count = Column(Integer, default=0)  # 文件数量
    chunk_count = Column(Integer, default=0)  # 分块数量
    is_indexed = Column(Boolean, default=False)  # 是否已建立索引
    es_index_name = Column(String(100))  # ES 索引名称
    last_scan_at = Column(DateTime)  # 上次扫描时间
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # 关系
    user = relationship("User", back_populates="knowledge_bases")


class Document(Base):
    """文档模型 - 知识库中的文件（已废弃，保留用于兼容）"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    content_text = Column(Text)
    chunk_count = Column(Integer, default=0)
    is_indexed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # 关系
    user = relationship("User", back_populates="documents")
    chunks = relationship("TextChunk", back_populates="document", cascade="all, delete-orphan")


class TextChunk(Base):
    """文本块模型 - 存储文档分块信息"""
    __tablename__ = "text_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # 块索引
    content = Column(Text, nullable=False)  # 块内容
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # 关系
    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    """对话模型 - 存储对话会话"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200))  # 对话标题
    mode = Column(String(20), default="agentic")  # 模式：agentic 或 traditional
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """消息模型 - 存储对话中的每条消息"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)  # 消息内容
    model_used = Column(String(50))  # 使用的模型
    token_count = Column(Integer)  # token数量
    latency = Column(Float)  # 响应延迟（秒）
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")


class UsageStats(Base):
    """使用统计模型"""
    __tablename__ = "usage_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_call_type = Column(String(50))  # API调用类型
    model_name = Column(String(50))  # 使用的模型
    tokens_used = Column(Integer, default=0)
    latency = Column(Float)  # 延迟（秒）
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==================== Pydantic模型（API请求/响应） ====================

class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int
    is_indexed: bool


class MessageCreate(BaseModel):
    conversation_id: int
    role: str
    content: str
    model_used: Optional[str] = None
    token_count: Optional[int] = None
    latency: Optional[float] = None


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    model_used: Optional[str]
    token_count: Optional[int]
    latency: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: Optional[str]
    mode: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    mode: str = "agentic"


class KnowledgeBaseResponse(BaseModel):
    id: int
    name: str
    path: str
    description: Optional[str]
    file_types: str
    file_count: int
    chunk_count: int
    is_indexed: bool
    es_index_name: Optional[str]
    last_scan_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class KnowledgeBaseCreate(BaseModel):
    name: str
    path: str
    description: Optional[str] = None
    file_types: Optional[str] = ".pdf,.txt,.md"


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    file_types: Optional[str] = None


# ==================== CRUD操作函数 ====================

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    # bcrypt 限制密码最大长度为 72 字节
    # 先将字符串编码为字节以准确计算长度（处理中文等多字节字符）
    password_bytes = password.encode('utf-8')
    
    if len(password_bytes) > 72:
        # 截断超过 72 字节的部分
        password_bytes = password_bytes[:72]
        # 解码回字符串，errors='ignore' 防止截断处正好是一个多字节字符的中间导致解码失败
        password = password_bytes.decode('utf-8', errors='ignore')
    
    return pwd_context.hash(password)


def create_user(db: Session, username: str, email: str, password: str) -> User:
    """创建用户"""
    hashed_password = get_password_hash(password)
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """根据用户名获取用户"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """根据ID获取用户"""
    return db.query(User).filter(User.id == user_id).first()


def create_document(
    db: Session,
    user_id: int,
    filename: str,
    file_path: str,
    file_size: int,
    content_text: str,
    chunk_count: int
) -> Document:
    """创建文档记录"""
    db_document = Document(
        user_id=user_id,
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        content_text=content_text,
        chunk_count=chunk_count,
        is_indexed=True
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document


def get_user_documents(db: Session, user_id: int) -> List[Document]:
    """获取用户的所有文档"""
    return db.query(Document).filter(Document.user_id == user_id).all()


def get_document_by_id(db: Session, doc_id: int, user_id: int) -> Optional[Document]:
    """根据ID获取文档"""
    return db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first()


def delete_document(db: Session, doc_id: int, user_id: int) -> bool:
    """删除文档"""
    doc = get_document_by_id(db, doc_id, user_id)
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False


def create_conversation(db: Session, user_id: int, title: str, mode: str = "agentic") -> Conversation:
    """创建对话"""
    db_conversation = Conversation(
        user_id=user_id,
        title=title,
        mode=mode
    )
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation


def get_conversation(db: Session, conversation_id: int, user_id: int) -> Optional[Conversation]:
    """获取对话"""
    return db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()


def get_user_conversations(db: Session, user_id: int, limit: int = 50) -> List[Conversation]:
    """获取用户的所有对话"""
    return db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).limit(limit).all()


def create_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
    model_used: str = None,
    token_count: int = None,
    latency: float = None
) -> Message:
    """创建消息"""
    db_message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model_used=model_used,
        token_count=token_count,
        latency=latency
    )
    db.add(db_message)
    
    # 更新对话的更新时间
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(db_message)
    return db_message


def get_conversation_messages(db: Session, conversation_id: int) -> List[Message]:
    """获取对话的所有消息"""
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).all()


def create_usage_stats(
    db: Session,
    user_id: int,
    api_call_type: str,
    model_name: str,
    tokens_used: int,
    latency: float
) -> UsageStats:
    """创建使用统计"""
    db_stats = UsageStats(
        user_id=user_id,
        api_call_type=api_call_type,
        model_name=model_name,
        tokens_used=tokens_used,
        latency=latency
    )
    db.add(db_stats)
    db.commit()
    return db_stats


def get_user_stats(db: Session, user_id: int, days: int = 7) -> dict:
    """获取用户使用统计"""
    from datetime import timedelta
    from sqlalchemy import func

    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    # 总对话数
    total_conversations = db.query(func.count(Conversation.id)).filter(
        Conversation.user_id == user_id,
        Conversation.created_at >= start_date
    ).scalar()

    # 总消息数
    total_messages = db.query(func.count(Message.id)).join(Conversation).filter(
        Conversation.user_id == user_id,
        Message.created_at >= start_date
    ).scalar()

    # 总token数
    total_tokens = db.query(func.sum(Message.token_count)).join(Conversation).filter(
        Conversation.user_id == user_id,
        Message.created_at >= start_date
    ).scalar() or 0

    # 平均延迟
    avg_latency = db.query(func.avg(Message.latency)).join(Conversation).filter(
        Conversation.user_id == user_id,
        Message.created_at >= start_date
    ).scalar() or 0

    # 知识库统计
    total_knowledge_bases = db.query(func.count(KnowledgeBase.id)).filter(
        KnowledgeBase.user_id == user_id
    ).scalar()

    total_chunks = db.query(func.sum(KnowledgeBase.chunk_count)).filter(
        KnowledgeBase.user_id == user_id
    ).scalar() or 0

    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "avg_latency": round(avg_latency, 2),
        "total_knowledge_bases": total_knowledge_bases,
        "total_chunks": total_chunks
    }


# ==================== KnowledgeBase CRUD ====================

def create_knowledge_base(
    db: Session,
    user_id: int,
    name: str,
    path: str,
    description: str = None,
    file_types: str = ".pdf,.txt,.md"
) -> KnowledgeBase:
    """创建知识库"""
    db_kb = KnowledgeBase(
        user_id=user_id,
        name=name,
        path=path,
        description=description,
        file_types=file_types,
        file_count=0,
        chunk_count=0,
        is_indexed=False
    )
    db.add(db_kb)
    db.commit()
    db.refresh(db_kb)
    return db_kb


def get_knowledge_base_by_id(db: Session, kb_id: int, user_id: int) -> Optional[KnowledgeBase]:
    """根据ID获取知识库"""
    return db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.user_id == user_id
    ).first()


def get_user_knowledge_bases(db: Session, user_id: int) -> List[KnowledgeBase]:
    """获取用户的所有知识库"""
    return db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == user_id
    ).order_by(KnowledgeBase.created_at.desc()).all()


def update_knowledge_base(
    db: Session,
    kb_id: int,
    user_id: int,
    name: str = None,
    description: str = None,
    file_types: str = None
) -> Optional[KnowledgeBase]:
    """更新知识库"""
    kb = get_knowledge_base_by_id(db, kb_id, user_id)
    if not kb:
        return None

    if name is not None:
        kb.name = name
    if description is not None:
        kb.description = description
    if file_types is not None:
        kb.file_types = file_types

    db.commit()
    db.refresh(kb)
    return kb


def update_knowledge_base_index_status(
    db: Session,
    kb_id: int,
    user_id: int,
    is_indexed: bool,
    file_count: int = None,
    chunk_count: int = None,
    es_index_name: str = None
) -> Optional[KnowledgeBase]:
    """更新知识库索引状态"""
    kb = get_knowledge_base_by_id(db, kb_id, user_id)
    if not kb:
        return None

    kb.is_indexed = is_indexed
    if file_count is not None:
        kb.file_count = file_count
    if chunk_count is not None:
        kb.chunk_count = chunk_count
    if es_index_name is not None:
        kb.es_index_name = es_index_name
    kb.last_scan_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(kb)
    return kb


def delete_knowledge_base(db: Session, kb_id: int, user_id: int) -> bool:
    """删除知识库"""
    kb = get_knowledge_base_by_id(db, kb_id, user_id)
    if kb:
        db.delete(kb)
        db.commit()
        return True
    return False
