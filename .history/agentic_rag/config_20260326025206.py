"""
配置文件 - 集中管理所有配置项
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent

# ==================== 数据库配置 ====================
# MySQL 配置（默认使用，替代 SQLite）
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "agentic_rag")

# 数据库 URL（优先使用环境变量，否则根据 MySQL 配置构建）
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

# ==================== Elasticsearch 配置 ====================
# ES 连接配置
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_USER = os.getenv("ES_USER", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "agentic_rag_knowledge")

# ES 混合检索配置
ES_VECTOR_DIM = int(os.getenv("ES_VECTOR_DIM", "768"))  # 向量维度
ES_TOP_K = int(os.getenv("ES_TOP_K", "20"))  # ES 初步检索数量
ES_BM25_WEIGHT = float(os.getenv("ES_BM25_WEIGHT", "0.3"))  # BM25 分数权重
ES_VECTOR_WEIGHT = float(os.getenv("ES_VECTOR_WEIGHT", "0.7"))  # 向量分数权重
ES_MIN_SCORE = float(os.getenv("ES_MIN_SCORE", "0.5"))  # 最小相似度阈值

# ==================== 模型来源配置 ====================
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

# ==================== Ollama模型配置 ====================
DECOMPOSE_MODEL = os.getenv("DECOMPOSE_MODEL", "deepseek-r1:8b")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen2.5:3b")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ==================== HuggingFace本地模型配置 ====================
HF_DECOMPOSE_MODEL = os.getenv("HF_DECOMPOSE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
HF_ANSWER_MODEL = os.getenv("HF_ANSWER_MODEL", "microsoft/Phi-3-mini-4k-instruct")
HF_SYNTHESIS_MODEL = os.getenv("HF_SYNTHESIS_MODEL", "microsoft/Phi-3-mini-4k-instruct")
HF_MODEL_CACHE_DIR = os.getenv("HF_MODEL_CACHE_DIR", None)

# ==================== 知识库配置 ====================
# 知识库根目录（可配置多个知识库路径）
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base"))
# 知识库扫描的文件类型
KNOWLEDGE_FILE_TYPES = os.getenv("KNOWLEDGE_FILE_TYPES", ".pdf,.txt,.md,.docx,.doc").split(",")

# ==================== 文档处理配置 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ==================== 检索配置 ====================
TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "10"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# ==================== Embedding模型配置 ====================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/root/autodl-tmp/models/maidalun/bce-embedding-base_v1")

# ==================== 重排序模型配置 ====================
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "/root/autodl-tmp/models/BAAI/bge-reranker-v2-m3")

# ==================== API配置 ====================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-12345678")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ==================== FastAPI配置 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ==================== Streamlit配置 ====================
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ==================== 日志配置 ====================
LOG_FILE = str(BASE_DIR / "logs" / "agentic_rag.log")

# ==================== Summary Memory配置 ====================
SUMMARY_LLM_MODEL = os.getenv("SUMMARY_LLM_MODEL", "qwen2.5:3b")
SUMMARY_LLM_TEMPERATURE = float(os.getenv("SUMMARY_LLM_TEMPERATURE", "0.3"))
SUMMARY_MAX_TOKEN_LIMIT = int(os.getenv("SUMMARY_MAX_TOKEN_LIMIT", "2000"))

# ==================== Mock LLM配置 ====================
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))

# ==================== 旧配置兼容（已废弃） ====================
# 以下配置保留用于向后兼容，实际不再使用
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
PDF_DIRECTORY = str(BASE_DIR / "documents")
