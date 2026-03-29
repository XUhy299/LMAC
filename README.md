# Agentic-RAG

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Agentic-RAG 是一个智能检索增强生成系统，结合了 Agent 工作流和 RAG 技术，支持 Elasticsearch 混合检索、重排序和多模型对话。

## 特性

- **Agentic 工作流**：使用 LangGraph 实现智能 RAG 流程，支持问题分解、多轮检索和答案综合
- **混合检索**：结合 BM25 和向量相似度的 Elasticsearch 混合检索
- **智能重排序**：使用 CrossEncoder 对检索结果进行重排序
- **多前端支持**：支持原生 JavaScript 前端和 Streamlit 前端
- **知识库管理**：支持多知识库管理和自动索引
- **用户认证**：JWT 认证系统，支持用户注册登录
- **Mock 模式**：无需配置 LLM 和 ES 即可快速测试

## 架构

```
Agentic-RAG/
├── agentic_rag/          # 主包
│   ├── api/              # FastAPI 后端
│   ├── core/             # 核心功能
│   │   ├── agent.py      # Agent 工作流
│   │   ├── cache.py      # 对话缓存
│   │   └── ...
│   ├── db/               # 数据库模型
│   ├── frontend/         # 前端文件
│   └── config.py         # 配置文件
├── tests/                # 测试代码
├── docs/                 # 文档
└── scripts/              # 工具脚本
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# 安装依赖
pip install -r requirements.txt
```

### 配置

复制环境变量模板并修改：

```bash
cp .env.example .env
```

编辑 `.env` 文件配置你的环境：

```env
# MySQL 数据库配置
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=agentic_rag

# Elasticsearch 配置
ES_HOST=localhost
ES_PORT=9200

# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434
DECOMPOSE_MODEL=deepseek-r1:8b
ANSWER_MODEL=qwen2.5:3b
```

### 启动

```bash
# 启动完整系统（默认 JS 前端）
python -m agentic_rag.main

# 启动 Streamlit 前端
python -m agentic_rag.main --frontend streamlit

# 仅启动 API 后端
python -m agentic_rag.main api

# Mock 模式（无需 Ollama 和 ES）
python -m agentic_rag.main --mock
```

## 使用 Docker

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d
```

## API 文档

启动后访问：`http://localhost:8000/docs`

### 主要接口

- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `POST /api/knowledge-bases` - 创建知识库
- `POST /api/knowledge-bases/{id}/scan` - 扫描知识库
- `POST /api/conversations` - 创建对话
- `POST /api/conversations/{id}/messages` - 发送消息
- `POST /api/rag/query` - 直接 RAG 查询

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

[MIT](LICENSE)
