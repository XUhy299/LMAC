# API 文档

## 认证

所有 API 接口（除登录注册外）都需要在请求头中携带 JWT Token：

```
Authorization: Bearer <access_token>
```

## 接口列表

### 认证接口

#### POST /api/auth/register
用户注册

**请求体：**
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**响应：**
```json
{
  "id": 1,
  "username": "string",
  "email": "string",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00"
}
```

#### POST /api/auth/login
用户登录

**请求体（form-data）：**
- username: string
- password: string

**响应：**
```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

### 知识库接口

#### POST /api/knowledge-bases
创建知识库

**请求体：**
```json
{
  "name": "string",
  "path": "/path/to/kb",
  "description": "string",
  "file_types": ".pdf,.txt,.md"
}
```

#### GET /api/knowledge-bases
获取知识库列表

#### POST /api/knowledge-bases/{kb_id}/scan
扫描知识库并建立索引

### 对话接口

#### POST /api/conversations
创建对话

**请求体：**
```json
{
  "title": "string",
  "mode": "agentic"
}
```

#### GET /api/conversations
获取对话列表

#### POST /api/conversations/{conversation_id}/messages
发送消息

**请求体：**
```json
{
  "content": "string",
  "mode": "agentic",
  "knowledge_base_id": 1
}
```

#### POST /api/chat/stream
流式对话

**查询参数：**
- message: string
- conversation_id: int (可选)
- mode: string (默认 agentic)
- knowledge_base_id: int (可选)

**响应：** SSE 流

```
data: {"content": "...", "done": false}
data: {"content": "...", "done": true, "conversation_id": 1}
```

### RAG 接口

#### POST /api/rag/query
直接 RAG 查询

**请求体：**
```json
{
  "query": "string",
  "mode": "agentic",
  "top_k": 5,
  "knowledge_base_id": 1
}
```

**响应：**
```json
{
  "query": "string",
  "answer": "string",
  "sub_questions": ["..."],
  "sources": [
    {
      "content": "...",
      "score": 0.95,
      "metadata": {}
    }
  ]
}
```
