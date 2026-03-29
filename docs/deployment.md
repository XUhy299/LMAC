# 部署指南

## Docker 部署

### 使用 Docker Compose（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 手动构建 Docker 镜像

```bash
# 构建镜像
docker build -t agentic-rag .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -e MOCK_LLM=true \
  -v $(pwd)/knowledge_base:/app/knowledge_base \
  agentic-rag
```

## 生产环境部署

### 使用 Gunicorn + Nginx

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务
gunicorn -w 4 -k uvicorn.workers.UvicornWorker agentic_rag.api:app -b 0.0.0.0:8000
```

### Nginx 配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 环境变量配置

生产环境需要配置的变量：

```bash
# 必需
SECRET_KEY=your-secret-key-here
MYSQL_HOST=your-mysql-host
MYSQL_PASSWORD=your-mysql-password
ES_HOST=your-es-host

# 可选（根据需求）
OLLAMA_BASE_URL=http://your-ollama-server:11434
EMBEDDING_MODEL=your-embedding-model
RERANKER_MODEL=your-reranker-model
```
