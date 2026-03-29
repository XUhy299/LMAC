FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY agentic_rag/ ./agentic_rag/
COPY .env.example .env

# 暴露端口
EXPOSE 8000 8501 3000

# 启动命令
CMD ["python", "-m", "agentic_rag.main", "--mock"]
