"""
入口脚本 - 一键启动 Agentic-RAG 系统
用法:
  python main.py                    # 启动完整系统 (FastAPI + JS前端)
  python main.py --frontend streamlit  # 启动完整系统 (FastAPI + Streamlit前端)
  python main.py api                # 仅启动 FastAPI 后端
  python main.py frontend           # 仅启动 JS 前端静态服务
  python main.py streamlit          # 仅启动 Streamlit 前端
  python main.py init               # 仅初始化数据库

环境变量:
  MOCK_LLM=true           # 跳过真实 LLM，使用 mock 响应（无需 Ollama）
"""
import os
import sys

# 必须在 import config 之前设置，否则 config.MOCK_LLM 会固化为 False
if "--mock" in sys.argv:
    os.environ["MOCK_LLM"] = "true"

import functools
import threading
import subprocess
import argparse
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

from agentic_rag import config
from agentic_rag.db import init_db


# ==================== 目录初始化 ====================

def create_directories():
    """创建必要的目录"""
    dirs = [
        os.path.dirname(config.LOG_FILE) if config.LOG_FILE else None,
        os.path.join(os.path.dirname(__file__), "data"),  # SQLite 数据目录
    ]
    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


def init_database():
    """初始化数据库（自动处理 MySQL/SQLite）"""
    print("=" * 50)
    print("初始化数据库")
    print("=" * 50)
    try:
        init_db()
        from agentic_rag.db import _using_sqlite_fallback
        if _using_sqlite_fallback:
            print("[DB] 使用 SQLite 本地数据库（无需 MySQL）")
        else:
            print("[DB] 使用 MySQL 数据库")
        print("数据库初始化完成！\n")
    except Exception as e:
        print(f"[ERROR] 数据库初始化失败: {e}")
        raise


# ==================== 后端服务 ====================

def start_api_server():
    """启动 FastAPI 后端（阻塞）"""
    print("=" * 50)
    print(f"启动 API 服务: http://{config.API_HOST}:{config.API_PORT}")
    print(f"Swagger 文档:  http://localhost:{config.API_PORT}/docs")
    print("=" * 50)
    import uvicorn
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False)


def start_api_in_background():
    """在子进程中启动 FastAPI，返回 Popen 对象"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "agentic_rag.api:app",
         "--host", config.API_HOST,
         "--port", str(config.API_PORT)],
        cwd=base_dir,
        env=env
    )
    return proc


# ==================== JS 前端服务 ====================

FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))


class QuietHandler(SimpleHTTPRequestHandler):
    """静默日志的静态文件服务器"""
    def log_message(self, fmt, *args):
        pass  # 不输出每条请求日志


def start_frontend_server():
    """在当前线程中启动前端静态服务（阻塞）"""
    frontend_dir = str(Path(__file__).parent / "frontend")
    handler = functools.partial(QuietHandler, directory=frontend_dir)
    server = HTTPServer(("0.0.0.0", FRONTEND_PORT), handler)
    print(f"前端地址: http://localhost:{FRONTEND_PORT}")
    server.serve_forever()


def start_frontend_in_thread():
    """在后台线程中启动前端静态服务"""
    t = threading.Thread(target=start_frontend_server, daemon=True)
    t.start()
    return t


# ==================== Streamlit 前端服务 ====================

def start_streamlit_server():
    """启动 Streamlit 前端（阻塞）"""
    print("=" * 50)
    print(f"启动 Streamlit 前端: http://localhost:{config.STREAMLIT_PORT}")
    print("=" * 50)
    import streamlit.web.cli as stcli
    frontend_dir = str(Path(__file__).parent)
    sys.argv = ["streamlit", "run", str(Path(frontend_dir) / "frontend" / "streamlit_app.py"),
                "--server.port", str(config.STREAMLIT_PORT),
                "--server.headless", "true"]
    stcli.main()


def start_streamlit_in_background():
    """在子进程中启动 Streamlit，返回 Popen 对象"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(Path(base_dir) / "frontend" / "streamlit_app.py"),
         "--server.port", str(config.STREAMLIT_PORT),
         "--server.headless", "true"],
        cwd=base_dir,
        env=env
    )
    return proc


# ==================== 一键启动 ====================

def start_all(frontend_type="js"):
    """启动完整系统：FastAPI 后端 + 前端（js 或 streamlit）"""
    mock = os.getenv("MOCK_LLM", "false").lower() == "true"

    print("\n" + "=" * 50)
    print("Agentic-RAG 系统启动中")
    if mock:
        print("[MOCK 模式] 无需 Ollama / ES / MySQL")
    print(f"[前端类型: {frontend_type}]")
    print("=" * 50 + "\n")

    init_database()
    create_directories()

    if frontend_type == "js":
        # 启动 JS 前端（后台线程）
        print(f"[1/2] 启动 JS 前端服务   -> http://localhost:{FRONTEND_PORT}")
        start_frontend_in_thread()
    elif frontend_type == "streamlit":
        # 启动 Streamlit（子进程）
        print(f"[1/2] 启动 Streamlit前端 -> http://localhost:{config.STREAMLIT_PORT}")
        start_streamlit_in_background()

    # 等一下确保端口绑定
    time.sleep(0.5)

    print(f"[2/2] 启动 API 后端     -> http://localhost:{config.API_PORT}")
    print(f"      Swagger 文档      -> http://localhost:{config.API_PORT}/docs\n")
    print("=" * 50)
    if frontend_type == "js":
        print("系统就绪！在浏览器打开: http://localhost:" + str(FRONTEND_PORT))
    elif frontend_type == "streamlit":
        print("系统就绪！在浏览器打开: http://localhost:" + str(config.STREAMLIT_PORT))
    print("按 Ctrl+C 停止服务")
    print("=" * 50 + "\n")

    try:
        import uvicorn
        uvicorn.run("agentic_rag.api:app", host=config.API_HOST, port=config.API_PORT, reload=False)
    except KeyboardInterrupt:
        print("\n系统已停止。")


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Agentic-RAG 系统启动器")
    parser.add_argument(
        "mode",
        choices=["all", "api", "frontend", "streamlit", "init"],
        default="all",
        nargs="?",
        help="all=完整系统(默认), api=仅后端, frontend=仅JS前端, streamlit=仅Streamlit前端, init=初始化DB"
    )
    parser.add_argument(
        "--frontend",
        choices=["js", "streamlit"],
        default="js",
        help="选择前端类型: js=静态HTML前端(默认), streamlit=Streamlit前端"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 Mock LLM 模式（无需 Ollama 和本地模型）"
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Agentic-RAG 智能问答系统")
    print("=" * 50 + "\n")

    if args.mode == "all":
        start_all(frontend_type=args.frontend)
    elif args.mode == "api":
        init_database()
        create_directories()
        start_api_server()
    elif args.mode == "frontend":
        print(f"启动 JS 前端 -> http://localhost:{FRONTEND_PORT}")
        print("按 Ctrl+C 停止")
        start_frontend_server()
    elif args.mode == "streamlit":
        print(f"启动 Streamlit 前端 -> http://localhost:{config.STREAMLIT_PORT}")
        print("按 Ctrl+C 停止")
        start_streamlit_server()
    elif args.mode == "init":
        init_database()


if __name__ == "__main__":
    main()
