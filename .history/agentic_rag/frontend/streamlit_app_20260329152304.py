
import streamlit as st
import requests
import json
import time
import os
from typing import Optional

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Agentic-RAG 智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []


def api_request(method: str, endpoint: str, data: dict = None, files: dict = None):
    """API请求封装"""
    headers = {}
    if st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, headers=headers, timeout=60)
            else:
                response = requests.post(url, json=data, headers=headers, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            return None
        
        if response.status_code in [200, 201]:
            return response.json()
        elif response.status_code == 401:
            st.error("登录已过期，请重新登录")
            st.session_state.access_token = None
            return None
        else:
            st.error(f"请求失败: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"无法连接到API服务器，请确保API服务运行在 {API_BASE_URL}")
        return None
    except Exception as e:
        st.error(f"请求错误: {str(e)}")
        return None


def login_form():
    """登录/注册表单"""
    st.title("🤖 Agentic-RAG 智能问答系统")
    
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submit = st.form_submit_button("登录", use_container_width=True)
            
            if submit:
                if username and password:
                    response = requests.post(
                        f"{API_BASE_URL}/api/auth/login",
                        data={"username": username, "password": password},
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    if response.status_code == 200:
                        token_data = response.json()
                        st.session_state.access_token = token_data["access_token"]
                        st.session_state.username = username
                        st.success("登录成功！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")
                else:
                    st.warning("请输入用户名和密码")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("用户名", key="register_username")
            email = st.text_input("邮箱(可选)", key="register_email")
            new_password = st.text_input("密码", type="password", key="register_password")
            confirm_password = st.text_input("确认密码", type="password", key="register_confirm")
            submit = st.form_submit_button("注册", use_container_width=True)
            
            if submit:
                if new_username and new_password:
                    if new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                    else:
                        response = api_request("POST", "/api/auth/register", {
                            "username": new_username,
                            "email": email,
                            "password": new_password
                        })
                        if response:
                            st.success("注册成功，请登录！")
                else:
                    st.warning("请填写完整信息")


def sidebar():
    """侧边栏：负责导航与对话历史管理"""
    with st.sidebar:
        st.title("📚 功能菜单")
        
        if st.session_state.access_token:
            st.write(f"👤 用户: {st.session_state.username}")
            
            # 退出登录逻辑：需完整清理会话状态
            if st.button("退出登录", use_container_width=True):
                st.session_state.access_token = None
                st.session_state.username = None
                st.session_state.conversation_id = None
                st.session_state.messages = []
                st.rerun()
            
            st.divider()
            
            # 选择核心功能
            menu = st.selectbox(
                "选择功能",
                ["💬 智能问答", "📁 文档管理", "📊 使用统计"]
            )
            
            # --- 关键修改：找回对话管理 ---
            # 只有在“智能问答”模式下，才在侧边栏渲染对话历史和“新建对话”按钮
            # 这样历史列表就回到了左侧，而聊天框在右侧主区域
            if menu == "💬 智能问答":
                show_conversations()
            
            # 注意：不要在这里调用 show_document_manager() 或 show_stats()
            # 它们已经在 main() 函数中被渲染到主界面了
            
            return menu
        else:
            st.info("请先登录")
            return None


def show_conversations():
    """对话管理"""
    st.subheader("💬 对话历史")
    
    if st.button("➕ 新建对话", use_container_width=True):
        response = api_request("POST", "/api/conversations", {
            "title": "新对话",
            "mode": "agentic"
        })
        if response:
            st.session_state.conversation_id = response["id"]
            st.session_state.messages = []
            st.rerun()
    
    conversations = api_request("GET", "/api/conversations")
    if conversations:
        st.write("**历史对话:**")
        for conv in conversations[:10]:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📝 {conv.get('title', '新对话')[:20]}", key=f"conv_{conv['id']}"):
                    st.session_state.conversation_id = conv["id"]
                    load_messages(conv["id"])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_conv_{conv['id']}"):
                    api_request("DELETE", f"/api/conversations/{conv['id']}")
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                        st.session_state.messages = []
                    st.rerun()


def load_messages(conversation_id: int):
    """加载对话消息"""
    messages = api_request("GET", f"/api/conversations/{conversation_id}/messages")
    if messages:
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
    else:
        st.session_state.messages = []


def show_document_manager():
    """文档管理 - 已改为知识库管理"""
    st.subheader("📁 知识库管理")

    st.info("📢 文档上传功能已迁移到知识库管理。请使用左侧菜单中的'知识库'功能来管理您的文档。")

    # 显示当前知识库列表（如果有API支持）
    st.write("**当前知识库:**")
    kbs = api_request("GET", "/api/knowledge-bases")
    if kbs:
        for kb in kbs:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📁 {kb.get('name', '未命名')}")
                    st.caption(f"路径: {kb.get('path', 'N/A')}")
                with col2:
                    st.write(f"{'✅ 已索引' if kb.get('is_indexed') else '⏳ 未索引'}")
                st.divider()
    else:
        st.info("暂无知识库，请前往知识库页面创建")


def show_stats():
    """使用统计"""
    st.subheader("📊 使用统计")
    
    stats = api_request("GET", "/api/stats")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("对话数", stats.get("total_conversations", 0))
        with col2:
            st.metric("消息数", stats.get("total_messages", 0))
        with col3:
            st.metric("Token数", stats.get("total_tokens", 0))
        with col4:
            st.metric("平均延迟", f"{stats.get('avg_latency', 0):.2f}s")
    else:
        st.info("暂无统计数据")


def chat_interface():
    """聊天界面"""
    st.title("💬 Agentic-RAG 智能问答")
    
    mode = st.radio(
        "选择模式",
        ["agentic", "traditional"],
        format_func=lambda x: "🤖 Agentic-RAG (推荐)" if x == "agentic" else "📖 传统RAG",
        horizontal=True
    )
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if st.session_state.conversation_id:
                endpoint = f"/api/conversations/{st.session_state.conversation_id}/messages"
                data = {"content": prompt, "mode": mode}
            else:
                response = api_request("POST", "/api/conversations", {
                    "title": prompt[:30],
                    "mode": mode
                })
                if response:
                    st.session_state.conversation_id = response["id"]
                    endpoint = f"/api/conversations/{response['id']}/messages"
                    data = {"content": prompt, "mode": mode}
                else:
                    endpoint = "/api/rag/query"
                    data = {"query": prompt, "mode": mode, "top_k": 5}
            
            try:
                if endpoint == "/api/rag/query":
                    response = api_request("POST", endpoint, data)
                    if response:
                        full_response = response.get("answer", "")
                        message_placeholder.write(full_response)
                else:
                    response = api_request("POST", endpoint, data)
                    if response:
                        full_response = response.get("content", "")
                        message_placeholder.write(full_response)
            except Exception as e:
                message_placeholder.error(f"错误: {str(e)}")
        
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()


def main():
    """主函数"""
    menu = sidebar()
    
    if st.session_state.access_token:
        if menu == "💬 智能问答" or menu is None:
            chat_interface()
        elif menu == "📁 文档管理":
            show_document_manager()
        elif menu == "📊 使用统计":
            show_stats()
    else:
        login_form()
