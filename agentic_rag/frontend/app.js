// app.js - Main application state and UI logic
import * as API from './api.js';

// ==================== State ====================
const state = {
  messages: [],          // in-memory, trimmed to last N rounds
  currentView: 'chat',   // 'chat' | 'kbs' | 'stats'
  conversations: [],
  knowledgeBases: [],    // 知识库列表
  currentKbId: null,     // 当前选中的知识库
  isLoading: false,
  chatMode: 'agentic',   // 'agentic' | 'traditional'
};

function getMaxRounds() {
  return parseInt(localStorage.getItem('max_history_rounds') || '10', 10);
}

function getCurrentKbId() {
  return localStorage.getItem('current_kb_id');
}

function setCurrentKbId(id) {
  if (id) localStorage.setItem('current_kb_id', id);
  else localStorage.removeItem('current_kb_id');
  state.currentKbId = id;
}

function trimMessages(msgs, n) {
  return msgs.slice(-n * 2);
}

function getCurrentConvId() {
  return localStorage.getItem('current_conv_id');
}

function setCurrentConvId(id) {
  if (id) localStorage.setItem('current_conv_id', id);
  else localStorage.removeItem('current_conv_id');
}

// ==================== Auth ====================
function isLoggedIn() {
  return !!localStorage.getItem('access_token');
}

function showApp() {
  document.getElementById('login-modal').style.display = 'none';
  document.getElementById('app').style.display = 'flex';
  loadConversations();
  // 加载保存的知识库选择
  const savedKbId = getCurrentKbId();
  if (savedKbId) {
    state.currentKbId = savedKbId;
  }
}

function showLogin() {
  document.getElementById('login-modal').style.display = 'flex';
  document.getElementById('app').style.display = 'none';
}

async function handleLogin(e) {
  e.preventDefault();
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');
  errEl.textContent = '';

  try {
    const data = await API.login(username, password);
    if (data && data.access_token) {
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('username', username);
      showApp();
    } else {
      errEl.textContent = data?.detail || '登录失败';
    }
  } catch {
    errEl.textContent = '网络错误，请重试';
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const username = document.getElementById('reg-username').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value;
  const errEl = document.getElementById('reg-error');
  errEl.textContent = '';

  try {
    const data = await API.register(username, email, password);
    if (data && data.id) {
      errEl.style.color = '#3ba55d';
      errEl.textContent = '注册成功，请登录';
      switchAuthTab('login');
    } else {
      errEl.textContent = data?.detail || '注册失败';
    }
  } catch {
    errEl.textContent = '网络错误，请重试';
  }
}

function handleLogout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('username');
  localStorage.removeItem('current_conv_id');
  localStorage.removeItem('current_kb_id');
  state.messages = [];
  state.conversations = [];
  state.knowledgeBases = [];
  showLogin();
}

// ==================== Auth Tabs ====================
function switchAuthTab(tab) {
  document.querySelectorAll('.auth-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('login-form').style.display = tab === 'login' ? 'block' : 'none';
  document.getElementById('register-form').style.display = tab === 'register' ? 'block' : 'none';
}

// ==================== Navigation ====================
function switchView(view) {
  state.currentView = view;
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.view === view));
  document.querySelectorAll('.view-panel').forEach(el => el.style.display = 'none');
  const panel = document.getElementById(`panel-${view}`);
  if (panel) panel.style.display = 'flex';

  if (view === 'kbs') loadKnowledgeBases();
  if (view === 'stats') loadStats();
}

// ==================== Conversations ====================
async function loadConversations() {
  state.conversations = await API.getConversations() || [];
  renderConversations();

  const convId = getCurrentConvId();
  if (convId && state.conversations.find(c => c.id == convId)) {
    loadConversation(convId);
  } else if (state.conversations.length > 0) {
    loadConversation(state.conversations[0].id);
  }
}

function renderConversations() {
  const list = document.getElementById('conv-list');
  if (!list) return;
  list.innerHTML = '';
  for (const conv of state.conversations) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (conv.id == getCurrentConvId() ? ' active' : '');
    item.innerHTML = `
      <span class="conv-title" data-id="${conv.id}">${escapeHtml(conv.title || '新对话')}</span>
      <button class="conv-delete" data-id="${conv.id}" title="删除">×</button>
    `;
    item.querySelector('.conv-title').addEventListener('click', () => loadConversation(conv.id));
    item.querySelector('.conv-delete').addEventListener('click', e => { e.stopPropagation(); confirmDeleteConv(conv.id); });
    list.appendChild(item);
  }
}

async function loadConversation(id) {
  setCurrentConvId(id);
  renderConversations();
  const rawMsgs = await API.getMessages(id) || [];
  state.messages = trimMessages(rawMsgs, getMaxRounds());
  renderMessages();
}

async function newConversation() {
  const title = `对话 ${new Date().toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`;
  const conv = await API.createConversation(title, state.chatMode);
  if (conv) {
    state.conversations.unshift(conv);
    renderConversations();
    setCurrentConvId(conv.id);
    state.messages = [];
    renderMessages();
  }
}

async function confirmDeleteConv(id) {
  if (!confirm('确认删除此对话？')) return;
  const ok = await API.deleteConversation(id);
  if (ok) {
    state.conversations = state.conversations.filter(c => c.id != id);
    if (getCurrentConvId() == id) {
      setCurrentConvId(null);
      state.messages = [];
      renderMessages();
    }
    renderConversations();
  }
}

// ==================== Chat ====================
function renderMessages() {
  const container = document.getElementById('messages');
  if (!container) return;
  container.innerHTML = '';
  for (const msg of state.messages) {
    appendMessageEl(msg.role, msg.content);
  }
  container.scrollTop = container.scrollHeight;
}

function appendMessageEl(role, content) {
  const container = document.getElementById('messages');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `message message-${role}`;
  el.innerHTML = `<div class="message-bubble">${escapeHtml(content)}</div>`;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  return el;
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const content = input.value.trim();
  if (!content || state.isLoading) return;

  let convId = getCurrentConvId();
  if (!convId) {
    await newConversation();
    convId = getCurrentConvId();
  }
  if (!convId) return;

  input.value = '';
  state.isLoading = true;
  setInputLoading(true);

  // Optimistic user message
  state.messages.push({ role: 'user', content });
  appendMessageEl('user', content);

  // 获取当前选中的知识库ID
  const kbId = state.currentKbId || getCurrentKbId();

  try {
    const result = await API.sendMessage(convId, content, state.chatMode, kbId);
    if (result) {
      const assistantContent = result.content || result.answer || result.response || JSON.stringify(result);
      state.messages.push({ role: 'assistant', content: assistantContent });
      appendMessageEl('assistant', assistantContent);
      state.messages = trimMessages(state.messages, getMaxRounds());
    }
  } catch (err) {
    appendMessageEl('assistant', `错误: ${err.message}`);
  } finally {
    state.isLoading = false;
    setInputLoading(false);
  }
}

function setInputLoading(loading) {
  const btn = document.getElementById('send-btn');
  const input = document.getElementById('chat-input');
  if (btn) btn.disabled = loading;
  if (input) input.disabled = loading;
}

function toggleChatMode() {
  state.chatMode = state.chatMode === 'agentic' ? 'traditional' : 'agentic';
  const btn = document.getElementById('mode-toggle');
  if (btn) btn.textContent = state.chatMode === 'agentic' ? 'Agentic' : 'Traditional';
  btn?.classList.toggle('mode-traditional', state.chatMode === 'traditional');
}

// ==================== 知识库管理 ====================
async function loadKnowledgeBases() {
  state.knowledgeBases = await API.getKnowledgeBases() || [];
  renderKnowledgeBases();
  renderKnowledgeBaseSelector();
}

function renderKnowledgeBases() {
  const list = document.getElementById('kb-list');
  if (!list) return;
  list.innerHTML = '';
  if (state.knowledgeBases.length === 0) {
    list.innerHTML = '<p class="empty-hint">暂无知识库，请创建</p>';
    return;
  }
  for (const kb of state.knowledgeBases) {
    const card = document.createElement('div');
    card.className = 'kb-card' + (kb.id == state.currentKbId ? ' selected' : '');
    card.innerHTML = `
      <div class="kb-icon">📁</div>
      <div class="kb-info">
        <div class="kb-name">${escapeHtml(kb.name)}</div>
        <div class="kb-path">${escapeHtml(kb.path)}</div>
        <div class="kb-meta">
          ${kb.is_indexed ? '<span class="kb-status indexed">已索引</span>' : '<span class="kb-status pending">未索引</span>'}
          <span>${kb.file_count || 0} 文件 · ${kb.chunk_count || 0} 块</span>
        </div>
      </div>
      <div class="kb-actions">
        <button class="btn-scan" data-id="${kb.id}" title="扫描索引">🔄</button>
        <button class="btn-delete" data-id="${kb.id}" title="删除">×</button>
      </div>
    `;
    card.addEventListener('click', (e) => {
      if (!e.target.closest('.kb-actions')) {
        selectKnowledgeBase(kb.id);
      }
    });
    card.querySelector('.btn-scan').addEventListener('click', (e) => {
      e.stopPropagation();
      scanKnowledgeBase(kb.id);
    });
    card.querySelector('.btn-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteKnowledgeBase(kb.id);
    });
    list.appendChild(card);
  }
}

function renderKnowledgeBaseSelector() {
  const selector = document.getElementById('kb-selector');
  if (!selector) return;

  // 保存当前选择
  const currentValue = selector.value;

  selector.innerHTML = '<option value="">全部知识库</option>';
  for (const kb of state.knowledgeBases) {
    if (kb.is_indexed) {
      const option = document.createElement('option');
      option.value = kb.id;
      option.textContent = kb.name;
      selector.appendChild(option);
    }
  }

  // 恢复选择
  if (state.currentKbId) {
    selector.value = state.currentKbId;
  }
}

function selectKnowledgeBase(kbId) {
  setCurrentKbId(kbId);
  const selector = document.getElementById('kb-selector');
  if (selector) selector.value = kbId;
  renderKnowledgeBases();

  // 显示提示
  const kb = state.knowledgeBases.find(k => k.id == kbId);
  if (kb) {
    showToast(`已选择知识库: ${kb.name}`);
  }
}

async function createKnowledgeBase() {
  const nameInput = document.getElementById('kb-name-input');
  const pathInput = document.getElementById('kb-path-input');
  const descInput = document.getElementById('kb-desc-input');
  const typesInput = document.getElementById('kb-types-input');

  const name = nameInput?.value.trim();
  const path = pathInput?.value.trim();
  const description = descInput?.value.trim();
  const fileTypes = typesInput?.value.trim() || '.pdf,.txt,.md';

  if (!name || !path) {
    showToast('请输入知识库名称和路径');
    return;
  }

  const result = await API.createKnowledgeBase(name, path, description, fileTypes);
  if (result && result.id) {
    showToast('知识库创建成功');
    nameInput.value = '';
    pathInput.value = '';
    descInput.value = '';
    loadKnowledgeBases();
  } else {
    showToast(result?.detail || '创建失败');
  }
}

async function scanKnowledgeBase(kbId) {
  showToast('扫描任务已启动...');
  const result = await API.scanKnowledgeBase(kbId);
  if (result) {
    showToast('扫描任务已在后台启动');
    // 轮询检查状态
    setTimeout(() => loadKnowledgeBases(), 2000);
  }
}

async function deleteKnowledgeBase(kbId) {
  if (!confirm('确认删除此知识库？这将删除所有相关索引，但不会影响本地文件。')) return;
  const ok = await API.deleteKnowledgeBase(kbId);
  if (ok) {
    if (state.currentKbId == kbId) {
      setCurrentKbId(null);
    }
    loadKnowledgeBases();
  }
}

function showToast(message) {
  // 简单的提示实现
  const existing = document.querySelector('.toast-message');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast-message';
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    z-index: 10000;
    font-size: 14px;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ==================== Stats ====================
async function loadStats() {
  const stats = await API.getStats();
  const grid = document.getElementById('stats-grid');
  if (!grid || !stats) return;
  grid.innerHTML = `
    <div class="stat-card"><div class="stat-value">${stats.total_conversations ?? '-'}</div><div class="stat-label">总对话数</div></div>
    <div class="stat-card"><div class="stat-value">${stats.total_messages ?? '-'}</div><div class="stat-label">总消息数</div></div>
    <div class="stat-card"><div class="stat-value">${stats.total_knowledge_bases ?? '-'}</div><div class="stat-label">知识库数量</div></div>
    <div class="stat-card"><div class="stat-value">${stats.total_chunks ?? '-'}</div><div class="stat-label">文本块数</div></div>
  `;
}

// ==================== Settings ====================
function initSettings() {
  const slider = document.getElementById('rounds-slider');
  const display = document.getElementById('rounds-display');
  if (!slider || !display) return;
  const current = getMaxRounds();
  slider.value = current;
  display.textContent = current;
  slider.addEventListener('input', () => {
    display.textContent = slider.value;
    localStorage.setItem('max_history_rounds', slider.value);
    state.messages = trimMessages(state.messages, parseInt(slider.value));
    renderMessages();
  });

  // 知识库选择器事件
  const kbSelector = document.getElementById('kb-selector');
  if (kbSelector) {
    kbSelector.addEventListener('change', (e) => {
      setCurrentKbId(e.target.value || null);
    });
  }
}

// ==================== Utils ====================
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

// ==================== Init ====================
document.addEventListener('DOMContentLoaded', () => {
  // Auth tabs
  document.querySelectorAll('.auth-tab').forEach(t => {
    t.addEventListener('click', () => switchAuthTab(t.dataset.tab));
  });

  // Auth forms
  document.getElementById('login-form')?.addEventListener('submit', handleLogin);
  document.getElementById('register-form')?.addEventListener('submit', handleRegister);
  document.getElementById('logout-btn')?.addEventListener('click', handleLogout);

  // Nav
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', () => switchView(el.dataset.view));
  });

  // New conversation
  document.getElementById('new-conv-btn')?.addEventListener('click', newConversation);

  // Chat mode toggle
  document.getElementById('mode-toggle')?.addEventListener('click', toggleChatMode);

  // Send message
  document.getElementById('send-btn')?.addEventListener('click', sendMessage);
  const chatInput = document.getElementById('chat-input');
  chatInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  chatInput?.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  });

  // Knowledge base create button
  document.getElementById('kb-create-btn')?.addEventListener('click', createKnowledgeBase);

  // Settings
  initSettings();

  // Check auth
  if (isLoggedIn()) {
    showApp();
  } else {
    showLogin();
  }
});
