// api.js - API wrapper with Bearer token auth
const API_BASE = 'http://localhost:8000';

function getToken() {
  return localStorage.getItem('access_token');
}

function authHeaders(extra = {}) {
  const token = getToken();
  const headers = { ...extra };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function apiFetch(path, options = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch (e) {
    throw new Error('无法连接到服务器，请确认后端已启动（http://localhost:8000）');
  }
  if (res.status === 401) {
    localStorage.removeItem('access_token');
    window.location.reload();
    return null;
  }
  return res;
}

async function safeJson(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(text || `服务器错误 (${res.status})`);
  }
}

export async function login(username, password) {
  const body = new URLSearchParams({ username, password });
  const res = await apiFetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body
  });
  return res ? safeJson(res) : null;
}

export async function register(username, email, password) {
  const res = await apiFetch('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password })
  });
  return res ? safeJson(res) : null;
}

export async function getConversations() {
  const res = await apiFetch('/api/conversations', {
    headers: authHeaders()
  });
  return res ? res.json() : [];
}

export async function createConversation(title, mode) {
  const res = await apiFetch('/api/conversations', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title, mode })
  });
  return res ? res.json() : null;
}

export async function deleteConversation(id) {
  const res = await apiFetch(`/api/conversations/${id}`, {
    method: 'DELETE',
    headers: authHeaders()
  });
  return res ? res.ok : false;
}

export async function getMessages(conversationId) {
  const res = await apiFetch(`/api/conversations/${conversationId}/messages`, {
    headers: authHeaders()
  });
  return res ? res.json() : [];
}

export async function sendMessage(conversationId, content, mode, knowledgeBaseId = null) {
  const res = await apiFetch(`/api/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ content, mode, knowledge_base_id: knowledgeBaseId })
  });
  if (!res || !res.ok) return null;
  return res.json();
}

export async function ragQuery(query, mode, topK = 5, knowledgeBaseId = null) {
  const res = await apiFetch('/api/rag/query', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ query, mode, top_k: topK, knowledge_base_id: knowledgeBaseId })
  });
  return res ? res.json() : null;
}

// ==================== 知识库管理 API ====================

export async function getKnowledgeBases() {
  const res = await apiFetch('/api/knowledge-bases', {
    headers: authHeaders()
  });
  return res ? res.json() : [];
}

export async function createKnowledgeBase(name, path, description = '', fileTypes = '.pdf,.txt,.md') {
  const res = await apiFetch('/api/knowledge-bases', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ name, path, description, file_types: fileTypes })
  });
  return res ? safeJson(res) : null;
}

export async function getKnowledgeBase(kbId) {
  const res = await apiFetch(`/api/knowledge-bases/${kbId}`, {
    headers: authHeaders()
  });
  return res ? res.json() : null;
}

export async function updateKnowledgeBase(kbId, name, description, fileTypes) {
  const res = await apiFetch(`/api/knowledge-bases/${kbId}`, {
    method: 'PUT',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ name, description, file_types: fileTypes })
  });
  return res ? res.json() : null;
}

export async function deleteKnowledgeBase(kbId) {
  const res = await apiFetch(`/api/knowledge-bases/${kbId}`, {
    method: 'DELETE',
    headers: authHeaders()
  });
  return res ? res.ok : false;
}

export async function scanKnowledgeBase(kbId) {
  const res = await apiFetch(`/api/knowledge-bases/${kbId}/scan`, {
    method: 'POST',
    headers: authHeaders()
  });
  return res ? safeJson(res) : null;
}

// ==================== 旧文档 API（已废弃，保留兼容） ====================

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await apiFetch('/api/documents/upload', {
    method: 'POST',
    headers: authHeaders(),
    body: formData
  });
  if (!res || !res.ok) return null;
  return res.json();
}

export async function getDocuments() {
  const res = await apiFetch('/api/documents', {
    headers: authHeaders()
  });
  return res ? res.json() : [];
}

export async function deleteDocument(id) {
  const res = await apiFetch(`/api/documents/${id}`, {
    method: 'DELETE',
    headers: authHeaders()
  });
  return res ? res.ok : false;
}

export async function getStats() {
  const res = await apiFetch('/api/stats', {
    headers: authHeaders()
  });
  return res ? res.json() : null;
}
