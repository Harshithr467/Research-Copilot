// app.js — main application logic

// ── State ──────────────────────────────────────────────────
const state = {
  chats: [],       // [{ id, title, messages: [], docs: [] }]
  activeChatId: null,
};

// ── Helpers ────────────────────────────────────────────────
function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function getActiveChat() {
  return state.chats.find(c => c.id === state.activeChatId) || null;
}

// ── Chat management ────────────────────────────────────────
function createChat(title) {
  const chat = { id: uid(), title: title || 'New chat', messages: [], docs: [] };
  state.chats.unshift(chat);
  switchChat(chat.id);
  renderSidebar();
}

function switchChat(id) {
  state.activeChatId = id;
  renderSidebar();
  renderTopbar();
  renderMessages();
}

// ── Document upload ────────────────────────────────────────
async function handleFileInput(files) {
  const chat = getActiveChat();
  if (!chat) return alert('Please start or select a chat first.');

  for (const file of files) {
    const result = await mockUpload(file);   // swap with real fetch() later
    chat.docs.push(result.name);
  }

  renderTopbar();
  renderUploadHint();
}

// ── Sending a query ────────────────────────────────────────
async function sendQuery() {
  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query) return;

  const chat = getActiveChat();
  if (!chat) return alert('Please start or select a chat first.');
  if (chat.docs.length === 0) return alert('Please upload at least one document first.');

  // Give chat a title from first message
  if (chat.messages.length === 0) {
    chat.title = query.slice(0, 36) + (query.length > 36 ? '…' : '');
    renderSidebar();
  }

  // Add user message
  chat.messages.push({ role: 'user', text: query });
  input.value = '';
  autoResizeTextarea(input);
  renderMessages();
  scrollToBottom();

  // Show typing indicator
  const typingId = showTyping();

  // Call mock (or real) API
  const response = await mockQuery(query, chat.docs);   // swap with real fetch() later

  removeTyping(typingId);

  // Add bot message
  chat.messages.push({ role: 'bot', ...response });
  renderMessages();
  scrollToBottom();
}

// ── Render: Sidebar ────────────────────────────────────────
function renderSidebar() {
  const list = document.getElementById('chatList');
  list.innerHTML = state.chats.map(chat => `
    <li class="${chat.id === state.activeChatId ? 'active' : ''}"
        onclick="switchChat('${chat.id}')">
      ${escapeHtml(chat.title)}
    </li>
  `).join('');
}

// ── Render: Topbar ─────────────────────────────────────────
function renderTopbar() {
  const chat = getActiveChat();
  document.getElementById('chatTitleLabel').textContent =
    chat ? chat.title : 'Select or start a chat';

  const docTags = document.getElementById('docTags');
  docTags.innerHTML = chat
    ? chat.docs.map(d => `<span class="doc-tag">📄 ${escapeHtml(d)}</span>`).join('')
    : '';

  renderUploadHint();
}

function renderUploadHint() {
  const chat = getActiveChat();
  const hint = document.getElementById('uploadHint');
  hint.textContent = chat && chat.docs.length > 0
    ? `${chat.docs.length} doc${chat.docs.length > 1 ? 's' : ''} attached`
    : 'No documents uploaded';
}

// ── Render: Messages ───────────────────────────────────────
function renderMessages() {
  const container = document.getElementById('messages');
  const chat = getActiveChat();

  if (!chat || chat.messages.length === 0) {
    container.innerHTML = `
      <div class="empty-state" id="emptyState">
        <p>Upload a document and ask a question to get started.</p>
      </div>`;
    return;
  }

  container.innerHTML = chat.messages.map(msg => {
    if (msg.role === 'user') return renderUserMessage(msg);
    if (msg.role === 'bot')  return renderBotMessage(msg);
    return '';
  }).join('');
}

function renderUserMessage(msg) {
  return `
    <div class="msg-row user">
      <div class="msg-avatar user">Y</div>
      <div class="msg-content">
        <div class="msg-bubble">${escapeHtml(msg.text)}</div>
      </div>
    </div>`;
}

function renderBotMessage(msg) {
  if (msg.insufficient) {
    return `
      <div class="msg-row bot insufficient">
        <div class="msg-avatar bot">⚠️</div>
        <div class="msg-content">
          <div class="msg-bubble">
            <strong>Not enough context in your documents.</strong><br>
            The uploaded papers don't contain enough information to answer this question.
            Try uploading a more relevant document.
          </div>
        </div>
      </div>`;
  }

  const citationsHtml = msg.citations.length > 0
    ? `<div class="citations">
        ${msg.citations.map(c => `
          <div class="citation-item">
            <span class="citation-badge">[${c.id}]</span>
            📄 ${escapeHtml(c.doc)} · Page ${c.page}
          </div>`).join('')}
       </div>`
    : '';

  return `
    <div class="msg-row bot">
      <div class="msg-avatar bot">🤖</div>
      <div class="msg-content">
        <div class="msg-bubble">${escapeHtml(msg.answer)}</div>
        ${citationsHtml}
      </div>
    </div>`;
}

// ── Typing indicator ───────────────────────────────────────
function showTyping() {
  const id = 'typing-' + uid();
  const container = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg-row bot';
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar bot">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble typing-bubble">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  container.appendChild(el);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  document.getElementById(id)?.remove();
}

// ── Utilities ──────────────────────────────────────────────
function scrollToBottom() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Event listeners ────────────────────────────────────────
document.getElementById('btnNewChat').addEventListener('click', () => createChat());

document.getElementById('btnSend').addEventListener('click', sendQuery);

document.getElementById('queryInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

document.getElementById('queryInput').addEventListener('input', (e) => {
  autoResizeTextarea(e.target);
});

document.getElementById('fileInput').addEventListener('change', (e) => {
  handleFileInput(Array.from(e.target.files));
  e.target.value = ''; // allow re-selecting same file
});

// ── Boot ───────────────────────────────────────────────────
// Start with one default chat so the screen isn't blank
createChat('My first research chat');