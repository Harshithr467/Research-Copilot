"use client";

import { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode, KeyboardEvent } from "react";
import { Plus, MessageSquare, BookOpen, FileText, Paperclip, ArrowUp, Loader2, AlertTriangle, Bot } from "lucide-react";

// Types including message, chats, documents, citations
interface Citation { id: number; doc: string; page: number; }
interface Message {
  id: string; role: "user" | "bot"; text?: string;
  answer?: string | null; citations?: Citation[]; insufficient?: boolean;
}
interface Doc { name: string; pages: number; }
interface Chat { id: string; title: string; messages: Message[]; docs: Doc[]; }

async function queryDocuments(query: string) {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const res = await fetch(`${backendUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.statusText}`);
  }

  return await res.json();
}
async function uploadDocument(file: File): Promise<Doc> {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${backendUrl}/upload-pdf`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Upload failed: ${res.statusText}`);
  }

  const data = await res.json();
  if (data.status === "error") {
    throw new Error(data.message || "Upload failed");
  }

  return { name: data.filename || file.name, pages: 1 };
}

// ── State (Context) ────────────────────────────────────────────────────────
function uid() { return Math.random().toString(36).slice(2); }

interface Ctx {
  chats: Chat[]; activeChatId: string | null; activeChat: Chat | null; isLoading: boolean;
  createChat: () => void; switchChat: (id: string) => void;
  sendMessage: (q: string) => Promise<void>; uploadFiles: (files: File[]) => Promise<void>;
}
const ChatCtx = createContext<Ctx>(null!);
function useChatCtx() { return useContext(ChatCtx); }

function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const activeChat = chats.find(c => c.id === activeChatId) ?? null;

  const update = useCallback((id: string, fn: (c: Chat) => Chat) =>
    setChats(prev => prev.map(c => c.id === id ? fn(c) : c)), []);

  const createChat = useCallback(() => {
    const id = uid();
    setChats(prev => [{ id, title: "New chat", messages: [], docs: [] }, ...prev]);
    setActiveChatId(id);
  }, []);

  const switchChat = useCallback((id: string) => setActiveChatId(id), []);

  const sendMessage = useCallback(async (query: string) => {
    if (!activeChatId) return;
    if (activeChat?.messages.length === 0) {
      const title = query.length > 40 ? query.slice(0, 40) + "…" : query;
      update(activeChatId, c => ({ ...c, title }));
    }
    const userMsg: Message = { id: uid(), role: "user", text: query };
    update(activeChatId, c => ({ ...c, messages: [...c.messages, userMsg] }));
    setIsLoading(true);
    try {
      const res = await queryDocuments(query);
      const botMsg: Message = { id: uid(), role: "bot", ...res };
      update(activeChatId, c => ({ ...c, messages: [...c.messages, botMsg] }));
    } finally { setIsLoading(false); }
  }, [activeChatId, activeChat, update]);

  const uploadFiles = useCallback(async (files: File[]) => {
    if (!activeChatId) return;
    for (const file of files) {
      const doc = await uploadDocument(file);
      update(activeChatId, c => ({ ...c, docs: [...c.docs, doc] }));
    }
  }, [activeChatId, update]);

  return (
    <ChatCtx.Provider value={{ chats, activeChatId, activeChat, isLoading, createChat, switchChat, sendMessage, uploadFiles }}>
      {children}
    </ChatCtx.Provider>
  );
}

// ── Sidebar
function Sidebar() {
  const { chats, activeChatId, createChat, switchChat } = useChatCtx();
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand"><BookOpen size={18} strokeWidth={1.5} /><span>ResearchBot</span></div>
        <button className="btn-new-chat" onClick={createChat}><Plus size={15} />New chat</button>
      </div>
      <div className="chat-section-label">Chats</div>
      <nav className="chat-list">
        {chats.length === 0 && <p className="chat-list-empty">No chats yet.</p>}
        {chats.map(chat => (
          <button key={chat.id} className={`chat-item ${chat.id === activeChatId ? "active" : ""}`} onClick={() => switchChat(chat.id)}>
            <MessageSquare size={13} strokeWidth={1.5} className="chat-item-icon" />
            <span className="chat-item-title">{chat.title}</span>
            <span className="chat-item-meta">{chat.docs.length} doc{chat.docs.length !== 1 ? "s" : ""}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}

// Topbar
function Topbar() {
  const { activeChat } = useChatCtx();
  return (
    <div className="topbar">
      <span className={`topbar-title ${!activeChat ? "muted" : ""}`}>
        {activeChat ? activeChat.title : "Select or start a chat"}
      </span>
      <div className="doc-chips">
        {activeChat?.docs.length === 0 && <span className="no-docs-hint">No documents attached yet</span>}
        {activeChat?.docs.map((doc, i) => (
          <span key={i} className="doc-chip"><FileText size={11} strokeWidth={1.5} />{doc.name}</span>
        ))}
      </div>
    </div>
  );
}

// ── Messages ───────────────────────────────────────────────────────────────
function MessageList() {
  const { activeChat, isLoading } = useChatCtx();
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [activeChat?.messages, isLoading]);

  if (!activeChat || activeChat.messages.length === 0) return (
    <div className="messages-empty">
      <div className="empty-card">
        <h2>{activeChat ? activeChat.title : "Research Assistant"}</h2>
        <p>{activeChat ? "Upload a document below and ask your first question." : "Start a new chat, upload your papers, and ask questions.Answers will be grounded strictly in your documents."}</p>
      </div>
    </div>
  );

  return (
    <div className="message-list">
      {activeChat.messages.map(msg => {
        if (msg.role === "user") return (
          <div key={msg.id} className="msg-row user">
            <div className="msg-avatar user">Y</div>
            <div className="msg-bubble user">{msg.text}</div>
          </div>
        );
        if (msg.insufficient) return (
          <div key={msg.id} className="msg-row bot">
            <div className="msg-avatar bot"><AlertTriangle size={14} strokeWidth={1.5} /></div>
            <div className="msg-content">
              <div className="msg-bubble insufficient">
                <strong>Not enough context in your documents.</strong>
                <p>The uploaded papers don&apos;t contain enough information. Try uploading a more relevant document.</p>
              </div>
            </div>
          </div>
        );
        return (
          <div key={msg.id} className="msg-row bot">
            <div className="msg-avatar bot"><Bot size={14} strokeWidth={1.5} /></div>
            <div className="msg-content">
              <div className="msg-bubble bot">{msg.answer}</div>
              {msg.citations && msg.citations.length > 0 && (
                <div className="citations">
                  {msg.citations.map(c => (
                    <div key={c.id} className="citation-item">
                      <span className="citation-badge">[{c.id}]</span>
                      <FileText size={12} strokeWidth={1.5} />
                      {c.doc} - Chunk {c.page}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
      {isLoading && (
        <div className="msg-row bot">
          <div className="msg-avatar bot"><Bot size={14} strokeWidth={1.5} /></div>
          <div className="msg-bubble bot typing-bubble"><span /><span /><span /></div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

// Input bar
function InputBar() {
  const { activeChat, isLoading, sendMessage, uploadFiles, createChat } = useChatCtx();
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }

  async function handleSend() {
    const q = query.trim();
    if (!q || isLoading) return;
    if (!activeChat) { createChat(); return; }
    if (activeChat.docs.length === 0) { alert("Upload at least one document first."); return; }
    setQuery("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    await sendMessage(q);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  async function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    if (!activeChat) createChat();
    await uploadFiles(files);
    e.target.value = "";
  }

  const docCount = activeChat?.docs.length ?? 0;
  return (
    <div className="input-area">
      <div className="upload-strip">
        <button className="attach-btn" onClick={() => fileInputRef.current?.click()}>
          <Paperclip size={14} strokeWidth={1.5} />Attach document
        </button>
        <span className="upload-strip-hint">
          {docCount > 0 ? `${docCount} document${docCount > 1 ? "s" : ""} attached` : "No documents attached — upload a PDF or DOCX"}
        </span>
        <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" multiple hidden onChange={handleFiles} />
      </div>
      <div className="input-row">
        <textarea ref={textareaRef} className="query-input" placeholder="Ask a question based on your uploaded documents…"
          value={query} rows={1} disabled={isLoading}
          onChange={e => { setQuery(e.target.value); autoResize(); }}
          onKeyDown={handleKeyDown} />
        <button className="send-btn" onClick={handleSend} disabled={!query.trim() || isLoading}>
          {isLoading ? <Loader2 size={16} strokeWidth={1.5} className="spin" /> : <ArrowUp size={16} strokeWidth={2} />}
        </button>
      </div>
    </div>
  );
}

// Page
export default function Page() {
  return (
    <ChatProvider>
      <div className="app-shell">
        <Sidebar />
        <main className="main-area">
          <Topbar />
          <MessageList />
          <InputBar />
        </main>
      </div>
    </ChatProvider>
  );
}
