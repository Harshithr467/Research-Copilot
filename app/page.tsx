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

//  Mock API (swap the 2 later) 
const MOCKS = [
  { answer: "The attention mechanism uses scaled dot-products between queries and keys. Scaling by √dₖ prevents vanishing gradients [1]. Multi-head attention replaces recurrence entirely, enabling full parallelisation [2].", citations: [{ id: 1, doc: "attention_is_all_you_need.pdf", page: 3 }, { id: 2, doc: "attention_is_all_you_need.pdf", page: 5 }], insufficient: false },
  { answer: null, citations: [], insufficient: true },
  { answer: "FlashAttention tiles the softmax to avoid materialising the full N×N matrix in HBM [1], reducing memory from O(N²) to O(N) and achieving 2–4× speedup [2].", citations: [{ id: 1, doc: "flash_attention.pdf", page: 2 }, { id: 2, doc: "flash_attention.pdf", page: 7 }], insufficient: false },
];
let mockIdx = 0;
async function queryDocuments(_query: string) {
  await new Promise(r => setTimeout(r, 900 + Math.random() * 600));
  return MOCKS[mockIdx++ % MOCKS.length];
}
async function uploadDocument(file: File): Promise<Doc> {
  await new Promise(r => setTimeout(r, 400));
  return { name: file.name, pages: Math.floor(Math.random() * 20) + 3 };
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