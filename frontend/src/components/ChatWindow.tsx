"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, ThumbsUp, ThumbsDown, AlertTriangle, Loader2 } from "lucide-react";
import { streamChat, submitFeedback } from "@/lib/api";
import type { Message, Citation } from "@/lib/types";

interface Props {
  conversationId: string | null;
  onCitationClick: (citations: Citation[]) => void;
  onConversationId: (id: string) => void;
}

const WELCOME = `Hi! I'm RAGStack — an agentic RAG assistant.

Upload documents using the **Upload** tab, then ask me anything about them. I'll search your knowledge base with hybrid vector + keyword search, rerank with Cohere, and generate a cited answer using GPT-4o.

Try asking: *"What are the main topics in the uploaded documents?"*`;

export default function ChatWindow({ conversationId, onCitationClick, onConversationId }: Props) {
  const [messages,   setMessages]   = useState<Message[]>([]);
  const [input,      setInput]      = useState("");
  const [streaming,  setStreaming]  = useState(false);
  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const convIdRef  = useRef<string | null>(conversationId);

  useEffect(() => { convIdRef.current = conversationId; }, [conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback(async () => {
    const query = input.trim();
    if (!query || streaming) return;

    setInput("");
    setStreaming(true);

    // Add user message
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      citations: [],
      guardrail_flags: [],
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    // Placeholder assistant message (streams in)
    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
      guardrail_flags: [],
      timestamp: new Date().toISOString(),
      streaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      await streamChat(query, convIdRef.current, {
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            )
          );
        },
        onCitations: (citations) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, citations } : m))
          );
        },
        onFlags: (flags) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, guardrail_flags: flags } : m
            )
          );
        },
        onDone: (_, newConvId) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m
            )
          );
          if (!convIdRef.current) onConversationId(newConvId);
          convIdRef.current = newConvId;
          setStreaming(false);
        },
        onError: (detail) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `Error: ${detail}`, streaming: false }
                : m
            )
          );
          setStreaming(false);
        },
      });
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: "Connection error — is the backend running?", streaming: false }
            : m
        )
      );
      setStreaming(false);
    }
  }, [input, streaming, onConversationId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* ── Message list ─────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 && (
          <WelcomeCard message={WELCOME} />
        )}
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onCitationClick={onCitationClick}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* ── Input bar ─────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-slate-700 bg-slate-900 px-4 py-3">
        <div className="flex items-end gap-3 max-w-4xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your documents…"
            rows={1}
            disabled={streaming}
            className="flex-1 resize-none rounded-xl bg-slate-800 border border-slate-600 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 disabled:opacity-50 max-h-36 overflow-y-auto"
            style={{ lineHeight: "1.5" }}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || streaming}
            className="shrink-0 w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
            aria-label="Send"
          >
            {streaming
              ? <Loader2 size={16} className="animate-spin text-white" />
              : <Send size={16} className="text-white" />
            }
          </button>
        </div>
        <p className="text-xs text-slate-600 text-center mt-2">
          Hybrid search → Cohere rerank → GPT-4o · LangSmith traced
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function WelcomeCard({ message }: { message: string }) {
  return (
    <div className="max-w-2xl mx-auto text-center py-12 px-6">
      <div className="w-12 h-12 rounded-2xl bg-blue-600 flex items-center justify-center text-xl font-bold mx-auto mb-4">
        R
      </div>
      <h1 className="text-xl font-semibold mb-2">RAGStack Assistant</h1>
      <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-line">
        {message.replace(/\*\*/g, "").replace(/\*/g, "")}
      </p>
    </div>
  );
}

function MessageBubble({
  message,
  onCitationClick,
}: {
  message: Message;
  onCitationClick: (c: Citation[]) => void;
}) {
  const isUser = message.role === "user";
  const hasFlags = message.guardrail_flags.length > 0;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} max-w-4xl mx-auto w-full`}>
      <div className={`max-w-[80%] ${isUser ? "order-2" : ""}`}>
        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-blue-600 text-white rounded-br-sm"
              : "bg-slate-800 text-slate-100 rounded-bl-sm border border-slate-700"
          }`}
        >
          <span className={message.streaming ? "streaming-cursor" : ""}>
            {message.content || (message.streaming ? "" : "…")}
          </span>
        </div>

        {/* Citations */}
        {message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            {message.citations.map((cit, i) => (
              <button
                key={i}
                onClick={() => onCitationClick(message.citations)}
                className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-700 hover:bg-slate-600 text-xs text-slate-300 transition-colors"
              >
                <span className="text-blue-400">↗</span>
                {cit.source_file} p.{cit.page_number}
              </button>
            ))}
          </div>
        )}

        {/* Guardrail flags */}
        {hasFlags && (
          <div className="flex items-center gap-1.5 mt-1.5 text-xs text-amber-400">
            <AlertTriangle size={11} />
            {message.guardrail_flags.join(", ")}
          </div>
        )}

        {/* Feedback (assistant messages only, after streaming) */}
        {!isUser && !message.streaming && (
          <div className="flex gap-2 mt-1.5">
            <button
              onClick={() => submitFeedback(message.id, 1)}
              className="text-slate-500 hover:text-emerald-400 transition-colors"
              title="Good response"
            >
              <ThumbsUp size={13} />
            </button>
            <button
              onClick={() => submitFeedback(message.id, -1)}
              className="text-slate-500 hover:text-red-400 transition-colors"
              title="Bad response"
            >
              <ThumbsDown size={13} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
