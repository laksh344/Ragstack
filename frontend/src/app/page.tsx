"use client";

import { useState } from "react";
import { Upload, MessageSquare, BarChart2 } from "lucide-react";
import ChatWindow from "@/components/ChatWindow";
import SourcePanel from "@/components/SourcePanel";
import UploadZone from "@/components/UploadZone";
import type { Citation } from "@/lib/types";

type Tab = "chat" | "upload";

export default function HomePage() {
  const [activeTab, setActiveTab]         = useState<Tab>("chat");
  const [sourcePanelOpen, setSourcePanelOpen] = useState(false);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const [conversationId, setConversationId]   = useState<string | null>(null);

  const handleCitationClick = (citations: Citation[]) => {
    setActiveCitations(citations);
    setSourcePanelOpen(true);
  };

  const handleConversationId = (id: string) => {
    setConversationId(id);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-700 bg-slate-900 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-blue-600 flex items-center justify-center text-sm font-bold">
            R
          </div>
          <span className="font-semibold text-slate-100 tracking-tight">RAGStack</span>
          <span className="text-xs text-slate-500 hidden sm:inline">
            Agentic RAG · Hybrid Search · Guardrails
          </span>
        </div>

        {/* Tab switcher */}
        <nav className="flex gap-1 bg-slate-800 rounded-lg p-1">
          {(["chat", "upload"] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab
                  ? "bg-blue-600 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab === "chat"   && <MessageSquare size={14} />}
              {tab === "upload" && <Upload size={14} />}
              {tab === "chat"   ? "Chat" : "Upload"}
            </button>
          ))}
        </nav>

        <a
          href="https://smith.langchain.com"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          <BarChart2 size={13} />
          LangSmith
        </a>
      </header>

      {/* ── Body ───────────────────────────────────────────────── */}
      <main className="flex flex-1 overflow-hidden">
        {activeTab === "chat" && (
          <>
            <div className="flex-1 min-w-0">
              <ChatWindow
                conversationId={conversationId}
                onCitationClick={handleCitationClick}
                onConversationId={handleConversationId}
              />
            </div>
            {sourcePanelOpen && (
              <SourcePanel
                citations={activeCitations}
                onClose={() => setSourcePanelOpen(false)}
              />
            )}
          </>
        )}

        {activeTab === "upload" && (
          <div className="flex-1 flex items-start justify-center p-8 overflow-y-auto">
            <div className="w-full max-w-2xl">
              <h2 className="text-xl font-semibold mb-1">Upload Documents</h2>
              <p className="text-slate-400 text-sm mb-6">
                PDF, DOCX, CSV, TXT — up to 50 MB. Documents are parsed, chunked,
                embedded, and indexed for hybrid search.
              </p>
              <UploadZone />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
