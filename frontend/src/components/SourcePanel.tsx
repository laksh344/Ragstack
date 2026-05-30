"use client";

import { X, FileText, Hash } from "lucide-react";
import type { Citation } from "@/lib/types";

interface Props {
  citations: Citation[];
  onClose: () => void;
}

export default function SourcePanel({ citations, onClose }: Props) {
  return (
    <aside className="w-80 shrink-0 border-l border-slate-700 bg-slate-900 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <span className="text-sm font-medium text-slate-200">
          Sources <span className="text-slate-500 font-normal">({citations.length})</span>
        </span>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-200 transition-colors"
          aria-label="Close sources"
        >
          <X size={16} />
        </button>
      </div>

      {/* Citation list */}
      <div className="flex-1 overflow-y-auto py-3 space-y-3 px-3">
        {citations.length === 0 && (
          <p className="text-sm text-slate-500 text-center py-8">No sources for this response.</p>
        )}
        {citations.map((cit, i) => (
          <SourceCard key={i} index={i + 1} citation={cit} />
        ))}
      </div>
    </aside>
  );
}

function SourceCard({ index, citation }: { index: number; citation: Citation }) {
  return (
    <div className="rounded-xl bg-slate-800 border border-slate-700 p-3 space-y-2">
      {/* File + page */}
      <div className="flex items-center gap-2">
        <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px] font-bold shrink-0">
          {index}
        </span>
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs text-slate-300 font-medium truncate">
            <FileText size={11} className="text-blue-400 shrink-0" />
            <span className="truncate">{citation.source_file}</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-slate-500 mt-0.5">
            <Hash size={10} />
            Page {citation.page_number}
          </div>
        </div>
      </div>

      {/* Excerpt */}
      {citation.excerpt && (
        <blockquote className="text-xs text-slate-400 leading-relaxed border-l-2 border-blue-600 pl-2 line-clamp-4">
          {citation.excerpt}
        </blockquote>
      )}
    </div>
  );
}
