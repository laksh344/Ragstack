"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, CheckCircle2, XCircle, Loader2, FileText } from "lucide-react";
import { uploadDocument } from "@/lib/api";
import type { IngestionResult } from "@/lib/types";

type Status = "idle" | "uploading" | "success" | "error";

interface UploadRecord {
  file: File;
  status: Status;
  result?: IngestionResult;
  error?: string;
}

const ACCEPTED = ".pdf,.docx,.csv,.txt,.xlsx";

export default function UploadZone() {
  const [records,   setRecords]   = useState<UploadRecord[]>([]);
  const [dragOver,  setDragOver]  = useState(false);
  const [strategy,  setStrategy]  = useState<"recursive" | "semantic">("recursive");
  const inputRef = useRef<HTMLInputElement>(null);

  const processFiles = useCallback(
    async (files: File[]) => {
      const newRecords: UploadRecord[] = files.map((f) => ({
        file: f,
        status: "uploading",
      }));
      setRecords((prev) => [...prev, ...newRecords]);

      await Promise.all(
        files.map(async (file, i) => {
          try {
            const result = await uploadDocument(file, strategy);
            setRecords((prev) =>
              prev.map((r) =>
                r.file === file ? { ...r, status: "success", result } : r
              )
            );
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            setRecords((prev) =>
              prev.map((r) =>
                r.file === file ? { ...r, status: "error", error: msg } : r
              )
            );
          }
        })
      );
    },
    [strategy]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length) processFiles(files);
    },
    [processFiles]
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      if (files.length) processFiles(files);
      e.target.value = "";
    },
    [processFiles]
  );

  return (
    <div className="space-y-4">
      {/* Strategy selector */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-slate-400">Chunking strategy:</span>
        {(["recursive", "semantic"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setStrategy(s)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              strategy === s
                ? "bg-blue-600 text-white"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-12 cursor-pointer transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-950/30"
            : "border-slate-600 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800"
        }`}
      >
        <Upload
          size={32}
          className={`transition-colors ${dragOver ? "text-blue-400" : "text-slate-500"}`}
        />
        <div className="text-center">
          <p className="text-sm font-medium text-slate-200">
            Drop files here or <span className="text-blue-400">browse</span>
          </p>
          <p className="text-xs text-slate-500 mt-1">PDF, DOCX, CSV, XLSX, TXT · max 50 MB</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          onChange={onInputChange}
          className="hidden"
        />
      </div>

      {/* Upload history */}
      {records.length > 0 && (
        <ul className="space-y-2">
          {records.map((rec, i) => (
            <UploadRow key={i} record={rec} />
          ))}
        </ul>
      )}
    </div>
  );
}

function UploadRow({ record }: { record: UploadRecord }) {
  const { file, status, result, error } = record;
  return (
    <li className="flex items-start gap-3 rounded-xl bg-slate-800 border border-slate-700 px-4 py-3">
      <div className="shrink-0 mt-0.5">
        {status === "uploading" && <Loader2 size={16} className="animate-spin text-blue-400" />}
        {status === "success"   && <CheckCircle2 size={16} className="text-emerald-400" />}
        {status === "error"     && <XCircle size={16} className="text-red-400" />}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-slate-200 truncate">{file.name}</p>
        {status === "uploading" && (
          <p className="text-xs text-slate-500 mt-0.5">Processing…</p>
        )}
        {status === "success" && result && (
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-1">
            {[
              ["Pages",   result.total_pages],
              ["Chunks",  result.total_chunks],
              ["Tokens",  result.estimated_tokens.toLocaleString()],
              ["Time",    `${result.processing_time_seconds}s`],
            ].map(([label, val]) => (
              <span key={label as string} className="text-xs text-slate-400">
                <span className="text-slate-500">{label}: </span>{val}
              </span>
            ))}
          </div>
        )}
        {status === "error" && (
          <p className="text-xs text-red-400 mt-0.5">{error}</p>
        )}
      </div>
    </li>
  );
}
