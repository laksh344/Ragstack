"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Upload, CheckCircle2, XCircle, Loader2, FileText, Trash2, Database } from "lucide-react";
import { uploadDocument, listDocuments, deleteDocument } from "@/lib/api";
import type { IngestionResult, DocumentInfo } from "@/lib/types";

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
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [deleting,  setDeleting]  = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [dragOver,  setDragOver]  = useState(false);
  const [strategy,  setStrategy]  = useState<"recursive" | "semantic">("recursive");
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch the persisted document list from the backend on mount.
  const refreshDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      setDocuments(await listDocuments());
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  const processFiles = useCallback(
    async (files: File[]) => {
      const newRecords: UploadRecord[] = files.map((f) => ({
        file: f,
        status: "uploading",
      }));
      setRecords((prev) => [...prev, ...newRecords]);

      await Promise.all(
        files.map(async (file) => {
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

      // Refresh the persisted list once uploads settle.
      refreshDocuments();
    },
    [strategy, refreshDocuments]
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

  const handleDelete = useCallback(
    async (sourceFile: string) => {
      setDeleteError(null);
      setDeleting(sourceFile);
      try {
        await deleteDocument(sourceFile);
        // Only refresh from the server after a confirmed delete.
        await refreshDocuments();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setDeleteError(`Failed to delete "${sourceFile}": ${msg}`);
      } finally {
        setDeleting(null);
      }
    },
    [refreshDocuments]
  );

  // Only show in-progress / failed uploads here; successful ones appear in the
  // persisted "Knowledge base" list below.
  const activeRecords = records.filter((r) => r.status !== "success");

  return (
    <div className="space-y-6">
      {/* Strategy selector */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-muted">Chunking strategy:</span>
        {(["recursive", "semantic"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setStrategy(s)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              strategy === s
                ? "bg-accent text-white"
                : "bg-surface border border-edge text-muted hover:text-content"
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
            ? "border-accent bg-accent/10"
            : "border-edge bg-surface/50 hover:border-accent hover:bg-surface"
        }`}
      >
        <Upload
          size={32}
          className={`transition-colors ${dragOver ? "text-accent" : "text-faint"}`}
        />
        <div className="text-center">
          <p className="text-sm font-medium text-content">
            Drop files here or <span className="text-accent">browse</span>
          </p>
          <p className="text-xs text-faint mt-1">PDF, DOCX, CSV, XLSX, TXT · max 50 MB</p>
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

      {/* In-progress / failed uploads */}
      {activeRecords.length > 0 && (
        <ul className="space-y-2">
          {activeRecords.map((rec, i) => (
            <UploadRow key={i} record={rec} />
          ))}
        </ul>
      )}

      {/* Persisted knowledge-base documents */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Database size={15} className="text-muted" />
          <h3 className="text-sm font-medium text-content">
            Knowledge Base
            {!loadingDocs && (
              <span className="text-faint font-normal"> · {documents.length} document
                {documents.length === 1 ? "" : "s"}</span>
            )}
          </h3>
        </div>

        {deleteError && (
          <p className="text-xs text-red-500 mb-2">{deleteError}</p>
        )}

        {loadingDocs ? (
          <div className="flex items-center gap-2 text-sm text-faint py-4">
            <Loader2 size={14} className="animate-spin" /> Loading documents…
          </div>
        ) : documents.length === 0 ? (
          <p className="text-sm text-faint py-4 text-center border border-dashed border-edge rounded-xl">
            No documents yet. Upload a file to build your knowledge base.
          </p>
        ) : (
          <ul className="space-y-2">
            {documents.map((doc) => (
              <DocumentRow
                key={doc.source_file}
                doc={doc}
                onDelete={handleDelete}
                deleting={deleting === doc.source_file}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function DocumentRow({
  doc,
  onDelete,
  deleting,
}: {
  doc: DocumentInfo;
  onDelete: (sourceFile: string) => void;
  deleting: boolean;
}) {
  return (
    <li className="group flex items-center gap-3 rounded-xl bg-surface border border-edge px-4 py-3">
      <FileText size={16} className="shrink-0 text-accent" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-content truncate">{doc.source_file}</p>
        <div className="flex flex-wrap gap-x-4 mt-0.5 text-xs text-faint">
          <span>{doc.chunk_count} chunks</span>
          {doc.file_type && <span>{doc.file_type.toUpperCase()}</span>}
          {doc.chunking_strategy && <span>{doc.chunking_strategy}</span>}
        </div>
      </div>
      <button
        onClick={() => onDelete(doc.source_file)}
        disabled={deleting}
        className="shrink-0 text-faint hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100 disabled:opacity-100"
        title="Remove from knowledge base"
      >
        {deleting ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
      </button>
    </li>
  );
}

function UploadRow({ record }: { record: UploadRecord }) {
  const { file, status, error } = record;
  return (
    <li className="flex items-start gap-3 rounded-xl bg-surface border border-edge px-4 py-3">
      <div className="shrink-0 mt-0.5">
        {status === "uploading" && <Loader2 size={16} className="animate-spin text-accent" />}
        {status === "success"   && <CheckCircle2 size={16} className="text-emerald-500" />}
        {status === "error"     && <XCircle size={16} className="text-red-500" />}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-content truncate">{file.name}</p>
        {status === "uploading" && (
          <p className="text-xs text-faint mt-0.5">Parsing, chunking, embedding…</p>
        )}
        {status === "error" && (
          <p className="text-xs text-red-500 mt-0.5">{error}</p>
        )}
      </div>
    </li>
  );
}
