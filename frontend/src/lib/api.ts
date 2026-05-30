import type { SSEEvent, Citation, IngestionResult } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatCallbacks {
  onToken:      (token: string) => void;
  onCitations:  (citations: Citation[]) => void;
  onFlags:      (flags: string[]) => void;
  onDone:       (messageId: string, conversationId: string) => void;
  onError:      (detail: string) => void;
}

export async function streamChat(
  query: string,
  conversationId: string | null,
  callbacks: ChatCallbacks
): Promise<void> {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, conversation_id: conversationId }),
  });

  if (!response.ok || !response.body) {
    callbacks.onError(`Server error: ${response.status}`);
    return;
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by double newline
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";  // last incomplete frame stays in buffer

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data: ")) continue;
      try {
        const event: SSEEvent = JSON.parse(line.slice(6));
        switch (event.type) {
          case "token":          callbacks.onToken(event.content);                          break;
          case "citations":      callbacks.onCitations(event.data);                         break;
          case "guardrail_flags":callbacks.onFlags(event.data);                             break;
          case "done":           callbacks.onDone(event.message_id, event.conversation_id); break;
          case "error":          callbacks.onError(event.detail);                           break;
        }
      } catch {
        // malformed JSON — skip
      }
    }
  }
}

export async function submitFeedback(
  messageId: string,
  rating: 1 | -1 | 0,
  comment = ""
): Promise<void> {
  await fetch(`${BASE}/chat/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId, rating, comment }),
  });
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

export async function uploadDocument(
  file: File,
  chunkingStrategy: "recursive" | "semantic" = "recursive"
): Promise<IngestionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("chunking_strategy", chunkingStrategy);
  form.append("use_vision", "false");   // vision needs OpenAI key in env

  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function getIngestionStats(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/ingest/stats`);
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}
