# RAGStack — 2-Minute Demo Script

> Use this for your Loom recording. Keep each beat tight — the goal is to
> show breadth in 120 seconds, not depth.

---

## Setup (before recording)

- [ ] Backend running: `make docker-up && make dev`
- [ ] Frontend running: `cd frontend && npm run dev`
- [ ] LangSmith project open in a browser tab (smith.langchain.com)
- [ ] Sample doc ready: `sample_docs/rag_technical_overview.txt` (or a PDF)
- [ ] Browser at `localhost:3000`, dark mode, 1080p

---

## Script

### Beat 1 — Open the app (0:00–0:12)

> "This is RAGStack — a production-grade agentic RAG platform I built from scratch.
> It ingests PDFs and documents, runs hybrid vector plus keyword search with reranking,
> uses a LangGraph agent for reasoning, and has full safety guardrails.
> Let me show you."

*[camera on browser — homepage with chat UI visible]*

---

### Beat 2 — Upload a document (0:12–0:30)

> "First, I'll upload a document. Switch to the Upload tab."

*[click Upload tab → drag rag_technical_overview.txt into the zone]*

> "It parses the file with PyMuPDF, chunks it recursively, embeds with
> text-embedding-3-small, and stores in both Qdrant for vector search
> and Elasticsearch for BM25 keyword search."

*[ingestion result appears: pages, chunks, tokens, processing time]*

> "Done — 12 chunks, indexed in both stores."

---

### Beat 3 — Ask a question, show streaming (0:30–1:00)

> "Back to chat. Let me ask something that requires reasoning across the doc."

*[click Chat tab, type: "What are the tradeoffs between recursive and semantic chunking?"]*

> "The router classifies this as a knowledge base query. The retriever runs
> hybrid search — vector plus BM25, fused with reciprocal rank fusion —
> then Cohere reranks the candidates."

*[watch tokens stream in]*

> "The response streams token by token via Server-Sent Events. Notice the
> citation badges — these link to the exact source chunks."

*[response finishes — citation badges visible]*

---

### Beat 4 — Click a citation (1:00–1:15)

> "Click any citation to open the source panel."

*[click a citation badge → source panel slides in on the right]*

> "You can see the exact chunk: source file, page number, the relevant
> excerpt highlighted. This is what the model actually used."

---

### Beat 5 — LangSmith traces (1:15–1:35)

> "Switch to LangSmith — every pipeline step is fully traced."

*[switch to LangSmith tab, click the latest run]*

> "Here's the full run: router classified the intent, retriever ran hybrid
> search, reranker picked the top 5 chunks, generator produced the response,
> guardrails checked for PII and hallucination. Total latency: about 2 seconds."

*[expand any node to show inputs/outputs]*

---

### Beat 6 — Evaluation results (1:35–1:50)

> "I also have a golden eval dataset — 20 hand-crafted QA pairs. Let me
> show the latest eval run."

*[switch to terminal, quickly show `make eval` output OR show a saved results JSON]*

> "Faithfulness 0.84, answer relevance 0.79, retrieval relevance 0.81.
> Adding the Cohere reranker improved faithfulness by 18 percentage points —
> that's measurable, real impact, not just a checkbox."

---

### Beat 7 — Wrap up (1:50–2:00)

> "That's RAGStack: hybrid search, LangGraph agent, streaming chat,
> LangSmith observability, Presidio PII guardrails, and deployed to
> GCP Cloud Run with Terraform and GitHub Actions CI/CD.
> Link to the repo is in the description."

---

## Key talking points (if asked to go deeper)

| Topic | 30-second answer |
|---|---|
| Why hybrid search? | Vector misses exact terms; BM25 misses semantics. RRF fusion gets both with no tuning. |
| Why LangGraph? | Explicit state machine with typed edges — inspectable, guardrailable, and LangSmith-traced by default. |
| Why Qdrant? | Self-hostable (no API cost in dev), richer metadata filtering than Pinecone, great Python SDK. |
| Why Cohere reranker? | Cross-encoder is provably better than bi-encoder for ranking; +18pp faithfulness measured. |
| Why SSE not WebSocket? | Chat is strictly server→client; SSE is simpler, proxy-friendly, and degrades gracefully. |
| Guardrails? | Presidio PII detect+redact, LLM-as-judge hallucination (single batched call), injection regex, token budget. |
| Evaluation? | 20 golden QA pairs, 4 metrics (faithfulness, relevance ×2, citation accuracy), A/B comparison CLI. |
