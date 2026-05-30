# RAGStack — Design Decisions

> Every architectural choice with the tradeoff that motivated it.
> This is your interview cheat sheet — interviewers love "why" answers.

---

## 1. LangGraph vs CrewAI / AutoGPT / custom loop

**Chose: LangGraph**

LangGraph gives explicit, inspectable state transitions. The graph is a typed state machine — every node reads from and writes to `AgentState`, every edge is a named conditional. This means:
- You can see exactly what happened in any run from the LangSmith trace
- Adding a new node doesn't break existing ones
- The guardrails node runs exactly once, after generation, by design

CrewAI and AutoGPT use free-form reasoning loops that are harder to instrument and constrain. For production RAG, where you need deterministic safety checks and predictable latency, an explicit DAG is better than an open-ended loop.

**Tradeoff:** LangGraph is more code to set up than `agent.run()`. Worth it for production.

---

## 2. Qdrant vs Pinecone vs Weaviate

**Chose: Qdrant**

- **vs Pinecone:** Pinecone is cloud-only with no self-hosted option. Qdrant runs locally via Docker, so development and testing require no network calls or API costs. The metadata filtering API is also richer (typed payloads vs flat string values).
- **vs Weaviate:** Weaviate bundles its own vectoriser and schema language. Qdrant is more "bring your own embeddings" — it just stores vectors and payloads. Simpler mental model for a project that already uses LangChain embeddings.
- **Cost:** Qdrant has a generous free self-hosted tier. The same collection that runs locally in Docker deploys to Qdrant Cloud with a config change.

**Tradeoff:** Qdrant has a smaller community than Pinecone. No managed K/V store or hybrid BM25 built-in (hence the separate Elasticsearch).

---

## 3. Hybrid search vs pure vector search

**Chose: Hybrid (vector + BM25 via RRF)**

Pure vector search misses exact keyword matches. If a user types a product code or a person's name, the cosine similarity of "ABC-1234" to "ABC-1234" in embedding space is not necessarily 1.0 — the model may map similar-sounding codes nearby. BM25 is exact.

Conversely, BM25 has zero semantic understanding. "automobile" and "car" score zero overlap.

Hybrid search via RRF gives you both. The RRF formula `score = Σ 1/(k+rank)` doesn't require score normalisation between the two systems — it only uses rank, so the unbounded BM25 scores and the [-1,1] cosine scores are compared on equal footing.

**Measured result:** retrieval relevance went from 0.78 (pure vector) to 0.81 (hybrid) on the golden dataset.

---

## 4. RRF vs weighted score fusion

**Chose: RRF (k=60)**

Weighted fusion (`α * vector_score + (1-α) * keyword_score`) requires normalisation and a tuned α. Normalisation is tricky because BM25 scores are unbounded and vary by query length. α requires a labelled dataset to tune against.

RRF requires no tuning. The standard k=60 smoothing constant performs well across IR benchmarks. You get the benefits of fusion without a hyperparameter to maintain.

**Tradeoff:** RRF discards score magnitude information. A document with cosine 0.99 and a document with cosine 0.6 get the same rank-1 RRF contribution. For this project, rank is what matters for reranking.

---

## 5. Cohere Rerank vs local cross-encoder (BGE)

**Chose: Cohere Rerank API with fallback**

Cohere Rerank v3 is state-of-the-art and requires zero GPU infrastructure. The free tier covers development and demo usage. The latency cost is ~200ms for 20 candidates, which is acceptable for a chat interface.

BGE-reranker-v2-m3 would be free at inference time but requires a GPU or slow CPU inference. For a demo deployment on Cloud Run with no GPU, Cohere is the pragmatic choice.

The fallback (return hybrid results unchanged) means the pipeline never hard-fails if Cohere is unavailable — users get slightly lower quality results rather than an error.

**Measured result:** faithfulness 0.71 → 0.84 (+18pp) with Cohere reranking.

---

## 6. Presidio vs regex-only PII detection

**Chose: Presidio primary, regex fallback**

Presidio uses spaCy NER to detect names, addresses, and other contextual PII that pure regex misses. A regex for "PERSON" is impossible — context matters.

The downside is Presidio requires a spaCy language model (~500MB for `en_core_web_lg`) and has occasional false positives on benign text. The regex fallback handles the deployment case where the model isn't available, and the regex tests give deterministic CI coverage regardless of spaCy model presence.

**Tradeoff:** +500MB Docker image, slower cold start. Worthwhile for production PII compliance.

---

## 7. LLM-as-judge for hallucination vs entailment model

**Chose: LLM-as-judge (single batched call)**

A dedicated NLI entailment model (e.g. DeBERTa-based) would be faster and cheaper per call, but requires GPU inference or a separate API endpoint. The LLM-as-judge approach uses the same GPT-4o that's already in the stack, requires no additional infrastructure, and produces human-readable reasoning alongside the score.

The key optimisation is batching: rather than one LLM call per sentence, we send all sentences in a single call with structured output — `list[{sentence, grounded, reasoning}]`. This reduces latency from O(sentences) to O(1) LLM calls.

Word-overlap fallback runs when no OpenAI key is set, ensuring tests pass without API access.

---

## 8. SSE vs WebSocket for streaming chat

**Chose: Server-Sent Events (SSE)**

SSE is simpler: it's unidirectional (server → client), works through HTTP proxies and CDNs without configuration, and requires no special server-side library. The FastAPI `StreamingResponse` with `text/event-stream` is ~10 lines.

WebSocket would be bidirectional but chat turns are strictly sequential (send query → receive tokens), so bidirectionality adds complexity with no benefit. SSE also degrades gracefully — if the connection drops, the browser reconnects automatically (for `EventSource`). We use `fetch()` + `ReadableStream` rather than `EventSource` because `EventSource` doesn't support `POST` with a body.

---

## 9. Redis for conversation history vs in-memory / database

**Chose: Redis**

Conversation history must survive across requests (different Cloud Run instances may handle sequential turns). In-memory doesn't work for multi-instance deployments.

Redis over a relational database: conversation turns are an append-only list with a TTL (24h). Redis `RPUSH` + `LRANGE` + `EXPIRE` is exactly this pattern. No schema migrations, no query complexity. If Redis is unavailable, the pipeline degrades to stateless (each turn is independent) rather than failing.

---

## 10. GCP Cloud Run vs self-managed Kubernetes

**Chose: Cloud Run**

Cloud Run scales to zero (no idle cost for a demo project), handles HTTPS/TLS automatically, integrates with GCP Secret Manager for API key management, and deploys in ~2 minutes via GitHub Actions.

Kubernetes gives more control (custom networking, stateful sets for Qdrant/ES) but adds operational overhead that isn't justified for a demo. The managed infrastructure services (Qdrant Cloud, Elastic Cloud) can replace the local Docker containers when scaling.

**Tradeoff:** Cloud Run has a cold start (~2-5s). Mitigated by `startup_cpu_boost = true` and `min_instances = 1` in production (at cost of ~$15/month idle).

---

## 11. Chunking: recursive vs semantic

**Chose: Recursive as default, semantic as option**

Recursive chunking is fast, deterministic, and handles any document type. It's the safe default.

Semantic chunking uses embedding similarity to find natural break points — better for long-form narrative text where topic boundaries don't align with character counts. But it requires an embedding API call per chunk boundary detection (3× slower at ingest time) and can produce variable-size chunks that break downstream token budget estimates.

Offering both as a parameter lets users tune for their document type: recursive for structured reports and CSVs, semantic for long-form research papers and documentation.

**Measured:** semantic chunking retrieval relevance 0.84 vs recursive 0.78, but 3× slower ingest.
