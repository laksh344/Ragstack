# RAGStack — Detailed Architecture

## Request lifecycle (chat turn)

```
User query
  │
  ▼
POST /api/v1/chat  (FastAPI)
  │  Load conversation history from Redis
  │  Build initial AgentState
  │
  ▼
LangGraph graph.ainvoke()
  │
  ├─► router node
  │     LLM (GPT-4o) with structured output → RouterDecision
  │     intent: knowledge_base | web_search | clarify | chitchat
  │
  ├─► retriever node  [if knowledge_base]
  │     HybridSearcher.search(query, k=20)
  │       ├─ VectorStore.search()  → Qdrant cosine, top-20
  │       └─ KeywordStore.search() → ES BM25 multi-match, top-20
  │     reciprocal_rank_fusion([vector_results, keyword_results])
  │     Reranker.rerank(query, fused, top_k=5)  → Cohere API
  │     score < 0.008? → needs_web_search=True
  │
  ├─► web_search node  [if web_search or retrieval insufficient]
  │     TavilySearchResults.ainvoke(query)
  │     Store in state["search_results"]
  │
  ├─► generator node
  │     Build context from retrieved_docs + search_results (12k char cap)
  │     ChatOpenAI.with_structured_output(GeneratedResponse)
  │     → {answer: str, citations: list[Citation]}
  │
  └─► guardrails node
        InputValidator.validate(query)     → injection, toxicity, length
        PiiRedactor.detect(query)          → flag PII in input
        PiiRedactor.redact(response)       → remove PII from output
        HallucinationDetector.check()      → word-overlap fallback / LLM judge
        TokenBudget.track()                → tiktoken count + cost estimate
        add_run_metadata(...)              → attach to LangSmith trace
  │
  ▼
StreamingResponse  (text/event-stream)
  │  word-by-word tokens
  │  citations event
  │  guardrail_flags event
  └─ done event  (message_id, conversation_id)
  │
  ▼
Redis: RPUSH conversation:{id}:messages, EXPIRE 86400
```

---

## Ingestion pipeline

```
POST /api/v1/ingest  (multipart file upload)
  │
  ├─ Validate: file type, size limit (50MB)
  │
  ├─ parse_document()
  │   PDF   → PyMuPDF page-by-page; find_tables() → to_pandas() → to_markdown()
  │   DOCX  → python-docx; paragraphs + tables → markdown
  │   CSV   → pandas; head(50).to_markdown() + describe() + value_counts()
  │   TXT   → split into 50-line pages
  │
  ├─ enrich_with_vision()  [optional, if has_tables or has_images]
  │   GPT-4o Vision: "Extract all text, tables, describe diagrams"
  │   Merge vision text with parsed content
  │
  ├─ chunk_document()
  │   recursive: RecursiveCharacterTextSplitter(size=1000, overlap=200)
  │   semantic:  SemanticChunker (LangChain experimental)
  │   Each chunk: content, source_file, page_number, chunk_index, strategy
  │
  └─ EmbeddingPipeline.embed_and_store()
      OpenAIEmbeddings.aembed_documents(texts, batch=100)
      Qdrant.upsert(points)     → vector search
      Elasticsearch.index(doc)  → BM25 keyword search
      Return: {chunks_stored, estimated_tokens, cost_usd}
```

---

## Data models

```
ParsedDocument
  source_file, file_type, title, total_pages
  pages: list[ParsedPage]
    page_number, content, has_tables, has_images, table_data

Chunk
  id (uuid), content, source_file, file_type
  page_number, chunk_index, chunking_strategy
  char_count, token_estimate, metadata

SearchResult
  chunk_id, content, source_file, page_number
  score, source: "vector"|"keyword"|"hybrid"|"reranked"

AgentState (TypedDict)
  messages, query, route, iteration_count
  retrieved_docs, needs_web_search, search_results
  response, citations, guardrail_flags

Message (Redis/API)
  id, role, content, citations, guardrail_flags, timestamp
```

---

## Storage layout

| Store | Key pattern | Contents |
|---|---|---|
| Qdrant | collection: `ragstack` | Vectors + payload (chunk metadata) |
| Elasticsearch | index: `ragstack` | Full-text docs for BM25 |
| Redis | `conversation:{uuid}:messages` | Serialised `Message` JSON list, TTL 24h |
| Redis | `feedback:{message_id}` | `{rating, comment, timestamp}`, TTL 30d |
| Filesystem | `uploads/` | Raw uploaded files |
| Filesystem | `eval/results/` | Timestamped JSON eval run results |

---

## Observability

Every LangChain / LangGraph call is automatically traced to LangSmith when `LANGCHAIN_TRACING_V2=true`. Additional metadata attached per run:

```python
add_run_metadata({
    "user_id":             ...,
    "guardrail_flags":     [...],
    "hallucination_score": 0.12,
    "token_usage":         {input_tokens, output_tokens, cost_usd},
    "pii_engine":          "presidio",
})
```

Custom evaluators (faithfulness, answer_relevance, retrieval_relevance, citation_accuracy) can be run programmatically via `backend/observability/datasets.py` or the `POST /api/v1/evaluate` endpoint.

---

## Security considerations

- **API keys** stored in GCP Secret Manager; never in Docker images or environment variables at build time
- **PII** redacted from responses before they leave the guardrails node; also detected in queries (logged, not blocked)
- **Prompt injection** detected by regex patterns; query is validated before entering the agent
- **CORS** set to `allow_origins=["*"]` for demo; restrict to your domain in production
- **File upload** validates extension and size; content is not executed
- **Redis** should be network-isolated (not exposed publicly) in production
