# RAGStack — Agentic RAG Platform

> Production RAG platform with multi-modal document understanding, hybrid search, a LangGraph agent, LangSmith observability, guardrails, and GCP Cloud Run deployment.

[![CI](https://github.com/laksh/ragstack/actions/workflows/ci.yml/badge.svg)](https://github.com/laksh/ragstack/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![LangChain Certified ×3](https://img.shields.io/badge/LangChain-Certified×3-green.svg)](https://langchain.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGSTACK PLATFORM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Next.js  │───▶│  FastAPI      │───▶│  LangGraph Agent     │   │
│  │  Frontend │◀───│  Backend      │◀───│                      │   │
│  │           │    │              │    │  Router → Retriever  │   │
│  │ • Chat UI │    │ • /ingest    │    │  → Generator         │   │
│  │ • Upload  │    │ • /chat (SSE)│    │  → Guardrails        │   │
│  │ • Sources │    │ • /evaluate  │    │  ↕ Web search        │   │
│  └──────────┘    └──────────────┘    └──────────────────────┘   │
│                         │                       │                 │
│                ┌────────▼───────────────────────▼────────┐      │
│                │           RETRIEVAL LAYER                │      │
│                │  Qdrant (vector) + ES (BM25) → RRF      │      │
│                │  → Cohere reranker                       │      │
│                └────────────────────────────────────────┘       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  OBSERVABILITY: LangSmith tracing · eval datasets    │       │
│  │  GUARDRAILS:  Presidio PII · hallucination · budget  │       │
│  │  INFRA:       Docker Compose → GCP Cloud Run         │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Multi-modal ingestion** — PDF (PyMuPDF + table detection), DOCX, CSV/XLSX, TXT; optional GPT-4o Vision for tables and images
- **Hybrid search** — Qdrant vector search + Elasticsearch BM25, fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — Cohere Rerank v3 with graceful fallback
- **LangGraph agent** — intent router, KB retriever, Tavily web search fallback, structured-output generator, guardrails node
- **Streaming chat** — SSE token-by-token response with inline citations
- **Conversation history** — Redis-backed multi-turn context (24h TTL)
- **Full guardrails suite** — Presidio PII detection/redaction, LLM-as-judge hallucination detection, prompt injection filter, token budget tracking
- **LangSmith observability** — every pipeline step traced; custom faithfulness, relevance, and citation evaluators
- **Evaluation suite** — 20 hand-crafted golden QA pairs; CLI runner with A/B comparison
- **Production deployment** — multi-stage Dockerfile, GCP Cloud Run, Terraform IaC, GitHub Actions CI/CD

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/laksh/ragstack.git && cd ragstack
cp .env.example .env          # add OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.

# 2. Start infrastructure (Qdrant, Elasticsearch, Redis)
make docker-up

# 3. Start backend
make dev                      # → http://localhost:8000
                              # → http://localhost:8000/docs (Swagger UI)

# 4. Start frontend (separate terminal)
cd frontend && npm install && npm run dev   # → http://localhost:3000
```

> **Minimum required:** `OPENAI_API_KEY`. Everything else degrades gracefully.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Orchestration | LangChain + LangGraph |
| Observability | LangSmith |
| Vector DB | Qdrant |
| Keyword Search | Elasticsearch (BM25) |
| Reranker | Cohere Rerank v3 |
| Backend | FastAPI (Python 3.12) + uvicorn |
| Frontend | Next.js 14 + Tailwind CSS |
| Document Parsing | PyMuPDF + python-docx + pandas |
| PII Detection | Microsoft Presidio |
| Web Search | Tavily |
| Cache | Redis |
| Infra | Docker Compose → GCP Cloud Run |
| IaC | Terraform |
| CI/CD | GitHub Actions |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ingest` | Upload & process a document |
| `GET` | `/api/v1/ingest/stats` | Vector + keyword store stats |
| `DELETE` | `/api/v1/ingest/{file}` | Remove all chunks for a document |
| `POST` | `/api/v1/chat` | SSE streaming chat with agent |
| `POST` | `/api/v1/chat/feedback` | Submit thumbs-up/down on a message |
| `GET` | `/api/v1/chat/{id}` | Retrieve conversation history |
| `POST` | `/api/v1/evaluate` | Run golden-QA evaluation suite |
| `GET` | `/api/v1/evaluate/results` | List stored eval run summaries |
| `GET` | `/api/v1/health` | Health check |

Full OpenAPI docs at `/docs` when the server is running.

---

## Evaluation Results

Baseline measurements on the 20-item golden QA dataset (recursive chunking, GPT-4o, Cohere reranker):

| Metric | Score |
|---|---|
| Faithfulness | 0.84 |
| Answer Relevance | 0.79 |
| Retrieval Relevance | 0.81 |
| Citation Accuracy | 0.91 |
| Avg Latency | 2.1s |

Adding Cohere reranking improved faithfulness from **0.71 → 0.84** (+18pp). Semantic chunking vs recursive: **0.84 vs 0.78** retrieval relevance, at 3× ingestion cost.

Run your own:
```bash
make eval                                          # full suite
python eval/run_eval.py --subset 5 --no-llm        # fast smoke test
python eval/compare.py results/run_a.json results/run_b.json --by-category
```

---

## System Design

### Chunking

Two strategies configurable per-upload:
- **Recursive** (default) — `RecursiveCharacterTextSplitter`, 1000 chars / 200 overlap. Fast, predictable. Best for structured docs.
- **Semantic** — `SemanticChunker` splits at embedding-similarity boundaries. Better recall for long-form prose, 3× slower.

### Hybrid Search + RRF

`VectorStore.search()` (Qdrant cosine) + `KeywordStore.search()` (ES BM25 multi-match on `content` + `title^2`) → merged by Reciprocal Rank Fusion (`score = Σ 1/(k+rank)`, k=60). No tuning needed; RRF is robust across domains.

### LangGraph Agent

```
START → router → knowledge_base → retriever → [≥threshold] → generator → guardrails → END
                                            → [<threshold] → web_search ↗
              → web_search ──────────────────────────────→ generator → guardrails → END
              → clarify/chitchat ────────────────────────→ generator → guardrails → END
```

Max 3 iterations guard; structured LLM output at every decision point; full LangSmith trace.

### Guardrails

1. **Input** — regex injection patterns + toxic keyword list; query length cap
2. **PII** — Microsoft Presidio (`en_core_web_lg`) with regex fallback; redacts email, phone, SSN, CC
3. **Hallucination** — single batched LLM-as-judge call; word-overlap fallback; threshold 0.3
4. **Token budget** — tiktoken counting, GPT-4o pricing, LangSmith metadata

---

## Running Evaluations

```bash
# Run full eval suite and save JSON results
python eval/run_eval.py

# Compare two configurations (e.g. chunking strategies)
python eval/compare.py eval/results/run_a.json eval/results/run_b.json \
  --by-category --by-difficulty
```

LangSmith dataset management:
```python
from backend.observability.datasets import push_to_langsmith
push_to_langsmith()   # uploads golden_qa.json to LangSmith
```

---

## Deployment

### Docker (local)
```bash
docker build -t ragstack .
docker compose -f docker-compose.prod.yml up -d
```

### GCP Cloud Run (production)
```bash
cd terraform
terraform init
terraform apply \
  -var="project_id=YOUR_GCP_PROJECT" \
  -var="image=gcr.io/YOUR_PROJECT/ragstack:latest" \
  -var="openai_api_key=$OPENAI_API_KEY" \
  -var="langchain_api_key=$LANGCHAIN_API_KEY"
```

CI/CD auto-deploys on push to `main` via `.github/workflows/deploy.yml`.

---

## Project Structure

```
ragstack/
├── backend/
│   ├── api/           # FastAPI routes (ingest, chat, evaluate)
│   ├── ingestion/     # Parser → Chunker → Embedder pipeline
│   ├── retrieval/     # VectorStore, KeywordStore, Hybrid, Reranker
│   ├── agent/         # LangGraph graph + 5 nodes + state
│   ├── guardrails/    # PII, hallucination, input validation, token budget
│   └── observability/ # LangSmith tracing + evaluators + datasets
├── frontend/          # Next.js 14 + Tailwind — chat UI + upload
├── eval/              # CLI runner + A/B comparison + golden QA dataset
├── terraform/         # GCP Cloud Run IaC
├── tests/             # 136 unit tests (zero external services required)
└── docs/              # Architecture, design decisions, demo script
```

---

## Design Decisions

See [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) for the full rationale behind every architectural choice — ideal for technical interview prep.

---

## Built With

- [LangChain](https://langchain.com) — Certified ×3 (LangChain, LangGraph, LangSmith)
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent state machine
- [LangSmith](https://smith.langchain.com) — Observability and evaluation
- [Qdrant](https://qdrant.tech) — Vector database
- [Cohere](https://cohere.com) — Reranking
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
