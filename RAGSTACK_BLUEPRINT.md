# RAGStack — Agentic RAG Platform
## Architecture Blueprint & AI IDE Build Plan

> **Project Codename:** RAGStack
> **Tagline:** "An agentic RAG platform with multi-modal document understanding, hybrid search, LangSmith observability, guardrails, and cloud-native deployment."
> **Built by:** Laksh | LangChain Certified × 3

---

## 1. PROJECT POSITIONING (Interview Narrative)

**The Story You Tell Recruiters:**
"I didn't just take three LangChain courses — I built a production RAG platform that ingests PDFs with tables and images, runs hybrid vector + keyword search with reranking, uses a LangGraph agent that reasons over multiple tools, has full LangSmith observability with evaluation datasets, includes guardrails for hallucination detection and PII redaction, and is deployed on GCP with Docker and CI/CD. Here's the live demo."

**Skills This Project Proves:**
| Interview Skill (2026 Hot) | Feature That Proves It |
|---|---|
| AI Agents & Autonomous Systems | LangGraph multi-step reasoning agent |
| RAG / Vector Databases | Hybrid search + reranking pipeline |
| Multi-Modal AI | PDF/DOCX parsing with vision for tables/images |
| MLOps / LangSmith | Tracing, eval datasets, regression testing |
| Production Deployment | Docker, GCP Cloud Run, Terraform, CI/CD |
| Responsible AI / Guardrails | Hallucination detection, PII redaction, citation verification |
| LLM Fine-tuning Awareness | Configurable model routing (GPT-4o / Claude / Gemini / local) |
| Data Engineering for AI | Chunking strategies, embedding pipelines, metadata filtering |

---

## 2. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGSTACK PLATFORM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Next.js  │───▶│  FastAPI      │───▶│  LangGraph Agent     │   │
│  │  Frontend │◀───│  Backend      │◀───│  (Orchestrator)      │   │
│  │           │    │              │    │                      │   │
│  │ • Chat UI │    │ • /ingest    │    │ • Router Node        │   │
│  │ • Upload  │    │ • /chat      │    │ • Retriever Node     │   │
│  │ • Chunks  │    │ • /evaluate  │    │ • Web Search Node    │   │
│  │ • Sources │    │ • /datasets  │    │ • Generator Node     │   │
│  │ • Traces  │    │ • /admin     │    │ • Guardrails Node    │   │
│  └──────────┘    └──────────────┘    └──────────────────────┘   │
│                         │                       │                 │
│                         ▼                       ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    DATA & RETRIEVAL LAYER                    │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │ │
│  │  │  Qdrant     │  │ Elasticsearch│  │  Cross-Encoder    │  │ │
│  │  │  (Vectors)  │  │ (BM25/Full   │  │  Reranker         │  │ │
│  │  │             │  │  Text)       │  │  (Cohere/BGE)     │  │ │
│  │  └─────────────┘  └──────────────┘  └───────────────────┘  │ │
│  │         ▲                  ▲                                  │ │
│  │         │                  │                                  │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │              INGESTION PIPELINE                          │ │ │
│  │  │                                                           │ │ │
│  │  │  PDF/DOCX/CSV ──▶ Parser ──▶ Chunker ──▶ Embedder       │ │ │
│  │  │       │                                                   │ │ │
│  │  │       ▼                                                   │ │ │
│  │  │  Vision Model (tables/images) ──▶ Text + Metadata        │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    OBSERVABILITY LAYER                        │ │
│  │  LangSmith Tracing │ Eval Datasets │ Cost Tracking           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    GUARDRAILS LAYER                           │ │
│  │  PII Redaction │ Citation Verify │ Hallucination Detection   │ │
│  │  Token Budget  │ Input Validation │ Content Safety            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

INFRASTRUCTURE: Docker Compose (local) → GCP Cloud Run (prod)
               Terraform │ GitHub Actions CI/CD
```

---

## 3. REPO STRUCTURE

```
ragstack/
│
├── README.md                          # Killer README (architecture diagram, demo GIF, quick start)
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # uv/poetry project config
├── Makefile                           # dev commands: make dev, make test, make deploy
├── Dockerfile                         # Multi-stage production build
├── docker-compose.yml                 # Local dev: FastAPI + Qdrant + ES + Redis
├── docker-compose.prod.yml            # Production config
├── .env.example                       # Template for secrets
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Lint + test + type-check on PR
│       └── deploy.yml                 # Build & push to GCP Cloud Run
│
├── terraform/                         # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   └── cloud_run.tf
│
├── backend/                           # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                        # FastAPI app entry
│   ├── config.py                      # Settings via pydantic-settings
│   │
│   ├── api/                           # API Routes
│   │   ├── __init__.py
│   │   ├── ingest.py                  # POST /ingest — upload & process docs
│   │   ├── chat.py                    # POST /chat — agentic RAG conversation
│   │   ├── datasets.py               # CRUD for knowledge bases
│   │   ├── evaluate.py               # POST /evaluate — run eval suite
│   │   └── admin.py                  # Health, stats, config
│   │
│   ├── ingestion/                     # Document Processing Pipeline
│   │   ├── __init__.py
│   │   ├── parser.py                  # PDF/DOCX/CSV parsing (reference: ragflow/deepdoc/)
│   │   ├── vision.py                  # Multi-modal: extract tables/images via vision LLM
│   │   ├── chunker.py                 # Recursive + Semantic chunking strategies
│   │   ├── embedder.py                # Embedding pipeline (OpenAI/Cohere/local)
│   │   └── metadata.py               # Metadata extraction (title, date, page, source)
│   │
│   ├── retrieval/                     # Search & Retrieval
│   │   ├── __init__.py
│   │   ├── vector_store.py            # Qdrant operations
│   │   ├── keyword_store.py           # Elasticsearch BM25 operations
│   │   ├── hybrid.py                  # Reciprocal Rank Fusion (RRF) merger
│   │   └── reranker.py               # Cross-encoder reranking (Cohere/BGE)
│   │
│   ├── agent/                         # LangGraph Agentic RAG
│   │   ├── __init__.py
│   │   ├── graph.py                   # LangGraph state machine definition
│   │   ├── nodes/
│   │   │   ├── router.py             # Intent classification → tool selection
│   │   │   ├── retriever.py          # Knowledge base retrieval node
│   │   │   ├── web_search.py         # Tavily/Brave web search fallback
│   │   │   ├── generator.py          # LLM response generation with citations
│   │   │   └── guardrails.py         # Pre/post generation safety checks
│   │   ├── state.py                   # TypedDict state schema
│   │   └── tools.py                   # Tool definitions for the agent
│   │
│   ├── guardrails/                    # Safety & Quality
│   │   ├── __init__.py
│   │   ├── pii_redactor.py            # Presidio-based PII detection/redaction
│   │   ├── hallucination.py           # Citation verification against source chunks
│   │   ├── input_validator.py         # Prompt injection detection, toxicity filter
│   │   └── token_budget.py            # Cost control & token tracking
│   │
│   ├── observability/                 # LangSmith Integration
│   │   ├── __init__.py
│   │   ├── tracing.py                 # LangSmith callback configuration
│   │   ├── evaluators.py              # Custom evaluators (faithfulness, relevance, etc.)
│   │   └── datasets.py               # Eval dataset management
│   │
│   ├── models/                        # Database Models
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── conversation.py
│   │   └── dataset.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── llm_router.py             # Multi-provider LLM routing (OpenAI/Claude/Gemini)
│       └── file_utils.py
│
├── frontend/                          # Next.js / React Frontend
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx              # Landing / Chat interface
│   │   │   ├── upload/page.tsx       # Document upload & status
│   │   │   ├── datasets/page.tsx     # Knowledge base management
│   │   │   ├── chunks/page.tsx       # Chunk visualization & inspection
│   │   │   └── traces/page.tsx       # LangSmith trace viewer embed
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx        # Main chat with streaming
│   │   │   ├── SourcePanel.tsx       # Citation/source side panel
│   │   │   ├── ChunkViewer.tsx       # Visual chunk inspector
│   │   │   ├── UploadZone.tsx        # Drag-drop file upload
│   │   │   └── EvalDashboard.tsx     # Evaluation results display
│   │   └── lib/
│   │       ├── api.ts                # Backend API client
│   │       └── types.ts             # TypeScript interfaces
│   └── public/
│
├── eval/                              # Evaluation Suite
│   ├── datasets/
│   │   ├── golden_qa.json            # Hand-crafted Q&A pairs
│   │   └── retrieval_relevance.json  # Retrieval quality test set
│   ├── run_eval.py                    # CLI eval runner
│   └── compare.py                    # A/B comparison between configs
│
├── scripts/
│   ├── seed_data.py                   # Load sample documents
│   ├── benchmark.py                   # Latency & throughput benchmarks
│   └── export_traces.py              # Export LangSmith traces for analysis
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_agent.py
│   ├── test_guardrails.py
│   └── test_api.py
│
├── docs/
│   ├── ARCHITECTURE.md               # Detailed architecture doc
│   ├── DESIGN_DECISIONS.md           # Why you chose X over Y (interview gold)
│   └── DEMO_SCRIPT.md               # 2-min Loom demo walkthrough script
│
└── sample_docs/                       # Demo documents for quick testing
    ├── annual_report.pdf
    ├── technical_spec.docx
    └── financial_data.csv
```

---

## 4. RAGFLOW CODE REFERENCE MAP

These are the specific RAGFlow modules to study/reference for each feature. **Don't copy-paste — understand the pattern, then build your own with LangChain.**

| Your Module | RAGFlow Reference | What to Learn |
|---|---|---|
| `ingestion/parser.py` | `ragflow/deepdoc/parser/` | PDF layout analysis, table detection, multi-format parsing |
| `ingestion/vision.py` | `ragflow/rag/nlp/` + DeepDoc vision pipeline | How they extract text from images/tables within PDFs |
| `ingestion/chunker.py` | `ragflow/rag/utils/` | Template-based chunking strategies, how they handle different doc types |
| `retrieval/hybrid.py` | `ragflow/rag/` retrieval modules | Their fusion of BM25 + vector search, reciprocal rank fusion |
| `retrieval/reranker.py` | `ragflow/rag/` reranker integration | How they integrate cross-encoder reranking post-retrieval |
| `agent/graph.py` | `ragflow/agent/` canvas/agent templates | Agent workflow patterns, tool orchestration |
| `api/chat.py` | `ragflow/api/apps/` conversation endpoints | Streaming chat, session management, citation formatting |
| `frontend/ChatWindow.tsx` | `ragflow/web/src/` React components | Chat UI patterns, source citation display, chunk visualization |
| Docker setup | `ragflow/docker/` compose files | Multi-service orchestration patterns |

### Key RAGFlow Files to Read First:
```
ragflow/
├── rag/                    # Core RAG logic — your primary reference
│   ├── nlp/                # NLP utilities, tokenization
│   ├── utils/              # Chunking, embedding utilities
│   └── app/                # RAG application logic
├── deepdoc/                # Document understanding — study the approach
│   ├── parser/             # PDF, DOCX, Excel parsers
│   └── vision/             # OCR, layout analysis
├── agent/                  # Agent framework — study patterns
│   ├── canvas/             # Visual agent builder
│   └── component/          # Agent components (tools, nodes)
├── api/                    # REST API — reference for your FastAPI routes
│   └── apps/               # Endpoint implementations
└── web/src/                # React frontend — reference for UI patterns
    ├── pages/              # Page components
    └── components/         # Reusable UI components
```

---

## 5. TECH STACK (Final)

| Layer | Technology | Why |
|---|---|---|
| **LLM Orchestration** | LangChain + LangGraph | Your certifications. Agent state machine. |
| **Observability** | LangSmith | Tracing, evals, datasets — direct cert proof |
| **Vector DB** | Qdrant (Docker) | Fast, typed, filter-friendly. Free tier for demo. |
| **Keyword Search** | Elasticsearch | BM25 hybrid search. Industry standard. |
| **Reranker** | Cohere Rerank API / BGE-reranker-v2 | Production reranking. Cohere has free tier. |
| **Backend** | FastAPI (Python 3.12) | Async, typed, OpenAPI docs auto-generated |
| **Frontend** | Next.js 14 + Tailwind + shadcn/ui | Fast to build with AI IDE. SSR for demo. |
| **Document Parsing** | PyMuPDF + Unstructured + GPT-4o Vision | Multi-modal pipeline |
| **PII Detection** | Microsoft Presidio | Industry-standard, open-source |
| **Web Search** | Tavily API | Built for LangChain agents. Free tier. |
| **Cache** | Redis | Session state, embedding cache |
| **Infra** | Docker Compose → GCP Cloud Run | Local dev → production path |
| **IaC** | Terraform | Interview signal for production thinking |
| **CI/CD** | GitHub Actions | Standard. Auto-deploy on main push. |

---

## 6. WEEK-BY-WEEK BUILD PLAN (AI IDE Optimized)

> **Optimization for Claude Code / Antigravity:**
> Each week has specific "AI IDE prompts" — copy these directly into your AI IDE session.
> Files are ordered by dependency chain so the AI has maximum context.

---

### WEEK 1: Foundation + Ingestion Pipeline (Days 1-7)

**Goal:** Documents go in → chunks with embeddings come out

**Day 1-2: Project Scaffold**
```
AI IDE Prompt:
"Create a Python FastAPI project called 'ragstack' with:
- pyproject.toml using uv with dependencies: fastapi, uvicorn, langchain,
  langchain-openai, langchain-community, langsmith, qdrant-client,
  elasticsearch, python-multipart, pydantic-settings, redis, pymupdf,
  unstructured, presidio-analyzer, presidio-anonymizer
- docker-compose.yml with services: qdrant (port 6333), elasticsearch
  (port 9200), redis (port 6379)
- backend/main.py with health check endpoint
- backend/config.py with pydantic-settings loading from .env
- .env.example with placeholders for OPENAI_API_KEY, LANGSMITH_API_KEY,
  COHERE_API_KEY, TAVILY_API_KEY
- Makefile with: dev, test, lint, docker-up, docker-down commands
Follow the repo structure I'll provide."
```

**Day 3-4: Document Parsers**
```
AI IDE Prompt:
"Build backend/ingestion/parser.py that:
- Accepts PDF, DOCX, CSV, TXT files
- For PDFs: use PyMuPDF to extract text page-by-page, preserving page numbers
  and detecting tables using pymupdf table extraction
- For DOCX: use python-docx to extract paragraphs and tables
- For CSV: use pandas to convert to structured text
- Returns a list of Document objects with: content, metadata (source, page,
  file_type, title)
- Reference pattern: RAGFlow's deepdoc/parser/ splits by format similarly

Build backend/ingestion/vision.py that:
- Takes pages where table/image detection triggered
- Sends page image to GPT-4o Vision with prompt: 'Extract all text,
  tables (as markdown), and describe any diagrams from this document page'
- Returns extracted text to merge with parser output
- This is the multi-modal differentiator feature
```

**Day 5-6: Chunking + Embedding**
```
AI IDE Prompt:
"Build backend/ingestion/chunker.py with two strategies:
1. RecursiveCharacterTextSplitter (LangChain) with configurable chunk_size
   (default 1000) and overlap (200)
2. SemanticChunker using embedding similarity to find natural break points
   (use langchain_experimental.text_splitter.SemanticChunker)
- Each chunk preserves metadata: source_file, page_number, chunk_index,
  chunking_strategy
- Expose a factory: get_chunker(strategy='recursive'|'semantic')

Build backend/ingestion/embedder.py that:
- Uses LangChain's OpenAIEmbeddings (default) with model text-embedding-3-small
- Batches chunks (max 100 per batch) for efficiency
- Stores vectors in Qdrant with payload metadata
- Also indexes full text in Elasticsearch for BM25
- Returns ingestion stats: num_chunks, avg_chunk_size, total_tokens
```

**Day 7: Ingest API + Test**
```
AI IDE Prompt:
"Build backend/api/ingest.py with:
- POST /api/v1/ingest — accepts file upload (multipart)
- Runs: parse → chunk → embed → store pipeline
- Returns job status with stats
- Add LangSmith tracing decorator to the full pipeline

Write tests/test_ingestion.py that:
- Tests PDF parsing with a sample file
- Tests both chunking strategies
- Verifies chunks are stored in Qdrant
- Verifies BM25 index in Elasticsearch
```

---

### WEEK 2: Retrieval + Agentic RAG (Days 8-14)

**Goal:** Query goes in → agent reasons → cited answer comes out

**Day 8-9: Hybrid Search**
```
AI IDE Prompt:
"Build the retrieval layer:

backend/retrieval/vector_store.py:
- QdrantVectorStore wrapper using langchain_qdrant
- search(query, k=10, filters=None) method
- Metadata filtering support (by source, date range, file type)

backend/retrieval/keyword_store.py:
- Elasticsearch BM25 search wrapper
- search(query, k=10) method returning scored results

backend/retrieval/hybrid.py:
- Takes results from both vector and keyword search
- Implements Reciprocal Rank Fusion (RRF) to merge results
- Formula: score = sum(1 / (k + rank)) across both result lists
- Returns unified ranked results with both scores

backend/retrieval/reranker.py:
- Uses Cohere Rerank API (cohere.Client.rerank)
- Takes top-N from hybrid search, reranks with cross-encoder
- Returns final top-K results with relevance scores
- Fallback: if Cohere unavailable, return hybrid results as-is
```

**Day 10-12: LangGraph Agent**
```
AI IDE Prompt:
"Build the agentic RAG system using LangGraph:

backend/agent/state.py:
- TypedDict 'AgentState' with fields: messages, query, retrieved_docs,
  search_results, response, citations, guardrail_flags, iteration_count

backend/agent/nodes/router.py:
- Classifies user intent: 'knowledge_base' | 'web_search' | 'clarify' | 'chitchat'
- Uses LLM with structured output to decide which tool to use
- If query is ambiguous, routes to 'clarify' node

backend/agent/nodes/retriever.py:
- Calls hybrid search pipeline (vector + BM25 + rerank)
- Evaluates if retrieved docs are sufficient (relevance threshold)
- If insufficient, signals to try web search as fallback

backend/agent/nodes/web_search.py:
- Uses Tavily search API via LangChain TavilySearchResults tool
- Formats web results as context documents
- Only triggered when KB retrieval is insufficient

backend/agent/nodes/generator.py:
- Takes retrieved context + user query
- Generates response with EXPLICIT citations [Source: filename, page X]
- Uses structured output to separate answer from citations

backend/agent/nodes/guardrails.py:
- Pre-generation: input validation, PII check on query
- Post-generation: hallucination detection (verify claims against sources),
  PII redaction on output
- Returns guardrail_flags in state

backend/agent/graph.py:
- LangGraph StateGraph connecting all nodes:
  START → router → (retriever|web_search|clarify) → generator → guardrails → END
- Conditional edges based on router output and retrieval sufficiency
- Max 3 iterations to prevent infinite loops
- Full LangSmith tracing on every node transition
```

**Day 13-14: Chat API + Streaming**
```
AI IDE Prompt:
"Build backend/api/chat.py with:
- POST /api/v1/chat — accepts {query, dataset_id, conversation_id}
- Invokes LangGraph agent
- Returns streaming response using FastAPI StreamingResponse
- Includes citations and source documents in response
- Stores conversation history in Redis for multi-turn context

- POST /api/v1/chat/feedback — accepts {message_id, rating, comment}
  for collecting user feedback (feeds into LangSmith datasets later)
```

---

### WEEK 3: Guardrails + Observability + Evaluation (Days 15-21)

**Goal:** Production safety + measurable quality

**Day 15-16: Guardrails Suite**
```
AI IDE Prompt:
"Build the guardrails layer:

backend/guardrails/pii_redactor.py:
- Uses Microsoft Presidio AnalyzerEngine + AnonymizerEngine
- Detects: email, phone, SSN, credit card, names, addresses
- Two modes: 'detect' (flag) and 'redact' (replace with <PII_TYPE>)
- Applied to both input queries and output responses

backend/guardrails/hallucination.py:
- Takes generated response + source chunks
- For each claim/sentence in response, checks if it's grounded in sources
- Uses LLM-as-judge: 'Is this claim supported by the provided context?'
- Returns hallucination_score (0-1) and flagged_sentences list
- Threshold: if score > 0.3, add warning to response

backend/guardrails/input_validator.py:
- Prompt injection detection (common patterns + LLM classification)
- Toxicity filtering using basic keyword + LLM check
- Query length limits and rate limiting helpers

backend/guardrails/token_budget.py:
- Track tokens per request (input + output)
- Configurable budget per user/session
- Log costs to LangSmith as metadata
```

**Day 17-19: LangSmith Observability**
```
AI IDE Prompt:
"Build the observability layer:

backend/observability/tracing.py:
- Configure LangSmith tracing globally for all LangChain calls
- Custom run metadata: user_id, dataset_id, model_used, total_cost
- Trace the full pipeline: ingest, retrieve, generate, guardrails

backend/observability/evaluators.py:
- Custom LangSmith evaluators:
  1. 'faithfulness' — is the answer grounded in retrieved docs?
  2. 'answer_relevance' — does the answer address the question?
  3. 'retrieval_relevance' — are retrieved chunks relevant to query?
  4. 'citation_accuracy' — do citations point to correct sources?
- Each evaluator returns a score (0-1) and reasoning

backend/observability/datasets.py:
- Create/manage LangSmith evaluation datasets
- Import from eval/datasets/golden_qa.json
- Run evaluations programmatically and store results

backend/api/evaluate.py:
- POST /api/v1/evaluate — runs eval suite against a dataset
- Returns per-question scores + aggregate metrics
- Compares two configurations side-by-side (A/B)
```

**Day 20-21: Evaluation Datasets**
```
AI IDE Prompt:
"Create eval/datasets/golden_qa.json with 20 hand-crafted Q&A pairs:
- 10 questions that should be answerable from the sample_docs
- 5 questions that require multi-doc reasoning
- 5 questions that should trigger 'I don't know' (out of scope)
- Each entry: {question, expected_answer, source_doc, difficulty}

Create eval/run_eval.py:
- CLI script that runs the full eval suite
- Outputs: per-question scores, aggregate metrics, latency stats
- Saves results to eval/results/ with timestamp

Create eval/compare.py:
- Compare two eval runs (e.g., recursive vs semantic chunking)
- Output side-by-side metrics table
- This is your 'I do A/B testing on my RAG pipeline' interview story
```

---

### WEEK 4: Frontend + Deployment + Polish (Days 22-30)

**Goal:** Live demo URL + polished GitHub repo

**Day 22-24: Frontend**
```
AI IDE Prompt:
"Create a Next.js 14 frontend for RAGStack:

src/app/page.tsx — Main chat interface:
- Full-screen chat with message bubbles
- Streaming response display
- Source citations panel on the right (click to expand)
- 'Upload Documents' button in header

src/components/ChatWindow.tsx:
- Input field with send button
- Streaming text display using fetch + ReadableStream
- Each AI message shows: response text, citation badges, confidence score
- Citation badges are clickable → expand source panel

src/components/SourcePanel.tsx:
- Shows retrieved chunks for current response
- Highlights relevant passages
- Shows: source file, page number, relevance score, chunk text

src/components/ChunkViewer.tsx:
- Visualize how a document was chunked
- Color-coded chunks with metadata overlay
- Toggle between recursive and semantic chunking view

src/components/UploadZone.tsx:
- Drag-and-drop file upload with progress
- File type validation (PDF, DOCX, CSV, TXT)
- Processing status indicator

Use Tailwind CSS + shadcn/ui components.
Clean, professional design — dark mode, good typography."
```

**Day 25-26: Docker + Cloud Deploy**
```
AI IDE Prompt:
"Create production deployment setup:

Dockerfile (multi-stage):
- Stage 1: Build frontend (node:20-alpine)
- Stage 2: Python backend (python:3.12-slim)
- Copies built frontend into backend static serving
- Exposes port 8000

docker-compose.prod.yml:
- All services with production configs
- Resource limits, restart policies, health checks
- Nginx reverse proxy for frontend

terraform/main.tf:
- GCP provider config
- Cloud Run service for the app
- Artifact Registry for Docker images
- Secret Manager for API keys

terraform/cloud_run.tf:
- Cloud Run service definition
- CPU/memory allocation
- Min 0 / Max 3 instances (cost-effective demo)
- Custom domain mapping (optional)

.github/workflows/ci.yml:
- On PR: lint (ruff), type-check (mypy), test (pytest)

.github/workflows/deploy.yml:
- On push to main: build Docker → push to GCR → deploy Cloud Run
- Environment secrets for API keys
```

**Day 27-28: README + Documentation**
```
AI IDE Prompt:
"Create a killer README.md for RAGStack that includes:
- Project logo/banner area
- One-line description
- Architecture diagram (link to image)
- Feature list with checkmarks
- 'Quick Start' section (3 commands to run locally)
- 'System Design' section with component explanation
- 'Design Decisions' section (why LangGraph over CrewAI, why Qdrant
  over Pinecone, why hybrid search, why Presidio for PII)
- 'Evaluation Results' section with metrics table
- 'Tech Stack' badges
- Screenshots/GIFs of the UI
- 'Built With' section linking to LangChain certifications
- License

Also create docs/DESIGN_DECISIONS.md:
- Every architectural choice with pros/cons/tradeoffs
- This is your interview cheat sheet — interviewers love 'why' answers
- Cover: chunking strategy, embedding model choice, reranking approach,
  agent architecture, guardrails design, deployment strategy

Also create docs/DEMO_SCRIPT.md:
- 2-minute Loom recording script
- Walk through: upload doc → ask question → show sources →
  show traces in LangSmith → show eval results
```

**Day 29-30: Record Demo + LinkedIn Post**
```
Tasks (not AI IDE — manual):
1. Record 2-min Loom demo following DEMO_SCRIPT.md
2. Take 3-4 polished screenshots of the UI
3. Draft LinkedIn post:
   - Hook: "I just built a production RAG platform from scratch..."
   - What it does (3 bullets)
   - Technical highlights (agent, hybrid search, guardrails)
   - Link to certifications
   - Link to GitHub + live demo
   - Call to action for recruiters/hiring managers
4. Pin GitHub repo
5. Update LinkedIn Featured section with repo + demo
```

---

## 7. AI IDE SESSION STRATEGY

### Claude Code / Antigravity Best Practices:

**Session 1 (per feature):** Always start with:
```
"I'm building [feature]. Here's my repo structure: [paste relevant section].
Here's what's already built: [list completed files].
Here's the RAGFlow reference I studied: [describe pattern].
Build [specific file] following this spec: [paste from blueprint]."
```

**Context Loading Order:**
1. Always load `backend/config.py` first (settings)
2. Then load the dependency files (e.g., load `retrieval/` before `agent/`)
3. Then build the new file
4. Then write tests

**RAGFlow Reference Workflow:**
```
1. Clone ragflow locally: git clone https://github.com/infiniflow/ragflow.git
2. Before building each module, read the corresponding ragflow code
3. Tell Claude Code: "I studied ragflow/deepdoc/parser/ — they do X.
   I want to achieve similar result but using LangChain + PyMuPDF.
   Build my version in backend/ingestion/parser.py"
```

**Iteration Pattern:**
```
Build → Test → Trace (LangSmith) → Evaluate → Improve → Commit
```

**Git Commit Strategy:**
- Commit after each working file (not end of day)
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`
- This creates a clean commit history for recruiters who check

---

## 8. DEMO-READY SAMPLE DATA

Prepare these documents for your demo (publicly available, impressive):
1. **Tesla Annual Report PDF** — complex tables, images, financial data
2. **A technical RFC/spec** — dense text, multi-section
3. **A CSV dataset** — structured data ingestion proof
4. **A DOCX with mixed content** — tables + images + text

These show multi-format handling and make the demo visually impressive.

---

## 9. INTERVIEW TALKING POINTS

When asked "Tell me about a project you built":

1. **Architecture:** "I designed a multi-layer system — ingestion pipeline with multi-modal parsing, hybrid retrieval with reranking, a LangGraph agent for autonomous reasoning, and a guardrails layer for safety."

2. **Trade-offs:** "I chose Qdrant over Pinecone because [free self-hosted, metadata filtering]. I used RRF over weighted fusion because [no tuning needed, robust]. I added a semantic chunker option because [better for long-form docs] but kept recursive as default because [faster, predictable]."

3. **Production thinking:** "I added LangSmith tracing on every pipeline stage, built custom evaluators for faithfulness and citation accuracy, created golden datasets for regression testing, and deployed with Terraform to GCP Cloud Run with CI/CD."

4. **Guardrails:** "I implemented PII redaction with Presidio, hallucination detection by verifying claims against source chunks, and prompt injection filtering — because production AI needs safety."

5. **Metrics:** "On my golden dataset, I achieved X% faithfulness, Y% retrieval relevance, and reduced hallucination rate from A% to B% by adding the reranker."

---

*Built for execution with AI IDEs. Every module is spec'd to be a single Claude Code session.*
