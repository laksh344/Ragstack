"""Seed sample documents for development and demo.

Creates sample PDF, DOCX, CSV, and TXT files in sample_docs/
for testing the full ingestion pipeline.

Usage: python scripts/seed_data.py
"""

import csv
import random
from pathlib import Path


SAMPLE_DIR = Path("sample_docs")
SAMPLE_DIR.mkdir(exist_ok=True)


def create_sample_txt():
    """Create a technical document about RAG systems."""
    content = """Retrieval-Augmented Generation: A Technical Overview

1. Introduction

Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language models
by incorporating external knowledge sources. Rather than relying solely on the parametric
knowledge encoded during training, RAG systems retrieve relevant documents at inference time
and use them as context for generation. This approach significantly reduces hallucinations
and enables the model to access up-to-date information.

2. Architecture Components

2.1 Document Ingestion Pipeline

The ingestion pipeline is responsible for processing raw documents into searchable chunks.
This involves several stages: parsing (extracting text from various formats), chunking
(splitting text into manageable segments), embedding (converting text to vector representations),
and indexing (storing vectors and metadata for efficient retrieval).

Parsing must handle diverse formats including PDF, DOCX, CSV, HTML, and plain text.
Each format presents unique challenges — PDFs may contain tables and images that require
special handling, while DOCX files have complex XML structures with embedded formatting.

2.2 Chunking Strategies

The choice of chunking strategy significantly impacts retrieval quality. Common approaches include:

Fixed-size chunking: Split text into chunks of a predetermined character or token count.
Simple and fast, but may split sentences or paragraphs unnaturally.

Recursive character splitting: Attempt to split on natural boundaries (paragraphs, then
sentences, then words) before falling back to character-level splits. This preserves
semantic coherence within chunks.

Semantic chunking: Use embedding similarity to detect topic shifts within a document,
placing chunk boundaries where the semantic content changes. This produces variable-size
chunks that align with natural topic boundaries.

2.3 Embedding Models

Text embeddings convert natural language into dense vector representations that capture
semantic meaning. Modern embedding models like OpenAI's text-embedding-3-small produce
1536-dimensional vectors. These vectors enable similarity search — semantically similar
texts produce vectors that are close together in the embedding space.

Key considerations for embedding model selection include: dimensionality (higher dimensions
capture more nuance but require more storage), multilingual support, maximum input length,
and cost per token.

2.4 Vector Databases

Vector databases are optimized for storing and searching high-dimensional vectors.
Popular options include Qdrant, Pinecone, Weaviate, Milvus, and ChromaDB. These databases
support approximate nearest neighbor (ANN) search algorithms like HNSW that enable
millisecond-level similarity search across millions of vectors.

Most vector databases also support metadata filtering, allowing queries like "find similar
documents published after 2024" by combining vector similarity with structured filters.

2.5 Hybrid Search

Pure vector search excels at semantic similarity but can miss exact keyword matches.
BM25 keyword search (used by Elasticsearch) handles exact terms, acronyms, and IDs well
but lacks semantic understanding. Hybrid search combines both approaches using techniques
like Reciprocal Rank Fusion (RRF), which merges ranked results from both systems without
requiring score normalization.

2.6 Reranking

Initial retrieval typically returns more candidates than needed, ranked by a lightweight
scoring function. Cross-encoder rerankers (like Cohere Rerank or BGE-reranker) then
score each query-document pair jointly, producing much more accurate relevance judgments.
This two-stage approach balances efficiency (fast initial retrieval) with precision
(accurate reranking of a smaller candidate set).

3. Agent-Based RAG

Modern RAG systems increasingly incorporate agentic behavior — the ability to plan,
reason, and execute multi-step tasks autonomously. Rather than a fixed retrieve-then-generate
pipeline, an agent can:

- Decide whether to search the knowledge base, search the web, or ask for clarification
- Evaluate retrieval quality and retry with reformulated queries if results are insufficient
- Combine information from multiple sources before generating a response
- Apply guardrails and safety checks at each step

Frameworks like LangGraph enable building these agent workflows as state machines with
conditional edges, making the decision logic explicit and debuggable.

4. Evaluation and Observability

Production RAG systems require continuous evaluation across multiple dimensions:

Faithfulness: Are generated answers grounded in the retrieved context?
Answer relevance: Does the response actually address the user's question?
Retrieval relevance: Are the retrieved chunks relevant to the query?
Citation accuracy: Do source references correctly point to supporting evidence?

Tools like LangSmith provide end-to-end tracing of RAG pipelines, allowing developers
to inspect every step from query to response, identify bottlenecks, and run evaluation
datasets for regression testing.

5. Guardrails

Responsible AI deployment requires multiple safety layers:

PII detection and redaction: Identifying and masking personally identifiable information
in both inputs and outputs using tools like Microsoft Presidio.

Hallucination detection: Verifying that generated claims are supported by the retrieved
context, typically using an LLM-as-judge approach.

Prompt injection defense: Detecting and blocking attempts to manipulate the LLM's behavior
through crafted inputs.

Token budget controls: Tracking and limiting API costs per request, session, and user.

6. Conclusion

Building a production RAG system requires careful attention to every stage of the pipeline,
from document parsing through retrieval to generation and safety. The most impactful
improvements often come not from model selection but from retrieval quality — better
chunking, hybrid search, and reranking consistently outperform larger or newer models
in RAG benchmarks.
"""
    path = SAMPLE_DIR / "rag_technical_overview.txt"
    path.write_text(content)
    print(f"Created: {path}")


def create_sample_csv():
    """Create a sample financial dataset."""
    path = SAMPLE_DIR / "quarterly_revenue.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quarter", "product", "region", "revenue_usd", "units_sold", "growth_pct"])

        products = ["RAGStack Pro", "RAGStack Enterprise", "RAGStack Cloud", "RAGStack API"]
        regions = ["North America", "Europe", "Asia Pacific", "Latin America"]

        for year in [2024, 2025]:
            for q in range(1, 5):
                for product in products:
                    for region in regions:
                        base_rev = random.randint(50000, 500000)
                        growth = round(random.uniform(-5, 25), 1)
                        units = random.randint(100, 5000)
                        writer.writerow([
                            f"Q{q} {year}",
                            product,
                            region,
                            base_rev,
                            units,
                            growth,
                        ])

    print(f"Created: {path}")


def create_sample_txt_meeting():
    """Create a sample meeting notes document."""
    content = """RAGStack Architecture Review — Meeting Notes
Date: March 15, 2026
Attendees: Engineering Team (Laksh, Sarah, Mike, Priya)

DECISIONS MADE:

1. Vector Database Selection
   We evaluated Qdrant, Pinecone, and Weaviate for our vector storage needs.
   Decision: Qdrant (self-hosted via Docker)
   Rationale: Best metadata filtering, no vendor lock-in, generous free tier,
   and excellent Python SDK. Pinecone was ruled out due to cloud-only deployment.

2. Chunking Strategy
   After A/B testing with our evaluation dataset:
   - Recursive splitting: 78% retrieval relevance on golden QA set
   - Semantic chunking: 84% retrieval relevance, but 3x slower ingestion
   Decision: Default to recursive (fast), offer semantic as option for research docs.

3. Reranking Pipeline
   Tested with and without Cohere reranking on 100 queries:
   - Without reranker: 71% answer faithfulness
   - With Cohere rerank: 89% answer faithfulness
   Decision: Always rerank. The 200ms latency increase is worth the quality gain.

4. Agent Architecture
   Debated between single-pass RAG and agentic RAG with LangGraph:
   - Single-pass: simpler, 400ms average response time
   - Agentic: handles edge cases (insufficient context, ambiguous queries), 800ms avg
   Decision: Agentic. The ability to fall back to web search and self-evaluate
   retrieval quality is critical for production reliability.

ACTION ITEMS:
- Laksh: Implement hybrid search with RRF fusion by Friday
- Sarah: Set up LangSmith evaluation datasets with 50 golden QA pairs
- Mike: Deploy Qdrant + Elasticsearch on GCP with Terraform
- Priya: Build PII detection pipeline using Presidio

NEXT MEETING: March 22, 2026 — Sprint review and demo preparation
"""
    path = SAMPLE_DIR / "architecture_meeting_notes.txt"
    path.write_text(content)
    print(f"Created: {path}")


def create_sample_txt_api_docs():
    """Create sample API documentation."""
    content = """RAGStack API Documentation v0.1.0

BASE URL: https://api.ragstack.dev/api/v1

AUTHENTICATION:
All API requests require a Bearer token in the Authorization header.
Example: Authorization: Bearer sk-ragstack-xxxx

RATE LIMITS:
- Free tier: 100 requests/hour, 10MB max file size
- Pro tier: 1000 requests/hour, 50MB max file size
- Enterprise: Custom limits, SLA guarantees

ENDPOINTS:

1. POST /ingest
   Upload and process a document through the RAG pipeline.

   Request (multipart/form-data):
     - file (required): Document file (PDF, DOCX, CSV, TXT)
     - chunking_strategy (optional): "recursive" or "semantic" (default: "recursive")
     - chunk_size (optional): Characters per chunk (default: 1000)
     - chunk_overlap (optional): Overlap between chunks (default: 200)
     - use_vision (optional): Enable GPT-4o vision for tables/images (default: true)

   Response:
     {
       "document_id": "uuid",
       "source_file": "report.pdf",
       "total_pages": 42,
       "total_chunks": 156,
       "chunking_strategy": "recursive",
       "avg_chunk_size": 876.3,
       "processing_time_seconds": 12.4,
       "vision_pages_processed": 8
     }

   Error Codes:
     400 - Unsupported file type or file too large
     422 - Document could not be parsed or produced no chunks
     500 - Internal processing error

2. POST /chat
   Send a query to the agentic RAG system.

   Request:
     {
       "query": "What were the key decisions from the architecture review?",
       "dataset_id": "optional-dataset-filter",
       "conversation_id": "optional-for-multi-turn",
       "stream": true
     }

   Response (streaming):
     Each SSE event contains a chunk of the response with citation metadata.

   Response (non-streaming):
     {
       "response": "The architecture review resulted in four key decisions...",
       "citations": [
         {
           "source_file": "architecture_meeting_notes.txt",
           "page_number": 1,
           "chunk_text": "...",
           "relevance_score": 0.94
         }
       ],
       "agent_trace": {
         "route": "knowledge_base",
         "retrieval_count": 5,
         "reranked": true,
         "guardrails_passed": true
       }
     }

3. POST /evaluate
   Run the evaluation suite against a dataset.

   Request:
     {
       "dataset_name": "golden_qa",
       "config": {
         "chunking_strategy": "semantic",
         "use_reranker": true
       }
     }

   Response:
     {
       "faithfulness": 0.89,
       "answer_relevance": 0.92,
       "retrieval_relevance": 0.85,
       "citation_accuracy": 0.91,
       "avg_latency_ms": 1240,
       "total_queries": 20
     }

4. GET /ingest/stats
   Get current statistics from vector and keyword stores.

5. DELETE /ingest/{source_file}
   Remove all chunks for a document from both stores.

6. GET /health
   Health check endpoint.
"""
    path = SAMPLE_DIR / "api_documentation.txt"
    path.write_text(content)
    print(f"Created: {path}")


if __name__ == "__main__":
    print("Seeding sample documents...")
    create_sample_txt()
    create_sample_csv()
    create_sample_txt_meeting()
    create_sample_txt_api_docs()
    print(f"\nDone! {len(list(SAMPLE_DIR.iterdir()))} files in {SAMPLE_DIR}/")
