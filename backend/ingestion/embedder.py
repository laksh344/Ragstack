"""Embedding pipeline — vectorize chunks and store in dual backends.

Stores vectors in Qdrant for semantic search and full text in
Elasticsearch for BM25 keyword search. This dual-store approach
enables hybrid retrieval (vector + keyword) with RRF fusion.

Reference: RAGFlow stores in ES for both full text and vectors.
We separate concerns: Qdrant for vectors, ES for BM25.
"""

import structlog
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from backend.config import settings
from backend.models.document import Chunk
from backend.utils.providers import get_embedding_dimensions, get_embeddings

logger = structlog.get_logger()

# Batch size for embedding API calls (OpenAI limit is ~2048 per request)
EMBEDDING_BATCH_SIZE = 100


class EmbeddingPipeline:
    """Manages embedding generation and dual-store persistence."""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.es = Elasticsearch(settings.es_host)
        self._ensure_stores()

    def _ensure_stores(self):
        """Create Qdrant collection and ES index if they don't exist.

        If a collection already exists with a different vector dimension than
        the configured embedding provider produces (e.g. after switching
        provider), it is recreated. This is safe because vectors of the wrong
        dimension cannot be queried against the new embeddings anyway.
        """
        dim = get_embedding_dimensions()

        # Qdrant collection — recreate if the vector size no longer matches.
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if settings.qdrant_collection in collections:
            existing_dim = self._existing_vector_size()
            if existing_dim is not None and existing_dim != dim:
                logger.warning(
                    "embedder.qdrant_dim_mismatch",
                    collection=settings.qdrant_collection,
                    existing=existing_dim,
                    expected=dim,
                )
                self.qdrant.delete_collection(settings.qdrant_collection)
                collections.remove(settings.qdrant_collection)

        if settings.qdrant_collection not in collections:
            self.qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(
                "embedder.qdrant_created",
                collection=settings.qdrant_collection,
                dimensions=dim,
            )

        # Elasticsearch index
        if not self.es.indices.exists(index=settings.es_index):
            self.es.indices.create(
                index=settings.es_index,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "default": {
                                    "type": "standard",
                                    "stopwords": "_english_",
                                }
                            }
                        },
                    },
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "content": {"type": "text", "analyzer": "default"},
                            "source_file": {"type": "keyword"},
                            "file_type": {"type": "keyword"},
                            "page_number": {"type": "integer"},
                            "chunk_index": {"type": "integer"},
                            "chunking_strategy": {"type": "keyword"},
                            "title": {"type": "text"},
                            "char_count": {"type": "integer"},
                            "metadata": {"type": "object", "enabled": False},
                        }
                    },
                },
            )
            logger.info("embedder.es_created", index=settings.es_index)

    def _existing_vector_size(self) -> int | None:
        """Return the vector dimension of the existing Qdrant collection."""
        try:
            info = self.qdrant.get_collection(settings.qdrant_collection)
            vectors = info.config.params.vectors
            # Single unnamed vector → VectorParams with .size
            return getattr(vectors, "size", None)
        except Exception:
            return None

    async def embed_and_store(self, chunks: list[Chunk]) -> dict:
        """Embed all chunks and store in both Qdrant and Elasticsearch.

        Returns stats about the embedding operation.
        """
        if not chunks:
            return {"chunks_stored": 0}

        logger.info("embedder.start", total_chunks=len(chunks))

        # Generate embeddings in batches
        all_vectors: list[list[float]] = []
        texts = [chunk.content for chunk in chunks]

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_vectors = await self.embeddings.aembed_documents(batch)
            all_vectors.extend(batch_vectors)
            logger.info(
                "embedder.batch",
                batch=i // EMBEDDING_BATCH_SIZE + 1,
                size=len(batch),
            )

        # Store in Qdrant
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "file_type": chunk.file_type,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "chunking_strategy": chunk.chunking_strategy,
                    "title": chunk.title,
                    "char_count": chunk.char_count,
                    "token_estimate": chunk.token_estimate,
                },
            )
            for chunk, vector in zip(chunks, all_vectors)
        ]

        # Qdrant upsert in batches of 100
        for i in range(0, len(points), 100):
            batch = points[i : i + 100]
            self.qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=batch,
            )

        logger.info("embedder.qdrant_stored", count=len(points))

        # Store in Elasticsearch
        for chunk in chunks:
            self.es.index(
                index=settings.es_index,
                id=chunk.id,
                body={
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "file_type": chunk.file_type,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "chunking_strategy": chunk.chunking_strategy,
                    "title": chunk.title,
                    "char_count": chunk.char_count,
                    "metadata": chunk.metadata,
                },
            )

        # Refresh ES index for immediate searchability
        self.es.indices.refresh(index=settings.es_index)
        logger.info("embedder.es_stored", count=len(chunks))

        total_tokens = sum(c.token_estimate for c in chunks)
        stats = {
            "chunks_stored": len(chunks),
            "total_characters": sum(c.char_count for c in chunks),
            "estimated_tokens": total_tokens,
            "estimated_embedding_cost_usd": round(total_tokens * 0.00002 / 1000, 4),
            "qdrant_collection": settings.qdrant_collection,
            "es_index": settings.es_index,
        }

        logger.info("embedder.complete", **stats)
        return stats

    def get_collection_stats(self) -> dict:
        """Get current stats from both stores."""
        qdrant_info = self.qdrant.get_collection(settings.qdrant_collection)
        es_count = self.es.count(index=settings.es_index)

        return {
            "qdrant": {
                "collection": settings.qdrant_collection,
                "points_count": qdrant_info.points_count,
                "vectors_count": getattr(qdrant_info, "vectors_count", None)
                or qdrant_info.points_count,
                "status": qdrant_info.status.value,
            },
            "elasticsearch": {
                "index": settings.es_index,
                "doc_count": es_count["count"],
            },
        }

    def list_documents(self) -> list[dict]:
        """List distinct ingested documents with per-document chunk counts.

        Aggregates over the Elasticsearch index by ``source_file`` so the UI
        can show what's currently in the knowledge base. Returns an empty list
        if the index doesn't exist yet.
        """
        if not self.es.indices.exists(index=settings.es_index):
            return []

        response = self.es.search(
            index=settings.es_index,
            body={
                "size": 0,
                "aggs": {
                    "documents": {
                        "terms": {"field": "source_file", "size": 1000},
                        "aggs": {
                            "file_type": {"terms": {"field": "file_type", "size": 1}},
                            "chunking_strategy": {
                                "terms": {"field": "chunking_strategy", "size": 1}
                            },
                        },
                    }
                },
            },
        )

        buckets = response["aggregations"]["documents"]["buckets"]
        documents = []
        for b in buckets:
            file_type_buckets = b.get("file_type", {}).get("buckets", [])
            strategy_buckets = b.get("chunking_strategy", {}).get("buckets", [])
            documents.append(
                {
                    "source_file": b["key"],
                    "chunk_count": b["doc_count"],
                    "file_type": file_type_buckets[0]["key"] if file_type_buckets else "",
                    "chunking_strategy": (
                        strategy_buckets[0]["key"] if strategy_buckets else ""
                    ),
                }
            )
        documents.sort(key=lambda d: d["source_file"].lower())
        return documents

    def delete_by_source(self, source_file: str) -> dict:
        """Delete all chunks for a given source file from both stores."""
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        flt = Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
        )

        # Delete from Qdrant. qdrant-client >= 1.12 takes points_selector
        # (a FilterSelector), not the old points_filter kwarg.
        self.qdrant.delete(
            collection_name=settings.qdrant_collection,
            points_selector=FilterSelector(filter=flt),
        )

        # Delete from Elasticsearch
        es_result = self.es.delete_by_query(
            index=settings.es_index,
            body={"query": {"term": {"source_file": source_file}}},
            refresh=True,
        )

        deleted = es_result.get("deleted", 0)
        logger.info("embedder.deleted", source_file=source_file, es_deleted=deleted)
        return {"deleted_source": source_file, "chunks_deleted": deleted}
