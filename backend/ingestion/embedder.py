"""Embedding pipeline — vectorize chunks and store in dual backends.

Stores vectors in Qdrant for semantic search and full text in
Elasticsearch for BM25 keyword search. This dual-store approach
enables hybrid retrieval (vector + keyword) with RRF fusion.

Reference: RAGFlow stores in ES for both full text and vectors.
We separate concerns: Qdrant for vectors, ES for BM25.
"""

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings

from backend.config import settings
from backend.models.document import Chunk

logger = structlog.get_logger()

# Batch size for embedding API calls (OpenAI limit is ~2048 per request)
EMBEDDING_BATCH_SIZE = 100


class EmbeddingPipeline:
    """Manages embedding generation and dual-store persistence."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.es = Elasticsearch(settings.es_host)
        self._ensure_stores()

    def _ensure_stores(self):
        """Create Qdrant collection and ES index if they don't exist."""
        # Qdrant collection
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if settings.qdrant_collection not in collections:
            self.qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("embedder.qdrant_created", collection=settings.qdrant_collection)

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
                "vectors_count": qdrant_info.vectors_count,
                "status": qdrant_info.status.value,
            },
            "elasticsearch": {
                "index": settings.es_index,
                "doc_count": es_count["count"],
            },
        }

    def delete_by_source(self, source_file: str) -> dict:
        """Delete all chunks for a given source file from both stores."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Delete from Qdrant
        self.qdrant.delete(
            collection_name=settings.qdrant_collection,
            points_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            ),
        )

        # Delete from Elasticsearch
        self.es.delete_by_query(
            index=settings.es_index,
            body={"query": {"term": {"source_file": source_file}}},
        )

        logger.info("embedder.deleted", source_file=source_file)
        return {"deleted_source": source_file}
