"""Qdrant vector store — semantic search via cosine similarity."""

import structlog
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.config import settings
from backend.retrieval import SearchResult

logger = structlog.get_logger()


class VectorStore:
    """Wraps Qdrant for semantic similarity search."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self._collection = settings.qdrant_collection

    async def search(
        self,
        query: str,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Embed query and return top-k semantically similar chunks.

        Args:
            query: Natural-language search query.
            k: Number of results to return.
            filters: Optional metadata filters, e.g.
                     {"source_file": "report.pdf", "file_type": "pdf"}.
        """
        vector = await self._embeddings.aembed_query(query)
        qdrant_filter = _build_filter(filters) if filters else None

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        )

        results = [_point_to_result(hit) for hit in hits]
        logger.debug("vector_store.search", query_len=len(query), hits=len(results))
        return results

    def collection_info(self) -> dict:
        """Return basic stats about the Qdrant collection."""
        info = self._client.get_collection(self._collection)
        return {
            "collection": self._collection,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filter(filters: dict) -> Filter:
    """Convert a flat dict of field→value pairs to a Qdrant must-Filter."""
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)


def _point_to_result(hit) -> SearchResult:
    p = hit.payload or {}
    return SearchResult(
        chunk_id=p.get("chunk_id", str(hit.id)),
        content=p.get("content", ""),
        source_file=p.get("source_file", ""),
        page_number=p.get("page_number", 0),
        chunk_index=p.get("chunk_index", 0),
        title=p.get("title", ""),
        file_type=p.get("file_type", ""),
        score=hit.score,
        source="vector",
        metadata={
            "char_count": p.get("char_count", 0),
            "token_estimate": p.get("token_estimate", 0),
            "chunking_strategy": p.get("chunking_strategy", ""),
        },
    )
