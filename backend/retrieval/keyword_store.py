"""Elasticsearch keyword store — BM25 full-text search."""

import structlog
from elasticsearch import Elasticsearch

from backend.config import settings
from backend.retrieval import SearchResult

logger = structlog.get_logger()

# BM25 score is unbounded; we cap normalisation at this value so RRF
# scores from both stores are comparable when logged.
_MAX_BM25_SCORE = 20.0


class KeywordStore:
    """Wraps Elasticsearch for BM25 keyword search."""

    def __init__(self) -> None:
        self._es = Elasticsearch(settings.es_host)
        self._index = settings.es_index

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """BM25 search over content and title fields.

        Args:
            query: Natural-language search query.
            k: Number of results to return.
            filters: Optional metadata filters, e.g.
                     {"source_file": "report.pdf", "file_type": "pdf"}.
        """
        body: dict = {
            "size": k,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "title^2"],
                            "type": "best_fields",
                        }
                    }
                }
            },
        }

        if filters:
            body["query"]["bool"]["filter"] = [
                {"term": {field: value}} for field, value in filters.items()
            ]

        response = self._es.search(index=self._index, body=body)
        hits = response["hits"]["hits"]

        results = [_hit_to_result(hit) for hit in hits]
        logger.debug("keyword_store.search", query_len=len(query), hits=len(results))
        return results

    def index_info(self) -> dict:
        """Return basic stats about the ES index."""
        count = self._es.count(index=self._index)
        return {
            "index": self._index,
            "doc_count": count["count"],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit_to_result(hit: dict) -> SearchResult:
    src = hit.get("_source", {})
    raw_score = hit.get("_score") or 0.0
    # Normalise to [0, 1] so the score is interpretable alongside cosine.
    normalised = min(raw_score / _MAX_BM25_SCORE, 1.0)
    return SearchResult(
        chunk_id=src.get("chunk_id", hit.get("_id", "")),
        content=src.get("content", ""),
        source_file=src.get("source_file", ""),
        page_number=src.get("page_number", 0),
        chunk_index=src.get("chunk_index", 0),
        title=src.get("title", ""),
        file_type=src.get("file_type", ""),
        score=normalised,
        source="keyword",
        metadata=src.get("metadata", {}),
    )
