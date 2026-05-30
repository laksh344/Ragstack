"""Cross-encoder reranker — Cohere Rerank with fallback.

Takes the fused candidates from hybrid search and re-scores them with a
cross-encoder model, which attends to both the query and each document
jointly and produces higher-quality relevance estimates than the bi-encoder
embedding scores used in the first-stage retrieval.

Fallback: if COHERE_API_KEY is unset or the API call fails, the original
ordering is returned unchanged so the pipeline never hard-fails.
"""

import structlog

from backend.config import settings
from backend.retrieval import SearchResult

logger = structlog.get_logger()

_COHERE_MODEL = "rerank-english-v3.0"


class Reranker:
    """Cross-encoder reranker backed by Cohere Rerank API."""

    def __init__(self) -> None:
        self._client = None
        if settings.cohere_api_key:
            try:
                import cohere  # noqa: PLC0415

                self._client = cohere.Client(api_key=settings.cohere_api_key)
            except ImportError:
                logger.warning("reranker.cohere_import_failed")

    @property
    def available(self) -> bool:
        return self._client is not None

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank results with cross-encoder and return top-k.

        Args:
            query: The original user query.
            results: Candidate results from hybrid search.
            top_k: How many to return after reranking.

        Returns:
            Reranked results with ``score`` set to Cohere relevance score
            and ``source`` set to "reranked".  Falls back to the first
            ``top_k`` candidates from the input list if Cohere is unavailable.
        """
        if not results:
            return []

        if not self._client:
            logger.info("reranker.fallback", reason="no_client", top_k=top_k)
            return _mark_reranked(results[:top_k])

        documents = [r.content for r in results]

        try:
            response = self._client.rerank(
                model=_COHERE_MODEL,
                query=query,
                documents=documents,
                top_n=top_k,
            )
        except Exception as exc:
            logger.warning("reranker.api_error", error=str(exc))
            return _mark_reranked(results[:top_k])

        reranked: list[SearchResult] = []
        for item in response.results:
            original = results[item.index].model_copy()
            original.score = round(item.relevance_score, 6)
            original.source = "reranked"
            reranked.append(original)

        logger.info(
            "reranker.complete",
            candidates=len(results),
            returned=len(reranked),
        )
        return reranked


def _mark_reranked(results: list[SearchResult]) -> list[SearchResult]:
    """Tag results as reranked without changing scores (fallback path)."""
    out = []
    for r in results:
        c = r.model_copy()
        c.source = "reranked"
        out.append(c)
    return out
