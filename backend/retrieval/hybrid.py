"""Hybrid search — Reciprocal Rank Fusion over vector + keyword results.

RRF formula: score(d) = Σ_r  1 / (k + rank_r(d))
where k=60 is the standard smoothing constant from the original RRF paper.
Documents that appear in only one result list still receive a partial score.
"""

import asyncio
from collections import defaultdict

import structlog

from backend.retrieval import SearchResult
from backend.retrieval.keyword_store import KeywordStore
from backend.retrieval.vector_store import VectorStore

logger = structlog.get_logger()

# Standard RRF smoothing constant (Cormack et al., 2009).
_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = _RRF_K,
) -> list[SearchResult]:
    """Merge multiple ranked result lists via Reciprocal Rank Fusion.

    Args:
        result_lists: One list per retrieval source, each already ranked
                      best-first.  Lists may overlap (same chunk_id).
        k: RRF smoothing constant.  60 works well across most tasks.

    Returns:
        A single list sorted by descending RRF score.  The ``score`` field
        holds the RRF value; ``source`` is set to "hybrid".
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    # Keep the best result object per chunk_id for payload reconstruction.
    best_result: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in best_result or result.score > best_result[cid].score:
                best_result[cid] = result

    merged: list[SearchResult] = []
    for cid, rrf_score in rrf_scores.items():
        r = best_result[cid].model_copy()
        r.score = round(rrf_score, 6)
        r.source = "hybrid"
        merged.append(r)

    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


class HybridSearcher:
    """Runs vector and keyword search in parallel, fuses with RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        keyword_store: KeywordStore | None = None,
    ) -> None:
        self._vector = vector_store or VectorStore()
        self._keyword = keyword_store or KeywordStore()

    async def search(
        self,
        query: str,
        k: int = 10,
        filters: dict | None = None,
        fetch_k: int | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search and return top-k fused results.

        Args:
            query: Natural-language search query.
            k: Final number of results after fusion.
            filters: Metadata filters forwarded to both stores.
            fetch_k: How many candidates to fetch from each store before
                     fusion (defaults to k * 2 for better recall).
        """
        n = fetch_k or k * 2

        # Keyword search is synchronous — run it in a thread to avoid blocking.
        vector_task = asyncio.create_task(
            self._vector.search(query, k=n, filters=filters)
        )
        keyword_results = await asyncio.to_thread(
            self._keyword.search, query, k=n, filters=filters
        )
        vector_results = await vector_task

        fused = reciprocal_rank_fusion([vector_results, keyword_results])
        top = fused[:k]

        logger.info(
            "hybrid.search",
            query_len=len(query),
            vector_hits=len(vector_results),
            keyword_hits=len(keyword_results),
            fused_hits=len(fused),
            returned=len(top),
        )
        return top
