"""Retriever node — hybrid search + relevance-based sufficiency check.

Calls the full hybrid pipeline (vector + BM25 → RRF → Cohere rerank).
If the top result's score is below the relevance threshold, it sets
``needs_web_search=True`` so the graph falls back to Tavily.
"""

import structlog

from backend.agent.state import AgentState
from backend.retrieval.hybrid import HybridSearcher
from backend.retrieval.reranker import Reranker

logger = structlog.get_logger()

# RRF scores top out around 0.033 for a rank-1 hit in both lists.
# 0.008 means "appears in at least one list within the top ~120 results".
# Below this we consider retrieval insufficient and fall back to web search.
_RELEVANCE_THRESHOLD = 0.008
_FETCH_K = 20   # candidates to retrieve before reranking
_FINAL_K = 5    # results to keep after reranking


async def retriever_node(state: AgentState) -> dict:
    """Run hybrid search, rerank, and evaluate sufficiency."""
    query = state["query"]
    searcher = HybridSearcher()
    reranker = Reranker()

    candidates = await searcher.search(query, k=_FETCH_K)
    results = await reranker.rerank(query, candidates, top_k=_FINAL_K)

    sufficient = bool(results and results[0].score >= _RELEVANCE_THRESHOLD)
    needs_web = not sufficient

    logger.info(
        "retriever.complete",
        candidates=len(candidates),
        results=len(results),
        top_score=results[0].score if results else 0.0,
        sufficient=sufficient,
    )

    return {
        "retrieved_docs": [r.model_dump() for r in results],
        "needs_web_search": needs_web,
    }


def route_after_retrieval(state: AgentState) -> str:
    """Conditional edge: insufficient retrieval falls back to web search."""
    return "web_search" if state.get("needs_web_search") else "generator"
