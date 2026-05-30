"""LangChain tool definitions for the RAGStack agent.

These are callable tool objects that nodes can invoke directly or that
could be exposed to a ReAct-style agent in future iterations.
"""

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from backend.config import settings


@tool
async def hybrid_search(query: str, k: int = 10) -> list[dict]:
    """Search the knowledge base using hybrid vector + BM25 retrieval with reranking.

    Args:
        query: Natural-language question or search query.
        k: Number of results to return after reranking.

    Returns:
        List of chunk dicts with content, source_file, page_number, score.
    """
    from backend.retrieval.hybrid import HybridSearcher
    from backend.retrieval.reranker import Reranker

    searcher = HybridSearcher()
    reranker = Reranker()

    candidates = await searcher.search(query, k=k * 2)
    reranked = await reranker.rerank(query, candidates, top_k=k)
    return [r.model_dump() for r in reranked]


def get_web_search_tool(max_results: int = 5) -> TavilySearchResults:
    """Return a configured Tavily web search tool."""
    return TavilySearchResults(
        max_results=max_results,
        api_key=settings.tavily_api_key,
        include_answer=True,
        include_raw_content=False,
    )
