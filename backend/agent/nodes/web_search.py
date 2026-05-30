"""Web search node — Tavily fallback when knowledge base retrieval is insufficient.

Triggered either directly (router intent = "web_search") or as a fallback
after the retriever decides the KB results are below the relevance threshold.
"""

import structlog

from backend.agent.state import AgentState
from backend.agent.tools import get_web_search_tool
from backend.config import settings

logger = structlog.get_logger()

_MAX_RESULTS = 5


async def web_search_node(state: AgentState) -> dict:
    """Run Tavily search and store results in state."""
    if not settings.tavily_api_key:
        logger.warning("web_search.no_api_key")
        return {
            "search_results": [],
            "guardrail_flags": state.get("guardrail_flags", []) + ["web_search_unavailable"],
        }

    tool = get_web_search_tool(max_results=_MAX_RESULTS)
    raw = await tool.ainvoke({"query": state["query"]})

    # Tavily returns a list of dicts: {"url", "content", "title", "score"}
    results: list[dict] = raw if isinstance(raw, list) else []

    logger.info("web_search.complete", results=len(results))
    return {"search_results": results}
