"""LangGraph agent graph — wires all nodes into a compiled StateGraph.

Flow:
  START
    └─► router
          ├─► [knowledge_base] ─► retriever ─► [sufficient] ─► generator ─► guardrails ─► END
          │                                 └─► [fallback]  ─► web_search ─┘
          ├─► [web_search]     ─► web_search ──────────────────► generator ─► guardrails ─► END
          └─► [clarify/chitchat] ──────────────────────────────► generator ─► guardrails ─► END

Import ``build_graph()`` to get the compiled graph, or use the module-level
``graph`` singleton (compiled once on first import).
"""

import structlog
from langgraph.graph import END, START, StateGraph

from backend.agent.nodes.generator import generator_node
from backend.agent.nodes.guardrails import guardrails_node
from backend.agent.nodes.retriever import retriever_node, route_after_retrieval
from backend.agent.nodes.router import route_after_router, router_node
from backend.agent.nodes.web_search import web_search_node
from backend.agent.state import AgentState

logger = structlog.get_logger()


def build_graph():
    """Construct and compile the RAGStack agent graph."""
    g = StateGraph(AgentState)

    # --- Register nodes ---
    g.add_node("router",     router_node)
    g.add_node("retriever",  retriever_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generator",  generator_node)
    g.add_node("guardrails", guardrails_node)

    # --- Entry point ---
    g.add_edge(START, "router")

    # --- Router → branch ---
    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "knowledge_base": "retriever",
            "web_search":     "web_search",
            "generator":      "generator",   # clarify + chitchat shortcut
        },
    )

    # --- Retriever → sufficient or web fallback ---
    g.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "generator":  "generator",
            "web_search": "web_search",
        },
    )

    # --- Linear tail ---
    g.add_edge("web_search", "generator")
    g.add_edge("generator",  "guardrails")
    g.add_edge("guardrails", END)

    compiled = g.compile()
    logger.info("agent.graph_compiled")
    return compiled


# Module-level singleton — imported by the chat API.
graph = build_graph()
