"""LangGraph agent state schema.

All node functions read from and write to this TypedDict.
Fields are additive — nodes only need to return the keys they change.
"""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Citation(TypedDict):
    source_file: str
    page_number: int
    excerpt: str


class AgentState(TypedDict):
    # --- Conversation ---
    # add_messages merges lists rather than overwriting, enabling multi-turn.
    messages: Annotated[list[BaseMessage], add_messages]
    query: str

    # --- Routing ---
    # "knowledge_base" | "web_search" | "clarify" | "chitchat"
    route: str
    # Guards against runaway loops if the graph is ever made cyclic.
    iteration_count: int

    # --- Retrieval ---
    # SearchResult.model_dump() dicts from the hybrid search pipeline.
    retrieved_docs: list[dict]
    # True when retrieval score falls below the relevance threshold.
    needs_web_search: bool

    # --- Web search ---
    # Raw Tavily result dicts: {"url": ..., "content": ..., "title": ...}
    search_results: list[dict]

    # --- Generation ---
    response: str
    citations: list[Citation]

    # --- Guardrails ---
    # Human-readable flag strings, e.g. "pii_detected", "low_faithfulness".
    guardrail_flags: list[str]


# Routing literals used in conditional edges.
Route = Literal["knowledge_base", "web_search", "clarify", "chitchat"]
