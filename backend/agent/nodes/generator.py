"""Generator node — LLM response with structured citations.

Builds context from whatever retrieval path ran (KB docs or web results),
then calls the LLM with structured output to produce a grounded answer
and explicit citation list.  Handles clarify/chitchat routes with
lightweight prompts that skip citation formatting entirely.
"""

from typing import cast

import structlog
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from backend.agent.state import AgentState, Citation
from backend.config import settings

logger = structlog.get_logger()

_MAX_CONTEXT_CHARS = 12_000  # keep prompt within ~3k tokens of context


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class GeneratedResponse(BaseModel):
    answer: str
    citations: list[Citation]


class SimpleResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RAG_SYSTEM = """\
You are a helpful AI assistant with access to a curated knowledge base.
Answer the user's question using ONLY the provided context chunks.
For every factual claim, include an inline citation in the format:
  [Source: <filename>, page <N>]
If the context does not contain enough information, say so honestly.
Do not fabricate facts.
"""

_CLARIFY_SYSTEM = """\
You are a helpful AI assistant. The user's query was too vague to answer.
Ask a concise clarifying question to understand what they need.
"""

_CHITCHAT_SYSTEM = """\
You are a friendly AI assistant. Respond naturally and briefly.
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def generator_node(state: AgentState) -> dict:
    """Generate a response given the current retrieval context."""
    route = state.get("route", "knowledge_base")
    query = state["query"]

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=0.2,
    )

    if route == "clarify":
        return await _generate_clarification(llm, query)

    if route == "chitchat":
        return await _generate_chitchat(llm, query)

    return await _generate_rag_response(llm, query, state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _generate_rag_response(llm: ChatOpenAI, query: str, state: AgentState) -> dict:
    context = _build_context(state)
    structured = llm.with_structured_output(GeneratedResponse)

    result = cast(GeneratedResponse, await structured.ainvoke(
        [
            {"role": "system", "content": _RAG_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
    ))

    logger.info(
        "generator.rag_response",
        answer_len=len(result.answer),
        citations=len(result.citations),
    )
    return {
        "response": result.answer,
        "citations": list(result.citations),
    }


async def _generate_clarification(llm: ChatOpenAI, query: str) -> dict:
    structured = llm.with_structured_output(SimpleResponse)
    result = cast(SimpleResponse, await structured.ainvoke(
        [
            {"role": "system", "content": _CLARIFY_SYSTEM},
            {"role": "user", "content": query},
        ]
    ))
    return {"response": result.answer, "citations": []}


async def _generate_chitchat(llm: ChatOpenAI, query: str) -> dict:
    structured = llm.with_structured_output(SimpleResponse)
    result = cast(SimpleResponse, await structured.ainvoke(
        [
            {"role": "system", "content": _CHITCHAT_SYSTEM},
            {"role": "user", "content": query},
        ]
    ))
    return {"response": result.answer, "citations": []}


def _build_context(state: AgentState) -> str:
    """Assemble a context string from KB docs and/or web results."""
    parts: list[str] = []

    for doc in state.get("retrieved_docs", []):
        header = f"[{doc.get('source_file', 'unknown')}, page {doc.get('page_number', 0)}]"
        parts.append(f"{header}\n{doc.get('content', '')}")

    for hit in state.get("search_results", []):
        url = hit.get("url", "web")
        parts.append(f"[Web: {url}]\n{hit.get('content', '')}")

    context = "\n\n---\n\n".join(parts)
    # Hard-truncate to avoid exceeding LLM context limits.
    return context[:_MAX_CONTEXT_CHARS]
