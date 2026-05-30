"""Router node — classifies user intent and selects the retrieval path.

Uses structured LLM output so the decision is always one of the four
valid route values.  Chitchat and clarify routes skip retrieval entirely.
"""

from typing import Literal

import structlog
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from backend.agent.state import AgentState
from backend.config import settings

logger = structlog.get_logger()


class RouterDecision(BaseModel):
    intent: Literal["knowledge_base", "web_search", "clarify", "chitchat"]
    reasoning: str


_SYSTEM_PROMPT = """\
You are a query router for an agentic RAG system.
Classify the user's intent into exactly one of these categories:

knowledge_base — The query asks about content that is likely in the
  ingested document knowledge base (reports, specs, internal docs).
web_search — The query asks about current events, live data, or topics
  clearly outside the knowledge base.
clarify — The query is too vague or ambiguous to answer; you need more
  detail from the user.
chitchat — The query is a greeting, small talk, or off-topic.

Respond with the intent and a one-sentence reasoning.
"""


async def router_node(state: AgentState) -> dict:
    """Classify query intent and set the route in state."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    structured = llm.with_structured_output(RouterDecision)

    decision: RouterDecision = await structured.ainvoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": state["query"]},
        ]
    )

    logger.info(
        "router.decision",
        intent=decision.intent,
        reasoning=decision.reasoning,
    )
    return {
        "route": decision.intent,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [HumanMessage(content=state["query"])],
    }


def route_after_router(state: AgentState) -> str:
    """Conditional edge function: maps route value to next node name."""
    route = state.get("route", "knowledge_base")
    # clarify and chitchat skip retrieval → go straight to generator
    if route in ("clarify", "chitchat"):
        return "generator"
    return route  # "knowledge_base" or "web_search"
