"""Guardrails node — delegates to the full backend/guardrails/ suite.

Pre-generation  (query):    InputValidator + PiiRedactor.detect
Post-generation (response): PiiRedactor.redact + HallucinationDetector + TokenBudget

Accumulates flags in AgentState["guardrail_flags"] so the chat API and
frontend can surface safety information per-message.
"""

import structlog

from backend.agent.state import AgentState
from backend.guardrails.hallucination import HallucinationDetector, HALLUCINATION_THRESHOLD
from backend.guardrails.input_validator import InputValidator
from backend.guardrails.pii_redactor import PiiRedactor
from backend.guardrails.token_budget import TokenBudget
from backend.observability.tracing import add_run_metadata

logger = structlog.get_logger()

_pii      = PiiRedactor()
_validator = InputValidator()
_detector  = HallucinationDetector()


async def guardrails_node(state: AgentState) -> dict:
    """Run all guardrail checks and return updated state fields."""
    flags: list[str] = list(state.get("guardrail_flags", []))
    query    = state.get("query", "")
    response = state.get("response", "")

    # ------------------------------------------------------------------
    # 1. Pre-generation: validate + detect PII in query
    # ------------------------------------------------------------------
    validation = _validator.validate(query)
    flags.extend(validation.flags)

    pii_query = _pii.detect(query)
    if pii_query.detected:
        flags.extend(f"pii_{e.lower()}_in_query" for e in pii_query.entity_types)

    # ------------------------------------------------------------------
    # 2. Post-generation: redact PII in response
    # ------------------------------------------------------------------
    pii_response = _pii.redact(response)
    if pii_response.detected:
        flags.extend(f"pii_{e.lower()}_in_response" for e in pii_response.entity_types)
        response = pii_response.redacted_text

    # ------------------------------------------------------------------
    # 3. Hallucination detection
    # ------------------------------------------------------------------
    context_parts = [d.get("content", "") for d in state.get("retrieved_docs", [])]
    context_parts += [h.get("content", "") for h in state.get("search_results", [])]
    context = " ".join(context_parts)

    hallucination = await _detector.check(response, context, use_llm=False)
    if hallucination.score > HALLUCINATION_THRESHOLD and context:
        flags.append("low_faithfulness")
        logger.warning(
            "guardrails.low_faithfulness",
            score=hallucination.score,
            flagged=len(hallucination.flagged_sentences),
        )

    # ------------------------------------------------------------------
    # 4. Token budget tracking + LangSmith metadata
    # ------------------------------------------------------------------
    budget = TokenBudget()
    usage  = budget.track(query=query, context=context, response=response)
    budget.log_to_langsmith(usage)
    add_run_metadata({
        "guardrail_flags":  flags,
        "hallucination_score": hallucination.score,
        "pii_engine":       _pii.engine,
        "token_usage":      usage.model_dump(),
    })

    logger.info(
        "guardrails.complete",
        flags=flags,
        hallucination_score=hallucination.score,
        total_tokens=usage.total_tokens,
        cost_usd=usage.estimated_cost_usd,
    )

    return {
        "response":       response,
        "guardrail_flags": flags,
    }
