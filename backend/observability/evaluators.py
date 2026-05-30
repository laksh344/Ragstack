"""Custom LangSmith-compatible evaluators for RAGStack.

Each evaluator is a standalone async function that can be:
  1. Called directly from eval/run_eval.py
  2. Wrapped for langsmith.evaluate() via the LangSmith evaluator protocol

Evaluators:
  faithfulness        — is the answer grounded in the retrieved documents?
  answer_relevance    — does the answer address the user's question?
  retrieval_relevance — are the retrieved chunks relevant to the query?
  citation_accuracy   — do citations reference real chunks from retrieval?

All return an EvalScore(key, score 0-1, reasoning, passed).
"""

import re
import structlog
from pydantic import BaseModel

from backend.guardrails import EvalScore

logger = structlog.get_logger()

_PASS_THRESHOLD = 0.5   # score >= this → passed=True


# ---------------------------------------------------------------------------
# Structured LLM output for judge-based evaluators
# ---------------------------------------------------------------------------

class _JudgeOutput(BaseModel):
    score: float    # 0-1
    reasoning: str


# ---------------------------------------------------------------------------
# 1. Faithfulness — is the answer grounded in retrieved docs?
# ---------------------------------------------------------------------------

async def evaluate_faithfulness(
    query: str,
    response: str,
    context: str,
    use_llm: bool = True,
) -> EvalScore:
    """Score how well the response is supported by retrieved context."""
    from backend.config import settings  # deferred

    if use_llm and settings.openai_api_key and context.strip():
        try:
            score, reasoning = await _llm_judge(
                query=query,
                response=response,
                context=context,
                task="faithfulness",
                prompt=(
                    "Rate how faithfully this response is grounded in the provided context. "
                    "Score 1.0 if every claim has a source in the context; "
                    "0.0 if the response contradicts or ignores the context. "
                    "Penalise fabricated facts not in the context."
                ),
                settings=settings,
            )
        except Exception as exc:
            logger.warning("evaluator.faithfulness.llm_error", error=str(exc))
            score, reasoning = _overlap_faithfulness(response, context)
    else:
        score, reasoning = _overlap_faithfulness(response, context)

    return EvalScore(
        key="faithfulness",
        score=score,
        reasoning=reasoning,
        passed=score >= _PASS_THRESHOLD,
    )


def _overlap_faithfulness(response: str, context: str) -> tuple[float, str]:
    resp_words = _words(response)
    ctx_words  = _words(context)
    if not resp_words:
        return 1.0, "empty_response"
    ratio = len(resp_words & ctx_words) / len(resp_words)
    return round(ratio, 3), f"word_overlap={ratio:.3f}"


# ---------------------------------------------------------------------------
# 2. Answer relevance — does the answer address the question?
# ---------------------------------------------------------------------------

async def evaluate_answer_relevance(
    query: str,
    response: str,
    use_llm: bool = True,
) -> EvalScore:
    """Score how directly the response addresses the query."""
    from backend.config import settings  # deferred

    if use_llm and settings.openai_api_key:
        try:
            score, reasoning = await _llm_judge(
                query=query,
                response=response,
                context="",
                task="answer_relevance",
                prompt=(
                    "Rate how directly and completely this response answers the user's question. "
                    "Score 1.0 if the response fully addresses the question; "
                    "0.0 if it is off-topic or does not answer at all."
                ),
                settings=settings,
            )
        except Exception as exc:
            logger.warning("evaluator.answer_relevance.llm_error", error=str(exc))
            score, reasoning = _overlap_relevance(query, response)
    else:
        score, reasoning = _overlap_relevance(query, response)

    return EvalScore(
        key="answer_relevance",
        score=score,
        reasoning=reasoning,
        passed=score >= _PASS_THRESHOLD,
    )


def _overlap_relevance(query: str, response: str) -> tuple[float, str]:
    q_words = _words(query)
    r_words  = _words(response)
    if not q_words:
        return 1.0, "empty_query"
    ratio = len(q_words & r_words) / len(q_words)
    return round(ratio, 3), f"query_response_overlap={ratio:.3f}"


# ---------------------------------------------------------------------------
# 3. Retrieval relevance — are retrieved chunks relevant to the query?
# ---------------------------------------------------------------------------

def evaluate_retrieval_relevance(
    query: str,
    chunks: list[dict],
) -> EvalScore:
    """Score how relevant the retrieved chunks are to the query (no LLM needed)."""
    if not chunks:
        return EvalScore(key="retrieval_relevance", score=0.0, reasoning="no_chunks_retrieved", passed=False)

    q_words = _words(query)
    if not q_words:
        return EvalScore(key="retrieval_relevance", score=1.0, reasoning="empty_query")

    scores: list[float] = []
    for chunk in chunks:
        c_words = _words(chunk.get("content", ""))
        if c_words:
            scores.append(len(q_words & c_words) / len(q_words))

    avg = round(sum(scores) / max(len(scores), 1), 3)
    return EvalScore(
        key="retrieval_relevance",
        score=avg,
        reasoning=f"avg_overlap={avg:.3f}_over_{len(chunks)}_chunks",
        passed=avg >= _PASS_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# 4. Citation accuracy — do citations reference real retrieved chunks?
# ---------------------------------------------------------------------------

def evaluate_citation_accuracy(
    citations: list[dict],
    chunks: list[dict],
) -> EvalScore:
    """Score the fraction of citations that match a retrieved chunk."""
    if not citations:
        return EvalScore(key="citation_accuracy", score=1.0, reasoning="no_citations_to_verify")

    source_files = {c.get("source_file", "") for c in chunks}
    matched = sum(
        1 for cit in citations
        if cit.get("source_file", "") in source_files
    )
    score = round(matched / len(citations), 3)
    return EvalScore(
        key="citation_accuracy",
        score=score,
        reasoning=f"{matched}/{len(citations)}_citations_verified",
        passed=score >= _PASS_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# LLM judge helper — single call for any evaluator
# ---------------------------------------------------------------------------

async def _llm_judge(
    query: str,
    response: str,
    context: str,
    task: str,
    prompt: str,
    settings,
) -> tuple[float, str]:
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    structured = llm.with_structured_output(_JudgeOutput)

    ctx_block = f"\n\nCONTEXT:\n{context[:4000]}" if context else ""
    user_content = (
        f"{prompt}\n\n"
        f"QUESTION: {query}\n\n"
        f"RESPONSE: {response}"
        f"{ctx_block}\n\n"
        "Return a score between 0 and 1 and a one-sentence reasoning."
    )

    result: _JudgeOutput = await structured.ainvoke(
        [{"role": "user", "content": user_content}]
    )
    clamped = round(max(0.0, min(1.0, result.score)), 3)
    logger.debug("evaluator.llm_judge", task=task, score=clamped)
    return clamped, result.reasoning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b\w{4,}\b", text.lower())}
