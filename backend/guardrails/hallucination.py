"""Hallucination detection — LLM-as-judge with word-overlap fallback.

Strategy:
  1. Split the response into sentences.
  2. Ask the LLM in a single batched call to classify each sentence as
     grounded or ungrounded given the retrieved context.
  3. hallucination_score = ungrounded_count / total_sentences
  4. If score > THRESHOLD, the guardrail flags "low_faithfulness".

Fallback (no API key or LLM call fails):
  Use word-overlap ratio: sentences that share few words with the context
  are considered potentially ungrounded.
"""

import re
import structlog
from pydantic import BaseModel

from backend.guardrails import HallucinationResult, SentenceVerdict

logger = structlog.get_logger()

HALLUCINATION_THRESHOLD = 0.3   # above this → flag response
_MIN_SENTENCE_WORDS = 4         # skip very short sentences (articles, etc.)
_MAX_CONTEXT_CHARS = 8_000      # truncate to stay within LLM context


# ---------------------------------------------------------------------------
# Structured output schema for LLM judge
# ---------------------------------------------------------------------------

class _SentenceJudgement(BaseModel):
    sentence: str
    grounded: bool
    reasoning: str


class _JudgementList(BaseModel):
    verdicts: list[_SentenceJudgement]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Detect hallucinations in generated responses."""

    async def check(
        self,
        response: str,
        context: str,
        use_llm: bool = True,
    ) -> HallucinationResult:
        """Evaluate how well the response is grounded in the context.

        Args:
            response: The generated assistant response.
            context:  Concatenated retrieved chunk text.
            use_llm:  If True and an OpenAI key is set, use LLM-as-judge.
                      Otherwise falls back to word-overlap heuristic.
        """
        sentences = _split_sentences(response)
        if not sentences:
            return HallucinationResult(score=0.0, total_sentences=0, grounded_count=0)

        from backend.config import settings  # deferred

        if use_llm and settings.openai_api_key:
            try:
                return await self._llm_check(sentences, context, settings)
            except Exception as exc:
                logger.warning("hallucination.llm_fallback", error=str(exc))

        return self._overlap_check(sentences, context)

    # ------------------------------------------------------------------
    # LLM judge
    # ------------------------------------------------------------------

    async def _llm_check(
        self, sentences: list[str], context: str, settings
    ) -> HallucinationResult:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        structured = llm.with_structured_output(_JudgementList)

        truncated_ctx = context[:_MAX_CONTEXT_CHARS]
        sentences_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

        prompt = (
            "You are a fact-checker. Given the following CONTEXT, evaluate each "
            "SENTENCE from the response and decide whether it is supported by the context.\n\n"
            f"CONTEXT:\n{truncated_ctx}\n\n"
            f"SENTENCES TO EVALUATE:\n{sentences_block}\n\n"
            "For each sentence set grounded=true if it is supported, false if not."
        )

        result: _JudgementList = await structured.ainvoke(
            [{"role": "user", "content": prompt}]
        )

        verdicts = [
            SentenceVerdict(
                sentence=j.sentence,
                grounded=j.grounded,
                reasoning=j.reasoning,
            )
            for j in result.verdicts
        ]
        flagged = [v.sentence for v in verdicts if not v.grounded]
        grounded_count = sum(1 for v in verdicts if v.grounded)
        score = round(1 - grounded_count / max(len(verdicts), 1), 3)

        logger.info(
            "hallucination.llm_check",
            total=len(verdicts),
            grounded=grounded_count,
            score=score,
        )
        return HallucinationResult(
            score=score,
            flagged_sentences=flagged,
            verdicts=verdicts,
            total_sentences=len(verdicts),
            grounded_count=grounded_count,
            engine="llm",
        )

    # ------------------------------------------------------------------
    # Word-overlap fallback
    # ------------------------------------------------------------------

    def _overlap_check(self, sentences: list[str], context: str) -> HallucinationResult:
        if not sentences:
            return HallucinationResult(score=0.0, total_sentences=0, grounded_count=0, engine="overlap")

        ctx_words = _meaningful_words(context)
        verdicts: list[SentenceVerdict] = []

        for sentence in sentences:
            resp_words = _meaningful_words(sentence)
            if not resp_words:
                verdicts.append(SentenceVerdict(sentence=sentence, grounded=True))
                continue
            overlap = len(resp_words & ctx_words) / len(resp_words)
            grounded = overlap >= 0.15   # at least 15% word overlap
            verdicts.append(SentenceVerdict(
                sentence=sentence,
                grounded=grounded,
                reasoning=f"word_overlap={overlap:.2f}",
            ))

        flagged = [v.sentence for v in verdicts if not v.grounded]
        grounded_count = sum(1 for v in verdicts if v.grounded)
        score = round(1 - grounded_count / max(len(verdicts), 1), 3)

        return HallucinationResult(
            score=score,
            flagged_sentences=flagged,
            verdicts=verdicts,
            total_sentences=len(verdicts),
            grounded_count=grounded_count,
            engine="overlap",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split response into non-trivial sentences."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [
        s.strip() for s in raw
        if len(s.split()) >= _MIN_SENTENCE_WORDS
    ]


def _meaningful_words(text: str) -> set[str]:
    """Return lowercase words of 4+ characters (filter stop words by length)."""
    return {w for w in re.findall(r"\b\w{4,}\b", text.lower())}
