"""Input validation — prompt injection, toxicity, and length limits.

Validation runs before the agent processes the query.  All checks are
deterministic regex / keyword-list so they add negligible latency.
An optional LLM classification pass can be enabled per-request.
"""

import re

import structlog

from backend.guardrails import ValidationResult

logger = structlog.get_logger()

MAX_QUERY_LENGTH = 2_000   # characters

# ---------------------------------------------------------------------------
# Injection patterns — common jailbreak phrases
# ---------------------------------------------------------------------------

_INJECTION_RE = re.compile(
    r"ignore (previous|prior|above|all) instructions?|"
    r"you are now|act as if you|pretend (you are|to be)|"
    r"jailbreak|disregard your|forget (all )?instructions?|"
    r"system prompt|bypass (your|the) (safety|filter|restriction)|"
    r"DAN mode|developer mode",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Toxicity keyword list (minimal; production would use a dedicated model)
# ---------------------------------------------------------------------------

_TOXIC_KEYWORDS: frozenset[str] = frozenset({
    "kill", "murder", "suicide", "bomb", "weapon", "drug",
    "hack", "exploit", "malware", "ransomware", "phishing",
    "racist", "sexist", "nazi", "terrorism", "terrorist",
})


class InputValidator:
    """Validate a user query before it enters the agent pipeline."""

    def validate(self, query: str) -> ValidationResult:
        """Run all checks and return a combined ValidationResult."""
        flags: list[str] = []

        # 1. Length check
        if len(query) > MAX_QUERY_LENGTH:
            flags.append("query_too_long")

        # 2. Empty / whitespace-only
        if not query.strip():
            flags.append("empty_query")

        # 3. Prompt injection
        if _INJECTION_RE.search(query):
            flags.append("prompt_injection_detected")
            logger.warning("input_validator.injection", query_snippet=query[:80])

        # 4. Toxicity
        if self._is_toxic(query):
            flags.append("toxicity_detected")
            logger.warning("input_validator.toxicity", query_snippet=query[:80])

        valid = len(flags) == 0
        reason = "; ".join(flags) if flags else ""
        return ValidationResult(valid=valid, flags=flags, reason=reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_toxic(text: str) -> bool:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        return bool(words & _TOXIC_KEYWORDS)
