"""Token counting, cost estimation, and LangSmith cost metadata logging.

Uses tiktoken for accurate token counts.  Falls back to a character-based
approximation (chars / 4) if tiktoken is unavailable for a model.

Pricing table is approximate (GPT-4o as of early 2026) and should be
updated as model pricing changes.
"""

import structlog

from backend.guardrails import TokenUsage

logger = structlog.get_logger()

# USD per 1 000 tokens — (input_cost, output_cost)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":                    (0.005,    0.015),
    "gpt-4o-mini":               (0.00015,  0.0006),
    "gpt-4-turbo":               (0.01,     0.03),
    "text-embedding-3-small":    (0.00002,  0.0),
    "text-embedding-3-large":    (0.00013,  0.0),
}

_DEFAULT_PRICING = (0.005, 0.015)   # fall back to gpt-4o rates


class TokenBudget:
    """Count tokens and estimate costs for a single request."""

    def __init__(self, model: str | None = None) -> None:
        from backend.config import settings  # deferred to avoid circular import
        self.model = model or settings.openai_model
        self._encoder = self._load_encoder(self.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        """Return the number of tokens in *text* for this model."""
        if self._encoder:
            return len(self._encoder.encode(text))
        return max(1, len(text) // 4)    # character-based fallback

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated USD cost for the given token counts."""
        in_rate, out_rate = _MODEL_PRICING.get(self.model, _DEFAULT_PRICING)
        return round(
            (input_tokens * in_rate + output_tokens * out_rate) / 1_000,
            6,
        )

    def track(
        self,
        query: str,
        context: str,
        response: str,
    ) -> TokenUsage:
        """Count tokens for a full request/response cycle and return usage."""
        input_text = query + "\n" + context
        input_tokens  = self.count(input_text)
        output_tokens = self.count(response)
        cost = self.estimate_cost(input_tokens, output_tokens)

        usage = TokenUsage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
        )
        logger.info(
            "token_budget.tracked",
            model=self.model,
            input=input_tokens,
            output=output_tokens,
            cost_usd=cost,
        )
        return usage

    def log_to_langsmith(self, usage: TokenUsage) -> None:
        """Attach cost metadata to the active LangSmith run (no-op if none)."""
        try:
            from langsmith.run_helpers import get_current_run_tree  # noqa: PLC0415

            run = get_current_run_tree()
            if run:
                run.add_metadata({
                    "token_usage": usage.model_dump(),
                    "estimated_cost_usd": usage.estimated_cost_usd,
                })
        except Exception:
            pass   # LangSmith not active — silently skip

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load_encoder(model: str):
        try:
            import tiktoken  # noqa: PLC0415

            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None
