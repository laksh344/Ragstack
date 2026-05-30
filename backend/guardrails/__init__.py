"""Shared types for the guardrails layer."""

from pydantic import BaseModel, Field


class PiiResult(BaseModel):
    detected: bool
    entity_types: list[str] = Field(default_factory=list)
    redacted_text: str = ""
    engine: str = "regex"   # "presidio" | "regex"


class ValidationResult(BaseModel):
    valid: bool
    flags: list[str] = Field(default_factory=list)
    reason: str = ""


class SentenceVerdict(BaseModel):
    sentence: str
    grounded: bool
    reasoning: str = ""


class HallucinationResult(BaseModel):
    score: float                    # 0 = fully grounded, 1 = fully hallucinated
    flagged_sentences: list[str] = Field(default_factory=list)
    verdicts: list[SentenceVerdict] = Field(default_factory=list)
    total_sentences: int = 0
    grounded_count: int = 0
    engine: str = "overlap"         # "llm" | "overlap"


class TokenUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class EvalScore(BaseModel):
    key: str
    score: float                    # 0-1
    reasoning: str = ""
    passed: bool = True             # score >= threshold
