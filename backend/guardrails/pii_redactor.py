"""PII detection and redaction.

Primary engine: Microsoft Presidio (AnalyzerEngine + AnonymizerEngine).
Fallback: regex patterns when Presidio / spaCy model is not installed.

Two modes:
  detect — scan text and report entity types found, no modification.
  redact — replace PII spans with <ENTITY_TYPE> placeholders.
"""

import re
import structlog

from backend.guardrails import PiiResult

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Regex fallback patterns
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL_ADDRESS", re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    )),
    ("PHONE_NUMBER", re.compile(
        r"\b(\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
    )),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
    ("IP_ADDRESS", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )),
]

# Presidio entities to analyse (superset of the regex list above)
_PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
    "PERSON", "LOCATION", "US_PASSPORT", "IBAN_CODE", "IP_ADDRESS",
]


class PiiRedactor:
    """Detect and redact PII using Presidio with a regex fallback."""

    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None
        self._engine_name = "regex"
        self._try_load_presidio()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> PiiResult:
        """Scan text and return detected entity types (text unchanged)."""
        if self._analyzer:
            return self._presidio_detect(text)
        return self._regex_detect(text)

    def redact(self, text: str) -> PiiResult:
        """Replace PII spans with <ENTITY_TYPE> placeholders."""
        if self._analyzer and self._anonymizer:
            return self._presidio_redact(text)
        return self._regex_redact(text)

    @property
    def engine(self) -> str:
        return self._engine_name

    # ------------------------------------------------------------------
    # Presidio implementation
    # ------------------------------------------------------------------

    def _try_load_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
            from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._engine_name = "presidio"
            logger.debug("pii_redactor.presidio_loaded")
        except Exception as exc:
            logger.debug("pii_redactor.presidio_unavailable", reason=str(exc))

    def _presidio_detect(self, text: str) -> PiiResult:
        results = self._analyzer.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES)
        entity_types = list({r.entity_type for r in results})
        return PiiResult(
            detected=bool(entity_types),
            entity_types=entity_types,
            redacted_text="",
            engine="presidio",
        )

    def _presidio_redact(self, text: str) -> PiiResult:
        from presidio_anonymizer.entities import OperatorConfig  # noqa: PLC0415

        results = self._analyzer.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES)
        entity_types = list({r.entity_type for r in results})

        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in _PRESIDIO_ENTITIES
        }
        anonymized = self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )
        return PiiResult(
            detected=bool(entity_types),
            entity_types=entity_types,
            redacted_text=anonymized.text,
            engine="presidio",
        )

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    def _regex_detect(self, text: str) -> PiiResult:
        found: list[str] = []
        for name, pattern in _REGEX_PATTERNS:
            if pattern.search(text):
                found.append(name)
        return PiiResult(detected=bool(found), entity_types=found, redacted_text="", engine="regex")

    def _regex_redact(self, text: str) -> PiiResult:
        found: list[str] = []
        redacted = text
        for name, pattern in _REGEX_PATTERNS:
            if pattern.search(redacted):
                found.append(name)
                redacted = pattern.sub(f"<{name}>", redacted)
        return PiiResult(detected=bool(found), entity_types=found, redacted_text=redacted, engine="regex")
