"""Test configuration."""

import backend.guardrails as _guardrails
import backend.guardrails.pii_redactor as _pii_module


def _disable_presidio(self) -> None:
    """Keep PII tests deterministic by forcing regex mode."""
    return None


_pii_module.PiiRedactor._try_load_presidio = _disable_presidio  # type: ignore[method-assign]

# Drop any cached singleton so it is rebuilt in regex mode on next use.
_guardrails.get_pii_redactor.cache_clear()
