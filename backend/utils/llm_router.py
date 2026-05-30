"""Multi-provider LLM routing (compatibility shim).

The real provider logic lives in backend.utils.providers. This module is
kept for backwards compatibility — it delegates to providers.get_llm so any
older imports of get_llm continue to work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.utils.providers import get_llm as _get_llm

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def get_llm(
    model: str | None = None,
    temperature: float = 0,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a configured chat model for the active provider.

    See backend.utils.providers.get_llm for the full implementation.
    """
    return _get_llm(temperature=temperature, streaming=streaming, model=model)
