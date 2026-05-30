"""LangSmith tracing configuration and metadata helpers.

LangSmith traces all LangChain calls automatically when
LANGCHAIN_TRACING_V2=true is set.  This module adds utilities for
attaching custom metadata to active runs and creating named spans for
non-LangChain code paths (e.g. the ingest pipeline).
"""

import os
from contextlib import contextmanager

import structlog

from backend.config import settings

logger = structlog.get_logger()


def configure_tracing() -> None:
    """Write LangSmith environment variables from settings."""
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"]     = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"]     = settings.langchain_project


def add_run_metadata(metadata: dict) -> None:
    """Attach extra key-value pairs to the current LangSmith run.

    Safe to call even when tracing is disabled — silently no-ops.
    """
    try:
        from langsmith.run_helpers import get_current_run_tree  # noqa: PLC0415

        run = get_current_run_tree()
        if run:
            run.add_metadata(metadata)
    except Exception:
        pass


def add_run_tags(tags: list[str]) -> None:
    """Attach tags to the current LangSmith run."""
    try:
        from langsmith.run_helpers import get_current_run_tree  # noqa: PLC0415

        run = get_current_run_tree()
        if run:
            run.add_tags(tags)
    except Exception:
        pass


@contextmanager
def trace_span(name: str, metadata: dict | None = None):
    """Context manager that creates a named child span in LangSmith.

    Usage::

        with trace_span("ingest.parse", {"file": filename}):
            parsed = parse_document(path)
    """
    try:
        from langsmith import trace  # noqa: PLC0415

        with trace(name=name, metadata=metadata or {}):
            yield
    except Exception:
        yield   # tracing unavailable — run the code without a span


def tag_request(user_id: str | None = None, dataset_id: str | None = None, model: str | None = None) -> None:
    """Attach standard per-request metadata to the active LangSmith trace."""
    meta: dict = {}
    if user_id:
        meta["user_id"] = user_id
    if dataset_id:
        meta["dataset_id"] = dataset_id
    if model:
        meta["model_used"] = model
    if meta:
        add_run_metadata(meta)


# Auto-configure on import
configure_tracing()
