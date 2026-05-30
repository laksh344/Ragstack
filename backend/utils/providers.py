"""Multi-provider LLM and embedding factory.

This is the single source of truth for *which* model backend the platform
uses. Selecting a provider here (via settings.llm_provider /
settings.embedding_provider) swaps the backend everywhere — ingestion,
retrieval, the agent nodes, guardrails, and evaluators — without touching
any call sites.

Supported providers: "openai" | "google" | "cohere".

Provider packages are imported lazily so that only the selected provider's
dependency needs to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from backend.config import settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Embedding dimensions — used to size the Qdrant collection.
# ---------------------------------------------------------------------------

_EMBEDDING_DIMENSIONS: dict[str, dict[str, int]] = {
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    },
    "google": {
        "models/gemini-embedding-001": 3072,
    },
    "cohere": {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
    },
}

_DEFAULT_DIMENSIONS: dict[str, int] = {"openai": 1536, "google": 3072, "cohere": 1024}


def _embedding_model_name(provider: str) -> str:
    return {
        "openai": settings.embedding_model,
        "google": settings.google_embedding_model,
        "cohere": settings.cohere_embedding_model,
    }.get(provider, settings.embedding_model)


def get_embedding_dimensions() -> int:
    """Return the vector dimension for the configured embedding provider."""
    provider = settings.embedding_provider
    model = _embedding_model_name(provider)
    return _EMBEDDING_DIMENSIONS.get(provider, {}).get(
        model, _DEFAULT_DIMENSIONS.get(provider, 1536)
    )


def llm_available() -> bool:
    """True if an API key is configured for the selected LLM provider."""
    key = {
        "openai": settings.openai_api_key,
        "google": settings.google_api_key,
        "cohere": settings.cohere_api_key,
    }.get(settings.llm_provider, "")
    return bool(key)


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def get_embeddings() -> Embeddings:
    """Return a LangChain Embeddings object for the configured provider."""
    provider = settings.embedding_provider
    logger.debug("providers.get_embeddings", provider=provider)

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=settings.cohere_embedding_model,
            cohere_api_key=settings.cohere_api_key,
        )

    # default: openai
    from langchain_openai import OpenAIEmbeddings
    from pydantic import SecretStr

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(settings.openai_api_key),
    )


# ---------------------------------------------------------------------------
# Chat LLM factory
# ---------------------------------------------------------------------------

def get_llm(
    temperature: float = 0.0,
    streaming: bool = False,
    model: str | None = None,
) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider.

    Args:
        temperature: Sampling temperature.
        streaming: Whether to enable token streaming (OpenAI only; ignored by
                   providers that don't take the kwarg).
        model: Override the provider's default model name.
    """
    provider = settings.llm_provider
    logger.debug("providers.get_llm", provider=provider, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    if provider == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(
            model=model or settings.cohere_model,
            cohere_api_key=settings.cohere_api_key,
            temperature=temperature,
        )

    # default: openai
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    return ChatOpenAI(
        model=model or settings.openai_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=temperature,
        streaming=streaming,
    )
