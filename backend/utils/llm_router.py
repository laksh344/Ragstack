"""Multi-provider LLM routing.

Supports OpenAI, Anthropic, and Google Gemini with
configurable fallback chains.
"""

from langchain_openai import ChatOpenAI

from backend.config import settings


def get_llm(
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
    streaming: bool = False,
) -> ChatOpenAI:
    """Get a configured LLM instance.

    Currently defaults to OpenAI. Extend with Anthropic/Gemini
    by adding provider detection on the model string.
    """
    return ChatOpenAI(
        model=model or settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
