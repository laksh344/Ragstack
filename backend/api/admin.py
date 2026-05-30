"""Admin endpoints — health check, stats, configuration."""

from datetime import UTC, datetime

from fastapi import APIRouter

from backend.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.1.0",
        "services": {
            "qdrant": f"{settings.qdrant_host}:{settings.qdrant_port}",
            "elasticsearch": settings.es_host,
            "redis": settings.redis_url,
        },
    }


@router.get("/config")
async def get_config():
    return {
        "llm_model": settings.openai_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "langsmith_project": settings.langchain_project,
        "qdrant_collection": settings.qdrant_collection,
        "es_index": settings.es_index,
    }
