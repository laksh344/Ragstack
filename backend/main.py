"""RAGStack — Agentic RAG Platform.

FastAPI application entry point.
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import admin, chat, evaluate, ingest
from backend.config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(
        "ragstack.starting",
        qdrant=f"{settings.qdrant_host}:{settings.qdrant_port}",
        elasticsearch=settings.es_host,
        langsmith_project=settings.langchain_project,
    )
    settings.upload_path  # ensure upload dir exists
    yield
    logger.info("ragstack.shutdown")


app = FastAPI(
    title="RAGStack",
    description="Agentic RAG platform with multi-modal document understanding, "
    "hybrid search, LangSmith observability, and guardrails.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request timing middleware ---
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# --- Register routers ---
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(chat.router,     prefix="/api/v1", tags=["Chat"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluation"])


# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
