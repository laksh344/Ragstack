"""Ingestion API — upload documents and process through the full pipeline.

POST /api/v1/ingest — Upload a file, parse, chunk, embed, and store.
GET  /api/v1/ingest/stats — Get current store statistics.
DELETE /api/v1/ingest/{source_file} — Remove a document from all stores.

The pipeline: Upload → Parse → (Vision enrich) → Chunk → Embed → Store
Each step is traced in LangSmith for full observability.
"""

import shutil
import time
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import EmbeddingPipeline
from backend.ingestion.metadata import extract_file_metadata
from backend.ingestion.parser import parse_document
from backend.ingestion.vision import enrich_with_vision
from backend.models.document import ChunkingStrategy, FileType, IngestionResult

logger = structlog.get_logger()
router = APIRouter()


@router.post("/ingest", response_model=IngestionResult)
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(default="recursive"),
    chunk_size: int = Form(default=None),
    chunk_overlap: int = Form(default=None),
    use_vision: bool = Form(default=True),
):
    """Upload and process a document through the full RAG pipeline.

    1. Validates file type and size
    2. Parses document into pages with metadata
    3. (Optional) Enriches visual pages with GPT-4o Vision
    4. Chunks document using selected strategy
    5. Embeds chunks and stores in Qdrant + Elasticsearch
    6. Returns detailed ingestion statistics
    """
    start_time = time.perf_counter()
    errors: list[str] = []

    # --- Validate ---
    ext = Path(file.filename or "unknown").suffix.lower()
    file_type = FileType.from_extension(ext)
    if file_type == FileType.UNKNOWN:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .pdf, .docx, .csv, .txt, .xlsx",
        )

    # Save uploaded file to disk
    filename = file.filename or "upload"
    upload_path = settings.upload_path / filename
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    file_meta = extract_file_metadata(upload_path)
    if file_meta["file_size_mb"] > settings.max_file_size_mb:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large: {file_meta['file_size_mb']}MB "
                f"(max: {settings.max_file_size_mb}MB)"
            ),
        )

    logger.info(
        "ingest.start",
        file=file.filename,
        type=file_type.value,
        size_mb=file_meta["file_size_mb"],
    )

    # --- Parse ---
    try:
        parsed_doc = parse_document(upload_path)
    except Exception as e:
        errors.append(f"Parse error: {e}")
        logger.error("ingest.parse_error", error=str(e))
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    # --- Vision enrichment (optional) ---
    # Vision uses GPT-4o multimodal; only attempt it when OpenAI is the
    # configured provider and a key is present.
    vision_pages = 0
    if use_vision and settings.llm_provider == "openai" and settings.openai_api_key:
        try:
            visual_pages = [p for p in parsed_doc.pages if p.has_tables or p.has_images]
            if visual_pages:
                parsed_doc = await enrich_with_vision(parsed_doc, upload_path)
                vision_pages = len(visual_pages)
        except Exception as e:
            errors.append(f"Vision enrichment error: {e}")
            logger.warning("ingest.vision_error", error=str(e))

    # --- Chunk ---
    try:
        strategy = ChunkingStrategy(chunking_strategy)
    except ValueError:
        strategy = ChunkingStrategy.RECURSIVE

    chunks = chunk_document(
        parsed_doc,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Document produced no chunks. File may be empty or unreadable.",
        )

    # --- Embed & Store ---
    try:
        pipeline = EmbeddingPipeline()
        embed_stats = await pipeline.embed_and_store(chunks)
    except Exception as e:
        errors.append(f"Embedding error: {e}")
        logger.error("ingest.embed_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to embed and store: {e}")

    # --- Result ---
    elapsed = time.perf_counter() - start_time
    total_chars = sum(c.char_count for c in chunks)

    result = IngestionResult(
        source_file=filename,
        file_type=file_type.value,
        total_pages=parsed_doc.total_pages,
        total_chunks=len(chunks),
        chunking_strategy=strategy.value,
        avg_chunk_size=round(total_chars / max(len(chunks), 1), 1),
        total_characters=total_chars,
        estimated_tokens=embed_stats.get("estimated_tokens", 0),
        processing_time_seconds=round(elapsed, 2),
        vision_pages_processed=vision_pages,
        errors=errors,
    )

    logger.info(
        "ingest.complete",
        file=file.filename,
        chunks=result.total_chunks,
        time_s=result.processing_time_seconds,
    )

    return result


@router.get("/ingest/stats")
async def get_ingestion_stats():
    """Get current statistics from vector and keyword stores."""
    try:
        pipeline = EmbeddingPipeline()
        return pipeline.get_collection_stats()
    except Exception as e:
        logger.error("ingest.stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/ingest/{source_file}")
async def delete_document(source_file: str):
    """Remove all chunks for a document from both stores."""
    try:
        pipeline = EmbeddingPipeline()
        result = pipeline.delete_by_source(source_file)
        return {"status": "deleted", **result}
    except Exception as e:
        logger.error("ingest.delete_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
