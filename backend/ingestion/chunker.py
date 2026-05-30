"""Dual chunking strategies for document splitting.

Provides two approaches:
1. Recursive Character Splitting — fast, predictable, good default
2. Semantic Chunking — embedding-based natural break detection

Both preserve metadata per chunk for downstream attribution.
A/B testable via the evaluation suite to find optimal strategy per doc type.

Reference: RAGFlow's rag/utils/ provides template-based chunking.
We use LangChain's splitters for consistency with the LangChain ecosystem.
"""

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings
from backend.models.document import Chunk, ChunkingStrategy, ParsedDocument

logger = structlog.get_logger()


def chunk_document(
    document: ParsedDocument,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a parsed document into chunks using the specified strategy.

    Args:
        document: Parsed document with pages
        strategy: 'recursive' or 'semantic'
        chunk_size: Override default chunk size (characters)
        chunk_overlap: Override default overlap (characters)

    Returns:
        List of Chunk objects ready for embedding
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    logger.info(
        "chunker.start",
        file=document.source_file,
        strategy=strategy.value,
        chunk_size=size,
        overlap=overlap,
        total_pages=document.total_pages,
    )

    if strategy == ChunkingStrategy.SEMANTIC:
        chunks = _semantic_chunk(document, size, overlap)
    else:
        chunks = _recursive_chunk(document, size, overlap)

    # Assign chunk indices and metadata
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.chunking_strategy = strategy.value
        chunk.title = document.title
        chunk.source_file = document.source_file
        chunk.file_type = document.file_type.value
        chunk.metadata.update(
            {
                "document_title": document.title,
                "total_chunks": len(chunks),
                **{
                    k: v
                    for k, v in document.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
        )

    logger.info(
        "chunker.complete",
        file=document.source_file,
        strategy=strategy.value,
        total_chunks=len(chunks),
        avg_size=sum(c.char_count for c in chunks) / max(len(chunks), 1),
    )

    return chunks


def _recursive_chunk(
    document: ParsedDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Recursive character text splitting.

    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    on natural boundaries (paragraphs, sentences, words) before falling
    back to character-level splits.

    Best for: general documents, predictable chunk sizes, fast processing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Chunk] = []

    for page in document.pages:
        if not page.content.strip():
            continue

        splits = splitter.split_text(page.content)

        for split_text in splits:
            if split_text.strip():
                chunks.append(
                    Chunk(
                        content=split_text.strip(),
                        source_file=document.source_file,
                        file_type=document.file_type.value,
                        page_number=page.page_number,
                    )
                )

    return chunks


def _semantic_chunk(
    document: ParsedDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Semantic chunking using embedding similarity.

    Uses LangChain's SemanticChunker which computes embeddings for
    sentences and groups them based on semantic similarity — splitting
    where the topic shifts.

    Best for: long-form documents, research papers, technical docs where
    natural topic boundaries don't align with character counts.

    Falls back to recursive chunking if embedding fails.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75,
        )

        # Combine all pages into one text for semantic analysis
        full_text = document.full_text
        if not full_text.strip():
            return []

        splits = semantic_splitter.split_text(full_text)

        # Map splits back to approximate page numbers
        chunks: list[Chunk] = []
        char_offset = 0
        page_boundaries = _build_page_boundaries(document)

        for split_text in splits:
            if not split_text.strip():
                continue

            page_num = _find_page_for_offset(char_offset, page_boundaries)
            chunks.append(
                Chunk(
                    content=split_text.strip(),
                    source_file=document.source_file,
                    file_type=document.file_type.value,
                    page_number=page_num,
                )
            )
            char_offset += len(split_text) + 2  # +2 for \n\n separator

        # If semantic chunks are too large, sub-split with recursive
        final_chunks: list[Chunk] = []
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in chunks:
            if chunk.char_count > chunk_size * 1.5:
                sub_splits = sub_splitter.split_text(chunk.content)
                for sub in sub_splits:
                    final_chunks.append(
                        Chunk(
                            content=sub.strip(),
                            source_file=chunk.source_file,
                            file_type=chunk.file_type,
                            page_number=chunk.page_number,
                        )
                    )
            else:
                final_chunks.append(chunk)

        return final_chunks

    except Exception as e:
        logger.warning(
            "chunker.semantic_fallback",
            error=str(e),
            fallback="recursive",
        )
        return _recursive_chunk(document, chunk_size, chunk_overlap)


def _build_page_boundaries(document: ParsedDocument) -> list[tuple[int, int, int]]:
    """Build (start_offset, end_offset, page_number) tuples."""
    boundaries = []
    offset = 0
    for page in document.pages:
        length = len(page.content)
        boundaries.append((offset, offset + length, page.page_number))
        offset += length + 2  # +2 for \n\n join separator
    return boundaries


def _find_page_for_offset(
    offset: int, boundaries: list[tuple[int, int, int]]
) -> int:
    """Find which page a character offset belongs to."""
    for start, end, page_num in boundaries:
        if start <= offset < end:
            return page_num
    return boundaries[-1][2] if boundaries else 1
