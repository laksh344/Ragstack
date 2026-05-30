"""Retrieval layer — vector, keyword, hybrid, and reranked search."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single retrieved chunk with score and provenance."""

    chunk_id: str
    content: str
    source_file: str
    page_number: int = 0
    chunk_index: int = 0
    title: str = ""
    file_type: str = ""
    score: float = 0.0
    # "vector" | "keyword" | "hybrid" | "reranked"
    source: str = "vector"
    metadata: dict = Field(default_factory=dict)
