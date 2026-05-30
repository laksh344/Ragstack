"""Data models for documents, chunks, and ingestion results."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"
    TXT = "txt"
    XLSX = "xlsx"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        mapping = {
            ".pdf": cls.PDF,
            ".docx": cls.DOCX,
            ".csv": cls.CSV,
            ".txt": cls.TXT,
            ".xlsx": cls.XLSX,
        }
        return mapping.get(ext.lower(), cls.UNKNOWN)


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class ParsedPage(BaseModel):
    """A single page/section extracted from a document."""

    page_number: int
    content: str
    has_tables: bool = False
    has_images: bool = False
    table_data: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Result of parsing a document file."""

    source_file: str
    file_type: FileType = FileType.UNKNOWN
    title: str = ""
    total_pages: int = 0
    pages: list[ParsedPage] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.content for p in self.pages if p.content)


class Chunk(BaseModel):
    """A text chunk ready for embedding and storage."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source_file: str
    file_type: str
    page_number: int = 0
    chunk_index: int = 0
    chunking_strategy: str = "recursive"
    title: str = ""
    char_count: int = 0
    token_estimate: int = 0
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        self.char_count = len(self.content)
        self.token_estimate = self.char_count // 4  # rough estimate


class IngestionResult(BaseModel):
    """Summary of an ingestion job."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    source_file: str
    file_type: str
    total_pages: int = 0
    total_chunks: int = 0
    chunking_strategy: str = "recursive"
    avg_chunk_size: float = 0.0
    total_characters: int = 0
    estimated_tokens: int = 0
    processing_time_seconds: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    errors: list[str] = Field(default_factory=list)
    vision_pages_processed: int = 0
