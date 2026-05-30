"""Tests for the ingestion pipeline.

Tests parsing, chunking, and the end-to-end flow.
Embedding/store tests require Docker services running (integration tests).
"""

import tempfile

import pytest

from backend.ingestion.chunker import chunk_document
from backend.ingestion.metadata import extract_file_metadata
from backend.models.document import ChunkingStrategy, FileType, ParsedDocument, ParsedPage

# --- File type detection ---

class TestFileType:
    def test_pdf_extension(self):
        assert FileType.from_extension(".pdf") == FileType.PDF

    def test_docx_extension(self):
        assert FileType.from_extension(".docx") == FileType.DOCX

    def test_csv_extension(self):
        assert FileType.from_extension(".csv") == FileType.CSV

    def test_txt_extension(self):
        assert FileType.from_extension(".txt") == FileType.TXT

    def test_unknown_extension(self):
        assert FileType.from_extension(".xyz") == FileType.UNKNOWN

    def test_case_insensitive(self):
        assert FileType.from_extension(".PDF") == FileType.PDF


# --- TXT Parsing ---

class TestTxtParser:
    def test_parse_txt(self):
        from backend.ingestion.parser import parse_document

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello world.\n" * 100)
            f.flush()
            doc = parse_document(f.name)

        assert doc.total_pages >= 1
        assert doc.file_type == FileType.TXT
        assert "Hello world" in doc.pages[0].content

    def test_parse_empty_txt(self):
        from backend.ingestion.parser import parse_document

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            doc = parse_document(f.name)

        assert doc.total_pages == 0


# --- CSV Parsing ---

class TestCsvParser:
    def test_parse_csv(self):
        from backend.ingestion.parser import parse_document

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
            f.flush()
            doc = parse_document(f.name)

        assert doc.total_pages == 1
        assert "Alice" in doc.pages[0].content
        assert doc.metadata["rows"] == 2
        assert doc.metadata["columns"] == 3


# --- Chunking ---

class TestChunker:
    @pytest.fixture
    def sample_document(self):
        return ParsedDocument(
            source_file="test.txt",
            file_type=FileType.TXT,
            title="Test Document",
            total_pages=2,
            pages=[
                ParsedPage(
                    page_number=1,
                    content="This is the first page with some content. " * 50,
                ),
                ParsedPage(
                    page_number=2,
                    content="This is the second page with different content. " * 50,
                ),
            ],
        )

    def test_recursive_chunking(self, sample_document):
        chunks = chunk_document(
            sample_document,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=50,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.char_count <= 250  # some tolerance
            assert chunk.source_file == "test.txt"
            assert chunk.chunking_strategy == "recursive"
            assert chunk.page_number in [1, 2]

    def test_chunk_metadata(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=500)
        for chunk in chunks:
            assert chunk.title == "Test Document"
            assert chunk.file_type == "txt"
            assert chunk.chunk_index >= 0

    def test_empty_document(self):
        doc = ParsedDocument(
            source_file="empty.txt",
            file_type=FileType.TXT,
            title="Empty",
            total_pages=0,
            pages=[],
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_indices_are_sequential(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# --- Metadata ---

class TestMetadata:
    def test_file_metadata(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("test content")
            f.flush()
            meta = extract_file_metadata(f.name)

        assert meta["file_name"].endswith(".txt")
        assert meta["file_extension"] == ".txt"
        assert meta["file_size_bytes"] > 0
        assert "ingested_at" in meta
