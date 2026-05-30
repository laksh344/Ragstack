"""Document and chunk metadata extraction utilities."""

from pathlib import Path
from datetime import datetime, timezone


def extract_file_metadata(file_path: str | Path) -> dict:
    """Extract file-system level metadata from a document."""
    path = Path(file_path)
    stat = path.stat()

    return {
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
