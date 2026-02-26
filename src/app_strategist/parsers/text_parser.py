"""Text-based parsers for .txt and .md files."""

import logging
from pathlib import Path

from app_strategist.parsers.base import DocumentParser, JobDescriptionParser

logger = logging.getLogger(__name__)


def _validate_and_read_file(path: Path) -> str:
    """Validate file exists, is readable, non-empty; return content. Raises on error."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise PermissionError(f"Cannot read file: {path}") from e
    if not content.strip():
        raise ValueError(f"File is empty: {path}")
    return content


class TextDocumentParser:
    """Parser for resume/cover letter in .txt or .md format."""

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def parse(self, path: Path) -> str:
        """Parse the document and return its text content."""
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}' for document. Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
        return _validate_and_read_file(path)


class TextJobDescriptionParser:
    """Parser for job description in .txt or .md format."""

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def parse(self, path: Path) -> str:
        """Parse the job description and return its text content."""
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}' for job description. Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
        return _validate_and_read_file(path)
