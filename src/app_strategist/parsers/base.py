"""Parser protocols - extensible interfaces for document and job description parsing."""

from pathlib import Path
from typing import Protocol


class DocumentParser(Protocol):
    """Protocol for parsing resume/cover letter documents."""

    def parse(self, path: Path) -> str:
        """Parse the file and return its text content."""
        ...


class JobDescriptionParser(Protocol):
    """Protocol for parsing job description sources (file, URL in future)."""

    def parse(self, path: Path) -> str:
        """Parse the job description and return its text content."""
        ...
