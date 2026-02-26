"""Parser registries - map file extensions to parsers."""

from pathlib import Path

from app_strategist.parsers.base import DocumentParser, JobDescriptionParser
from app_strategist.parsers.text_parser import TextDocumentParser, TextJobDescriptionParser


class DocumentParserRegistry:
    """Registry mapping file extensions to document parsers."""

    def __init__(self) -> None:
        self._text_parser = TextDocumentParser()
        self._parsers: dict[str, DocumentParser] = {
            ext: self._text_parser for ext in TextDocumentParser.SUPPORTED_EXTENSIONS
        }

    def parse(self, path: Path) -> str:
        """Parse the document at path. Raises ValueError for unsupported extensions."""
        ext = path.suffix.lower()
        if ext not in self._parsers:
            supported = ", ".join(sorted(self._parsers.keys()))
            raise ValueError(
                f"Unsupported document extension '{ext}'. Supported: {supported}. "
                "Future versions may support .pdf and .docx."
            )
        return self._parsers[ext].parse(path)


class JobDescriptionParserRegistry:
    """Registry mapping file extensions to job description parsers."""

    def __init__(self) -> None:
        self._text_parser = TextJobDescriptionParser()
        self._parsers: dict[str, JobDescriptionParser] = {
            ext: self._text_parser for ext in TextJobDescriptionParser.SUPPORTED_EXTENSIONS
        }

    def parse(self, path: Path) -> str:
        """Parse the job description at path. Raises ValueError for unsupported extensions."""
        ext = path.suffix.lower()
        if ext not in self._parsers:
            supported = ", ".join(sorted(self._parsers.keys()))
            raise ValueError(
                f"Unsupported job description extension '{ext}'. Supported: {supported}. "
                "Future versions may support URLs."
            )
        return self._parsers[ext].parse(path)
