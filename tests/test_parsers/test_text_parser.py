"""Tests for text parsers."""

import pytest
from pathlib import Path

from app_strategist.parsers.text_parser import TextDocumentParser, TextJobDescriptionParser


def test_text_document_parser_reads_md(tmp_path: Path) -> None:
    content = "# Resume\nHello world"
    f = tmp_path / "resume.md"
    f.write_text(content)
    parser = TextDocumentParser()
    assert parser.parse(f) == content


def test_text_document_parser_reads_txt(tmp_path: Path) -> None:
    content = "Resume content"
    f = tmp_path / "resume.txt"
    f.write_text(content)
    parser = TextDocumentParser()
    assert parser.parse(f) == content


def test_text_document_parser_rejects_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "resume.pdf"
    f.write_text("x")
    parser = TextDocumentParser()
    with pytest.raises(ValueError, match="Unsupported extension"):
        parser.parse(f)


def test_text_document_parser_raises_on_missing_file(tmp_path: Path) -> None:
    parser = TextDocumentParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "nonexistent.md")


def test_text_document_parser_raises_on_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.md"
    f.write_text("")
    parser = TextDocumentParser()
    with pytest.raises(ValueError, match="empty"):
        parser.parse(f)


def test_text_job_description_parser_reads_txt(tmp_path: Path) -> None:
    content = "Job description here"
    f = tmp_path / "job.txt"
    f.write_text(content)
    parser = TextJobDescriptionParser()
    assert parser.parse(f) == content


def test_text_job_description_parser_rejects_pdf(tmp_path: Path) -> None:
    f = tmp_path / "job.pdf"
    f.write_text("x")
    parser = TextJobDescriptionParser()
    with pytest.raises(ValueError, match="Unsupported extension"):
        parser.parse(f)
