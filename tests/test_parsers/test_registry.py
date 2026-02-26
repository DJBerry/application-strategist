"""Tests for parser registries."""

import pytest
from pathlib import Path

from app_strategist.parsers.registry import DocumentParserRegistry, JobDescriptionParserRegistry


def test_document_registry_parses_md(tmp_path: Path) -> None:
    content = "Resume content"
    f = tmp_path / "doc.md"
    f.write_text(content)
    registry = DocumentParserRegistry()
    assert registry.parse(f) == content


def test_document_registry_parses_txt(tmp_path: Path) -> None:
    content = "Resume content"
    f = tmp_path / "doc.txt"
    f.write_text(content)
    registry = DocumentParserRegistry()
    assert registry.parse(f) == content


def test_document_registry_rejects_unsupported(tmp_path: Path) -> None:
    f = tmp_path / "doc.pdf"
    f.write_text("x")
    registry = DocumentParserRegistry()
    with pytest.raises(ValueError, match="Unsupported document extension"):
        registry.parse(f)


def test_job_registry_parses_txt(tmp_path: Path) -> None:
    content = "Job description"
    f = tmp_path / "job.txt"
    f.write_text(content)
    registry = JobDescriptionParserRegistry()
    assert registry.parse(f) == content


def test_job_registry_rejects_unsupported(tmp_path: Path) -> None:
    f = tmp_path / "job.pdf"
    f.write_text("x")
    registry = JobDescriptionParserRegistry()
    with pytest.raises(ValueError, match="Unsupported job description extension"):
        registry.parse(f)
