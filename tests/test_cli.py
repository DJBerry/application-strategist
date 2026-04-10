"""Tests for CLI - Typer analyze command."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from app_strategist.main import app


runner = CliRunner()

_EMPLOYER_JSON = '{"strengths":["x"],"gaps":[],"suggested_improvements":[],"wording_suggestions":[],"fit_score":{"value":75,"components":[{"name":"Skills/experience alignment","weight":0.35,"score":80,"explanation":"x"},{"name":"Keyword/match density","weight":0.25,"score":70,"explanation":"x"},{"name":"Clarity and structure","weight":0.2,"score":75,"explanation":"x"},{"name":"Quantified achievements","weight":0.15,"score":60,"explanation":"x"},{"name":"Tailoring to role","weight":0.05,"score":80,"explanation":"x"}],"score_rationale":"OK"}'
_CANDIDATE_JSON = '{"positive_alignments":[],"concerns":[],"questions_to_ask":[],"worker_fit_score":{"value":70,"components":[{"name":"Likely interest alignment","weight":0.2,"score":80,"explanation":"x"},{"name":"Role clarity","weight":0.15,"score":80,"explanation":"x"},{"name":"Compensation/benefits transparency","weight":0.2,"score":50,"explanation":"x"},{"name":"Work-life balance signals","weight":0.15,"score":70,"explanation":"x"},{"name":"Growth/career path","weight":0.15,"score":75,"explanation":"x"},{"name":"Red-flag absence","weight":0.15,"score":75,"explanation":"x"}],"score_rationale":"OK"}'


def _make_mock_llm():
    class MockLLM:
        def __init__(self):
            self._responses = [_EMPLOYER_JSON, _CANDIDATE_JSON]
            self._call_count = 0

        def complete(self, system_prompt: str, messages: list) -> str:
            if self._call_count >= len(self._responses):
                raise AssertionError(
                    f"MockLLM.complete() called {self._call_count + 1} times, "
                    f"but only {len(self._responses)} responses configured"
                )
            idx = self._call_count
            self._call_count += 1
            return self._responses[idx]

    return MockLLM()


def test_analyze_missing_resume_file(sample_job: Path, tmp_path: Path) -> None:
    """Analyze command exits with error when resume file does not exist."""
    fake_resume = tmp_path / "nonexistent.md"
    mock_llm = _make_mock_llm()

    with (
        patch("app_strategist.main.get_llm_provider", return_value=mock_llm),
        patch("app_strategist.main._run_repl"),
    ):
        result = runner.invoke(
            app,
            ["analyze", "--resume", str(fake_resume), "--job", str(sample_job)],
        )

    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_analyze_missing_api_key(sample_resume: Path, sample_job: Path) -> None:
    """Analyze command exits with error when get_llm_provider raises (e.g. missing API key)."""
    with patch(
        "app_strategist.main.get_llm_provider",
        side_effect=ValueError("ANTHROPIC_API_KEY is not set"),
    ):
        result = runner.invoke(
            app,
            ["analyze", "--resume", str(sample_resume), "--job", str(sample_job)],
        )

    assert result.exit_code == 1
    assert "Error" in result.stdout
    assert "ANTHROPIC_API_KEY" in result.stdout
