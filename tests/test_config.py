"""Tests for config module - get_api_key and get_llm_provider."""

import os
from unittest.mock import patch

import pytest

from app_strategist.config import get_api_key, get_llm_provider


class TestGetApiKey:
    def test_anthropic_with_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        assert get_api_key("anthropic") == "sk-test-123"

    def test_anthropic_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-abc")
        assert get_api_key("ANTHROPIC") == "key-abc"
        assert get_api_key("  anthropic  ") == "key-abc"

    def test_anthropic_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  key  ")
        assert get_api_key("anthropic") == "key"

    def test_openai_with_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-456")
        assert get_api_key("openai") == "sk-openai-456"

    def test_raises_for_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider.*Supported: anthropic, openai"):
            get_api_key("unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_api_key("gemini")

    def test_raises_when_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            get_api_key("anthropic")

    def test_raises_when_key_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            get_api_key("anthropic")

    def test_raises_when_key_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            get_api_key("anthropic")


class TestGetLlmProvider:
    def test_returns_anthropic_provider_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from app_strategist.llm import AnthropicProvider

        provider = get_llm_provider(provider="anthropic")
        assert isinstance(provider, AnthropicProvider)
        assert provider.model

    def test_returns_openai_provider_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        from app_strategist.llm import OpenAIProvider

        provider = get_llm_provider(provider="openai")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model

    def test_reads_llm_provider_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
        from app_strategist.llm import OpenAIProvider

        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_defaults_to_anthropic_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-default")
        from app_strategist.llm import AnthropicProvider

        provider = get_llm_provider()
        assert isinstance(provider, AnthropicProvider)

    def test_raises_for_unknown_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        with pytest.raises(ValueError, match="Unknown LLM provider.*Supported"):
            get_llm_provider(provider="unknown")

    def test_raises_when_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            get_llm_provider(provider="anthropic")
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            get_llm_provider(provider="openai")
