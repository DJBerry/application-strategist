"""Tests for LLM providers - AnthropicProvider and OpenAIProvider."""

from unittest.mock import MagicMock, patch

import pytest

from app_strategist.llm.anthropic_provider import AnthropicProvider
from app_strategist.llm.openai_provider import OpenAIProvider


class TestAnthropicProvider:
    def test_complete_returns_response_text(self) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("app_strategist.llm.anthropic_provider.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-test")
            result = provider.complete("You are helpful.", [{"role": "user", "content": "Hi"}])

        assert result == "Hello from Claude"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["system"] == "You are helpful."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    def test_complete_handles_empty_content(self) -> None:
        mock_response = MagicMock()
        mock_response.content = []
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("app_strategist.llm.anthropic_provider.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-test")
            result = provider.complete("System", [])

        assert result == ""

    def test_uses_custom_model_and_max_tokens(self) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("app_strategist.llm.anthropic_provider.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(
                api_key="sk-test",
                model="claude-3-haiku",
                max_tokens=1024,
            )
            provider.complete("Sys", [])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-haiku"
        assert call_kwargs["max_tokens"] == 1024


class TestOpenAIProvider:
    def test_complete_returns_response_text(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from GPT"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("app_strategist.llm.openai_provider.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            result = provider.complete("You are helpful.", [{"role": "user", "content": "Hi"}])

        assert result == "Hello from GPT"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.2"
        assert call_kwargs["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Hi"}

    def test_complete_handles_empty_choices(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("app_strategist.llm.openai_provider.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            result = provider.complete("System", [])

        assert result == ""

    def test_complete_handles_none_content(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("app_strategist.llm.openai_provider.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            result = provider.complete("System", [])

        assert result == ""

    def test_uses_custom_model_and_max_tokens(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("app_strategist.llm.openai_provider.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(
                api_key="sk-test",
                model="gpt-4o-mini",
                max_tokens=512,
            )
            provider.complete("Sys", [])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_completion_tokens"] == 512
