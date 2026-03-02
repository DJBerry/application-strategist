"""Anthropic Claude LLM provider implementation."""

import logging

from anthropic import Anthropic

from app_strategist.config import get_api_key
from app_strategist.llm.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    """Anthropic Claude implementation of LLMProvider."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key or get_api_key("anthropic"))

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        """Call Claude and return the assistant text."""
        logger.debug("Calling Anthropic API with model=%s", self.model)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        text = response.content[0].text if response.content else ""
        logger.debug("Received response (%d chars)", len(text))
        return text
