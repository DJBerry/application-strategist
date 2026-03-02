"""OpenAI LLM provider implementation."""

import logging

from openai import OpenAI

from app_strategist.llm.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"  # "gpt-5.2"  # "gpt-4o"
DEFAULT_MAX_TOKENS = 4096


class OpenAIProvider:
    """OpenAI implementation of LLMProvider."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            from app_strategist.config import get_api_key

            api_key = get_api_key("openai")
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        """Call OpenAI and return the assistant text."""
        logger.debug("Calling OpenAI API with model=%s", self.model)
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=all_messages,
        )
        content = response.choices[0].message.content if response.choices else ""
        text = content or ""
        logger.debug("Received response (%d chars)", len(text))
        return text
