"""LLM provider protocol - abstract interface for LLM backends."""

from typing import Protocol


class LLMProvider(Protocol):
    """Protocol for LLM providers. Implement this to add new backends (e.g., OpenAI)."""

    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        """
        Send messages to the LLM and return the assistant's text response.

        Args:
            system_prompt: System/instruction prompt for the model.
            messages: List of message dicts with "role" ("user"|"assistant") and "content" (str).

        Returns:
            The assistant's response text.
        """
        ...
