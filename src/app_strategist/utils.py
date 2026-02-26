"""Shared utilities."""

import json


class JSONExtractionError(Exception):
    """Raised when no valid JSON object could be extracted from LLM response text."""


def _extract_json_object(text: str) -> str | None:
    """Extract a complete JSON object using the standard library. Handles escape
    sequences, nested structures, and all valid JSON correctly.
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        decoder = json.JSONDecoder()
        _, end_idx = decoder.raw_decode(text, start)
        return text[start:end_idx]
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> str:
    """Extract JSON from LLM response (may be wrapped in markdown code blocks).
    Uses brace matching to correctly handle content with triple backticks or
    other markdown inside string values.
    """
    text = text.strip()
    # Prefer content after ```json or ``` if present, then use brace matching
    for marker in ("```json", "```"):
        if marker in text:
            idx = text.find(marker)
            after_marker = text[idx + len(marker) :].lstrip()
            result = _extract_json_object(after_marker)
            if result:
                return result
    # No markdown wrapper: find first JSON object
    result = _extract_json_object(text)
    if result is None:
        raise JSONExtractionError(
            "Could not extract a valid JSON object from the LLM response. "
            "The model may have returned non-JSON content (e.g., explanatory text or malformed output)."
        )
    return result
