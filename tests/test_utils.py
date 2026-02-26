"""Tests for utils."""

import json
import pytest

from app_strategist.utils import extract_json, JSONExtractionError


def test_extract_json_plain() -> None:
    text = '{"a": 1}'
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_in_markdown() -> None:
    text = """Some text
```json
{"a": 1, "b": 2}
```
More text"""
    assert extract_json(text) == '{"a": 1, "b": 2}'


def test_extract_json_code_block_no_json_tag() -> None:
    text = """```
{"x": "y"}
```"""
    assert extract_json(text) == '{"x": "y"}'


def test_extract_json_finds_object() -> None:
    text = "Here is the result: {\"key\": \"value\"} end"
    result = extract_json(text)
    assert "key" in result
    assert "value" in result


def test_extract_json_handles_backticks_in_content() -> None:
    """JSON with triple backticks inside a string value - split-based extraction would truncate."""
    text = '''```json
{"description": "Use ```code``` here", "count": 1}
```'''
    result = extract_json(text)
    parsed = json.loads(result)
    assert parsed["description"] == "Use ```code``` here"
    assert parsed["count"] == 1


def test_extract_json_handles_braces_in_strings() -> None:
    """JSON with braces inside string values."""
    text = '{"msg": "Use {braces} and } here"}'
    result = extract_json(text)
    parsed = json.loads(result)
    assert parsed["msg"] == "Use {braces} and } here"


def test_extract_json_handles_escaped_quotes() -> None:
    """JSON with escaped quotes inside string values - regression for double-increment bug."""
    text = r'{"desc": "Say \"hello\" here"}'
    result = extract_json(text)
    parsed = json.loads(result)
    assert parsed["desc"] == 'Say "hello" here'


def test_extract_json_raises_on_no_json() -> None:
    """Raises JSONExtractionError when no valid JSON object can be extracted."""
    with pytest.raises(JSONExtractionError, match="Could not extract"):
        extract_json("Here is my response: no JSON here")
