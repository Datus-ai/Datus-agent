import json
from unittest.mock import patch

import pytest

from datus.utils.message_utils import (
    build_structured_content,
    extract_enhanced_context,
    extract_user_input,
    is_structured_content,
)

# ---------------------------------------------------------------------------
# build_structured_content
# ---------------------------------------------------------------------------


def test_build_structured_content_roundtrip_with_unicode():
    """Serialized parts can be round-tripped back through json.loads and preserve unicode."""
    parts = [
        {"type": "user", "content": "你好世界"},
        {"type": "enhanced", "content": "Context: greeting"},
    ]
    result = build_structured_content(parts)
    parsed = json.loads(result)

    assert parsed == parts
    assert "你好世界" in result  # ensure_ascii=False keeps raw characters


def test_build_structured_content_empty_list():
    """An empty parts list produces a valid JSON array string."""
    result = build_structured_content([])

    assert result == "[]"
    assert json.loads(result) == []


# ---------------------------------------------------------------------------
# is_structured_content  (covers lines 50, 55-57)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain text message",
        '{"type": "user", "content": "hello"}',  # dict, not array
        "  { not array }",
        "123",
        "null",
    ],
    ids=[
        "empty_string",
        "plain_text",
        "json_object_not_array",
        "curly_brace_start",
        "numeric_string",
        "null_string",
    ],
)
def test_is_structured_content_string_not_starting_with_bracket_returns_false(value):
    """Non-array-like strings must return False (covers line 50)."""
    assert is_structured_content(value) is False
    assert isinstance(is_structured_content(value), bool)


@pytest.mark.parametrize(
    "value",
    [
        None,
        42,
        3.14,
        True,
        ["not", "a", "string"],
        {"key": "value"},
    ],
    ids=[
        "none",
        "int",
        "float",
        "bool",
        "list",
        "dict",
    ],
)
def test_is_structured_content_non_string_input_returns_false(value):
    """Non-string inputs must return False without raising (covers line 50)."""
    assert is_structured_content(value) is False
    assert isinstance(is_structured_content(value), bool)


def test_is_structured_content_malformed_json_returns_false():
    """Malformed JSON that starts with '[' must return False (covers lines 55-57)."""
    malformed = "[{this is not valid json!!!"
    result = is_structured_content(malformed)

    assert result is False
    assert isinstance(result, bool)


def test_is_structured_content_array_without_user_type_returns_false():
    """A valid JSON array without a 'type: user' element returns False."""
    content = json.dumps([{"type": "enhanced", "content": "ctx"}])
    result = is_structured_content(content)

    assert result is False
    assert isinstance(result, bool)


def test_is_structured_content_empty_json_array_returns_false():
    """An empty JSON array returns False since there is no user part."""
    result = is_structured_content("[]")

    assert result is False
    assert isinstance(result, bool)


def test_is_structured_content_valid_structured_returns_true():
    """A well-formed structured message returns True."""
    content = json.dumps([{"type": "user", "content": "hello"}])
    result = is_structured_content(content)

    assert result is True
    assert isinstance(result, bool)


def test_is_structured_content_array_of_non_dict_items_returns_false():
    """A JSON array of primitives (no dicts) returns False (covers lines 55-57 TypeError path)."""
    content = json.dumps(["just", "a", "list", "of", "strings"])
    result = is_structured_content(content)

    assert result is False
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# extract_user_input  (covers lines 68, 74-76)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("hello world", "hello world"),
        ("", ""),
        ("  some plain text  ", "  some plain text  "),
        ('{"key": "value"}', '{"key": "value"}'),
    ],
    ids=[
        "plain_text",
        "empty_string",
        "whitespace_padded",
        "json_object_string",
    ],
)
def test_extract_user_input_non_structured_returns_content_unchanged(value, expected):
    """When content is not structured, return it unchanged (covers line 68)."""
    result = extract_user_input(value)

    assert result == expected
    assert isinstance(result, str)


def test_extract_user_input_valid_structured_returns_user_part():
    """When content is structured, extract the user part."""
    content = json.dumps(
        [
            {"type": "user", "content": "original question"},
            {"type": "enhanced", "content": "Context: extra info"},
        ]
    )
    result = extract_user_input(content)

    assert result == "original question"
    assert "Context" not in result


def test_extract_user_input_structured_missing_content_key_returns_fallback():
    """User part without a 'content' key falls back to the raw content string."""
    parts = [{"type": "user"}]
    content = json.dumps(parts)
    result = extract_user_input(content)

    # part.get("content", content) returns the full raw content as fallback
    assert result == content
    assert isinstance(result, str)


def test_extract_user_input_malformed_json_returns_content_unchanged():
    """A string starting with '[' but containing invalid JSON returns unchanged (line 68)."""
    broken = "[not valid json"
    result = extract_user_input(broken)

    assert result == broken
    assert isinstance(result, str)


def test_extract_user_input_json_decode_error_returns_fallback():
    """When json.loads raises inside extract_user_input, the raw content is returned (covers lines 74-76).

    Patch is_structured_content to return True so we enter the try block, then
    patch json.loads to raise JSONDecodeError to exercise the except branch.
    """
    raw = '[{"type": "user", "content": "hello"}]'
    with patch("datus.utils.message_utils.is_structured_content", return_value=True):
        with patch("datus.utils.message_utils.json.loads", side_effect=json.JSONDecodeError("err", "doc", 0)):
            result = extract_user_input(raw)

    assert result == raw
    assert isinstance(result, str)


def test_extract_user_input_type_error_returns_fallback():
    """When json.loads raises TypeError inside extract_user_input, the raw content is returned (covers lines 74-76)."""
    raw = '[{"type": "user", "content": "hello"}]'
    with patch("datus.utils.message_utils.is_structured_content", return_value=True):
        with patch("datus.utils.message_utils.json.loads", side_effect=TypeError("bad type")):
            result = extract_user_input(raw)

    assert result == raw
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_enhanced_context  (covers lines 86, 92-94)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "hello world",
        "",
        '{"type": "enhanced", "content": "ctx"}',
        "42",
        "[not valid json",
    ],
    ids=[
        "plain_text",
        "empty_string",
        "json_object_string",
        "numeric_string",
        "broken_array_prefix",
    ],
)
def test_extract_enhanced_context_non_structured_returns_none(value):
    """When content is not structured, return None (covers line 86)."""
    result = extract_enhanced_context(value)

    assert result is None
    assert not isinstance(result, str)


def test_extract_enhanced_context_valid_structured_returns_enhanced_part():
    """When content is structured with an enhanced part, extract it."""
    content = json.dumps(
        [
            {"type": "user", "content": "question"},
            {"type": "enhanced", "content": "Context: relevant info"},
        ]
    )
    result = extract_enhanced_context(content)

    assert result == "Context: relevant info"
    assert "question" not in result


def test_extract_enhanced_context_structured_without_enhanced_returns_none():
    """Structured content without an enhanced part returns None."""
    content = json.dumps([{"type": "user", "content": "just a question"}])
    result = extract_enhanced_context(content)

    assert result is None
    assert not isinstance(result, str)


def test_extract_enhanced_context_multiple_enhanced_returns_first():
    """When multiple enhanced parts exist, the first one is returned."""
    content = json.dumps(
        [
            {"type": "user", "content": "q"},
            {"type": "enhanced", "content": "first context"},
            {"type": "enhanced", "content": "second context"},
        ]
    )
    result = extract_enhanced_context(content)

    assert result == "first context"
    assert result != "second context"


def test_extract_enhanced_context_json_decode_error_returns_none():
    """When json.loads raises inside extract_enhanced_context, None is returned (covers lines 92-94).

    Patch is_structured_content to return True so we enter the try block, then
    patch json.loads to raise JSONDecodeError to exercise the except branch.
    """
    raw = '[{"type": "user", "content": "q"}, {"type": "enhanced", "content": "ctx"}]'
    with patch("datus.utils.message_utils.is_structured_content", return_value=True):
        with patch("datus.utils.message_utils.json.loads", side_effect=json.JSONDecodeError("err", "doc", 0)):
            result = extract_enhanced_context(raw)

    assert result is None
    assert not isinstance(result, str)


def test_extract_enhanced_context_type_error_returns_none():
    """When json.loads raises TypeError inside extract_enhanced_context, None is returned (covers lines 92-94)."""
    raw = '[{"type": "user", "content": "q"}, {"type": "enhanced", "content": "ctx"}]'
    with patch("datus.utils.message_utils.is_structured_content", return_value=True):
        with patch("datus.utils.message_utils.json.loads", side_effect=TypeError("bad type")):
            result = extract_enhanced_context(raw)

    assert result is None
    assert not isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration: round-trip through build -> extract
# ---------------------------------------------------------------------------


def test_roundtrip_build_then_extract_user_and_context():
    """Build structured content and verify both user input and context can be extracted."""
    parts = [
        {"type": "user", "content": "What is 2+2?"},
        {"type": "enhanced", "content": "The user is asking a math question."},
    ]
    structured = build_structured_content(parts)

    assert extract_user_input(structured) == "What is 2+2?"
    assert extract_enhanced_context(structured) == "The user is asking a math question."
