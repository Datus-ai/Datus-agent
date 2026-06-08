import pytest

from datus.utils.message_utils import (
    SYSTEM_REMINDER_CLOSE,
    SYSTEM_REMINDER_OPEN,
    build_structured_content,
    extract_enhanced_context,
    extract_user_input,
    is_structured_content,
)

# ---------------------------------------------------------------------------
# build_structured_content
# ---------------------------------------------------------------------------


def test_build_structured_content_produces_expected_layout():
    result = build_structured_content("Context: greeting", "你好世界")

    assert result == "<system_reminder>Context: greeting</system_reminder>\n你好世界"
    assert result.startswith(SYSTEM_REMINDER_OPEN)
    assert SYSTEM_REMINDER_CLOSE in result


def test_build_structured_content_preserves_newlines_inside_enhanced():
    enhanced = "Database Context: mysql\n\nAvailable tables:\n- t1\n- t2"
    result = build_structured_content(enhanced, "list tables")

    assert result == f"{SYSTEM_REMINDER_OPEN}{enhanced}{SYSTEM_REMINDER_CLOSE}list tables"
    # Closing tag separates enhanced from user verbatim — newlines inside
    # the enhanced section must NOT be collapsed.
    assert "Available tables:\n- t1\n- t2" in result


def test_build_structured_content_empty_enhanced_still_wraps():
    """build helper itself does not skip empty enhanced — callers must guard."""
    result = build_structured_content("", "raw question")

    assert result == "<system_reminder></system_reminder>\nraw question"


def test_build_structured_content_empty_user_input():
    result = build_structured_content("ctx", "")

    assert result == "<system_reminder>ctx</system_reminder>\n"


# ---------------------------------------------------------------------------
# is_structured_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain text message",
        "<system_reminder>only opening tag",
        "no opening</system_reminder>\n suffix",
        '[{"type": "user", "content": "legacy json"}]',  # legacy JSON envelope — no longer recognised
        "  <system_reminder>leading whitespace</system_reminder>\nuser",
    ],
    ids=[
        "empty_string",
        "plain_text",
        "only_open_tag",
        "only_close_tag",
        "legacy_json_array",
        "leading_whitespace",
    ],
)
def test_is_structured_content_returns_false_for_non_envelopes(value):
    assert is_structured_content(value) is False


@pytest.mark.parametrize(
    "value",
    [None, 42, 3.14, True, ["not", "a", "string"], {"key": "value"}],
    ids=["none", "int", "float", "bool", "list", "dict"],
)
def test_is_structured_content_non_string_returns_false(value):
    assert is_structured_content(value) is False


def test_is_structured_content_complete_envelope_returns_true():
    content = "<system_reminder>ctx</system_reminder>\nuser"

    assert is_structured_content(content) is True


def test_is_structured_content_envelope_with_multiline_enhanced():
    content = "<system_reminder>line1\nline2</system_reminder>\nuser"

    assert is_structured_content(content) is True


# ---------------------------------------------------------------------------
# extract_user_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("hello world", "hello world"),
        ("", ""),
        ("  padded  ", "  padded  "),
        ('{"key": "value"}', '{"key": "value"}'),
        ('[{"type": "user", "content": "legacy"}]', "legacy"),
    ],
    ids=["plain", "empty", "padded", "json_object", "legacy_json_array"],
)
def test_extract_user_input_non_envelope_returns_unchanged(value, expected):
    result = extract_user_input(value)

    assert result == expected
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "value",
    [
        '["hello", "world"]',
        "[1, 2, 3]",
        '[{"foo": "bar"}]',
        '[{"type": "unknown", "content": "x"}]',
        "[]",
    ],
    ids=[
        "string_array",
        "int_array",
        "dict_no_recognized_type",
        "dict_unknown_type",
        "empty_array",
    ],
)
def test_extract_user_input_non_legacy_json_array_returned_unchanged(value):
    """Non-legacy JSON arrays must pass through as-is, never return empty string."""
    assert extract_user_input(value) == value


def test_extract_user_input_envelope_returns_user_portion():
    content = "<system_reminder>Context: greeting</system_reminder>\noriginal question"

    result = extract_user_input(content)

    assert result == "original question"
    assert "Context" not in result


def test_extract_user_input_envelope_with_empty_user_section():
    content = build_structured_content("ctx", "")

    assert extract_user_input(content) == ""


def test_extract_user_input_envelope_user_section_contains_newlines():
    user = "line1\nline2\nline3"
    content = build_structured_content("ctx", user)

    assert extract_user_input(content) == user


def test_extract_user_input_envelope_user_section_contains_literal_close_tag():
    """A second ``</system_reminder>\\n`` inside user text is left untouched —
    only the first occurrence terminates the envelope."""
    user = "weird user text </system_reminder>\nstill user"
    content = build_structured_content("ctx", user)

    assert extract_user_input(content) == user


def test_extract_user_input_none_returns_empty_string():
    assert extract_user_input(None) == ""


def test_extract_user_input_non_string_scalar_is_coerced_to_str():
    assert extract_user_input(42) == "42"


# ---------------------------------------------------------------------------
# extract_user_input — provider content-block list shape
# (Anthropic / OpenAI Responses style, persisted by ClaudeModel OAuth path)
# ---------------------------------------------------------------------------


def test_extract_user_input_anthropic_single_text_block():
    blocks = [{"type": "text", "text": "未来两年的趋势是什么？"}]

    assert extract_user_input(blocks) == "未来两年的趋势是什么？"


def test_extract_user_input_anthropic_multiple_text_blocks_join_with_newline():
    blocks = [
        {"type": "text", "text": "first line"},
        {"type": "text", "text": "second line"},
    ]

    assert extract_user_input(blocks) == "first line\nsecond line"


def test_extract_user_input_openai_input_and_output_text_blocks():
    blocks = [
        {"type": "input_text", "text": "user typed this"},
        {"type": "output_text", "text": "assistant said this"},
    ]

    assert extract_user_input(blocks) == "user typed this\nassistant said this"


def test_extract_user_input_text_block_wrapping_envelope_recursively_unwraps():
    inner = build_structured_content("Database Context: mysql", "现在有哪些表")
    blocks = [{"type": "input_text", "text": inner}]

    assert extract_user_input(blocks) == "现在有哪些表"


def test_extract_user_input_block_list_skips_non_text_blocks():
    blocks = [
        {"type": "text", "text": "explain"},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
        {"type": "tool_result", "tool_use_id": "t1", "content": "ignored"},
    ]

    assert extract_user_input(blocks) == "explain"


def test_extract_user_input_empty_list_returns_empty_string():
    assert extract_user_input([]) == ""


def test_extract_user_input_block_list_with_non_dict_items_is_skipped():
    blocks = [
        "raw string slipped in",
        {"type": "text", "text": "real content"},
        42,
    ]

    assert extract_user_input(blocks) == "real content"


def test_extract_user_input_block_list_text_field_non_string_is_skipped():
    blocks = [
        {"type": "text", "text": ["not", "a", "string"]},
        {"type": "text", "text": "real text"},
    ]

    assert extract_user_input(blocks) == "real text"


def test_extract_user_input_legacy_json_string_enhanced_plus_user():
    """Full issue #887 scenario: legacy JSON array with enhanced + user blocks."""
    content = (
        '[{"type": "enhanced", "content": "Database Context: starrocks"},'
        ' {"type": "user", "content": "What can you do?"}]'
    )

    assert extract_user_input(content) == "What can you do?"


def test_extract_user_input_legacy_list_user_block_directly():
    """Already-parsed list path: legacy user block bypassing the string branch."""
    blocks = [{"type": "user", "content": "direct list input"}]

    assert extract_user_input(blocks) == "direct list input"


def test_extract_user_input_legacy_user_block_empty_content_uses_empty():
    """When content is empty string, should NOT fallback to text field."""
    blocks = [{"type": "user", "content": "", "text": "fallback"}]

    assert extract_user_input(blocks) == ""


def test_extract_user_input_legacy_user_block_none_content_uses_text():
    """When content is None, should fallback to text field."""
    blocks = [{"type": "user", "text": "from text field"}]

    assert extract_user_input(blocks) == "from text field"


def test_extract_user_input_pydantic_compatible_for_chat_session_item_info():
    """Regression for the warning seen in chat_service.list_sessions:

    ``ChatSessionItemInfo(user_query=extract_user_input(list_content))`` must
    produce a str, otherwise pydantic raises ``string_type`` validation error.
    """
    blocks = [{"type": "text", "text": "anything"}]
    result = extract_user_input(blocks)

    assert isinstance(result, str), f"expected str, got {type(result).__name__}: {result!r}"


# ---------------------------------------------------------------------------
# extract_enhanced_context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "hello world",
        "",
        '{"type": "enhanced", "content": "ctx"}',
        "42",
        "<system_reminder>missing close",
        '[{"type": "enhanced", "content": "legacy"}]',
    ],
    ids=["plain", "empty", "json_object", "numeric", "open_tag_only", "legacy_json_array"],
)
def test_extract_enhanced_context_non_envelope_returns_none(value):
    assert extract_enhanced_context(value) is None


def test_extract_enhanced_context_returns_enhanced_section():
    content = "<system_reminder>Context: relevant info</system_reminder>\nquestion"

    assert extract_enhanced_context(content) == "Context: relevant info"


def test_extract_enhanced_context_preserves_multiline_and_unicode():
    enhanced = "数据库上下文：mysql\n表：用户表, 订单表"
    content = build_structured_content(enhanced, "查一下")

    assert extract_enhanced_context(content) == enhanced


def test_extract_enhanced_context_first_close_tag_terminates_section():
    """If the user portion happens to contain another ``</system_reminder>\\n``,
    the enhanced section ends at the FIRST close tag (str.index)."""
    user = "user text </system_reminder>\nmore"
    content = build_structured_content("real-enhanced", user)

    assert extract_enhanced_context(content) == "real-enhanced"


def test_extract_enhanced_context_empty_enhanced_section():
    content = "<system_reminder></system_reminder>\nuser"

    assert extract_enhanced_context(content) == ""


# ---------------------------------------------------------------------------
# Integration: round-trip through build -> extract
# ---------------------------------------------------------------------------


def test_roundtrip_build_then_extract_both_sides():
    enhanced = "The user is asking a math question."
    user = "What is 2+2?"
    structured = build_structured_content(enhanced, user)

    assert extract_user_input(structured) == user
    assert extract_enhanced_context(structured) == enhanced
    assert is_structured_content(structured) is True
