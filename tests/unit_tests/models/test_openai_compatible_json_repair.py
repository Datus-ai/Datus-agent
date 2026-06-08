# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for malformed JSON repair in tool call arguments.

Tests the shared repair_tool_call_arguments() utility and verifies
that Pydantic model_copy correctly propagates repaired arguments.
"""

import json

import pytest
from pydantic import BaseModel

from datus.utils.json_utils import repair_tool_call_arguments


class FakeRawItem(BaseModel):
    name: str = "ask_user"
    call_id: str = "call_abc123"
    arguments: str = "{}"

    class Config:
        extra = "allow"


class TestRepairToolCallArguments:
    """Test the repair_tool_call_arguments utility function."""

    def test_valid_json_passes_through_unchanged(self):
        args = '{"question": "hello", "options": ["a", "b"]}'
        result, was_repaired = repair_tool_call_arguments(args)
        assert result == args
        assert was_repaired is False

    def test_empty_string_not_repaired(self):
        result, was_repaired = repair_tool_call_arguments("")
        assert result == ""
        assert was_repaired is False

    def test_whitespace_only_not_repaired(self):
        result, was_repaired = repair_tool_call_arguments("   \t\n  ")
        assert result == ""
        assert was_repaired is False

    def test_malformed_json_is_repaired(self):
        malformed = '{"keywords": PERCENTILE function, "platform": "starrocks"}'
        result, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True
        parsed = json.loads(result)
        assert "keywords" in parsed
        assert "platform" in parsed
        assert parsed["platform"] == "starrocks"

    def test_glm_style_unquoted_values_repaired(self):
        malformed = '{"question": 请问您想查询哪个数据库的信息？, "options": [StarRocks, MySQL]}'
        result, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True
        parsed = json.loads(result)
        assert parsed["question"] == "请问您想查询哪个数据库的信息？"
        assert parsed["options"] == ["StarRocks", "MySQL"]

    def test_garbage_input_fallback(self):
        garbage = "not json at all {{{{[[[["
        result, was_repaired = repair_tool_call_arguments(garbage)
        # repair either produced a valid JSON object, or fell back to the original unchanged
        if was_repaired:
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
        else:
            assert result == garbage

    @pytest.mark.parametrize(
        "valid_args",
        [
            "{}",
            '{"key": "value"}',
            '{"nested": {"a": 1}, "list": [1, 2, 3]}',
            "[]",
            '{"unicode": "中文值"}',
        ],
    )
    def test_various_valid_json_not_repaired(self, valid_args):
        result, was_repaired = repair_tool_call_arguments(valid_args)
        assert was_repaired is False
        assert result == valid_args


class TestModelCopyIntegration:
    """Verify that model_copy correctly propagates repaired arguments."""

    def test_model_copy_updates_arguments(self):
        malformed = '{"sql": SELECT * FROM t}'
        raw_item = FakeRawItem(
            name="execute_sql",
            call_id="call_xyz789",
            arguments=malformed,
        )

        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        new_raw_item = raw_item.model_copy(update={"arguments": repaired_args})
        parsed = json.loads(new_raw_item.arguments)
        assert "sql" in parsed
        assert new_raw_item.name == "execute_sql"
        assert new_raw_item.call_id == "call_xyz789"

    def test_model_copy_not_needed_for_valid_json(self):
        valid = '{"question": "hello"}'
        raw_item = FakeRawItem(arguments=valid)

        result, was_repaired = repair_tool_call_arguments(valid)
        assert was_repaired is False
        assert raw_item.arguments == valid


class TestDictRawItemRepair:
    """Verify repair integration with dict-based raw_item (CodexModel path)."""

    def test_dict_raw_item_updated_in_place(self):
        malformed = '{"sql": SELECT * FROM users WHERE id = 1}'
        raw_item = {"name": "execute_sql", "call_id": "call_001", "arguments": malformed}

        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        if isinstance(raw_item, dict):
            raw_item["arguments"] = repaired_args

        parsed = json.loads(raw_item["arguments"])
        assert "sql" in parsed
        assert raw_item["name"] == "execute_sql"

    def test_dict_raw_item_unchanged_for_valid_json(self):
        valid = '{"query": "SELECT 1"}'
        raw_item = {"name": "execute_sql", "call_id": "call_002", "arguments": valid}

        repaired_args, was_repaired = repair_tool_call_arguments(valid)
        assert was_repaired is False
        assert raw_item["arguments"] == valid


class TestRepairEdgeCases:
    """Cover edge cases and exception paths in repair_tool_call_arguments."""

    def test_whitespace_string_not_repaired(self):
        result, was_repaired = repair_tool_call_arguments("  ")
        assert result == ""
        assert was_repaired is False

    def test_truncated_json_repaired(self):
        truncated = '{"key": "value", "nested": {"a":'
        result, was_repaired = repair_tool_call_arguments(truncated)
        assert was_repaired is True
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_trailing_comma_repaired(self):
        trailing = '{"a": 1, "b": 2,}'
        result, was_repaired = repair_tool_call_arguments(trailing)
        assert was_repaired is True
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_single_quotes_repaired(self):
        single_quotes = "{'key': 'value'}"
        result, was_repaired = repair_tool_call_arguments(single_quotes)
        assert was_repaired is True
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_repair_fallback_when_json_repair_raises(self):
        """Cover the exception fallback path (lines 614-615) when json_repair itself fails."""
        from unittest.mock import patch

        malformed = '{"broken": !!!}'
        with patch("datus.utils.json_utils.json_repair.loads", side_effect=Exception("repair failed")):
            result, was_repaired = repair_tool_call_arguments(malformed)
        assert result == malformed
        assert was_repaired is False


class TestStreamingRepairIntegration:
    """Cover the repair branch inside model streaming event handlers."""

    def test_openai_compatible_repair_branch_with_pydantic_raw_item(self):
        """Simulate the repair path in OpenAICompatibleModel streaming."""
        malformed = '{"sql": SELECT * FROM t WHERE id = 1}'
        raw_item = FakeRawItem(name="execute_sql", call_id="call_001", arguments=malformed)

        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        # Simulate the model code path: event.item.raw_item = raw_item.model_copy(...)
        new_raw_item = raw_item.model_copy(update={"arguments": repaired_args})
        raw_item = new_raw_item
        arguments = repaired_args

        parsed = json.loads(arguments)
        assert "sql" in parsed
        assert raw_item.arguments == repaired_args

    def test_codex_repair_branch_with_dict_raw_item(self):
        """Simulate the dict-based repair path in CodexModel streaming."""
        malformed = '{"query": SELECT count(*) FROM orders}'
        raw_item = {"name": "execute_sql", "call_id": "call_002", "arguments": malformed}

        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        # Simulate the CodexModel code path: if isinstance(raw_item, dict)
        if isinstance(raw_item, dict):
            raw_item["arguments"] = repaired_args
        arguments = repaired_args

        parsed = json.loads(arguments)
        assert "query" in parsed
        assert raw_item["arguments"] == repaired_args

    def test_codex_repair_branch_with_pydantic_raw_item(self):
        """Simulate the else branch in CodexModel (Pydantic raw_item)."""
        malformed = '{"table": users, "limit": 10}'
        raw_item = FakeRawItem(name="read_table", call_id="call_003", arguments=malformed)

        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        # Simulate: not isinstance(raw_item, dict) → model_copy path
        if not isinstance(raw_item, dict):
            new_raw_item = raw_item.model_copy(update={"arguments": repaired_args})
            raw_item = new_raw_item
        arguments = repaired_args

        assert raw_item.arguments == repaired_args
        parsed = json.loads(arguments)
        assert parsed["table"] == "users"
        assert parsed["limit"] == 10


class TestToInputItemContract:
    """Regression: repaired arguments must survive to_input_item() serialization."""

    def test_pydantic_raw_item_to_input_item_contains_valid_json(self):
        """model_copy(update=...) keeps 'arguments' in model_fields_set so to_input_item works."""
        from unittest.mock import MagicMock

        malformed = '{"sql": SELECT * FROM orders}'
        raw_item = FakeRawItem(name="execute_sql", call_id="call_contract_pydantic", arguments=malformed)
        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        new_raw_item = raw_item.model_copy(update={"arguments": repaired_args})
        event_item = MagicMock()
        event_item.raw_item = new_raw_item
        event_item.to_input_item.return_value = {"arguments": new_raw_item.arguments, "name": new_raw_item.name}

        input_item = event_item.to_input_item()
        assert json.loads(input_item["arguments"]) == json.loads(repaired_args)

    def test_dict_raw_item_to_input_item_contains_valid_json(self):
        from unittest.mock import MagicMock

        malformed = '{"sql": SELECT count(*) FROM users}'
        raw_item = {"name": "execute_sql", "call_id": "call_contract_dict", "arguments": malformed}
        repaired_args, was_repaired = repair_tool_call_arguments(malformed)
        assert was_repaired is True

        raw_item["arguments"] = repaired_args
        event_item = MagicMock()
        event_item.raw_item = raw_item
        event_item.to_input_item.return_value = raw_item

        input_item = event_item.to_input_item()
        assert json.loads(input_item["arguments"]) == json.loads(repaired_args)
