# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus.gateway.formatters."""

import pytest

from datus.gateway.formatters import ToolOutputFormatter, _format_params, _format_result_default, _truncate
from datus.gateway.models import Verbose

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CALL_PAYLOAD = {
    "callToolId": "t1",
    "toolName": "search_table",
    "toolParams": {"query": "revenue", "database": "analytics"},
}

RESULT_PAYLOAD = {
    "callToolId": "t1",
    "toolName": "search_table",
    "duration": 1.2,
    "shortDesc": "Found tables",
    "result": {"metadata": [{"name": "orders"}], "sample_data": [{"id": 1}, {"id": 2}]},
}


@pytest.fixture
def formatter():
    return ToolOutputFormatter()


# ---------------------------------------------------------------------------
# format_tool_complete — three verbose levels
# ---------------------------------------------------------------------------


class TestFormatToolComplete:
    def test_off_returns_none(self, formatter):
        assert formatter.format_tool_complete(CALL_PAYLOAD, RESULT_PAYLOAD, Verbose.OFF) is None

    def test_on_summary_only(self, formatter):
        result = formatter.format_tool_complete(CALL_PAYLOAD, RESULT_PAYLOAD, Verbose.ON)
        assert "search_table" in result
        assert "1.2s" in result
        # ON mode should NOT include params or result detail
        assert "revenue" not in result
        assert "metadata" not in result

    def test_full_includes_params_and_result(self, formatter):
        result = formatter.format_tool_complete(CALL_PAYLOAD, RESULT_PAYLOAD, Verbose.FULL)
        assert "search_table" in result
        assert "1.2s" in result
        # Params visible ("query" is a SQL key, wrapped in code block)
        assert "revenue" in result
        assert "database: analytics" in result
        # Result visible (search_table special formatter)
        assert "metadata: 1" in result
        assert "sample rows: 2" in result

    def test_empty_params(self, formatter):
        call = {"callToolId": "t1", "toolName": "foo", "toolParams": {}}
        res = {"callToolId": "t1", "toolName": "foo", "duration": 0.5, "result": None}
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "foo" in result
        assert "0.5s" in result

    def test_empty_result(self, formatter):
        call = {"callToolId": "t1", "toolName": "bar", "toolParams": {"x": 1}}
        res = {"callToolId": "t1", "toolName": "bar", "duration": 0.1, "result": None}
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "bar" in result
        assert "x: 1" in result

    def test_missing_call_payload_fields(self, formatter):
        result = formatter.format_tool_complete({}, RESULT_PAYLOAD, Verbose.ON)
        assert "search_table" in result

    def test_tool_name_falls_back_to_call(self, formatter):
        call = {"callToolId": "t1", "toolName": "my_tool", "toolParams": {}}
        res = {"callToolId": "t1", "duration": 0.3, "result": {}}
        result = formatter.format_tool_complete(call, res, Verbose.ON)
        assert "my_tool" in result


# ---------------------------------------------------------------------------
# Registered tool formatters (full mode)
# ---------------------------------------------------------------------------


class TestReadQueryFormatter:
    def test_with_rows_and_columns(self, formatter):
        call = {"callToolId": "t1", "toolName": "read_query", "toolParams": {"sql": "SELECT 1"}}
        res = {
            "callToolId": "t1",
            "toolName": "read_query",
            "duration": 0.8,
            "result": {"columns": ["id", "name"], "rows": [{"id": 1}, {"id": 2}, {"id": 3}]},
        }
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "columns: 2" in result
        assert "rows: 3" in result
        assert "preview:" in result

    def test_empty_rows(self, formatter):
        call = {"callToolId": "t1", "toolName": "read_query", "toolParams": {}}
        res = {"callToolId": "t1", "toolName": "read_query", "duration": 0.1, "result": {"columns": [], "rows": []}}
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "rows: 0" in result

    def test_non_dict_result(self, formatter):
        call = {"callToolId": "t1", "toolName": "read_query", "toolParams": {}}
        res = {"callToolId": "t1", "toolName": "read_query", "duration": 0.1, "result": "raw string"}
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "raw string" in result


class TestDescribeTableFormatter:
    def test_with_columns(self, formatter):
        call = {"callToolId": "t1", "toolName": "describe_table", "toolParams": {}}
        res = {
            "callToolId": "t1",
            "toolName": "describe_table",
            "duration": 0.3,
            "result": {"columns": ["id", "name", "age"]},
        }
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "columns: 3" in result

    def test_non_dict_result(self, formatter):
        call = {"callToolId": "t1", "toolName": "describe_table", "toolParams": {}}
        res = {"callToolId": "t1", "toolName": "describe_table", "duration": 0.1, "result": [1, 2]}
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "[2 items]" in result


class TestSearchTableFormatter:
    def test_with_metadata_and_samples(self, formatter):
        call = {"callToolId": "t1", "toolName": "search_table", "toolParams": {}}
        res = {
            "callToolId": "t1",
            "toolName": "search_table",
            "duration": 1.0,
            "result": {"metadata": [1, 2, 3], "sample_data": [1, 2]},
        }
        result = formatter.format_tool_complete(call, res, Verbose.FULL)
        assert "metadata: 3" in result
        assert "sample rows: 2" in result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestFormatParams:
    def test_basic_params(self):
        result = _format_params({"key": "value", "count": 5})
        assert "> key: value" in result
        assert "> count: 5" in result

    def test_sql_key_wrapped_in_code_block(self):
        result = _format_params({"sql": "SELECT 1"})
        assert "```" in result
        assert "SELECT 1" in result

    def test_query_key_wrapped_in_code_block(self):
        result = _format_params({"query": "SELECT * FROM t"})
        assert "```" in result

    def test_empty_params(self):
        assert _format_params({}) == ""


class TestFormatResultDefault:
    def test_dict_result(self):
        result = _format_result_default({"key1": "val1", "key2": [1, 2, 3]})
        assert "> key1: val1" in result
        assert "> key2: [3 items]" in result

    def test_list_result(self):
        result = _format_result_default([1, 2, 3, 4])
        assert "[4 items]" in result

    def test_string_result(self):
        result = _format_result_default("hello")
        assert result == "hello"

    def test_nested_dict(self):
        result = _format_result_default({"outer": {"inner": "val"}})
        assert "> outer:" in result

    def test_empty_dict(self):
        assert _format_result_default({}) == ""


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("abc", 3) == "abc"

    def test_long_text_truncated(self):
        result = _truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_very_long_value_in_formatter(self, formatter):
        long_val = "x" * 3000
        call = {"callToolId": "t1", "toolName": "generic", "toolParams": {"data": long_val}}
        res = {"callToolId": "t1", "toolName": "generic", "duration": 0.1, "result": {"big": long_val}}
        result = formatter.format_tool_complete(call, res, Verbose.FULL)

        quoted_lines = [line for line in result.split("\n") if line.startswith("> ")]
        assert quoted_lines, "formatter should emit at least one quoted value line"
        assert all(len(line) < 2200 for line in quoted_lines)
