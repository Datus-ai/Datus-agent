# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/models/mcp_result_extractors.py

CI-level: zero external dependencies. RunResultBase is mocked.
"""

from unittest.mock import MagicMock

from datus.models.mcp_result_extractors import extract_sql_contexts, get_function_call_names
from datus.utils.constants import DBType


class TestGetFunctionCallNames:
    def test_snowflake(self):
        funcs = get_function_call_names("snowflake")
        assert "read_query" in funcs
        assert "list_tables" in funcs

    def test_sqlite(self):
        funcs = get_function_call_names(DBType.SQLITE)
        assert "write_query" in funcs

    def test_duckdb(self):
        funcs = get_function_call_names(DBType.DUCKDB)
        assert "query" in funcs

    def test_unknown_returns_empty(self):
        funcs = get_function_call_names("unknown_db")
        assert funcs == set()

    def test_starrocks(self):
        funcs = get_function_call_names("starrocks")
        assert "read_query" in funcs
        assert "db_overview" in funcs


class TestExtractSqlContexts:
    def _make_result(self, items):
        mock_result = MagicMock()
        mock_result.to_input_list.return_value = items
        return mock_result

    def test_no_to_input_list(self):
        """If result has no to_input_list, return empty list."""
        mock_result = MagicMock(spec=[])  # no to_input_list attribute
        result = extract_sql_contexts(mock_result)
        assert result == []

    def test_empty_input_list(self):
        result = extract_sql_contexts(self._make_result([]))
        assert result == []

    def test_extracts_snowflake_query(self):
        items = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "c1",
                "arguments": '{"query": "SELECT 1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "col1\n1\n",
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type="snowflake")
        assert len(result) == 1
        assert "read_query" in result[0].sql_query
        assert result[0].sql_return == "col1\n1\n"

    def test_extracts_with_reflection(self):
        items = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "c2",
                "arguments": '{"query": "SELECT 2"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c2",
                "output": "result_data",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"text": "This query returned 1 row."}],
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type="snowflake")
        assert len(result) == 1
        assert result[0].reflection_explanation == "This query returned 1 row."

    def test_skips_non_matching_function_calls(self):
        items = [
            {
                "type": "function_call",
                "name": "non_query_func",
                "call_id": "c3",
                "arguments": "{}",
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type="snowflake")
        assert result == []

    def test_handles_missing_output(self):
        """Function call with no corresponding output should still produce a context."""
        items = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "c4",
                "arguments": '{"query": "SELECT 3"}',
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type="snowflake")
        assert len(result) == 1
        assert result[0].sql_return is None

    def test_multiple_function_calls(self):
        items = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "c5",
                "arguments": '{"query": "SELECT 5"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c5",
                "output": "data5",
            },
            {
                "type": "function_call",
                "name": "list_tables",
                "call_id": "c6",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c6",
                "output": "tables_data",
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type="snowflake")
        assert len(result) == 2

    def test_duckdb_query_function(self):
        items = [
            {
                "type": "function_call",
                "name": "query",
                "call_id": "c7",
                "arguments": '{"sql": "SELECT 7"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c7",
                "output": "duck_result",
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type=DBType.DUCKDB)
        assert len(result) == 1
        assert result[0].sql_return == "duck_result"

    def test_reflection_stops_at_next_function_call(self):
        """When another function call follows immediately, no reflection is captured."""
        items = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "c8",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c8",
                "output": "data8",
            },
            {
                "type": "function_call",
                "name": "list_tables",
                "call_id": "c9",
                "arguments": "{}",
            },
        ]
        result = extract_sql_contexts(self._make_result(items), db_type="snowflake")
        assert result[0].reflection_explanation is None

    def test_default_db_type_snowflake(self):
        """Default db_type should be snowflake."""
        items = [
            {
                "type": "function_call",
                "name": "read_query",
                "call_id": "c10",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c10",
                "output": "default_result",
            },
        ]
        result = extract_sql_contexts(self._make_result(items))
        assert len(result) == 1
