# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/llms_tools/match_schema.py"""

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.tools.llms_tools.match_schema import MatchSchemaTool, gen_all_table_dict, parse_matched_tables


def _make_pyarrow_table(records):
    """Create a pyarrow table from a list of dicts."""
    if not records:
        return pa.table(
            {
                "identifier": pa.array([], type=pa.string()),
                "catalog_name": pa.array([], type=pa.string()),
                "database_name": pa.array([], type=pa.string()),
                "schema_name": pa.array([], type=pa.string()),
                "table_name": pa.array([], type=pa.string()),
                "definition": pa.array([], type=pa.string()),
                "table_type": pa.array([], type=pa.string()),
                "schema_text": pa.array([], type=pa.string()),
            }
        )
    return pa.Table.from_pylist(records)


def _make_schema_linking_input(**kwargs):
    defaults = dict(
        input_text="Show all users",
        database_type="sqlite",
        database_name="test_db",
        catalog_name="",
        schema_name="",
    )
    defaults.update(kwargs)
    return SchemaLinkingInput(**defaults)


def _make_table_record(table_name="users", schema_name="public", db="test_db"):
    fq = f"{db}.{schema_name}.{table_name}"
    return {
        "identifier": fq,
        "catalog_name": "",
        "database_name": db,
        "schema_name": schema_name,
        "table_name": table_name,
        "definition": f"CREATE TABLE {table_name} (id INT)",
        "table_type": "table",
        "schema_text": f"CREATE TABLE {table_name} (id INT)",
    }


class TestGenAllTableDict:
    def test_empty_table_returns_empty_dict(self):
        empty = _make_pyarrow_table([])
        result = gen_all_table_dict(empty)
        assert result == {}

    def test_single_record(self):
        rec = _make_table_record()
        arrow = _make_pyarrow_table([rec])
        result = gen_all_table_dict(arrow)
        assert "test_db.public.users" in result
        assert result["test_db.public.users"]["table_name"] == "users"

    def test_multiple_records(self):
        records = [
            _make_table_record("users"),
            _make_table_record("orders"),
        ]
        arrow = _make_pyarrow_table(records)
        result = gen_all_table_dict(arrow)
        assert "test_db.public.users" in result
        assert "test_db.public.orders" in result

    def test_batch_processing_large_table(self):
        # 1500 rows to force 2 batches (batch_size=1000)
        records = [_make_table_record(f"t_{i}") for i in range(1500)]
        arrow = _make_pyarrow_table(records)
        result = gen_all_table_dict(arrow)
        assert len(result) == 1500


class TestParseMatchedTables:
    def test_single_table_string(self):
        all_table_dict = {
            "test_db.public.users": {
                "schema_text": "CREATE TABLE users (id INT)",
            }
        }
        tables = {"table": "public.users", "score": 0.9, "reasons": ["relevant"]}
        result = parse_matched_tables("test_db", tables, all_table_dict)
        assert len(result) == 1
        assert result[0]["table_name"] == "users"
        assert result[0]["score"] == 0.9

    def test_list_of_tables(self):
        all_table_dict = {
            "test_db.public.users": {"schema_text": "CREATE TABLE users (id INT)"},
            "test_db.public.orders": {"schema_text": "CREATE TABLE orders (id INT)"},
        }
        tables = {
            "table": ["public.users", "public.orders"],
            "score": 0.8,
            "reasons": ["relevant"],
        }
        result = parse_matched_tables("test_db", tables, all_table_dict)
        assert len(result) == 2

    def test_unknown_table_skipped(self):
        all_table_dict = {}
        tables = {"table": "public.unknown_table", "score": 0.5, "reasons": []}
        result = parse_matched_tables("test_db", tables, all_table_dict)
        assert result == []


class TestMatchSchemaTool:
    def _make_tool(self):
        mock_model = MagicMock()
        mock_storage = MagicMock()
        return MatchSchemaTool(model=mock_model, storage=mock_storage), mock_model, mock_storage

    def test_execute_returns_failure_when_no_metadata(self):
        tool, _, mock_storage = self._make_tool()
        empty_table = _make_pyarrow_table([])
        mock_storage.search_all.return_value = empty_table

        input_data = _make_schema_linking_input()
        result = tool.execute(input_data)

        assert result.success is False
        assert "No table metadata found" in result.error

    def test_execute_success_with_llm_match(self):
        tool, mock_model, mock_storage = self._make_tool()
        rec = _make_table_record()
        arrow = _make_pyarrow_table([rec])
        mock_storage.search_all.return_value = arrow

        match_result = [{"table": "public.users", "score": 0.95, "reasons": ["match"]}]

        with patch.object(tool, "match_schema", return_value=match_result):
            input_data = _make_schema_linking_input()
            result = tool.execute(input_data)

        assert result.success is True
        assert result.schema_count == 1
        assert result.table_schemas[0].table_name == "users"

    def test_execute_exception_raises_datus_exception(self):
        from datus.utils.exceptions import DatusException

        tool, mock_model, mock_storage = self._make_tool()
        rec = _make_table_record()
        arrow = _make_pyarrow_table([rec])
        mock_storage.search_all.return_value = arrow

        with patch.object(tool, "match_schema", side_effect=Exception("boom")):
            with pytest.raises(DatusException):
                input_data = _make_schema_linking_input()
                tool.execute(input_data)

    def test_process_match_result_empty_returns_failure(self):
        tool, _, _ = self._make_tool()
        input_data = _make_schema_linking_input()
        result = tool._process_match_result(input_data, [], "test_db", {})
        assert result.success is False
        assert "No match result found" in result.error

    def test_process_match_result_none_returns_failure(self):
        tool, _, _ = self._make_tool()
        input_data = _make_schema_linking_input()
        result = tool._process_match_result(input_data, None, "test_db", {})
        assert result.success is False

    def test_count_task_size_returns_1_when_within_limit(self):
        tool, mock_model, _ = self._make_tool()
        mock_model.token_count.return_value = 100
        mock_model.max_tokens.return_value = 1000
        size = tool.count_task_size(mock_model, "question", "prompt")
        assert size == 1

    def test_count_task_size_returns_multiple_when_over_limit(self):
        tool, mock_model, _ = self._make_tool()
        mock_model.token_count.return_value = 3000
        mock_model.max_tokens.return_value = 1000
        size = tool.count_task_size(mock_model, "question", "prompt")
        assert size == 3

    def test_match_schema_uses_direct_match_for_small_table(self):
        tool, mock_model, mock_storage = self._make_tool()
        records = [_make_table_record(f"t_{i}") for i in range(5)]
        arrow = _make_pyarrow_table(records)

        mock_model.token_count.return_value = 100
        mock_model.max_tokens.return_value = 1000

        with (
            patch("datus.tools.llms_tools.match_schema.gen_prompt", return_value="prompt"),
            patch("datus.tools.llms_tools.match_schema.llm_result2json", return_value=[]),
        ):
            mock_model.generate.return_value = "[]"
            input_data = _make_schema_linking_input()
            all_tables = gen_all_table_dict(arrow)
            result = tool.match_schema(input_data, arrow, all_tables)

        assert result == []

    def test_match_schema_uses_map_reduce_for_large_table(self):
        tool, mock_model, mock_storage = self._make_tool()
        # 201 rows triggers map-reduce
        records = [_make_table_record(f"t_{i}") for i in range(201)]
        arrow = _make_pyarrow_table(records)

        input_data = _make_schema_linking_input()
        all_tables = gen_all_table_dict(arrow)

        with patch.object(tool, "map_reduce_match_schema", return_value=[]) as mock_mr:
            tool.match_schema(input_data, arrow, all_tables)

        mock_mr.assert_called_once()

    def test_validate_input_returns_true_for_schema_linking_input(self):
        tool, _, _ = self._make_tool()
        input_data = _make_schema_linking_input()
        assert tool.validate_input(input_data) is True

    def test_validate_input_returns_false_for_other(self):
        tool, _, _ = self._make_tool()
        assert tool.validate_input("not valid") is False
