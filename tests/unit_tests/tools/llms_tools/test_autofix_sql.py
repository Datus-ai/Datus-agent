# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/llms_tools/autofix_sql.py"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.fix_node_models import FixInput, FixResult
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema
from datus.tools.llms_tools.autofix_sql import autofix_sql


def _make_fix_input(
    task_str="Show all users",
    sql_query="SELECT * FROM users",
    explanation="select all",
    sql_return="id,name\n1,Alice",
    sql_error="",
    prompt_version=None,
):
    sql_task = SqlTask(task=task_str, database_type="sqlite", database_name="test_db")
    sql_context = SQLContext(
        sql_query=sql_query,
        explanation=explanation,
        sql_return=sql_return,
        sql_error=sql_error,
    )
    return FixInput(
        sql_task=sql_task,
        sql_context=sql_context,
        schemas=[],
        prompt_version=prompt_version,
    )


def _make_mock_model(return_value):
    mock_model = MagicMock()
    mock_model.generate_with_json_output.return_value = return_value
    return mock_model


class TestAutofixSqlValidation:
    def test_raises_value_error_for_non_fix_input(self):
        mock_model = _make_mock_model({})
        with pytest.raises(ValueError, match="Input must be a FixInput instance"):
            autofix_sql(mock_model, "not a FixInput", [])

    def test_raises_value_error_for_dict_input(self):
        mock_model = _make_mock_model({})
        with pytest.raises(ValueError):
            autofix_sql(mock_model, {"sql_task": "foo"}, [])


class TestAutofixSqlSuccess:
    def test_returns_fix_result_with_dict_response(self):
        mock_model = _make_mock_model({"sql": "SELECT id FROM users", "explanation": "fixed"})
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert isinstance(result, FixResult)
        assert result.success is True
        assert result.sql_query == "SELECT id FROM users"
        assert result.explanation == "fixed"

    def test_returns_fix_result_with_json_string_response(self):
        json_str = json.dumps({"sql": "SELECT name FROM users", "explanation": "corrected"})
        mock_model = _make_mock_model(json_str)
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is True
        assert result.sql_query == "SELECT name FROM users"
        assert result.explanation == "corrected"

    def test_strips_markdown_code_blocks(self):
        json_str = "```json\n" + json.dumps({"sql": "SELECT 1", "explanation": "test"}) + "\n```"
        mock_model = _make_mock_model(json_str)
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is True
        assert result.sql_query == "SELECT 1"

    def test_removes_sql_comment_lines(self):
        json_body = json.dumps({"sql": "SELECT 1", "explanation": "commented"})
        json_str = "-- This is a comment\n" + json_body
        mock_model = _make_mock_model(json_str)
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is True
        assert result.sql_query == "SELECT 1"

    def test_docs_passed_to_prompt(self):
        mock_model = _make_mock_model({"sql": "SELECT 1", "explanation": "ok"})
        input_data = _make_fix_input()
        docs = ["doc1", "doc2"]

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="prompt") as mock_prompt:
            autofix_sql(mock_model, input_data, docs)

        call_kwargs = mock_prompt.call_args[1]
        assert call_kwargs["docs"] == docs

    def test_schemas_passed_to_prompt(self):
        mock_model = _make_mock_model({"sql": "SELECT 1", "explanation": "ok"})
        schema = TableSchema(
            table_name="users", database_name="test_db", schema_name="public", definition="CREATE TABLE users (id INT)"
        )
        input_data = _make_fix_input()
        input_data = FixInput(
            sql_task=input_data.sql_task,
            sql_context=input_data.sql_context,
            schemas=[schema],
        )

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="prompt") as mock_prompt:
            autofix_sql(mock_model, input_data, [])

        call_kwargs = mock_prompt.call_args[1]
        assert call_kwargs["schemas"] == [schema]


class TestAutofixSqlFailures:
    def test_invalid_json_string_returns_failure(self):
        # When JSON parsing fails, the production code falls into the except Exception
        # block (line 75-77) which returns success=False with the exception message.
        mock_model = _make_mock_model("not valid json at all")
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is False
        # The exception message comes from the validation error or JSON decode error
        assert result.error is not None

    def test_empty_dict_response_returns_failure(self):
        mock_model = _make_mock_model({})
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is False

    def test_none_dict_response_returns_failure(self):
        mock_model = _make_mock_model(None)
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is False

    def test_exception_in_model_returns_failure(self):
        mock_model = MagicMock()
        mock_model.generate_with_json_output.side_effect = Exception("model error")
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is False
        assert "model error" in result.error

    def test_missing_sql_key_returns_empty_sql(self):
        mock_model = _make_mock_model({"explanation": "only explanation, no sql"})
        input_data = _make_fix_input()

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="mock_prompt"):
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is True
        assert result.sql_query == ""
        assert result.explanation == "only explanation, no sql"

    def test_sql_error_context_is_included(self):
        mock_model = _make_mock_model({"sql": "SELECT 1", "explanation": "fixed error"})
        input_data = _make_fix_input(sql_error="column not found")

        with patch("datus.tools.llms_tools.autofix_sql.fix_sql_prompt", return_value="prompt") as mock_prompt:
            result = autofix_sql(mock_model, input_data, [])

        assert result.success is True
        call_kwargs = mock_prompt.call_args[1]
        assert "column not found" in call_kwargs["sql_context"]
