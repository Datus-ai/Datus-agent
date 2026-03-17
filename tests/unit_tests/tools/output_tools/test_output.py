# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/output_tools/output.py"""

import json
import os
from unittest.mock import MagicMock, patch

from datus.schemas.node_models import OutputInput
from datus.tools.output_tools.output import OutputTool, save_csv, save_json, save_sql


def _make_output_input(tmp_dir, finished=True, error=None, file_type="all", check_result=False, sql_result="id\n1"):
    return OutputInput(
        finished=finished,
        error=error,
        task_id="task_001",
        task="Show all users",
        database_name="test_db",
        output_dir=str(tmp_dir),
        gen_sql="SELECT * FROM users",
        sql_result=sql_result,
        row_count=1,
        table_schemas=[],
        metrics=[],
        file_type=file_type,
        check_result=check_result,
    )


class TestSaveSql:
    def test_creates_sql_file(self, tmp_path):
        result = save_sql(str(tmp_path), "test_task", "SELECT 1")
        assert result.endswith(".sql")
        assert os.path.exists(result)
        with open(result) as f:
            assert f.read() == "SELECT 1"


class TestSaveCsv:
    def test_creates_csv_file(self, tmp_path):
        result = save_csv(str(tmp_path), "test_task", "id,name\n1,Alice")
        assert result.endswith(".csv")
        assert os.path.exists(result)
        with open(result) as f:
            assert "Alice" in f.read()


class TestSaveJson:
    def test_creates_json_file(self, tmp_path):
        input_data = _make_output_input(tmp_path)
        result = save_json(str(tmp_path), input_data, "SELECT * FROM users", "id\n1")
        assert result.endswith(".json")
        assert os.path.exists(result)
        with open(result) as f:
            data = json.load(f)
        assert data["finished"] is True
        assert data["database_name"] == "test_db"

    def test_includes_gen_sql_final_when_different(self, tmp_path):
        input_data = _make_output_input(tmp_path)
        save_json(str(tmp_path), input_data, "SELECT id FROM users", "id\n1")
        with open(os.path.join(str(tmp_path), "task_001.json")) as f:
            data = json.load(f)
        assert "gen_sql_final" in data

    def test_no_gen_sql_final_when_same(self, tmp_path):
        input_data = _make_output_input(tmp_path)
        save_json(str(tmp_path), input_data, "SELECT * FROM users", "id\n1")
        with open(os.path.join(str(tmp_path), "task_001.json")) as f:
            data = json.load(f)
        assert "gen_sql_final" not in data

    def test_includes_result_file_name_when_provided(self, tmp_path):
        input_data = _make_output_input(tmp_path)
        save_json(str(tmp_path), input_data, "SELECT * FROM users", "id\n1", result_file_name="result.csv")
        with open(os.path.join(str(tmp_path), "task_001.json")) as f:
            data = json.load(f)
        assert data["result"] == "result.csv"

    def test_includes_sql_result_final_when_different(self, tmp_path):
        input_data = _make_output_input(tmp_path, sql_result="id\n1")
        save_json(str(tmp_path), input_data, "SELECT * FROM users", "id\n2")
        with open(os.path.join(str(tmp_path), "task_001.json")) as f:
            data = json.load(f)
        assert "sql_result_final" in data


class TestOutputToolExecute:
    def test_execute_finished_all_file_type(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, file_type="all")
        mock_connector = MagicMock()

        with patch.object(tool, "check_sql", return_value=("SELECT * FROM users", "id\n1")):
            result = tool.execute(input_data, mock_connector)

        assert result.success is True
        assert result.output is not None

    def test_execute_finished_sql_file_type(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, file_type="sql")
        mock_connector = MagicMock()

        with patch.object(tool, "check_sql", return_value=("SELECT * FROM users", "id\n1")):
            result = tool.execute(input_data, mock_connector)

        assert result.success is True
        assert result.output.endswith(".sql")

    def test_execute_finished_csv_file_type(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, file_type="csv")
        mock_connector = MagicMock()

        with patch.object(tool, "check_sql", return_value=("SELECT * FROM users", "id\n1")):
            result = tool.execute(input_data, mock_connector)

        assert result.success is True
        assert result.output.endswith(".csv")

    def test_execute_finished_json_file_type(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, file_type="json")
        mock_connector = MagicMock()

        with patch.object(tool, "check_sql", return_value=("SELECT * FROM users", "id\n1")):
            result = tool.execute(input_data, mock_connector)

        assert result.success is True
        assert result.output.endswith(".json")

    def test_execute_not_finished_writes_error_json(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, finished=False, error="SQL failed")
        mock_connector = MagicMock()

        result = tool.execute(input_data, mock_connector)

        assert result.success is False
        assert result.output == "SQL failed"
        error_file = os.path.join(str(tmp_path), "task_001.json")
        assert os.path.exists(error_file)
        with open(error_file) as f:
            data = json.load(f)
        assert data["finished"] is False

    def test_execute_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "subdir" / "output"
        tool = OutputTool()
        input_data = _make_output_input(new_dir, file_type="sql")
        mock_connector = MagicMock()

        with patch.object(tool, "check_sql", return_value=("SELECT 1", "1")):
            tool.execute(input_data, mock_connector)

        assert new_dir.exists()

    def test_check_sql_returns_original_when_check_result_false(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, check_result=False)
        mock_connector = MagicMock()

        sql, result = tool.check_sql(input_data, mock_connector)
        assert sql == input_data.gen_sql
        assert result == input_data.sql_result

    def test_check_sql_returns_original_when_no_model(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, check_result=True)
        mock_connector = MagicMock()

        sql, result = tool.check_sql(input_data, mock_connector, model=None)
        assert sql == input_data.gen_sql

    def test_check_sql_returns_original_when_model_says_correct(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, check_result=True)
        mock_connector = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {"is_correct": True}

        with patch("datus.tools.output_tools.output.gen_prompt", return_value="prompt"):
            sql, result = tool.check_sql(input_data, mock_connector, model=mock_model)

        assert sql == input_data.gen_sql
        assert result == input_data.sql_result

    def test_check_sql_uses_final_columns_when_provided(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, check_result=True, sql_result="id,name\n1,Alice\n2,Bob")
        mock_connector = MagicMock()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "is_correct": False,
            "revised_sql": "SELECT id FROM users",
            "final_columns": ["id"],
        }

        with patch("datus.tools.output_tools.output.gen_prompt", return_value="prompt"):
            sql, result = tool.check_sql(input_data, mock_connector, model=mock_model)

        assert sql == "SELECT id FROM users"
        # result should only have 'id' column
        assert "name" not in result

    def test_check_sql_executes_sql_when_no_current_result(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, check_result=True, sql_result=None)
        mock_connector = MagicMock()
        execute_result = MagicMock()
        execute_result.success = True
        execute_result.sql_return = "id\n1"
        mock_connector.execute.return_value = execute_result
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "is_correct": False,
            "revised_sql": "SELECT id FROM users",
        }

        with patch("datus.tools.output_tools.output.gen_prompt", return_value="prompt"):
            sql, result = tool.check_sql(input_data, mock_connector, model=mock_model)

        assert sql == "SELECT id FROM users"
        assert result == "id\n1"

    def test_check_sql_falls_back_on_execute_failure(self, tmp_path):
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, check_result=True, sql_result=None)
        mock_connector = MagicMock()
        execute_result = MagicMock()
        execute_result.success = False
        execute_result.error = "syntax error"
        mock_connector.execute.return_value = execute_result
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "is_correct": False,
            "revised_sql": "BAD SQL",
        }

        with patch("datus.tools.output_tools.output.gen_prompt", return_value="prompt"):
            sql, result = tool.check_sql(input_data, mock_connector, model=mock_model)

        # Falls back to original
        assert sql == input_data.gen_sql

    def test_execute_check_sql_none_result_falls_back_to_original(self, tmp_path):
        """When check_sql returns (non_null_sql, None), fallback to original input data."""
        tool = OutputTool()
        input_data = _make_output_input(tmp_path, file_type="sql")
        mock_connector = MagicMock()

        # Return a sql but no result - should fall back to input_data values
        with patch.object(tool, "check_sql", return_value=("SELECT 1", None)):
            result = tool.execute(input_data, mock_connector)

        assert result.success is True
