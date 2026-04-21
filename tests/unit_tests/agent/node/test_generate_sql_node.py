# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for GenerateSQLNode - zero external dependencies."""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import GenerateSQLInput, GenerateSQLResult, SqlTask, TableSchema, TableValue


def make_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_datasource = "test"
    cfg.nodes = {}
    return cfg


def make_sql_task(task="Count rows", db="mydb"):
    return SqlTask(task=task, database_type="sqlite", database_name=db)


def make_generate_sql_input():
    return GenerateSQLInput(
        sql_task=make_sql_task(),
        table_schemas=[
            TableSchema(
                table_name="orders",
                database_name="mydb",
                definition="CREATE TABLE orders (id INT, total FLOAT)",
            )
        ],
        data_details=[],
        metrics=[],
        contexts=[],
        external_knowledge="",
    )


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "gen_sql_1",
        "Generate SQL",
        NodeType.TYPE_GENERATE_SQL,
        input_data=input_data,
        agent_config=cfg,
    )


class TestGenerateSQLNodeExecute:
    """Test _execute_generate_sql() / execute()."""

    def test_execute_no_model_returns_failure(self):
        node = make_node(make_generate_sql_input())
        node.model = None
        node.execute()
        assert node.result.success is False
        assert "model not provided" in node.result.error.lower()

    def test_execute_calls_generate_sql_function(self):
        node = make_node(make_generate_sql_input())
        mock_model = MagicMock()
        node.model = mock_model

        expected = GenerateSQLResult(success=True, sql_query="SELECT COUNT(*) FROM orders", tables=["orders"])
        with patch("datus.agent.node.generate_sql_node.generate_sql", return_value=expected):
            node.execute()

        assert node.result.success is True
        assert node.result.sql_query == "SELECT COUNT(*) FROM orders"

    def test_execute_handles_exception(self):
        node = make_node(make_generate_sql_input())
        node.model = MagicMock()

        with patch(
            "datus.agent.node.generate_sql_node.generate_sql",
            side_effect=RuntimeError("llm error"),
        ):
            node.execute()

        assert node.result.success is False
        assert "llm error" in node.result.error


class TestGenerateSQLFunction:
    """Test the module-level generate_sql() function."""

    def test_generate_sql_valid_json_response(self):
        from datus.agent.node.generate_sql_node import generate_sql

        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "sql": "SELECT * FROM t",
            "tables": ["t"],
            "explanation": "all rows",
        }
        input_data = make_generate_sql_input()
        result = generate_sql(mock_model, input_data)
        assert result.success is True
        assert result.sql_query == "SELECT * FROM t"
        assert result.tables == ["t"]

    def test_generate_sql_string_json_response(self):
        from datus.agent.node.generate_sql_node import generate_sql

        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = json.dumps(
            {"sql": "SELECT 1", "tables": [], "explanation": "test"}
        )
        input_data = make_generate_sql_input()
        result = generate_sql(mock_model, input_data)
        assert result.success is True
        assert result.sql_query == "SELECT 1"

    def test_generate_sql_invalid_json(self):
        from datus.agent.node.generate_sql_node import generate_sql

        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = "not json at all {{"
        input_data = make_generate_sql_input()
        result = generate_sql(mock_model, input_data)
        assert result.success is False

    def test_generate_sql_empty_dict_response(self):
        from datus.agent.node.generate_sql_node import generate_sql

        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {}
        input_data = make_generate_sql_input()
        result = generate_sql(mock_model, input_data)
        assert result.success is False

    def test_generate_sql_wrong_input_type(self):
        from datus.agent.node.generate_sql_node import generate_sql

        mock_model = MagicMock()
        with pytest.raises(TypeError):
            generate_sql(mock_model, {"not": "a GenerateSQLInput"})

    def test_generate_sql_model_raises(self):
        from datus.agent.node.generate_sql_node import generate_sql

        mock_model = MagicMock()
        mock_model.generate_with_json_output.side_effect = Exception("network error")
        input_data = make_generate_sql_input()
        result = generate_sql(mock_model, input_data)
        assert result.success is False
        assert "network error" in result.error


class TestGenerateSQLNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_builds_generate_sql_input(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task = make_sql_task()
        workflow.context.document_result = None
        workflow.context.table_schemas = []
        workflow.context.table_values = []
        workflow.context.metrics = []
        workflow.context.sql_contexts = []

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, GenerateSQLInput)

    def test_setup_input_includes_doc_result(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task = make_sql_task()
        workflow.context.document_result.docs = {"ns": ["doc content"]}
        workflow.context.table_schemas = []
        workflow.context.table_values = []
        workflow.context.metrics = []
        workflow.context.sql_contexts = []

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert "doc content" in node.input.database_docs


class TestGenerateSQLNodeUpdateContext:
    """Test update_context."""

    def test_update_context_appends_sql_context(self):
        node = make_node(make_generate_sql_input())
        node.result = GenerateSQLResult(success=True, sql_query="SELECT 1", tables=["orders"], explanation="test")
        workflow = MagicMock()
        workflow.task = make_sql_task()
        workflow.context.sql_contexts = []

        mock_connector = MagicMock()
        mock_connector.catalog_name = ""
        mock_connector.database_name = "mydb"
        mock_connector.schema_name = ""

        table_schema = TableSchema(table_name="orders", database_name="mydb", definition="CREATE TABLE orders (id INT)")
        table_value = TableValue(table_name="orders", database_name="mydb", table_values="1,2")

        with patch.object(node, "_sql_connector", return_value=mock_connector):
            with patch.object(node, "_get_schema_and_values", return_value=([table_schema], [table_value])):
                result = node.update_context(workflow)

        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1

    def test_update_context_schema_count_mismatch(self):
        node = make_node(make_generate_sql_input())
        node.result = GenerateSQLResult(
            success=True, sql_query="SELECT 1", tables=["orders", "users"], explanation="test"
        )
        workflow = MagicMock()
        workflow.task = make_sql_task()
        workflow.context.sql_contexts = []

        with patch.object(node, "_get_schema_and_values", return_value=([], [])):
            result = node.update_context(workflow)

        # Should still succeed but with warning
        assert result["success"] is True

    def test_update_context_exception(self):
        node = make_node(make_generate_sql_input())
        node.result = GenerateSQLResult(success=True, sql_query="SELECT 1", tables=[], explanation="test")
        workflow = MagicMock()
        workflow.task = make_sql_task()
        workflow.context.sql_contexts = MagicMock()
        workflow.context.sql_contexts.append.side_effect = RuntimeError("ctx error")

        result = node.update_context(workflow)
        assert result["success"] is False


class TestGenerateSQLNodeStream:
    """Test execute_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_model_returns_empty(self):
        node = make_node(make_generate_sql_input())
        node.model = None
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        node = make_node(make_generate_sql_input())
        node.model = MagicMock()
        mock_result = GenerateSQLResult(success=True, sql_query="SELECT 1", tables=[])

        with patch("datus.agent.node.generate_sql_node.generate_sql", return_value=mock_result):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert len(actions) >= 1
        assert node.result.sql_query == "SELECT 1"
