# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ExecuteSQLNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult, SQLContext


def make_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_database = "test"
    cfg.nodes = {}
    return cfg


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "exec_sql_1",
        "Execute SQL",
        NodeType.TYPE_EXECUTE_SQL,
        input_data=input_data,
        agent_config=cfg,
    )


class TestExecuteSQLNodeExecute:
    """Test ExecuteSQLNode.execute() / _execute_sql()."""

    def test_execute_success(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="testdb"))
        mock_connector = MagicMock()
        mock_result = ExecuteSQLResult(success=True, sql_query="SELECT 1", row_count=1, sql_return="1")
        mock_connector.execute.return_value = mock_result

        with patch.object(node, "_sql_connector", return_value=mock_connector):
            node.execute()

        assert node.result.success is True
        assert node.result.row_count == 1

    def test_execute_no_connector(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="db"))
        with patch.object(node, "_sql_connector", return_value=None):
            node.execute()

        assert node.result.success is False
        assert "not initialized" in node.result.error.lower()

    def test_execute_connector_raises_exception(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT bad", database_name="db"))
        with patch.object(node, "_sql_connector", side_effect=RuntimeError("db error")):
            node.execute()

        assert node.result.success is False
        assert "db error" in node.result.error

    def test_execute_validation_error(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="db"))
        mock_connector = MagicMock()
        mock_connector.execute.side_effect = Exception("validation-like error")
        with patch.object(node, "_sql_connector", return_value=mock_connector):
            node.execute()

        assert node.result.success is False


class TestExecuteSQLNodeSetupInput:
    """Test setup_input."""

    def test_setup_input(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task.database_name = "mydb"
        workflow.get_last_sqlcontext.return_value = SQLContext(sql_query="SELECT 1")

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, ExecuteSQLInput)
        assert node.input.sql_query == "SELECT 1"
        assert node.input.database_name == "mydb"

    def test_setup_input_strips_markdown(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task.database_name = "db"
        workflow.get_last_sqlcontext.return_value = SQLContext(sql_query="```sql\nSELECT 1\n```")

        node.setup_input(workflow)
        assert node.input.sql_query == "SELECT 1"


class TestExecuteSQLNodeUpdateContext:
    """Test update_context."""

    def test_update_context_success(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="db"))
        node.result = ExecuteSQLResult(success=True, sql_query="SELECT 1", row_count=5, sql_return="data")
        workflow = MagicMock()
        sql_ctx = SQLContext(sql_query="SELECT 1")
        workflow.context.sql_contexts = [sql_ctx]

        result = node.update_context(workflow)
        assert result["success"] is True
        assert sql_ctx.row_count == 5
        assert sql_ctx.sql_return == "data"

    def test_update_context_no_sql_contexts(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="db"))
        node.result = ExecuteSQLResult(success=True, sql_query="SELECT 1", row_count=1, sql_return="data")
        workflow = MagicMock()
        workflow.context.sql_contexts = []  # will raise IndexError
        result = node.update_context(workflow)
        assert result["success"] is False


class TestStripSQLMarkdown:
    """Test _strip_sql_markdown utility."""

    def test_strips_sql_markdown(self):
        node = make_node()
        text = "```sql\nSELECT * FROM t\n```"
        assert node._strip_sql_markdown(text) == "SELECT * FROM t"

    def test_no_markdown(self):
        node = make_node()
        text = "SELECT * FROM t"
        assert node._strip_sql_markdown(text) == "SELECT * FROM t"

    def test_non_string_input(self):
        node = make_node()
        result = node._strip_sql_markdown(None)
        assert result is None

    def test_multiline_sql(self):
        node = make_node()
        text = "```sql\nSELECT a,\n  b\nFROM t\n```"
        stripped = node._strip_sql_markdown(text)
        assert "```" not in stripped
        assert "SELECT" in stripped


class TestExecuteSQLNodeStream:
    """Test execute_stream yields ActionHistory objects."""

    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="db"))
        mock_result = ExecuteSQLResult(success=True, sql_query="SELECT 1", row_count=1, sql_return="1")
        mock_connector = MagicMock()
        mock_connector.execute.return_value = mock_result

        actions = []
        with patch.object(node, "_sql_connector", return_value=mock_connector):
            async for action in node.execute_stream():
                actions.append(action)

        assert len(actions) >= 1

    @pytest.mark.asyncio
    async def test_execute_stream_no_connector_raises(self):
        node = make_node(ExecuteSQLInput(sql_query="SELECT 1", database_name="db"))
        with patch.object(node, "_sql_connector", side_effect=RuntimeError("no conn")):
            with pytest.raises(RuntimeError):
                async for _ in node.execute_stream():
                    pass
