# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ReasonSQLNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult


def make_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_datasource = "test"
    cfg.nodes = {}
    return cfg


def make_reasoning_input():
    return ReasoningInput(
        sql_task=SqlTask(task="Count orders", database_type="sqlite", database_name="db"),
        table_schemas=[
            TableSchema(table_name="orders", database_name="db", definition="CREATE TABLE orders (id INT, total FLOAT)")
        ],
        data_details=[],
        metrics=[],
        contexts=[],
        external_knowledge="",
        database_type="sqlite",
    )


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "reason_1",
        "Reason SQL",
        NodeType.TYPE_REASONING,
        input_data=input_data,
        agent_config=cfg,
    )


class TestReasonSQLNodeInit:
    def test_init_sets_action_history_manager(self):
        node = make_node()
        assert node.action_history_manager is None


class TestReasonSQLNodeExecute:
    """Test execute() / _reason_sql()."""

    def test_execute_success(self):
        node = make_node(make_reasoning_input())
        node.model = MagicMock()
        expected = ReasoningResult(
            success=True,
            sql_query="SELECT COUNT(*) FROM orders",
            sql_return="42",
            row_count=1,
        )
        with patch("datus.agent.node.reason_sql_node.reasoning_sql_with_mcp", return_value=expected):
            node.execute()

        assert node.result.success is True
        assert "COUNT" in node.result.sql_query

    def test_execute_returns_failure_on_exception(self):
        node = make_node(make_reasoning_input())
        node.model = MagicMock()

        with patch(
            "datus.agent.node.reason_sql_node.reasoning_sql_with_mcp",
            side_effect=RuntimeError("mcp error"),
        ):
            node.execute()

        assert node.result.success is False
        assert "mcp error" in node.result.error


class TestReasonSQLNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_builds_reasoning_input(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task = SqlTask(task="Count orders", database_type="sqlite", database_name="db")
        workflow.context.table_schemas = []
        workflow.context.table_values = []
        workflow.context.metrics = []
        workflow.context.sql_contexts = []

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, ReasoningInput)
        assert node.input.sql_task.task == "Count orders"

    def test_setup_input_uses_last_sql_context(self):
        node = make_node()
        sql_ctx = SQLContext(sql_query="SELECT 1", sql_error="bad")
        workflow = MagicMock()
        workflow.task = SqlTask(task="Retry", database_type="sqlite", database_name="db")
        workflow.context.table_schemas = []
        workflow.context.table_values = []
        workflow.context.metrics = []
        workflow.context.sql_contexts = [sql_ctx]

        node.setup_input(workflow)
        assert len(node.input.contexts) == 1
        assert node.input.contexts[0].sql_query == "SELECT 1"


class TestReasonSQLNodeUpdateContext:
    """Test update_context."""

    def test_update_context_success_appends_sql(self):
        node = make_node(make_reasoning_input())
        inner_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM orders", sql_error="")
        node.result = ReasoningResult(
            success=True,
            sql_query="SELECT COUNT(*) FROM orders",
            sql_return="42",
            row_count=1,
            sql_contexts=[inner_ctx],
        )
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        # Both inner_ctx and the final new_record should be added
        assert len(workflow.context.sql_contexts) >= 1

    def test_update_context_filters_failed_inner_contexts(self):
        node = make_node(make_reasoning_input())
        bad_ctx = SQLContext(sql_query="SELECT bad", sql_error="syntax error")
        node.result = ReasoningResult(
            success=True,
            sql_query="SELECT 1",
            sql_return="1",
            row_count=1,
            sql_contexts=[bad_ctx],
        )
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        node.update_context(workflow)
        # bad_ctx should be skipped, only final result appended
        assert len(workflow.context.sql_contexts) == 1

    def test_update_context_failure_triggers_regenerate(self):
        node = make_node(make_reasoning_input())
        node.result = ReasoningResult(success=False, sql_query="", error="reasoning failed")
        node.agent_config = make_agent_config()
        workflow = MagicMock()
        workflow.context.sql_contexts = []
        workflow.reflection_round = 0
        workflow.tools = []
        workflow.current_node_index = 0

        with patch.object(Node, "new_instance") as mock_new_instance:
            mock_new_instance.return_value = MagicMock()
            result = node.update_context(workflow)

        assert result["success"] is True  # regenerate path returns success

    def test_update_context_uses_streaming_results(self):
        node = make_node(make_reasoning_input())
        node.result = ReasoningResult(success=True, sql_query="SELECT 1")
        sql_ctx = SQLContext(sql_query="SELECT 1", sql_error="")
        mock_manager = MagicMock()
        mock_manager.sql_contexts = [sql_ctx]
        node.action_history_manager = mock_manager

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True

    def test_update_context_exception(self):
        node = make_node(make_reasoning_input())
        node.result = ReasoningResult(success=True, sql_query="SELECT 1")
        workflow = MagicMock()
        workflow.context.sql_contexts = MagicMock()
        workflow.context.sql_contexts.__bool__ = MagicMock(return_value=False)
        # Trigger exception by making sql_contexts.append raise
        workflow.context.sql_contexts.append.side_effect = RuntimeError("ctx error")

        result = node.update_context(workflow)
        assert result["success"] is False
