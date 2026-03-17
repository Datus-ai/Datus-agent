# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for FixNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.fix_node_models import FixInput, FixResult
from datus.schemas.node_models import SQLContext, SqlTask, TableSchema


def make_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_namespace = "test"
    cfg.nodes = {}
    return cfg


def make_fix_input():
    return FixInput(
        sql_task=SqlTask(task="Fix the query", database_type="sqlite", database_name="db"),
        sql_context=SQLContext(
            sql_query="SELECT * FROM t WHERE x = 1",
            explanation="original",
            sql_error="no such column: x",
        ),
        schemas=[TableSchema(table_name="t", database_name="db", definition="CREATE TABLE t (id INT)")],
    )


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "fix_1",
        "Fix SQL",
        NodeType.TYPE_FIX,
        input_data=input_data,
        agent_config=cfg,
    )


class TestFixNodeExecute:
    """Test FixNode._execute_fix() / execute()."""

    def test_execute_no_model_returns_failure(self):
        node = make_node(make_fix_input())
        node.model = None
        node.execute()
        assert node.result.success is False
        assert "model not provided" in node.result.error.lower()

    def test_execute_calls_autofix_sql(self):
        node = make_node(make_fix_input())
        node.model = MagicMock()
        expected = FixResult(
            success=True,
            sql_query="SELECT * FROM t WHERE id = 1",
            explanation="Fixed column name",
        )
        with patch("datus.agent.node.fix_node.autofix_sql", return_value=expected):
            node.execute()

        assert node.result.success is True
        assert "id = 1" in node.result.sql_query

    def test_execute_handles_exception(self):
        node = make_node(make_fix_input())
        node.model = MagicMock()

        with patch("datus.agent.node.fix_node.autofix_sql", side_effect=RuntimeError("fix error")):
            node.execute()

        assert node.result.success is False
        assert "fix error" in node.result.error


class TestFixNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_builds_fix_input(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task = SqlTask(task="Fix SQL", database_type="sqlite", database_name="db")
        workflow.get_last_sqlcontext.return_value = SQLContext(sql_query="SELECT bad", sql_error="syntax error")
        workflow.context.table_schemas = []

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, FixInput)
        assert node.input.sql_context.sql_query == "SELECT bad"


class TestFixNodeUpdateContext:
    """Test update_context."""

    def test_update_context_appends_sql_context(self):
        node = make_node(make_fix_input())
        node.result = FixResult(
            success=True,
            sql_query="SELECT id FROM t",
            explanation="Fixed",
        )
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1
        assert workflow.context.sql_contexts[0].sql_query == "SELECT id FROM t"

    def test_update_context_exception(self):
        node = make_node(make_fix_input())
        node.result = FixResult(success=True, sql_query="SELECT 1", explanation="ok")
        workflow = MagicMock()
        workflow.context.sql_contexts = MagicMock()
        workflow.context.sql_contexts.append.side_effect = RuntimeError("ctx error")

        result = node.update_context(workflow)
        assert result["success"] is False


class TestFixNodeStream:
    """Test execute_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_model_yields_nothing(self):
        node = make_node(make_fix_input())
        node.model = None
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        node = make_node(make_fix_input())
        node.model = MagicMock()
        expected = FixResult(success=True, sql_query="SELECT id FROM t", explanation="Fixed")
        with patch("datus.agent.node.fix_node.autofix_sql", return_value=expected):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert len(actions) >= 1
        assert node.result.success is True
