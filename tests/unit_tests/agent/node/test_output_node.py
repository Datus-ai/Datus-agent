# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for OutputNode.

CI-level: zero external deps, zero network, zero API keys.
OutputTool and SQL connector are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.output_node import OutputNode
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.node_models import OutputInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config():
    cfg = MagicMock()
    cfg.agentic_nodes = {}
    cfg.permissions_config = None
    cfg.skills_config = None
    return cfg


def _make_output_input(**kwargs):
    defaults = dict(
        finished=True,
        task_id="task_1",
        task="Show total sales",
        database_name="test_db",
        output_dir="output",
        gen_sql="SELECT SUM(sales) FROM orders",
        sql_result="100000",
        row_count=1,
        table_schemas=[],
        metrics=[],
        external_knowledge="",
        error=None,
    )
    defaults.update(kwargs)
    return OutputInput(**defaults)


def _make_node(agent_config=None):
    cfg = agent_config or _make_agent_config()
    node = OutputNode(
        node_id="output_1",
        description="Output node",
        node_type="output",
        agent_config=cfg,
    )
    node.input = _make_output_input()
    return node


def _make_workflow():
    wf = MagicMock()
    wf.task.id = "task_1"
    wf.task.database_name = "test_db"
    wf.task.output_dir = "output"
    wf.task.external_knowledge = "some knowledge"
    wf.context.table_schemas = []
    wf.context.metrics = []
    sql_ctx = MagicMock()
    sql_ctx.sql_query = "SELECT 1"
    sql_ctx.sql_return = "1"
    sql_ctx.row_count = 1
    sql_ctx.sql_error = None
    wf.get_last_sqlcontext.return_value = sql_ctx
    wf.get_task.return_value = "Show total sales"
    return wf


# ---------------------------------------------------------------------------
# TestOutputNodeInit
# ---------------------------------------------------------------------------


class TestOutputNodeInit:
    def test_node_creates(self):
        node = _make_node()
        assert node.id == "output_1"
        assert node.description == "Output node"


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputOutputNode:
    def test_setup_input_builds_output_input(self):
        node = _make_node()
        wf = _make_workflow()
        result = node.setup_input(wf)

        assert result["success"] is True
        assert isinstance(node.input, OutputInput)
        assert node.input.task_id == "task_1"
        assert node.input.database_name == "test_db"
        assert node.input.gen_sql == "SELECT 1"

    def test_setup_input_uses_sql_context(self):
        node = _make_node()
        wf = _make_workflow()
        wf.get_last_sqlcontext.return_value.sql_query = "SELECT COUNT(*) FROM orders"
        node.setup_input(wf)

        assert node.input.gen_sql == "SELECT COUNT(*) FROM orders"

    def test_setup_input_finished_is_true(self):
        node = _make_node()
        wf = _make_workflow()
        node.setup_input(wf)

        assert node.input.finished is True


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextOutputNode:
    def test_update_context_returns_success(self):
        node = _make_node()
        wf = _make_workflow()
        result = node.update_context(wf)

        assert result["success"] is True
        assert "no context update" in result["message"].lower()


# ---------------------------------------------------------------------------
# TestExecuteOutput
# ---------------------------------------------------------------------------


class TestExecuteOutput:
    def test_execute_calls_output_tool(self):
        node = _make_node()
        mock_result = MagicMock()
        mock_result.success = True

        with patch("datus.agent.node.output_node.OutputTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = mock_result
            with patch.object(node, "_sql_connector", return_value=MagicMock()):
                result = node._execute_output()

        mock_tool.execute.assert_called_once()
        assert result == mock_result

    def test_execute_sets_result(self):
        node = _make_node()
        mock_result = MagicMock()
        mock_result.success = True

        with patch("datus.agent.node.output_node.OutputTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = mock_result
            with patch.object(node, "_sql_connector", return_value=MagicMock()):
                node.execute()

        assert node.result == mock_result


# ---------------------------------------------------------------------------
# TestExecuteStream
# ---------------------------------------------------------------------------


class TestExecuteStreamOutputNode:
    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        """execute_stream yields at least two actions (processing + success)."""
        node = _make_node()
        mock_result = MagicMock()
        mock_result.success = True

        with patch("datus.agent.node.output_node.OutputTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = mock_result
            with patch.object(node, "_sql_connector", return_value=MagicMock()):
                actions = []
                async for action in node.execute_stream():
                    actions.append(action)

        assert len(actions) >= 2

    @pytest.mark.asyncio
    async def test_execute_stream_final_action_success(self):
        """Last action has status SUCCESS."""
        node = _make_node()
        mock_result = MagicMock()
        mock_result.success = True

        with patch("datus.agent.node.output_node.OutputTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = mock_result
            with patch.object(node, "_sql_connector", return_value=MagicMock()):
                actions = []
                async for action in node.execute_stream():
                    actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_sets_result(self):
        """execute_stream stores result on self.result."""
        node = _make_node()
        mock_result = MagicMock()
        mock_result.success = True

        with patch("datus.agent.node.output_node.OutputTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = mock_result
            with patch.object(node, "_sql_connector", return_value=MagicMock()):
                async for _ in node.execute_stream():
                    pass

        assert node.result == mock_result

    @pytest.mark.asyncio
    async def test_execute_stream_propagates_exception(self):
        """execute_stream re-raises exceptions from _execute_output."""
        node = _make_node()

        with patch.object(node, "_execute_output", side_effect=RuntimeError("output error")):
            with pytest.raises(RuntimeError, match="output error"):
                async for _ in node.execute_stream():
                    pass

    @pytest.mark.asyncio
    async def test_execute_stream_action_has_workflow_role(self):
        """First yielded action has WORKFLOW role."""
        node = _make_node()
        mock_result = MagicMock()
        mock_result.success = True

        with patch("datus.agent.node.output_node.OutputTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = mock_result
            with patch.object(node, "_sql_connector", return_value=MagicMock()):
                actions = []
                async for action in node.execute_stream():
                    actions.append(action)

        assert actions[0].role == ActionRole.WORKFLOW
