# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ParallelNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import GenerateSQLResult
from datus.schemas.parallel_node_models import ParallelInput, ParallelResult


def make_agent_config():
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_namespace = "test"
    cfg.nodes = {}
    cfg.custom_workflows = {}
    return cfg


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "par_1",
        "Parallel",
        NodeType.TYPE_PARALLEL,
        input_data=input_data,
        agent_config=cfg,
    )


class TestParallelNodeExecute:
    """Test execute() / _execute_child_node()."""

    def test_execute_no_input_returns_failure(self):
        node = make_node(ParallelInput(child_nodes=[]))
        result = node.execute()
        # execute() returns ParallelResult directly; node.result may or may not be set
        # Check the returned value
        assert result is not None
        assert result.success is False
        assert "No child nodes" in result.error

    def test_execute_no_workflow_context_returns_failure(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        # workflow attr not set
        node.execute()
        assert node.result.success is False

    def test_execute_children_succeed(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql", "generate_sql"]))
        workflow = MagicMock()
        workflow.tools = []
        node.workflow = workflow

        mock_child_result = {
            "success": True,
            "status": "completed",
            "result": GenerateSQLResult(success=True, sql_query="SELECT 1", tables=[]),
            "node_id": "par_1_child_0_generate_sql",
            "node_type": "generate_sql",
            "start_time": 0.0,
            "end_time": 1.0,
        }

        with patch.object(node, "_create_child_nodes") as mock_create:
            child_node_a = MagicMock()
            child_node_a.id = "par_1_child_0_generate_sql"
            child_node_b = MagicMock()
            child_node_b.id = "par_1_child_1_generate_sql"
            mock_create.return_value = [child_node_a, child_node_b]

            with patch.object(node, "_execute_child_node", return_value=mock_child_result):
                node.execute()

        assert node.result.success is True
        assert len(node.result.child_results) == 2

    def test_execute_all_children_fail(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        workflow = MagicMock()
        workflow.tools = []
        node.workflow = workflow

        fail_result = {
            "success": False,
            "status": "failed",
            "error": "child failed",
            "node_id": "par_1_child_0_generate_sql",
            "node_type": "generate_sql",
            "start_time": 0.0,
            "end_time": 1.0,
        }

        with patch.object(node, "_create_child_nodes") as mock_create:
            child_node = MagicMock()
            child_node.id = "par_1_child_0_generate_sql"
            mock_create.return_value = [child_node]

            with patch.object(node, "_execute_child_node", return_value=fail_result):
                node.execute()

        assert node.result.success is False
        assert "All child nodes failed" in node.result.error

    def test_execute_partial_success(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql", "generate_sql"]))
        workflow = MagicMock()
        workflow.tools = []
        node.workflow = workflow

        success_result = {
            "success": True,
            "status": "completed",
            "node_id": "c0",
            "node_type": "generate_sql",
            "result": None,
            "start_time": 0.0,
            "end_time": 1.0,
        }
        fail_result = {
            "success": False,
            "status": "failed",
            "error": "fail",
            "node_id": "c1",
            "node_type": "generate_sql",
            "start_time": 0.0,
            "end_time": 1.0,
        }

        with patch.object(node, "_create_child_nodes") as mock_create:
            c0 = MagicMock()
            c0.id = "c0"
            c1 = MagicMock()
            c1.id = "c1"
            mock_create.return_value = [c0, c1]

            results_iter = iter([success_result, fail_result])
            with patch.object(node, "_execute_child_node", side_effect=results_iter):
                node.execute()

        # any_success = True since c0 succeeded
        assert node.result.success is True

    def test_execute_create_child_nodes_exception(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        node.workflow = MagicMock()

        with patch.object(node, "_create_child_nodes", side_effect=RuntimeError("create error")):
            node.execute()

        assert node.result.success is False
        assert "create error" in node.result.error


class TestParallelNodeExecuteChildNode:
    """Test _execute_child_node()."""

    def test_execute_child_node_success(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        child = MagicMock()
        child.id = "child_id"
        child.type = "generate_sql"
        child.start_time = 0.0
        child.end_time = 1.0
        child.status = "completed"
        child.result = GenerateSQLResult(success=True, sql_query="SELECT 1", tables=[])

        def mock_execute():
            child.result = GenerateSQLResult(success=True, sql_query="SELECT 1", tables=[])

        child.execute = mock_execute

        result = node._execute_child_node(child)
        assert result["success"] is True

    def test_execute_child_node_returns_null_result(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        child = MagicMock()
        child.id = "child_id"
        child.type = "generate_sql"
        child.start_time = 0.0
        child.end_time = 1.0
        child.result = None

        def mock_execute():
            child.result = None

        child.execute = mock_execute

        result = node._execute_child_node(child)
        assert result["success"] is False

    def test_execute_child_node_raises_exception(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        child = MagicMock()
        child.id = "child_id"
        child.type = "generate_sql"
        child.start_time = 0.0
        child.end_time = 1.0
        child._initialize.side_effect = RuntimeError("child error")

        result = node._execute_child_node(child)
        assert result["success"] is False
        assert "child error" in result["error"]


class TestParallelNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_valid_action_types(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql", "reasoning"]))
        workflow = MagicMock()

        result = node.setup_input(workflow)
        assert result["success"] is True

    def test_setup_input_invalid_child_type(self):
        node = make_node(ParallelInput(child_nodes=["invalid_type"]))
        workflow = MagicMock()

        result = node.setup_input(workflow)
        assert result["success"] is False
        assert "Unsupported child entry" in result["message"]

    def test_setup_input_wrong_input_type(self):
        node = make_node()
        from datus.schemas.node_models import GenerateSQLInput, SqlTask

        node.input = GenerateSQLInput(
            sql_task=SqlTask(task="t", database_type="sqlite", database_name="db"),
            table_schemas=[],
        )
        workflow = MagicMock()
        result = node.setup_input(workflow)
        assert result["success"] is False

    def test_setup_input_subworkflow_name_valid(self):
        cfg = make_agent_config()
        cfg.custom_workflows = {"my_workflow": {}}
        node = Node.new_instance(
            "par_sw",
            "Parallel",
            NodeType.TYPE_PARALLEL,
            input_data=ParallelInput(child_nodes=["my_workflow"]),
            agent_config=cfg,
        )
        node.agent_config = cfg
        workflow = MagicMock()

        result = node.setup_input(workflow)
        assert result["success"] is True

    def test_setup_input_subworkflow_type_valid(self):
        node = make_node(ParallelInput(child_nodes=["subworkflow"]))
        result = node.setup_input(MagicMock())
        assert result["success"] is True


class TestParallelNodeUpdateContext:
    """Test update_context."""

    def test_update_context_stores_parallel_results(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        node.result = ParallelResult(
            success=True,
            child_results={"c0": {"success": True}},
            execution_order=["c0"],
        )
        workflow = MagicMock()

        result = node.update_context(workflow)
        assert result["success"] is True
        workflow.context.update_parallel_results.assert_called_once()

    def test_update_context_no_result(self):
        node = make_node()
        node.result = None
        workflow = MagicMock()

        result = node.update_context(workflow)
        assert result["success"] is True


class TestParallelNodeStream:
    """Test execute_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_manager_yields_nothing(self):
        node = make_node()
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_with_manager(self):
        node = make_node(ParallelInput(child_nodes=["generate_sql"]))
        node.workflow = MagicMock()
        fail_result = {
            "success": False,
            "status": "failed",
            "error": "fail",
            "node_id": "c0",
            "node_type": "generate_sql",
            "start_time": 0.0,
            "end_time": 1.0,
        }
        # Use MagicMock for manager to avoid ActionHistoryManager API constraints
        mgr = MagicMock()
        mgr.create.return_value = MagicMock()
        mgr.update.return_value = MagicMock()
        with patch.object(node, "_create_child_nodes") as mock_create:
            c0 = MagicMock()
            c0.id = "c0"
            mock_create.return_value = [c0]
            with patch.object(node, "_execute_child_node", return_value=fail_result):
                actions = []
                async for action in node.execute_stream(action_history_manager=mgr):
                    actions.append(action)

        assert len(actions) >= 2
