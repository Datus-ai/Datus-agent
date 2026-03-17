# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for SubworkflowNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.subworkflow_node_models import SubworkflowInput, SubworkflowResult


def make_agent_config(has_workflow=True, workflow_name="my_wf"):
    cfg = MagicMock()
    cfg.namespaces = {}
    cfg.current_namespace = "test"
    cfg.nodes = {}
    if has_workflow:
        cfg.custom_workflows = {workflow_name: {}}
    else:
        cfg.custom_workflows = {}
    return cfg


def make_node(input_data=None, agent_config=None):
    cfg = agent_config or make_agent_config()
    return Node.new_instance(
        "sw_1",
        "Subworkflow",
        NodeType.TYPE_SUBWORKFLOW,
        input_data=input_data,
        agent_config=cfg,
    )


class TestSubworkflowNodeExecute:
    """Test execute()."""

    def test_execute_none_input_result_is_none_or_failure(self):
        """When input is None, execute returns early. result stays None."""
        node = make_node()
        node.execute()
        # execute() returns early with a SubworkflowResult(success=False) via the first guard
        assert node.result is None or node.result.success is False

    def test_execute_empty_workflow_name_returns_failure_or_none(self):
        """Empty workflow_name causes early return."""
        node = make_node(SubworkflowInput(workflow_name=""))
        node.execute()
        # Either result is None (early return) or a SubworkflowResult(success=False)
        assert node.result is None or node.result.success is False

    def test_execute_no_parent_workflow_returns_failure(self):
        """If parent workflow not attached, execute() catches the error."""
        node = make_node(SubworkflowInput(workflow_name="my_wf"))
        # workflow attribute not set → raises ValueError internally
        node.execute()
        assert node.result.success is False
        assert "not available" in node.result.error.lower()

    def test_execute_workflow_not_in_config(self):
        cfg = make_agent_config(has_workflow=False)
        node = make_node(SubworkflowInput(workflow_name="missing_wf"), agent_config=cfg)
        node.workflow = MagicMock()
        node.workflow.task = MagicMock()
        node.execute()
        assert node.result.success is False
        assert "not found" in node.result.error.lower()

    def test_execute_success(self):
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)

        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        parent_workflow.context = MagicMock()
        node.workflow = parent_workflow

        # Build a mock subworkflow that completes immediately
        mock_subworkflow = MagicMock()
        mock_subworkflow.is_complete.return_value = True
        mock_subworkflow.get_current_node.return_value = None
        mock_subworkflow.workflow_config = None

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            node.execute()

        assert node.result.success is True
        assert node.result.workflow_name == "my_wf"

    def test_execute_passes_context_when_requested(self):
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf", pass_context=True), agent_config=cfg)

        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        ctx_copy = {"key": "value"}
        parent_workflow.context.copy.return_value = ctx_copy
        node.workflow = parent_workflow

        mock_subworkflow = MagicMock()
        mock_subworkflow.is_complete.return_value = True
        mock_subworkflow.get_current_node.return_value = None
        mock_subworkflow.workflow_config = None
        mock_subworkflow.context = {}

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            node.execute()

        assert mock_subworkflow.context == ctx_copy

    def test_execute_with_start_node_and_one_child(self):
        """Full execution path: start node → skip → action node → complete."""
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)
        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        node.workflow = parent_workflow

        start_node = MagicMock()
        start_node.id = "start"

        action_node = MagicMock()
        action_node.id = "action_1"
        action_node.type = NodeType.TYPE_GENERATE_SQL
        action_node.description = "Generate SQL"
        action_node.status = "completed"
        action_node.result = MagicMock(success=True)
        action_node.execute = MagicMock()

        mock_subworkflow = MagicMock()
        mock_subworkflow.workflow_config = None
        # get_current_node: first call returns start_node (for the skip block),
        # then action_node in the loop, then None once is_complete is True
        mock_subworkflow.get_current_node.side_effect = [start_node, action_node]
        mock_subworkflow.advance_to_next_node.return_value = None
        # is_complete: False first iteration, True after
        mock_subworkflow.is_complete.side_effect = [False, True]

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            # patch setup_node_input at datus.agent.evaluate since it's lazily imported
            with patch("datus.agent.evaluate.setup_node_input", return_value={"success": True}):
                node.execute()

        assert node.result is not None

    def test_execute_child_node_fails_returns_early(self):
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)
        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        node.workflow = parent_workflow

        start_node = MagicMock()
        start_node.id = "start"

        action_node = MagicMock()
        action_node.id = "fail_node"
        action_node.type = NodeType.TYPE_GENERATE_SQL
        action_node.description = "Fail"
        action_node.status = "failed"
        action_node.result = MagicMock(success=False, error="child failed")
        action_node.execute = MagicMock()

        mock_subworkflow = MagicMock()
        mock_subworkflow.workflow_config = None
        mock_subworkflow.get_current_node.side_effect = [start_node, action_node]
        mock_subworkflow.advance_to_next_node.return_value = None
        mock_subworkflow.is_complete.return_value = False

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            with patch("datus.agent.evaluate.setup_node_input", return_value={"success": True}):
                node.execute()

        assert node.result.success is False

    def test_execute_exceeds_max_iterations(self):
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)
        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        node.workflow = parent_workflow

        start_node = MagicMock()
        start_node.id = "start"

        action_node = MagicMock()
        action_node.id = "loop_node"
        action_node.type = NodeType.TYPE_GENERATE_SQL
        action_node.description = "Looping"
        action_node.status = "completed"
        action_node.result = MagicMock(success=True)
        action_node.execute = MagicMock()

        mock_subworkflow = MagicMock()
        mock_subworkflow.workflow_config = None
        # start_node for initial skip, then always returns action_node for the loop
        mock_subworkflow.get_current_node.side_effect = [start_node] + [action_node] * 60
        mock_subworkflow.advance_to_next_node.return_value = None
        mock_subworkflow.is_complete.return_value = False  # never completes

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            with patch("datus.agent.evaluate.setup_node_input", return_value={"success": True}):
                node.execute()

        assert node.result.success is False
        assert "infinite loop" in node.result.error.lower() or "maximum" in node.result.error.lower()

    def test_execute_child_node_raises_exception(self):
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)
        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        node.workflow = parent_workflow

        start_node = MagicMock()
        start_node.id = "start"

        action_node = MagicMock()
        action_node.id = "exc_node"
        action_node.type = NodeType.TYPE_GENERATE_SQL
        action_node.description = "Crash"
        action_node._initialize = MagicMock(side_effect=RuntimeError("crash!"))

        mock_subworkflow = MagicMock()
        mock_subworkflow.workflow_config = None
        mock_subworkflow.get_current_node.side_effect = [start_node, action_node]
        mock_subworkflow.advance_to_next_node.return_value = None
        mock_subworkflow.is_complete.return_value = False

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            with patch("datus.agent.evaluate.setup_node_input", return_value={"success": True}):
                node.execute()

        assert node.result.success is False
        assert "crash!" in node.result.error

    def test_execute_setup_node_input_fails(self):
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)
        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        node.workflow = parent_workflow

        start_node = MagicMock()
        start_node.id = "start"

        next_node = MagicMock()
        next_node.id = "next"

        action_node = MagicMock()
        action_node.id = "action"
        action_node.type = NodeType.TYPE_GENERATE_SQL
        action_node.description = "Action"
        action_node.status = "completed"
        action_node.result = MagicMock(success=True)
        action_node.execute = MagicMock()

        mock_subworkflow = MagicMock()
        mock_subworkflow.workflow_config = None
        mock_subworkflow.get_current_node.side_effect = [start_node, action_node]
        mock_subworkflow.advance_to_next_node.return_value = next_node
        mock_subworkflow.is_complete.return_value = False

        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            with patch(
                "datus.agent.evaluate.setup_node_input",
                side_effect=[{"success": True}, {"success": False, "message": "setup failed"}],
            ):
                node.execute()

        assert node.result.success is False
        assert "setup failed" in node.result.error


class TestSubworkflowNodeApplyNodeParams:
    """Test _apply_node_params."""

    def test_apply_node_params_sets_attributes(self):
        node = make_node(
            SubworkflowInput(
                workflow_name="wf",
                node_params={"generate_sql": {"max_context_length": 100}},
            )
        )
        workflow = MagicMock()

        # Use a real object for node.input so hasattr works and setattr doesn't fail
        class FakeInput:
            max_context_length = 8000

        fake_child = MagicMock()
        fake_child.type = "generate_sql"
        fake_child.id = "gen_1"
        fake_child.input = FakeInput()
        workflow.nodes.values.return_value = [fake_child]

        node._apply_node_params(workflow)
        assert fake_child.input.max_context_length == 100

    def test_apply_node_params_missing_attribute_warns(self):
        node = make_node(
            SubworkflowInput(
                workflow_name="wf",
                node_params={"generate_sql": {"nonexistent_param": "value"}},
            )
        )
        workflow = MagicMock()

        class FakeInput:
            pass

        fake_child = MagicMock()
        fake_child.type = "generate_sql"
        fake_child.input = FakeInput()
        workflow.nodes.values.return_value = [fake_child]

        # Should not raise, just warn
        node._apply_node_params(workflow)

    def test_apply_node_params_empty(self):
        node = make_node(SubworkflowInput(workflow_name="wf", node_params=None))
        workflow = MagicMock()
        # Should be a no-op
        node._apply_node_params(workflow)


class TestSubworkflowNodeApplyConfigParams:
    """Test _apply_config_params."""

    def test_apply_config_dict(self):
        node = make_node()
        workflow = MagicMock()

        class FakeInput:
            max_context_length = 8000

        mock_child = MagicMock()
        mock_child.type = "generate_sql"
        mock_child.id = "gen_1"
        mock_child.input = FakeInput()
        workflow.nodes.values.return_value = [mock_child]

        node._apply_config_params(workflow, {"generate_sql": {"model": "gpt-4", "max_context_length": 1000}})
        assert mock_child.model == "gpt-4"
        assert mock_child.input.max_context_length == 1000

    def test_apply_config_none(self):
        node = make_node()
        workflow = MagicMock()
        # Should be a no-op
        node._apply_config_params(workflow, None)

    def test_apply_config_string_file_not_found(self):
        node = make_node()
        workflow = MagicMock()
        # File doesn't exist, should log warning but not raise
        node._apply_config_params(workflow, "/nonexistent/path/config.yaml")


class TestSubworkflowNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_valid(self):
        node = make_node(SubworkflowInput(workflow_name="my_wf"))
        workflow = MagicMock()

        result = node.setup_input(workflow)
        assert result["success"] is True

    def test_setup_input_no_workflow_name(self):
        node = make_node(SubworkflowInput(workflow_name="my_wf"))
        node.input.workflow_name = ""
        workflow = MagicMock()

        result = node.setup_input(workflow)
        assert result["success"] is False

    def test_setup_input_wrong_type(self):
        node = make_node()
        workflow = MagicMock()
        result = node.setup_input(workflow)
        assert result["success"] is False


class TestSubworkflowNodeUpdateContext:
    """Test update_context."""

    def test_update_context_with_success_result(self):
        node = make_node(SubworkflowInput(workflow_name="my_wf"))
        node.result = SubworkflowResult(
            success=True,
            workflow_name="my_wf",
            node_results={"n1": {"success": True}},
            execution_order=["n1"],
        )
        workflow = MagicMock()
        workflow.context = {}

        result = node.update_context(workflow)
        assert result["success"] is True
        assert "subworkflow_my_wf" in workflow.context

    def test_update_context_no_result(self):
        node = make_node()
        node.result = None
        workflow = MagicMock()

        result = node.update_context(workflow)
        assert result["success"] is True


class TestSubworkflowNodeStream:
    """Test execute_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_manager_no_input(self):
        node = make_node()
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_with_mock_manager(self):
        """Use MagicMock for manager since ActionHistoryManager has no .create()/.update()."""
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)

        mgr = MagicMock()
        start_action = MagicMock()
        end_action = MagicMock()
        mgr.create.return_value = start_action
        mgr.update.return_value = end_action

        actions = []
        async for action in node.execute_stream(action_history_manager=mgr):
            actions.append(action)

        assert len(actions) >= 2

    @pytest.mark.asyncio
    async def test_execute_stream_success_path(self):
        """Full stream success: subworkflow completes immediately."""
        cfg = make_agent_config(workflow_name="my_wf")
        node = make_node(SubworkflowInput(workflow_name="my_wf"), agent_config=cfg)
        parent_workflow = MagicMock()
        parent_workflow.task = MagicMock()
        parent_workflow.context = MagicMock()
        node.workflow = parent_workflow

        mock_subworkflow = MagicMock()
        mock_subworkflow.is_complete.return_value = True
        mock_subworkflow.get_current_node.return_value = None
        mock_subworkflow.workflow_config = None

        mgr = MagicMock()
        start_action = MagicMock()
        end_action = MagicMock()
        mgr.create.return_value = start_action
        mgr.update.return_value = end_action

        actions = []
        with patch("datus.agent.node.subworkflow_node.generate_workflow", return_value=mock_subworkflow):
            async for action in node.execute_stream(action_history_manager=mgr):
                actions.append(action)

        assert len(actions) >= 2
        assert node.result.success is True
