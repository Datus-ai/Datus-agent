# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/agent/workflow_runner.py.

CI-level: zero external dependencies. All LLM, DB, storage, and workflow
execution paths are mocked.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.workflow_runner import WorkflowRunner
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.base import BaseResult
from datus.schemas.node_models import SqlTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    defaults = dict(
        max_steps=20,
        workflow="reflection",
        load_cp=None,
        debug=False,
        plan_mode=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_config():
    cfg = MagicMock()
    cfg.workflow_plan = "reflection"
    cfg.get_trajectory_run_dir.return_value = "/tmp/test_traj"
    cfg.check_init_storage_config.return_value = None
    return cfg


def _make_sql_task():
    return SqlTask(task="find all users", database_name="mydb")


def _make_runner(args=None, config=None, pre_run=None, run_id="test-run"):
    return WorkflowRunner(
        args=args or _make_args(),
        agent_config=config or _make_config(),
        pre_run_callable=pre_run,
        run_id=run_id,
    )


def _make_mock_workflow(is_complete=True, node_count=2):
    wf = MagicMock()
    wf.is_complete.return_value = is_complete
    wf.nodes = {f"n{i}": MagicMock() for i in range(node_count)}
    wf.current_node_index = 0
    wf.task = MagicMock()
    wf.task.id = "task_001"
    wf.metadata = {}
    wf.get_current_node.return_value = None
    wf.get_final_result.return_value = {"status": "completed", "nodes": {}}
    return wf


# ---------------------------------------------------------------------------
# __init__ / basic attributes
# ---------------------------------------------------------------------------


class TestWorkflowRunnerInit:
    def test_attributes_set(self):
        args = _make_args()
        cfg = _make_config()
        runner = WorkflowRunner(args, cfg, run_id="r1")
        assert runner.args is args
        assert runner.global_config is cfg
        assert runner.workflow is None
        assert runner.workflow_ready is False
        assert runner.run_id == "r1"

    def test_pre_run_callable_stored(self):
        fn = MagicMock()
        runner = _make_runner(pre_run=fn)
        assert runner._pre_run is fn


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


class TestIsComplete:
    def test_no_workflow_returns_true(self):
        runner = _make_runner()
        assert runner.is_complete() is True

    def test_with_complete_workflow(self):
        runner = _make_runner()
        runner.workflow = _make_mock_workflow(is_complete=True)
        assert runner.is_complete() is True

    def test_with_incomplete_workflow(self):
        runner = _make_runner()
        runner.workflow = _make_mock_workflow(is_complete=False)
        assert runner.is_complete() is False


# ---------------------------------------------------------------------------
# _create_action_history / _update_action_status
# ---------------------------------------------------------------------------


class TestActionHistoryHelpers:
    def test_create_action_history(self):
        runner = _make_runner()
        action = runner._create_action_history("aid1", "msg", "type1", {"k": "v"})
        assert action.action_id == "aid1"
        assert action.messages == "msg"
        assert action.action_type == "type1"
        assert action.input == {"k": "v"}
        assert action.status == ActionStatus.PROCESSING
        assert action.role == ActionRole.WORKFLOW

    def test_create_action_history_no_input(self):
        runner = _make_runner()
        action = runner._create_action_history("aid2", "msg2", "t2")
        assert action.input == {}

    def test_update_action_status_success(self):
        runner = _make_runner()
        action = runner._create_action_history("a", "m", "t")
        runner._update_action_status(action, success=True, output_data={"key": "val"})
        assert action.status == ActionStatus.SUCCESS
        assert action.output == {"key": "val"}

    def test_update_action_status_failure(self):
        runner = _make_runner()
        action = runner._create_action_history("a", "m", "t")
        runner._update_action_status(action, success=False, error="boom")
        assert action.status == ActionStatus.FAILED
        assert "boom" in action.output.get("error", "")

    def test_update_action_status_failure_with_output(self):
        runner = _make_runner()
        action = runner._create_action_history("a", "m", "t")
        runner._update_action_status(action, success=False, error="err", output_data={"extra": 1})
        assert action.output.get("extra") == 1


# ---------------------------------------------------------------------------
# initialize_workflow
# ---------------------------------------------------------------------------


class TestInitializeWorkflow:
    def test_initialize_workflow_creates_workflow(self):
        runner = _make_runner()
        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.metadata = {}
        sql_task = _make_sql_task()

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            runner.initialize_workflow(sql_task)

        assert runner.workflow is mock_wf
        mock_wf.display.assert_called_once()

    def test_plan_mode_stored_in_metadata(self):
        args = _make_args(plan_mode="auto")
        runner = _make_runner(args=args)
        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.metadata = {}

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            runner.initialize_workflow(_make_sql_task())

        assert mock_wf.metadata.get("plan_mode") == "auto"
        assert mock_wf.metadata.get("auto_execute_plan") is True


# ---------------------------------------------------------------------------
# resume_workflow
# ---------------------------------------------------------------------------


class TestResumeWorkflow:
    def test_resume_workflow_success(self):
        args = _make_args(load_cp="/tmp/some_checkpoint.yaml")
        runner = _make_runner(args=args)
        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.resume = MagicMock()

        with patch("datus.agent.workflow_runner.Workflow.load", return_value=mock_wf):
            runner.resume_workflow(args)

        assert runner.workflow is mock_wf
        mock_wf.resume.assert_called_once()
        mock_wf.display.assert_called_once()

    def test_resume_workflow_load_failure_raises(self):
        args = _make_args(load_cp="/tmp/missing.yaml")
        runner = _make_runner(args=args)

        with patch("datus.agent.workflow_runner.Workflow.load", side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileNotFoundError):
                runner.resume_workflow(args)


# ---------------------------------------------------------------------------
# init_or_load_workflow
# ---------------------------------------------------------------------------


class TestInitOrLoadWorkflow:
    def test_uses_checkpoint_when_load_cp_set(self):
        args = _make_args(load_cp="/tmp/cp.yaml")
        runner = _make_runner(args=args)
        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.resume = MagicMock()

        with patch("datus.agent.workflow_runner.Workflow.load", return_value=mock_wf):
            result = runner.init_or_load_workflow(None)

        assert result is True
        assert runner.workflow_ready is True

    def test_uses_sql_task_when_provided(self):
        runner = _make_runner()
        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.metadata = {}

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            result = runner.init_or_load_workflow(_make_sql_task())

        assert result is True
        assert runner.workflow_ready is True

    def test_no_task_no_checkpoint_returns_none(self):
        runner = _make_runner()
        result = runner.init_or_load_workflow(None)
        assert result is None


# ---------------------------------------------------------------------------
# _finalize_workflow
# ---------------------------------------------------------------------------


class TestFinalizeWorkflow:
    def test_no_workflow_returns_empty(self):
        runner = _make_runner()
        result = runner._finalize_workflow(0)
        assert result == {}

    def test_with_workflow_saves_and_returns_metadata(self, tmp_path):
        runner = _make_runner()
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.get_final_result.return_value = {"status": "completed"}
        runner.workflow = mock_wf

        with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
            result = runner._finalize_workflow(3)

        assert result["steps"] == 3
        assert result["run_id"] == runner.run_id
        assert "save_path" in result
        mock_wf.save.assert_called_once()

    def test_trace_url_stored_in_metadata(self, tmp_path):
        runner = _make_runner()
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow()
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.get_final_result.return_value = {}
        runner.workflow = mock_wf

        with patch("datus.agent.workflow_runner.get_trace_url", return_value="http://trace.url/123"):
            runner._finalize_workflow(1)

        assert mock_wf.metadata.get("trace_url") == "http://trace.url/123"


# ---------------------------------------------------------------------------
# run (synchronous)
# ---------------------------------------------------------------------------


class TestWorkflowRunnerRun:
    def test_run_returns_empty_when_prerequisites_fail(self):
        runner = _make_runner()
        # No task, no checkpoint -> _ensure_prerequisites returns False
        result = runner.run(sql_task=None, check_storage=False)
        assert result == {}

    def test_run_with_completed_workflow(self, tmp_path):
        runner = _make_runner()
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow(is_complete=True)
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.metadata = {}
        mock_wf.is_complete.return_value = True
        mock_wf.get_final_result.return_value = {"status": "completed"}

        # _prepare_first_node needs get_current_node() to return a node with .complete()
        mock_first_node = MagicMock()
        mock_first_node.complete = MagicMock()
        mock_wf.get_current_node.return_value = mock_first_node
        mock_wf.advance_to_next_node.return_value = None

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
                with patch("datus.agent.workflow_runner.setup_node_input"):
                    result = runner.run(sql_task=_make_sql_task(), check_storage=False)

        assert isinstance(result, dict)

    def test_run_max_steps_stops_loop(self, tmp_path):
        """Workflow that never completes stops at max_steps."""
        args = _make_args(max_steps=2)
        runner = _make_runner(args=args)
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow(is_complete=False)
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.metadata = {}

        mock_node = MagicMock()
        mock_node.status = "completed"
        mock_node.type = "generate_sql"
        mock_node.description = "Generate SQL"
        mock_node.result = BaseResult(success=True)
        mock_node.run = MagicMock()

        # First call: _prepare_first_node; subsequent: execution loop
        call_idx = [0]

        def _get_current():
            call_idx[0] += 1
            if call_idx[0] == 1:
                first = MagicMock()
                first.complete = MagicMock()
                return first
            return mock_node

        mock_wf.get_current_node.side_effect = _get_current
        mock_wf.advance_to_next_node.return_value = None

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            with patch("datus.agent.workflow_runner.evaluate_result", return_value={"success": True}):
                with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
                    with patch("datus.agent.workflow_runner.setup_node_input"):
                        runner.run(sql_task=_make_sql_task(), check_storage=False)

    def test_run_node_failure_breaks_loop(self, tmp_path):
        """When a node fails, the run loop breaks."""
        runner = _make_runner()
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow(is_complete=False)
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.metadata = {}
        mock_wf.is_complete.side_effect = [False, False, True]

        mock_node = MagicMock()
        mock_node.status = "failed"
        mock_node.type = "generate_sql"
        mock_node.description = "Generate SQL"
        mock_node.result = None
        mock_node.run = MagicMock()

        call_idx = [0]

        def _get_current():
            call_idx[0] += 1
            if call_idx[0] == 1:
                first = MagicMock()
                first.complete = MagicMock()
                return first
            return mock_node

        mock_wf.get_current_node.side_effect = _get_current
        mock_wf.advance_to_next_node.return_value = None

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
                with patch("datus.agent.workflow_runner.setup_node_input"):
                    result = runner.run(sql_task=_make_sql_task(), check_storage=False)

        assert isinstance(result, dict)

    def test_pre_run_callable_called(self, tmp_path):
        pre_run = MagicMock()
        runner = _make_runner(pre_run=pre_run)
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow(is_complete=True)
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.metadata = {}
        mock_first_node = MagicMock()
        mock_first_node.complete = MagicMock()
        mock_wf.get_current_node.return_value = mock_first_node
        mock_wf.advance_to_next_node.return_value = None

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
                with patch("datus.agent.workflow_runner.setup_node_input"):
                    runner.run(sql_task=_make_sql_task(), check_storage=False)

        pre_run.assert_called_once()


# ---------------------------------------------------------------------------
# run_stream (async)
# ---------------------------------------------------------------------------


class TestWorkflowRunnerRunStream:
    @pytest.mark.asyncio
    async def test_run_stream_yields_init_action(self):
        runner = _make_runner()
        # No task, no checkpoint: yields init_action then returns early
        actions = []
        async for action in runner.run_stream(sql_task=None, check_storage=False):
            actions.append(action)

        assert len(actions) >= 1
        assert actions[0].action_id == "workflow_initialization"

    @pytest.mark.asyncio
    async def test_run_stream_yields_completion_action(self, tmp_path):
        runner = _make_runner()
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow(is_complete=True)
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.metadata = {}
        mock_wf.is_complete.return_value = True

        # _prepare_first_node needs get_current_node() to return a node with .complete()
        mock_first_node = MagicMock()
        mock_first_node.complete = MagicMock()
        mock_wf.get_current_node.return_value = mock_first_node
        mock_wf.advance_to_next_node.return_value = None

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
                with patch("datus.agent.workflow_runner.setup_node_input"):
                    actions = []
                    async for action in runner.run_stream(sql_task=_make_sql_task()):
                        actions.append(action)

        action_ids = [a.action_id for a in actions]
        assert "workflow_completion" in action_ids

    @pytest.mark.asyncio
    async def test_run_stream_fails_gracefully_on_bad_prerequisite(self):
        runner = _make_runner()
        # Force _ensure_prerequisites to fail by raising during check_init_storage_config
        runner.global_config.check_init_storage_config.side_effect = RuntimeError("storage error")

        actions = []
        async for action in runner.run_stream(sql_task=None, check_storage=True):
            actions.append(action)

        assert len(actions) >= 1
        init = actions[0]
        assert init.action_id == "workflow_initialization"

    @pytest.mark.asyncio
    async def test_run_stream_node_execution_yields_node_action(self, tmp_path):
        runner = _make_runner()
        runner.global_config.get_trajectory_run_dir.return_value = str(tmp_path)

        mock_wf = _make_mock_workflow(is_complete=False)
        mock_wf.display = MagicMock()
        mock_wf.save = MagicMock()
        mock_wf.metadata = {}

        mock_node = MagicMock()
        mock_node.id = "node_1"
        mock_node.status = "completed"
        mock_node.type = "generate_sql"
        mock_node.description = "Generate SQL"

        # run_stream on the node returns an async generator
        async def _node_stream(_):
            node_action = ActionHistory(
                action_id="node_1_action",
                role=ActionRole.WORKFLOW,
                messages="done",
                action_type="sql_generation",
                status=ActionStatus.SUCCESS,
            )
            yield node_action

        mock_node.run_stream = _node_stream

        call_count = 0

        def _is_complete():
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_wf.is_complete.side_effect = _is_complete

        # First call to get_current_node is from _prepare_first_node (needs .complete())
        # Subsequent calls are from the execution loop (needs .run_stream)
        get_current_calls = [0]

        def _get_current_node():
            get_current_calls[0] += 1
            if get_current_calls[0] == 1:
                # _prepare_first_node call - return node with complete()
                first = MagicMock()
                first.complete = MagicMock()
                return first
            return mock_node

        mock_wf.get_current_node.side_effect = _get_current_node
        mock_wf.advance_to_next_node.return_value = None

        with patch("datus.agent.workflow_runner.generate_workflow", return_value=mock_wf):
            with patch("datus.agent.workflow_runner.evaluate_result", return_value={"success": True}):
                with patch("datus.agent.workflow_runner.get_trace_url", return_value=None):
                    with patch("datus.agent.workflow_runner.setup_node_input"):
                        actions = []
                        async for action in runner.run_stream(sql_task=_make_sql_task()):
                            actions.append(action)

        action_ids = [a.action_id for a in actions]
        assert "node_1_action" in action_ids
