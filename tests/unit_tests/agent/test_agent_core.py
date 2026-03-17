# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/agent/agent.py — Agent class core methods.

CI-level: zero external dependencies. All LLM, DB, storage calls mocked.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.agent import Agent
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    defaults = dict(
        max_steps=10,
        workflow="reflection",
        load_cp=None,
        debug=False,
        force=False,
        yes=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_agent_config(namespace="test_ns"):
    cfg = MagicMock()
    cfg.current_namespace = namespace
    cfg.namespaces = {namespace: {"type": "sqlite", "dbs": []}}
    cfg.workflow_plan = "reflection"
    cfg.get_trajectory_run_dir.return_value = "/tmp/traj"
    cfg.output_dir = "/tmp/output"
    return cfg


def _make_agent(args=None, config=None):
    """Create Agent with mocked DB manager to avoid real DB connections."""
    mock_db_manager = MagicMock()
    with patch("datus.agent.agent.db_manager_instance", return_value=mock_db_manager):
        agent = Agent(
            args=args or _make_args(),
            agent_config=config or _make_agent_config(),
            db_manager=mock_db_manager,
        )
    return agent


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestAgentInit:
    def test_attributes_set(self):
        args = _make_args()
        cfg = _make_agent_config()
        agent = _make_agent(args=args, config=cfg)
        assert agent.args is args
        assert agent.global_config is cfg
        assert agent.tools == {}
        assert agent.storage_modules == {}

    def test_db_manager_from_argument(self):
        mock_db = MagicMock()
        args = _make_args()
        cfg = _make_agent_config()
        with patch("datus.agent.agent.db_manager_instance"):
            agent = Agent(args=args, agent_config=cfg, db_manager=mock_db)
        assert agent.db_manager is mock_db

    def test_db_manager_created_when_not_provided(self):
        args = _make_args()
        cfg = _make_agent_config()
        mock_db = MagicMock()
        with patch("datus.agent.agent.db_manager_instance", return_value=mock_db):
            agent = Agent(args=args, agent_config=cfg)
        assert agent.db_manager is mock_db


# ---------------------------------------------------------------------------
# _force_delete property
# ---------------------------------------------------------------------------


class TestForceDeleteProperty:
    def test_force_false_by_default(self):
        agent = _make_agent()
        assert agent._force_delete is False

    def test_force_true_when_force_set(self):
        agent = _make_agent(args=_make_args(force=True))
        assert agent._force_delete is True

    def test_force_true_when_yes_set(self):
        agent = _make_agent(args=_make_args(yes=True))
        assert agent._force_delete is True


# ---------------------------------------------------------------------------
# _check_storage_modules
# ---------------------------------------------------------------------------


class TestCheckStorageModules:
    def test_no_storage_dirs(self, tmp_path, monkeypatch):
        """When storage dirs don't exist, no modules registered."""
        monkeypatch.chdir(tmp_path)
        agent = _make_agent()
        # No storage dirs exist under tmp_path -> storage_modules should be empty
        assert agent.storage_modules == {}

    def test_schema_metadata_dir_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "storage" / "schema_metadata").mkdir(parents=True)
        agent = _make_agent()
        assert "schema_metadata" in agent.storage_modules

    def test_metric_store_dir_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "storage" / "metric_store").mkdir(parents=True)
        agent = _make_agent()
        assert "metric_store" in agent.storage_modules


# ---------------------------------------------------------------------------
# create_workflow_runner
# ---------------------------------------------------------------------------


class TestCreateWorkflowRunner:
    def test_returns_workflow_runner(self):
        from datus.agent.workflow_runner import WorkflowRunner

        agent = _make_agent()
        runner = agent.create_workflow_runner(check_db=False)
        assert isinstance(runner, WorkflowRunner)

    def test_run_id_passed_through(self):
        pass

        agent = _make_agent()
        runner = agent.create_workflow_runner(check_db=False, run_id="my-run-id")
        assert runner.run_id == "my-run-id"

    def test_pre_run_callable_set_when_check_db(self):
        agent = _make_agent()
        runner = agent.create_workflow_runner(check_db=True)
        assert runner._pre_run is not None

    def test_pre_run_callable_none_when_no_check_db(self):
        agent = _make_agent()
        runner = agent.create_workflow_runner(check_db=False)
        assert runner._pre_run is None


# ---------------------------------------------------------------------------
# check_db
# ---------------------------------------------------------------------------


class TestCheckDb:
    def test_success_when_connections_found(self):
        agent = _make_agent()
        mock_conn = MagicMock()
        mock_conn.test_connection.return_value = None
        agent.db_manager.get_connections.return_value = {"mydb": mock_conn}

        result = agent.check_db()
        assert result["status"] == "success"

    def test_error_when_namespace_not_in_config(self):
        cfg = _make_agent_config(namespace="test_ns")
        cfg.namespaces = {}  # empty namespaces
        agent = _make_agent(config=cfg)

        result = agent.check_db()
        assert result["status"] == "error"

    def test_error_when_no_connections_returned(self):
        agent = _make_agent()
        agent.db_manager.get_connections.return_value = {}

        result = agent.check_db()
        assert result["status"] == "error"

    def test_single_connection_tested(self):
        agent = _make_agent()
        mock_conn = MagicMock()
        agent.db_manager.get_connections.return_value = mock_conn  # not a dict

        result = agent.check_db()
        mock_conn.test_connection.assert_called_once()
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# probe_llm
# ---------------------------------------------------------------------------


class TestProbeLlm:
    def test_success(self):
        agent = _make_agent()
        mock_model = MagicMock()
        mock_model.model_config.type = "openai"
        mock_model.model_config.model = "gpt-4"
        mock_model.generate.return_value = "I can hear you!"

        with patch("datus.agent.agent.LLMBaseModel.create_model", return_value=mock_model):
            result = agent.probe_llm()

        assert result["status"] == "success"
        assert "response" in result

    def test_failure_returns_error(self):
        agent = _make_agent()

        with patch("datus.agent.agent.LLMBaseModel.create_model", side_effect=RuntimeError("no key")):
            result = agent.probe_llm()

        assert result["status"] == "error"
        assert "no key" in result["message"]


# ---------------------------------------------------------------------------
# run (delegates to runner)
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_run_delegates_to_runner(self, tmp_path):
        agent = _make_agent()

        mock_runner = MagicMock()
        mock_runner.run.return_value = {"status": "completed"}

        with patch.object(agent, "create_workflow_runner", return_value=mock_runner):
            result = agent.run(sql_task=SqlTask(task="test"), check_storage=False, check_db=False)

        mock_runner.run.assert_called_once()
        assert result == {"status": "completed"}


# ---------------------------------------------------------------------------
# run_stream (async, delegates to runner)
# ---------------------------------------------------------------------------


class TestAgentRunStream:
    @pytest.mark.asyncio
    async def test_run_stream_yields_actions(self):
        agent = _make_agent()

        async def _fake_stream(*args, **kwargs):
            yield ActionHistory(
                action_id="a1",
                role=ActionRole.WORKFLOW,
                messages="test",
                action_type="workflow_init",
                status=ActionStatus.SUCCESS,
            )

        mock_runner = MagicMock()
        mock_runner.run_stream = _fake_stream

        with patch.object(agent, "create_workflow_runner", return_value=mock_runner):
            actions = []
            async for action in agent.run_stream(sql_task=SqlTask(task="test")):
                actions.append(action)

        assert len(actions) == 1
        assert actions[0].action_id == "a1"
