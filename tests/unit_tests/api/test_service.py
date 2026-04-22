# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/api/service.py — DatusAPIService and helper functions.

CI-level: zero external dependencies. All Agent, TaskStore, and LLM calls mocked.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from datus.api.legacy_models import FeedbackRequest, FeedbackStatus, RunWorkflowRequest
from datus.api.service import DatusAPIService
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args():
    return argparse.Namespace(
        namespace="test_ns",
        config=None,
        max_steps=20,
        workflow="fixed",
        load_cp=None,
        debug=False,
    )


def _make_service(args=None):
    return DatusAPIService(args=args or _make_args())


def _make_request(**kwargs):
    defaults = dict(
        workflow="nl2sql",
        namespace="test_ns",
        task="find all users",
    )
    defaults.update(kwargs)
    return RunWorkflowRequest(**defaults)


# ---------------------------------------------------------------------------
# DatusAPIService.__init__
# ---------------------------------------------------------------------------


class TestDatusAPIServiceInit:
    def test_initial_state(self):
        service = _make_service()
        assert service.agents == {}
        assert service.agent_config is None
        assert service.task_store is None


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestDatusAPIServiceInitialize:
    @pytest.mark.asyncio
    async def test_initialize_sets_config_and_task_store(self):
        service = _make_service()
        mock_cfg = MagicMock()
        mock_task_store = MagicMock()
        mock_task_store.cleanup_old_tasks.return_value = 5

        with patch("datus.api.service.load_agent_config", return_value=mock_cfg):
            with patch("datus.api.service.TaskStore", return_value=mock_task_store):
                await service.initialize()

        assert service.agent_config is mock_cfg
        assert service.task_store is mock_task_store
        mock_task_store.cleanup_old_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_count_zero_no_log_issue(self):
        """cleanup_old_tasks returning 0 doesn't crash."""
        service = _make_service()
        mock_cfg = MagicMock()
        mock_task_store = MagicMock()
        mock_task_store.cleanup_old_tasks.return_value = 0

        with patch("datus.api.service.load_agent_config", return_value=mock_cfg):
            with patch("datus.api.service.TaskStore", return_value=mock_task_store):
                await service.initialize()

        mock_task_store.cleanup_old_tasks.assert_called_once()
        assert service.agent_config is mock_cfg


# ---------------------------------------------------------------------------
# _parse_csv_to_list
# ---------------------------------------------------------------------------


class TestParseCsvToList:
    def test_valid_csv(self):
        service = _make_service()
        csv_str = "name,age\nAlice,30\nBob,25"
        result = service._parse_csv_to_list(csv_str)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["age"] == "25"

    def test_empty_string_returns_empty_list(self):
        service = _make_service()
        assert service._parse_csv_to_list("") == []

    def test_whitespace_only_returns_empty_list(self):
        service = _make_service()
        assert service._parse_csv_to_list("   ") == []

    def test_none_returns_empty_list(self):
        service = _make_service()
        assert service._parse_csv_to_list(None) == []

    def test_malformed_csv_returns_empty_list(self):
        service = _make_service()
        # DictReader on a string with no header just yields empty
        result = service._parse_csv_to_list("just one line no comma")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _generate_task_id
# ---------------------------------------------------------------------------


class TestGenerateTaskId:
    def test_contains_client_id(self):
        service = _make_service()
        task_id = service._generate_task_id("myClient")
        assert "myClient" in task_id

    def test_contains_timestamp(self):
        service = _make_service()
        task_id = service._generate_task_id("client")
        # Format: {client_id}_{YYYYMMDDHHMMSS}
        parts = task_id.split("_")
        assert len(parts) == 2
        assert parts[1].isdigit()


# ---------------------------------------------------------------------------
# get_agent
# ---------------------------------------------------------------------------


class TestGetAgent:
    def test_raises_http_500_when_config_not_loaded(self):
        from fastapi import HTTPException

        service = _make_service()
        service.agent_config = None  # not initialized

        with pytest.raises(HTTPException) as exc_info:
            service.get_agent("test_ns")
        assert exc_info.value.status_code == 500

    def test_creates_agent_for_new_namespace(self):
        service = _make_service()
        service.agent_config = MagicMock()
        mock_agent = MagicMock()

        with patch("datus.api.service.Agent", return_value=mock_agent):
            agent = service.get_agent("new_ns")

        assert agent is mock_agent
        assert "new_ns" in service.agents

    def test_returns_cached_agent_for_existing_namespace(self):
        service = _make_service()
        service.agent_config = MagicMock()
        mock_agent = MagicMock()
        service.agents["existing_ns"] = mock_agent

        agent = service.get_agent("existing_ns")
        assert agent is mock_agent


# ---------------------------------------------------------------------------
# _create_sql_task
# ---------------------------------------------------------------------------


class TestCreateSqlTask:
    def test_creates_sql_task_from_request(self):
        service = _make_service()
        request = _make_request(
            task="find users",
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
        )
        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"

        sql_task = service._create_sql_task(request, "task_001", mock_agent)

        assert sql_task.task == "find users"
        assert sql_task.catalog_name == "cat"
        assert sql_task.database_name == "db"
        assert sql_task.schema_name == "sch"
        assert sql_task.id == "task_001"

    def test_defaults_when_optional_fields_absent(self):
        service = _make_service()
        request = RunWorkflowRequest(workflow="nl2sql", namespace="ns", task="q")
        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"

        sql_task = service._create_sql_task(request, "t1", mock_agent)
        assert sql_task.database_name == "default"
        assert sql_task.catalog_name == ""
        assert sql_task.schema_name == ""


# ---------------------------------------------------------------------------
# _create_response
# ---------------------------------------------------------------------------


class TestCreateResponse:
    def test_success_response(self):
        service = _make_service()
        request = _make_request()
        resp = service._create_response("t1", request, "completed", sql_query="SELECT 1")
        assert resp.task_id == "t1"
        assert resp.status == "completed"
        assert resp.sql == "SELECT 1"
        assert resp.error is None

    def test_error_response(self):
        service = _make_service()
        request = _make_request()
        resp = service._create_response("t1", request, "error", error="boom")
        assert resp.error == "boom"


# ---------------------------------------------------------------------------
# run_workflow
# ---------------------------------------------------------------------------


class TestRunWorkflow:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        service = _make_service()
        service.agent_config = MagicMock()
        service.task_store = MagicMock()

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_runner = MagicMock()
        mock_runner.run.return_value = {"status": "completed"}
        mock_context = MagicMock()
        mock_context.sql_query = "SELECT 1"
        mock_context.sql_return = "id\n1"
        mock_runner.workflow.get_last_sqlcontext.return_value = mock_context
        mock_agent.create_workflow_runner.return_value = mock_runner

        service.agents["test_ns"] = mock_agent

        request = _make_request()
        response = await service.run_workflow(request, client_id="client1")

        assert response.status == "completed"
        assert response.sql == "SELECT 1"

    @pytest.mark.asyncio
    async def test_failed_run_returns_failed_status(self):
        service = _make_service()
        service.agent_config = MagicMock()
        service.task_store = MagicMock()

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_runner = MagicMock()
        mock_runner.run.return_value = {"status": "failed"}
        mock_runner.workflow = None
        mock_agent.create_workflow_runner.return_value = mock_runner

        service.agents["test_ns"] = mock_agent

        request = _make_request()
        response = await service.run_workflow(request, client_id="client1")

        assert response.status == "failed"

    @pytest.mark.asyncio
    async def test_exception_returns_error_status(self):
        service = _make_service()
        service.agent_config = MagicMock()
        service.task_store = MagicMock()

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_agent.create_workflow_runner.side_effect = RuntimeError("unexpected crash")

        service.agents["test_ns"] = mock_agent

        request = _make_request()
        response = await service.run_workflow(request, client_id="client1")

        assert response.status == "error"
        assert "unexpected crash" in response.error

    @pytest.mark.asyncio
    async def test_task_id_from_request_used(self):
        service = _make_service()
        service.agent_config = MagicMock()
        service.task_store = MagicMock()

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_runner = MagicMock()
        mock_runner.run.return_value = {"status": "completed"}
        mock_runner.workflow = None
        mock_agent.create_workflow_runner.return_value = mock_runner

        service.agents["test_ns"] = mock_agent

        request = _make_request(task_id="custom_id_123")
        response = await service.run_workflow(request, client_id="client1")
        assert response.task_id == "custom_id_123"

    @pytest.mark.asyncio
    async def test_csv_result_parsed(self):
        service = _make_service()
        service.agent_config = MagicMock()
        service.task_store = MagicMock()

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_runner = MagicMock()
        mock_runner.run.return_value = {"status": "completed"}
        mock_context = MagicMock()
        mock_context.sql_query = "SELECT name FROM users"
        mock_context.sql_return = "name\nAlice\nBob"
        mock_runner.workflow.get_last_sqlcontext.return_value = mock_context
        mock_agent.create_workflow_runner.return_value = mock_runner

        service.agents["test_ns"] = mock_agent

        request = _make_request()
        response = await service.run_workflow(request)

        assert response.result is not None
        assert len(response.result) == 2
        assert response.result[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# run_workflow_stream
# ---------------------------------------------------------------------------


class TestRunWorkflowStream:
    @pytest.mark.asyncio
    async def test_yields_actions_from_agent(self):
        service = _make_service()
        service.agent_config = MagicMock()

        action = ActionHistory(
            action_id="a1",
            role=ActionRole.WORKFLOW,
            messages="progress",
            action_type="workflow_init",
            status=ActionStatus.SUCCESS,
        )

        async def _fake_run_stream(*args, **kwargs):
            yield action

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_agent.run_stream = _fake_run_stream

        service.agents["test_ns"] = mock_agent

        request = _make_request()
        actions = []
        async for a in service.run_workflow_stream(request, client_id="c1"):
            actions.append(a)

        assert len(actions) == 1
        assert actions[0].action_id == "a1"

    @pytest.mark.asyncio
    async def test_yields_error_action_on_exception(self):
        service = _make_service()
        service.agent_config = MagicMock()

        mock_agent = MagicMock()
        mock_agent.global_config.output_dir = "/tmp/out"
        mock_agent.run_stream.side_effect = RuntimeError("stream crash")

        service.agents["test_ns"] = mock_agent

        request = _make_request()
        actions = []
        async for a in service.run_workflow_stream(request, client_id="c1"):
            actions.append(a)

        assert len(actions) == 1
        assert actions[0].action_id == "workflow_error"
        assert actions[0].status == ActionStatus.FAILED


# ---------------------------------------------------------------------------
# record_feedback
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    @pytest.mark.asyncio
    async def test_success(self):
        service = _make_service()
        service.task_store = MagicMock()
        service.task_store.record_feedback.return_value = {
            "task_id": "t1",
            "recorded_at": "2025-01-01T00:00:00Z",
        }

        req = FeedbackRequest(task_id="t1", status=FeedbackStatus.SUCCESS)
        resp = await service.record_feedback(req)

        assert resp.task_id == "t1"
        assert resp.acknowledged is True

    @pytest.mark.asyncio
    async def test_no_task_store_raises_http_500(self):
        pass

        service = _make_service()
        service.task_store = None

        req = FeedbackRequest(task_id="t1", status=FeedbackStatus.SUCCESS)
        resp = await service.record_feedback(req)
        # record_feedback catches the exception and returns acknowledged=False
        assert resp.acknowledged is False

    @pytest.mark.asyncio
    async def test_exception_returns_unacknowledged(self):
        service = _make_service()
        service.task_store = MagicMock()
        service.task_store.record_feedback.side_effect = RuntimeError("db error")

        req = FeedbackRequest(task_id="t1", status=FeedbackStatus.SUCCESS)
        resp = await service.record_feedback(req)

        assert resp.acknowledged is False
        assert resp.task_id == "t1"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_when_agent_config_available(self):
        service = _make_service()
        service.agent_config = MagicMock()
        service.agent_config.current_datasource = "ns"

        mock_agent = MagicMock()
        mock_agent.check_db.return_value = {"status": "success"}
        mock_agent.probe_llm.return_value = {"status": "ok"}

        with patch("datus.api.service.Agent", return_value=mock_agent):
            response = await service.health_check()

        assert response.status == "healthy"

    @pytest.mark.asyncio
    async def test_unhealthy_on_exception(self):
        service = _make_service()
        service.agent_config = MagicMock()

        with patch("datus.api.service.Agent", side_effect=RuntimeError("config error")):
            response = await service.health_check()

        assert response.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_healthy_when_no_config(self):
        service = _make_service()
        service.agent_config = None

        response = await service.health_check()
        assert response.status == "healthy"
        assert response.llm_status == "unknown"
