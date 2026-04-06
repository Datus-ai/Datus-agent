# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-08: FastAPI endpoint contract tests.

Verifies request/response formats, authentication, and error codes
using TestClient with a fully mocked Agent backend.
No external dependencies (LLM, DB) required.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from datus.api.legacy_models import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatus,
    HealthResponse,
    Mode,
    RunWorkflowRequest,
    RunWorkflowResponse,
    TokenResponse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.current_database = "test_ns"
    config.rag_base_path = "/tmp/test_rag"
    config.api_config = {}
    return config


@pytest.fixture
def app(mock_agent_config):
    """Create a FastAPI app with mocked dependencies."""
    agent_args = argparse.Namespace(
        namespace="test_ns",
        config=None,
        max_steps=20,
        workflow="fixed",
        load_cp=None,
        debug=False,
    )

    mock_task_store_cls = MagicMock()
    mock_task_store_instance = MagicMock()
    mock_task_store_instance.cleanup_old_tasks.return_value = 0
    mock_task_store_cls.return_value = mock_task_store_instance

    mock_agent_cls = MagicMock()
    mock_agent_instance = MagicMock()
    mock_agent_instance.global_config.output_dir = "/tmp/test_output"
    mock_runner = MagicMock()
    mock_runner.run.return_value = {"status": "completed"}
    mock_sql_context = MagicMock()
    mock_sql_context.sql_query = "SELECT 1"
    mock_sql_context.sql_return = None
    mock_runner.workflow.get_last_sqlcontext.return_value = mock_sql_context
    mock_agent_instance.create_workflow_runner.return_value = mock_runner
    mock_agent_cls.return_value = mock_agent_instance

    with (
        patch("datus.api.service.load_agent_config", return_value=mock_agent_config),
        patch("datus.api.service.TaskStore", mock_task_store_cls),
        patch("datus.api.service.Agent", mock_agent_cls),
    ):
        from datus.api.service import create_app

        application = create_app(agent_args)
        # Force lifespan initialization by using TestClient as context manager
        yield application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_token():
    """Generate a valid JWT token for testing."""
    from datus.api.legacy_auth import auth_service

    token_data = auth_service.generate_access_token("datus_client")
    return token_data["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_format(self, client):
        data = client.get("/").json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_matches_model(self, client):
        data = client.get("/health").json()
        # Should match HealthResponse schema
        assert "status" in data
        assert "version" in data
        assert "database_status" in data
        assert "llm_status" in data


# ---------------------------------------------------------------------------
# Auth endpoint
# ---------------------------------------------------------------------------


class TestAuthEndpoint:
    def test_auth_valid_credentials(self, client):
        response = client.post(
            "/auth/token",
            data={
                "client_id": "datus_client",
                "client_secret": "datus_secret_key",
                "grant_type": "client_credentials",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "Bearer"
        assert "expires_in" in data

    def test_auth_invalid_credentials(self, client):
        response = client.post(
            "/auth/token",
            data={
                "client_id": "wrong_client",
                "client_secret": "wrong_secret",
                "grant_type": "client_credentials",
            },
        )
        assert response.status_code == 401

    def test_auth_invalid_grant_type(self, client):
        response = client.post(
            "/auth/token",
            data={
                "client_id": "datus_client",
                "client_secret": "datus_secret_key",
                "grant_type": "password",
            },
        )
        assert response.status_code == 400
        assert "grant_type" in response.json()["detail"].lower()

    def test_auth_missing_fields(self, client):
        response = client.post("/auth/token", data={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Workflow endpoint - auth required
# ---------------------------------------------------------------------------


class TestWorkflowEndpoint:
    def test_workflow_requires_auth(self, client):
        response = client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "namespace": "test_ns",
                "task": "show all tables",
            },
        )
        assert response.status_code in (401, 403)

    def test_workflow_invalid_token(self, client):
        response = client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "namespace": "test_ns",
                "task": "show all tables",
            },
            headers={"Authorization": "Bearer invalid_token_here"},
        )
        assert response.status_code == 401

    def test_workflow_request_validation(self, client, auth_headers):
        # Missing required fields
        response = client.post(
            "/workflows/run",
            json={"workflow": "nl2sql"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_workflow_valid_sync_request(self, client, auth_headers):
        response = client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "namespace": "test_ns",
                "task": "show all tables",
                "mode": "sync",
            },
            headers=auth_headers,
        )
        # Valid request should reach the handler and return a success response
        assert response.status_code == 200, (
            f"Expected 200 for valid sync request, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["status"] != "error", f"Workflow should not return error status: {data.get('error')}"

    def test_workflow_async_requires_sse_accept(self, client, auth_headers):
        response = client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "namespace": "test_ns",
                "task": "show all tables",
                "mode": "async",
            },
            headers=auth_headers,
        )
        # Should fail because Accept header doesn't include text/event-stream
        assert response.status_code == 400, (
            f"Expected 400 for missing SSE Accept header, got {response.status_code}: {response.text}"
        )


# ---------------------------------------------------------------------------
# Feedback endpoint - auth required
# ---------------------------------------------------------------------------


class TestFeedbackEndpoint:
    def test_feedback_requires_auth(self, client):
        response = client.post(
            "/workflows/feedback",
            json={"task_id": "test_123", "status": "success"},
        )
        assert response.status_code in (401, 403)

    def test_feedback_request_validation(self, client, auth_headers):
        # Missing required fields
        response = client.post(
            "/workflows/feedback",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_feedback_invalid_status(self, client, auth_headers):
        response = client.post(
            "/workflows/feedback",
            json={"task_id": "test_123", "status": "invalid_status"},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestAPIModels:
    def test_run_workflow_request_defaults(self):
        req = RunWorkflowRequest(workflow="nl2sql", namespace="test", task="query")
        assert req.mode == Mode.SYNC
        assert req.task_id is None
        assert req.subject_path is None

    def test_run_workflow_request_full(self):
        req = RunWorkflowRequest(
            workflow="nl2sql",
            namespace="test",
            task="query",
            mode=Mode.ASYNC,
            task_id="custom_id",
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            subject_path=["a", "b"],
            ext_knowledge="some context",
        )
        assert req.mode == Mode.ASYNC
        assert req.task_id == "custom_id"
        assert req.subject_path == ["a", "b"]

    def test_run_workflow_response(self):
        resp = RunWorkflowResponse(
            task_id="t1",
            status="completed",
            workflow="nl2sql",
            sql="SELECT 1",
        )
        assert resp.task_id == "t1"
        assert resp.error is None

    def test_health_response(self):
        resp = HealthResponse(
            status="healthy",
            version="1.0.0",
            database_status={"test": "connected"},
            llm_status="ok",
        )
        assert resp.status == "healthy"

    def test_token_response(self):
        resp = TokenResponse(access_token="abc", token_type="Bearer", expires_in=7200)
        assert resp.expires_in == 7200

    def test_feedback_request(self):
        req = FeedbackRequest(task_id="t1", status=FeedbackStatus.SUCCESS)
        assert req.status == FeedbackStatus.SUCCESS

    def test_feedback_response(self):
        resp = FeedbackResponse(task_id="t1", acknowledged=True, recorded_at="2025-01-01T00:00:00Z")
        assert resp.acknowledged is True
