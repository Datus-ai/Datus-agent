# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-08: FastAPI endpoint contract tests.

Verifies request/response formats, authentication, and error codes
using TestClient with a fully mocked Agent backend.
No external dependencies (LLM, DB) required.
"""

import argparse
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

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
def legacy_auth_service(tmp_path, monkeypatch, request):
    import secrets

    import yaml

    from datus.api import legacy_auth

    service = importlib.import_module("datus.api.service")

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    path = tmp_path / "auth.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "clients": {"datus_client": "explicit-test-client-secret"},
                "jwt": {"secret_key": secrets.token_urlsafe(32)},
            }
        )
    )
    if getattr(request, "param", None) == "missing":
        path.unlink()
    elif getattr(request, "param", None) == "invalid":
        path.write_text("clients: [")
    configured = legacy_auth.AuthService(str(path))
    monkeypatch.setattr(legacy_auth, "auth_service", configured)
    monkeypatch.setattr(service, "auth_service", configured)
    return configured


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.current_datasource = "test_ns"
    config.rag_base_path = "/tmp/test_rag"
    config.api_config = {}
    return config


@pytest.fixture
def app(mock_agent_config, legacy_auth_service):
    """Create a FastAPI app with mocked dependencies."""
    agent_args = argparse.Namespace(
        datasource="test_ns",
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
def auth_token(legacy_auth_service):
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
                "client_secret": "explicit-test-client-secret",
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
                "client_secret": "explicit-test-client-secret",
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
                "datasource": "test_ns",
                "task": "show all tables",
            },
        )
        assert response.status_code in (401, 403)

    def test_workflow_invalid_token(self, client):
        response = client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "datasource": "test_ns",
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
                "datasource": "test_ns",
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
                "datasource": "test_ns",
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
        req = RunWorkflowRequest(workflow="nl2sql", datasource="test", task="query")
        assert req.mode == Mode.SYNC
        assert req.task_id is None
        assert req.subject_path is None

    def test_run_workflow_request_full(self):
        req = RunWorkflowRequest(
            workflow="nl2sql",
            datasource="test",
            task="query",
            mode=Mode.ASYNC,
            task_id="custom_id",
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            subject_path=["a", "b"],
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


class TestUnavailableLegacyAuth:
    @pytest.mark.parametrize("authorization", [None, "Bearer untrusted", "Basic untrusted"])
    @pytest.mark.parametrize("config_contents", [None, "", "{}", "clients: ["])
    def test_legacy_routes_fail_before_business_calls(
        self, client, monkeypatch, tmp_path, authorization, config_contents
    ):
        from datus.api import legacy_auth

        service = importlib.import_module("datus.api.service")

        path = tmp_path / "unavailable-auth.yml"
        if config_contents is not None:
            path.write_text(config_contents)
        disabled = legacy_auth.AuthService(str(path))
        monkeypatch.setattr(legacy_auth, "auth_service", disabled)
        monkeypatch.setattr(service, "auth_service", disabled)
        headers = {"Authorization": authorization} if authorization else {}
        with (
            patch.object(service.service, "run_workflow", new_callable=AsyncMock) as run,
            patch.object(service.service, "record_feedback", new_callable=AsyncMock) as feedback,
            patch.object(disabled, "generate_access_token") as token,
            patch.object(service, "generate_sse_stream") as stream,
        ):
            responses = [
                client.post(
                    "/auth/token",
                    data={
                        "client_id": "datus_client",
                        "client_secret": "datus_secret_key",
                        "grant_type": "client_credentials",
                    },
                ),
                client.post("/auth/token", data={}),
                client.post(
                    "/workflows/run",
                    json={"task": "blocked", "datasource": "test_ns", "workflow": "nl2sql"},
                    headers=headers,
                ),
                client.post(
                    "/workflows/run",
                    json={"task": "blocked", "datasource": "test_ns", "workflow": "nl2sql", "mode": "async"},
                    headers={**headers, "Accept": "text/event-stream"},
                ),
                client.post("/workflows/feedback", json={"task_id": "test", "status": "success"}, headers=headers),
            ]
            assert all(response.status_code == 503 for response in responses)
            assert all(response.json()["detail"] == legacy_auth.LEGACY_AUTH_UNAVAILABLE for response in responses)
            run.assert_not_called()
            feedback.assert_not_called()
            token.assert_not_called()
            stream.assert_not_called()

    def test_public_default_credentials_do_not_authenticate(self, client):
        response = client.post(
            "/auth/token",
            data={"client_id": "datus_client", "client_secret": "datus_secret_key", "grant_type": "client_credentials"},
        )
        assert response.status_code == 401


@pytest.mark.parametrize("legacy_auth_service", ["missing", "invalid"], indirect=True)
@pytest.mark.parametrize("custom_provider", [False, True])
def test_web_and_new_api_work_when_legacy_auth_is_disabled(app, legacy_auth_service, monkeypatch, custom_provider):
    from datus.api import deps
    from datus.api.auth.context import AppContext
    from datus.api.auth.header_context_provider import HeaderContextProvider
    from datus.cli.web.chatbot import create_web_app

    assert not legacy_auth_service.available
    service = importlib.import_module("datus.api.service")
    provider = HeaderContextProvider()
    if custom_provider:
        provider = MagicMock()
        provider.authenticate = AsyncMock(return_value=AppContext(user_id="configured-user"))
        provider.on_evict = MagicMock()
        monkeypatch.setattr(service, "load_auth_provider", lambda *args, **kwargs: provider)
    authenticate = AsyncMock(side_effect=provider.authenticate)
    monkeypatch.setattr(provider, "authenticate", authenticate)
    web = create_web_app(argparse.Namespace(datasource="test_ns", config=None))
    with TestClient(web) as browser:
        response = browser.get("/")
        assert response.status_code == 200 and "DatusChatbot" in response.text
        assert browser.get("/health").status_code == 200
        assert browser.post("/auth/token", data={}).status_code == 503
        facade = MagicMock()
        facade.agent_config = MagicMock()
        cache = MagicMock()
        cache.get_or_create = AsyncMock(return_value=facade)
        monkeypatch.setattr(deps, "_service_cache", cache)
        monkeypatch.setattr(deps, "_auth_provider", provider)
        with patch("datus.api.services.agent_service.AgentService.list_agents", new_callable=AsyncMock) as list_agents:
            list_agents.return_value = {"success": True, "data": {"agents": []}}
            response = browser.get("/api/v1/agent/list")
        assert response.status_code == 200, response.text
        assert cache.get_or_create.await_count == 1
        authenticate.assert_awaited_once()


def test_token_endpoint_token_reaches_both_legacy_handlers(client):
    service = importlib.import_module("datus.api.service")
    token = client.post(
        "/auth/token",
        data={
            "client_id": "datus_client",
            "client_secret": "explicit-test-client-secret",
            "grant_type": "client_credentials",
        },
    )
    assert token.status_code == 200
    headers = {"Authorization": "Bearer " + token.json()["access_token"]}
    with (
        patch.object(
            service.service, "run_workflow", new_callable=AsyncMock, return_value={"status": "completed"}
        ) as run,
        patch.object(service.service, "record_feedback", new_callable=AsyncMock) as feedback,
    ):
        feedback.return_value = FeedbackResponse(task_id="test", acknowledged=True, recorded_at="2026-01-01T00:00:00Z")
        response = client.post(
            "/workflows/run", json={"task": "test", "datasource": "test_ns", "workflow": "nl2sql"}, headers=headers
        )
        assert response.status_code == 200
        response = client.post("/workflows/feedback", json={"task_id": "test", "status": "success"}, headers=headers)
        assert response.status_code == 200
        run.assert_awaited_once()
        feedback.assert_awaited_once()


@pytest.mark.parametrize("failure", ["expired", "wrong_signature", "missing_identity"])
def test_invalid_jwt_never_reaches_workflow_or_feedback(client, legacy_auth_service, failure):
    import secrets
    from datetime import datetime, timedelta, timezone

    import jwt

    service = importlib.import_module("datus.api.service")
    payload = {"client_id": "datus_client", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    if failure == "expired":
        payload["exp"] = datetime.now(timezone.utc) - timedelta(hours=1)
    elif failure == "missing_identity":
        del payload["client_id"]
    key = secrets.token_urlsafe(32) if failure == "wrong_signature" else legacy_auth_service.jwt_secret
    token = jwt.encode(payload, key, algorithm="HS256")
    headers = {"Authorization": "Bearer " + token}
    with (
        patch.object(service.service, "run_workflow", new_callable=AsyncMock) as run,
        patch.object(service.service, "record_feedback", new_callable=AsyncMock) as feedback,
    ):
        assert (
            client.post(
                "/workflows/run",
                json={"task": "blocked", "datasource": "test_ns", "workflow": "nl2sql"},
                headers=headers,
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/workflows/feedback", json={"task_id": "test", "status": "success"}, headers=headers
            ).status_code
            == 401
        )
        run.assert_not_called()
        feedback.assert_not_called()
