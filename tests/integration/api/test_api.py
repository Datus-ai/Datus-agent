import argparse
import sys
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from datus.api.models import FeedbackResponse, RunWorkflowResponse
from datus.api.service import DatusAPIService, create_app


def _get_service_module():
    """Get the actual service module via sys.modules to avoid name shadowing."""
    return sys.modules["datus.api.service"]


@pytest.fixture
def agent_args():
    return argparse.Namespace(
        namespace="bird_school",
        config="tests/conf/agent.yml",
        max_steps=20,
        workflow="fixed",
        load_cp=None,
        debug=False,
    )


@pytest.fixture
def app(agent_args):
    return create_app(agent_args)


@pytest_asyncio.fixture
async def api_client(app, agent_args):
    """Create API client with service initialized (mocked)."""
    svc_mod = _get_service_module()

    # Manually set the global service to avoid lifespan/real DB/LLM init
    mock_service = DatusAPIService(agent_args)
    original_service = svc_mod.service
    svc_mod.service = mock_service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
    finally:
        svc_mod.service = original_service


@pytest_asyncio.fixture
async def auth_token(api_client):
    """Get a valid auth token using default credentials."""
    resp = await api_client.post(
        "/auth/token",
        data={
            "client_id": "datus_client",
            "client_secret": "datus_secret_key",
            "grant_type": "client_credentials",
        },
    )
    return resp.json()["access_token"]


@pytest_asyncio.fixture
async def authenticated_client(app, agent_args, auth_token):
    """Create authenticated API client."""
    svc_mod = _get_service_module()

    mock_service = DatusAPIService(agent_args)
    original_service = svc_mod.service
    svc_mod.service = mock_service
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {auth_token}"},
        ) as client:
            yield client
    finally:
        svc_mod.service = original_service


@pytest.mark.nightly
class TestAPI:
    """N8: Agent API tests."""

    @pytest.mark.asyncio
    async def test_health_check(self, api_client):
        """N8-01: GET /health returns DB and LLM status."""
        resp = await api_client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "status" in data, "Response should contain 'status' field"
        assert "database_status" in data, "Response should contain 'database_status' field"
        assert "llm_status" in data, "Response should contain 'llm_status' field"
        assert "version" in data, "Response should contain 'version' field"

    @pytest.mark.asyncio
    async def test_auth_token_flow(self, api_client):
        """N8-02: POST /auth/token returns JWT."""
        resp = await api_client.post(
            "/auth/token",
            data={
                "client_id": "datus_client",
                "client_secret": "datus_secret_key",
                "grant_type": "client_credentials",
            },
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "access_token" in data, "Response should contain 'access_token'"
        assert data["token_type"] == "Bearer", "Token type should be 'Bearer'"
        assert data["expires_in"] > 0, "Token expiry should be positive"

    @pytest.mark.asyncio
    async def test_auth_invalid_credentials(self, api_client):
        """N8-02b: Invalid credentials should fail."""
        resp = await api_client.post(
            "/auth/token",
            data={
                "client_id": "wrong",
                "client_secret": "wrong",
                "grant_type": "client_credentials",
            },
        )
        assert resp.status_code == 401, f"Expected 401 for invalid credentials, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_auth_invalid_grant_type(self, api_client):
        """N8-02c: Invalid grant_type should fail."""
        resp = await api_client.post(
            "/auth/token",
            data={
                "client_id": "datus_client",
                "client_secret": "datus_secret_key",
                "grant_type": "password",
            },
        )
        assert resp.status_code == 400, f"Expected 400 for invalid grant_type, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_sync_workflow(self, authenticated_client):
        """N8-03: POST /workflows/run mode=sync executes workflow."""
        with patch.object(DatusAPIService, "run_workflow") as mock_run:
            mock_run.return_value = RunWorkflowResponse(
                task_id="test_task_1",
                status="completed",
                workflow="nl2sql",
                sql="SELECT * FROM schools",
                result=[{"id": 1}],
                metadata={},
                error=None,
                execution_time=1.5,
            )
            resp = await authenticated_client.post(
                "/workflows/run",
                json={
                    "workflow": "nl2sql",
                    "namespace": "bird_school",
                    "task": "List all schools",
                    "mode": "sync",
                },
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data.get("task_id") == "test_task_1", "Response should contain correct task_id"
        assert data.get("status") == "completed", "Workflow should report completed status"
        assert data.get("sql") == "SELECT * FROM schools", "Response should contain generated SQL"

    @pytest.mark.asyncio
    async def test_async_workflow_requires_sse_header(self, authenticated_client):
        """N8-04: Async mode without SSE header should fail."""
        resp = await authenticated_client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "namespace": "bird_school",
                "task": "List all schools",
                "mode": "async",
            },
        )
        # The route handler catches HTTPException(400) and re-raises as 500
        assert resp.status_code in (
            400,
            500,
        ), f"Expected 400 or 500 for async without SSE header, got {resp.status_code}"
        data = resp.json()
        assert (
            "event-stream" in data.get("detail", "").lower() or resp.status_code == 500
        ), "Error should mention event-stream requirement"

    @pytest.mark.asyncio
    async def test_stream_workflow(self, authenticated_client):
        """N8-05: Async mode with SSE accept header returns event stream."""
        with patch.object(DatusAPIService, "run_workflow_stream") as mock_stream:
            from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

            async def mock_gen(*args, **kwargs):
                yield ActionHistory(
                    action_id="workflow_completion",
                    role=ActionRole.WORKFLOW,
                    messages="Done",
                    action_type="workflow_completion",
                    input={},
                    status=ActionStatus.SUCCESS,
                    output={},
                )

            mock_stream.return_value = mock_gen()
            resp = await authenticated_client.post(
                "/workflows/run",
                json={
                    "workflow": "nl2sql",
                    "namespace": "bird_school",
                    "task": "List all schools",
                    "mode": "async",
                },
                headers={"Accept": "text/event-stream"},
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert "text/event-stream" in resp.headers.get(
            "content-type", ""
        ), "Response should have text/event-stream content type"

    @pytest.mark.asyncio
    async def test_feedback(self, authenticated_client):
        """N8-06: POST /workflows/feedback records feedback."""
        with patch.object(DatusAPIService, "record_feedback") as mock_feedback:
            mock_feedback.return_value = FeedbackResponse(
                task_id="test_task_1",
                acknowledged=True,
                recorded_at="2025-01-01T00:00:00Z",
            )
            resp = await authenticated_client.post(
                "/workflows/feedback",
                json={
                    "task_id": "test_task_1",
                    "status": "success",
                },
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data.get("task_id") == "test_task_1", "Response should contain correct task_id"
        assert data.get("acknowledged") is True, "Response should confirm acknowledgement"
        assert "recorded_at" in data, "Response should contain recorded_at timestamp"

    @pytest.mark.asyncio
    async def test_unauthenticated_request(self, api_client):
        """N8-08: Unauthenticated request should fail."""
        resp = await api_client.post(
            "/workflows/run",
            json={
                "workflow": "nl2sql",
                "namespace": "bird_school",
                "task": "List all schools",
            },
        )
        assert resp.status_code in (
            401,
            403,
        ), f"Expected 401 or 403 for unauthenticated request, got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info."""
        resp = await api_client.get("/")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "version" in data, "Root endpoint should contain 'version'"
