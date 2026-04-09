"""CI-level tests for datus/api/routes/kb_routes.py.

All external dependencies are mocked. Zero API keys, zero network access required.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datus.api.models.kb_models import BootstrapKbEvent
from datus.api.routes.kb_routes import router
from datus.utils.exceptions import DatusException, ErrorCode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_datus_service():
    """Create a mock DatusService with agent_config and kb."""
    svc = MagicMock()
    svc.agent_config = MagicMock()
    svc.agent_config.home = "/tmp/test_home"
    svc.agent_config.document_configs = {}
    svc.kb = MagicMock()
    return svc


@pytest.fixture
def client(mock_datus_service):
    """Create a TestClient with mocked dependencies."""
    from datus.api.deps import get_datus_service

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_datus_service] = lambda: mock_datus_service
    with TestClient(app) as c:
        yield c


def _make_kb_events():
    """Return two sample BootstrapKbEvent instances."""
    return [
        BootstrapKbEvent(
            stream_id="s1",
            component="platform_doc",
            stage="task_started",
            timestamp="2025-01-01T00:00:00",
        ),
        BootstrapKbEvent(
            stream_id="s1",
            component="platform_doc",
            stage="task_completed",
            timestamp="2025-01-01T00:00:01",
        ),
    ]


# ---------------------------------------------------------------------------
# POST /api/v1/kb/bootstrap-docs
# ---------------------------------------------------------------------------


class TestBootstrapDocs:
    def test_bootstrap_docs_returns_sse_stream(self, client, mock_datus_service):
        """SSE stream is returned with correct media type and event lines."""
        events = _make_kb_events()

        async def mock_stream(*args, **kwargs):
            for event in events:
                yield event

        mock_datus_service.kb.bootstrap_doc_stream = mock_stream
        mock_datus_service.agent_config.document_configs = {"myplatform": MagicMock(type="github")}

        response = client.post(
            "/api/v1/kb/bootstrap-docs",
            json={"platform": "myplatform"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "task_started" in body
        assert "task_completed" in body
        # Each SSE block starts with "event:"
        assert body.count("event:") == 2

    def test_bootstrap_docs_unknown_platform_no_source_returns_422(self, client, mock_datus_service):
        """Unknown platform with no source → 422 because config is missing."""
        mock_datus_service.agent_config.document_configs = {}

        response = client.post(
            "/api/v1/kb/bootstrap-docs",
            json={"platform": "unknown"},
        )

        assert response.status_code == 422
        assert "unknown" in response.json()["detail"]

    def test_bootstrap_docs_known_platform_succeeds(self, client, mock_datus_service):
        """Platform present in document_configs → 200 SSE response."""
        events = _make_kb_events()

        async def mock_stream(*args, **kwargs):
            for event in events:
                yield event

        mock_datus_service.kb.bootstrap_doc_stream = mock_stream
        doc_cfg = MagicMock()
        doc_cfg.type = "website"
        mock_datus_service.agent_config.document_configs = {"snowflake": doc_cfg}

        response = client.post(
            "/api/v1/kb/bootstrap-docs",
            json={"platform": "snowflake"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_bootstrap_docs_local_path_traversal_returns_422(self, client, mock_datus_service):
        """Local source with path traversal → safe_resolve raises DatusException → 422."""
        # Platform must exist so we pass the first validation check
        doc_cfg = MagicMock()
        doc_cfg.type = "local"
        mock_datus_service.agent_config.document_configs = {"myplatform": doc_cfg}

        with patch(
            "datus.api.routes.kb_routes.safe_resolve",
            side_effect=DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message="Path '../../../etc/passwd' escapes the project root",
            ),
        ):
            response = client.post(
                "/api/v1/kb/bootstrap-docs",
                json={
                    "platform": "myplatform",
                    "source": "../../../etc/passwd",
                    "source_type": "local",
                },
            )

        assert response.status_code == 422
        assert "escapes" in response.json()["detail"]

    def test_bootstrap_docs_local_source_from_config_validates_path(self, client, mock_datus_service):
        """Local source type from config triggers path validation → DatusException → 422."""
        doc_cfg = MagicMock()
        doc_cfg.type = "local"
        mock_datus_service.agent_config.document_configs = {"testplatform": doc_cfg}

        with patch(
            "datus.api.routes.kb_routes.safe_resolve",
            side_effect=DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message="escapes the project root",
            ),
        ):
            response = client.post(
                "/api/v1/kb/bootstrap-docs",
                json={
                    "platform": "testplatform",
                    "source": "../secret",
                },
            )

        assert response.status_code == 422


class TestCancelDocBootstrap:
    def test_cancel_doc_bootstrap_success(self, client):
        """cancel_stream returns True → response success=True."""
        with patch("datus.api.routes.kb_routes.cancel_stream", return_value=True):
            response = client.post("/api/v1/kb/bootstrap-docs/my-stream-id/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["stream_id"] == "my-stream-id"
        assert data["data"]["cancelled"] is True

    def test_cancel_doc_bootstrap_unknown_stream(self, client):
        """cancel_stream returns False → response success=False."""
        with patch("datus.api.routes.kb_routes.cancel_stream", return_value=False):
            response = client.post("/api/v1/kb/bootstrap-docs/nonexistent-stream/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["data"]["cancelled"] is False


# ---------------------------------------------------------------------------
# POST /api/v1/kb/bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapKb:
    def _valid_bootstrap_payload(self):
        return {"components": ["metadata"]}

    def test_bootstrap_kb_returns_sse_stream(self, client, mock_datus_service):
        """bootstrap_stream async generator → 200 SSE with event lines."""
        events = _make_kb_events()

        async def mock_stream(*args, **kwargs):
            for event in events:
                yield event

        mock_datus_service.kb.bootstrap_stream = mock_stream

        response = client.post(
            "/api/v1/kb/bootstrap",
            json=self._valid_bootstrap_payload(),
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "task_started" in body
        assert "task_completed" in body

    def test_bootstrap_kb_path_validation_error(self, client, mock_datus_service):
        """safe_resolve raises DatusException for success_story path → 422."""
        with patch(
            "datus.api.routes.kb_routes.safe_resolve",
            side_effect=DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message="Path '../../etc/passwd' escapes the project root",
            ),
        ):
            response = client.post(
                "/api/v1/kb/bootstrap",
                json={
                    "components": ["semantic_model"],
                    "success_story": "../../etc/passwd",
                },
            )

        assert response.status_code == 422
        assert "escapes" in response.json()["detail"]

    def test_bootstrap_kb_missing_components_returns_422(self, client):
        """components field is required with min_length=1; empty list → 422."""
        response = client.post(
            "/api/v1/kb/bootstrap",
            json={"components": []},
        )

        assert response.status_code == 422

    def test_bootstrap_kb_invalid_strategy_returns_422(self, client):
        """strategy must be one of overwrite/check/incremental → 422 for invalid."""
        response = client.post(
            "/api/v1/kb/bootstrap",
            json={"components": ["metadata"], "strategy": "invalid_strategy"},
        )

        assert response.status_code == 422

    def test_bootstrap_kb_sse_format_contains_id_and_event(self, client, mock_datus_service):
        """Each yielded event produces SSE lines with id:, event:, data: prefixes."""
        events = _make_kb_events()

        async def mock_stream(*args, **kwargs):
            for event in events:
                yield event

        mock_datus_service.kb.bootstrap_stream = mock_stream

        response = client.post(
            "/api/v1/kb/bootstrap",
            json=self._valid_bootstrap_payload(),
        )

        assert response.status_code == 200
        body = response.text
        assert "id:" in body
        assert "event:" in body
        assert "data:" in body


class TestCancelBootstrap:
    def test_cancel_bootstrap_success(self, client):
        """cancel_stream returns True → success=True."""
        with patch("datus.api.routes.kb_routes.cancel_stream", return_value=True):
            response = client.post("/api/v1/kb/bootstrap/active-stream-id/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["stream_id"] == "active-stream-id"
        assert data["data"]["cancelled"] is True

    def test_cancel_bootstrap_unknown_stream(self, client):
        """cancel_stream returns False → success=False."""
        with patch("datus.api.routes.kb_routes.cancel_stream", return_value=False):
            response = client.post("/api/v1/kb/bootstrap/ghost-stream/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["data"]["cancelled"] is False
