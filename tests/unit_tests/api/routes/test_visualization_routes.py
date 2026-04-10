# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/routes/visualization_routes.py — CI level, zero external deps."""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datus.api.routes.visualization_routes import (
    _CHART_TYPE_MAP,
    _build_visualization_tool,
    router,
)

# ── helpers ──────────────────────────────────────────────────────

_VIZ_TOOL_PATH = "datus.api.routes.visualization_routes.VisualizationTool"
_LLM_MODEL_PATH = "datus.api.routes.visualization_routes.LLMBaseModel"


def _make_app() -> FastAPI:
    """Return a minimal FastAPI app with the visualization router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


def _mock_svc():
    svc = Mock()
    svc.agent_config = Mock()
    return svc


@pytest.fixture
def client():
    """TestClient with a mocked ServiceDep."""
    app = _make_app()
    # Override the service dependency
    from datus.api.deps import get_datus_service

    app.dependency_overrides[get_datus_service] = _mock_svc
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_payload():
    return {
        "csv_data": {
            "columns": ["date", "sales", "profit"],
            "data": [
                {"date": "2024-01-01", "sales": 100, "profit": 20},
                {"date": "2024-01-02", "sales": 150, "profit": 35},
            ],
        }
    }


# ═══════════════════════════════════════════════════════════════════
# 1. _CHART_TYPE_MAP
# ═══════════════════════════════════════════════════════════════════


class TestChartTypeMap:
    def test_all_known_types_mapped(self):
        assert _CHART_TYPE_MAP["Bar Chart"] == "Bar"
        assert _CHART_TYPE_MAP["Line Chart"] == "Line"
        assert _CHART_TYPE_MAP["Pie Chart"] == "Pie"
        assert _CHART_TYPE_MAP["Scatter Plot"] == "Scatter"
        assert _CHART_TYPE_MAP["Unknown"] == "Unknown"


# ═══════════════════════════════════════════════════════════════════
# 2. _build_visualization_tool
# ═══════════════════════════════════════════════════════════════════


class TestBuildVisualizationTool:
    def test_creates_tool_with_model(self):
        svc = _mock_svc()
        with patch(_LLM_MODEL_PATH) as mock_llm:
            mock_llm.create_model.return_value = Mock()
            tool = _build_visualization_tool(svc)
        assert tool is not None
        assert tool.model is not None

    def test_falls_back_when_model_creation_fails(self):
        svc = _mock_svc()
        with patch(_LLM_MODEL_PATH) as mock_llm:
            mock_llm.create_model.side_effect = Exception("no key")
            tool = _build_visualization_tool(svc)
        assert tool is not None
        assert tool.model is None


# ═══════════════════════════════════════════════════════════════════
# 3. POST /api/v1/data_visualization — success cases
# ═══════════════════════════════════════════════════════════════════


class TestDataVisualizationSuccess:
    def test_returns_line_chart(self, client, valid_payload):
        mock_result = Mock()
        mock_result.success = True
        mock_result.chart_type = "Line Chart"
        mock_result.x_col = "date"
        mock_result.y_cols = ["sales", "profit"]
        mock_result.reason = "Datetime column detected"

        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH) as mock_cls,
        ):
            mock_cls.return_value.execute.return_value = mock_result
            resp = client.post("/api/v1/data_visualization", json=valid_payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["data"]["chart_type"] == "Line"
        assert body["data"]["data"]["x_col"] == "date"
        assert body["data"]["data"]["y_cols"] == ["sales", "profit"]

    def test_returns_bar_chart(self, client):
        payload = {
            "csv_data": {
                "columns": ["category", "amount"],
                "data": [{"category": "A", "amount": 10}, {"category": "B", "amount": 20}],
            }
        }
        mock_result = Mock()
        mock_result.success = True
        mock_result.chart_type = "Bar Chart"
        mock_result.x_col = "category"
        mock_result.y_cols = ["amount"]
        mock_result.reason = "Categorical"

        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH) as mock_cls,
        ):
            mock_cls.return_value.execute.return_value = mock_result
            resp = client.post("/api/v1/data_visualization", json=payload)

        body = resp.json()
        assert body["success"] is True
        assert body["data"]["data"]["chart_type"] == "Bar"

    def test_returns_unknown_with_reason(self, client, valid_payload):
        mock_result = Mock()
        mock_result.success = True
        mock_result.chart_type = "Unknown"
        mock_result.x_col = ""
        mock_result.y_cols = []
        mock_result.reason = "Cannot determine chart"

        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH) as mock_cls,
        ):
            mock_cls.return_value.execute.return_value = mock_result
            resp = client.post("/api/v1/data_visualization", json=valid_payload)

        body = resp.json()
        assert body["success"] is True
        assert body["data"]["data"]["chart_type"] == "Unknown"
        assert body["data"]["data"]["reason"] == "Cannot determine chart"
        # Unknown should NOT include x_col / y_cols
        assert "x_col" not in body["data"]["data"]

    def test_caller_overrides_chart_type(self, client, valid_payload):
        valid_payload["chart_type"] = "Bar"

        mock_result = Mock()
        mock_result.success = True
        mock_result.chart_type = "Line Chart"  # tool says Line
        mock_result.x_col = "date"
        mock_result.y_cols = ["sales"]
        mock_result.reason = "Overridden"

        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH) as mock_cls,
        ):
            mock_cls.return_value.execute.return_value = mock_result
            resp = client.post("/api/v1/data_visualization", json=valid_payload)

        body = resp.json()
        assert body["success"] is True
        assert body["data"]["data"]["chart_type"] == "Bar"  # caller's choice


# ═══════════════════════════════════════════════════════════════════
# 4. POST /api/v1/data_visualization — error cases
# ═══════════════════════════════════════════════════════════════════


class TestDataVisualizationErrors:
    def test_empty_data(self, client):
        payload = {"csv_data": {"columns": [], "data": []}}
        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH),
        ):
            resp = client.post("/api/v1/data_visualization", json=payload)

        body = resp.json()
        assert body["success"] is False
        assert body["errorCode"] == "EMPTY_DATA"

    def test_tool_execution_exception(self, client, valid_payload):
        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH) as mock_cls,
        ):
            mock_cls.return_value.execute.side_effect = Exception("boom")
            resp = client.post("/api/v1/data_visualization", json=valid_payload)

        body = resp.json()
        assert body["success"] is False
        assert body["errorCode"] == "VISUALIZATION_FAILED"
        assert "boom" in body["errorMessage"]

    def test_tool_returns_failure(self, client, valid_payload):
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "LLM unavailable"

        with (
            patch(_LLM_MODEL_PATH),
            patch(_VIZ_TOOL_PATH) as mock_cls,
        ):
            mock_cls.return_value.execute.return_value = mock_result
            resp = client.post("/api/v1/data_visualization", json=valid_payload)

        body = resp.json()
        assert body["success"] is False
        assert body["errorCode"] == "VISUALIZATION_FAILED"

    def test_invalid_request_body(self, client):
        resp = client.post("/api/v1/data_visualization", json={"wrong": "shape"})
        assert resp.status_code == 422  # FastAPI validation error

    def test_missing_csv_data(self, client):
        resp = client.post("/api/v1/data_visualization", json={})
        assert resp.status_code == 422
