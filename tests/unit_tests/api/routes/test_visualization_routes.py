# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/routes/visualization_routes.py — CI level, zero external deps."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datus.api.routes.visualization_routes import router

# ── helpers ──────────────────────────────────────────────────────


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _mock_svc(generate_return=None):
    svc = Mock()
    svc.visualization.generate.return_value = generate_return or {}
    return svc


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


def _client_with(generate_return):
    """Return a TestClient whose ServiceDep.visualization.generate returns the given dict."""
    from datus.api.deps import get_datus_service

    app = _make_app()
    app.dependency_overrides[get_datus_service] = lambda: _mock_svc(generate_return)
    return TestClient(app, raise_server_exceptions=False)


# ═══════════════════════════════════════════════════════════════════
# 1. Success cases
# ═══════════════════════════════════════════════════════════════════


class TestDataVisualizationSuccess:
    def test_returns_line_chart(self, valid_payload):
        client = _client_with(
            {
                "success": True,
                "data": {
                    "data": {
                        "chart_type": "Line",
                        "columns": ["date", "sales", "profit"],
                        "numeric_columns": ["sales", "profit"],
                        "x_col": "date",
                        "y_cols": ["sales", "profit"],
                        "reason": "Datetime column detected",
                    }
                },
            }
        )
        resp = client.post("/api/v1/data_visualization", json=valid_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        chart = body["data"]["data"]
        assert chart["chart_type"] == "Line"
        assert chart["x_col"] == "date"
        assert chart["y_cols"] == ["sales", "profit"]
        assert chart["columns"] == ["date", "sales", "profit"]
        assert chart["numeric_columns"] == ["sales", "profit"]

    def test_returns_unknown_with_reason(self, valid_payload):
        client = _client_with(
            {
                "success": True,
                "data": {
                    "data": {
                        "chart_type": "Unknown",
                        "columns": ["date", "sales", "profit"],
                        "numeric_columns": ["sales", "profit"],
                        "reason": "Cannot determine chart",
                    }
                },
            }
        )
        resp = client.post("/api/v1/data_visualization", json=valid_payload)
        body = resp.json()
        assert body["success"] is True
        chart = body["data"]["data"]
        assert chart["chart_type"] == "Unknown"
        assert chart["reason"] == "Cannot determine chart"
        assert chart["x_col"] is None
        assert chart["y_cols"] is None


# ═══════════════════════════════════════════════════════════════════
# 2. Error cases
# ═══════════════════════════════════════════════════════════════════


class TestDataVisualizationErrors:
    def test_service_returns_failure(self, valid_payload):
        client = _client_with(
            {
                "success": False,
                "errorCode": "EMPTY_DATA",
                "errorMessage": "Provided dataset is empty or has no columns.",
            }
        )
        resp = client.post("/api/v1/data_visualization", json=valid_payload)
        body = resp.json()
        assert body["success"] is False
        assert body["errorCode"] == "EMPTY_DATA"

    def test_invalid_request_body(self):
        client = _client_with({})
        resp = client.post("/api/v1/data_visualization", json={"wrong": "shape"})
        assert resp.status_code == 422

    def test_missing_csv_data(self):
        client = _client_with({})
        resp = client.post("/api/v1/data_visualization", json={})
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════
# 3. Service delegation
# ═══════════════════════════════════════════════════════════════════


class TestServiceDelegation:
    def _make_success_return(self):
        return {
            "success": True,
            "data": {
                "data": {
                    "chart_type": "Bar",
                    "columns": ["date", "sales"],
                    "numeric_columns": ["sales"],
                    "x_col": "date",
                    "y_cols": ["sales"],
                    "reason": "ok",
                }
            },
        }

    def test_passes_chart_type_to_service(self, valid_payload):
        from datus.api.deps import get_datus_service

        svc = _mock_svc(self._make_success_return())
        app = _make_app()
        app.dependency_overrides[get_datus_service] = lambda: svc
        client = TestClient(app, raise_server_exceptions=False)

        valid_payload["chart_type"] = "Bar"
        client.post("/api/v1/data_visualization", json=valid_payload)

        call_kwargs = svc.visualization.generate.call_args.kwargs
        assert call_kwargs["chart_type"] == "Bar"

    def test_passes_sql_and_user_question_to_service(self, valid_payload):
        from datus.api.deps import get_datus_service

        svc = _mock_svc(self._make_success_return())
        app = _make_app()
        app.dependency_overrides[get_datus_service] = lambda: svc
        client = TestClient(app, raise_server_exceptions=False)

        valid_payload["sql"] = "SELECT date, sales FROM t"
        valid_payload["user_question"] = "Show me sales"
        client.post("/api/v1/data_visualization", json=valid_payload)

        call_kwargs = svc.visualization.generate.call_args.kwargs
        assert call_kwargs["sql"] == "SELECT date, sales FROM t"
        assert call_kwargs["user_question"] == "Show me sales"

    def test_sql_and_user_question_default_to_none(self, valid_payload):
        from datus.api.deps import get_datus_service

        svc = _mock_svc(self._make_success_return())
        app = _make_app()
        app.dependency_overrides[get_datus_service] = lambda: svc
        client = TestClient(app, raise_server_exceptions=False)

        client.post("/api/v1/data_visualization", json=valid_payload)

        call_kwargs = svc.visualization.generate.call_args.kwargs
        assert call_kwargs["sql"] is None
        assert call_kwargs["user_question"] is None
