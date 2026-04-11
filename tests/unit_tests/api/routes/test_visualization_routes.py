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


def _success_return(chart, data_insight=None):
    return {
        "success": True,
        "data": {"chart": chart, "data_insight": data_insight},
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Success cases
# ═══════════════════════════════════════════════════════════════════


class TestDataVisualizationSuccess:
    def test_returns_line_chart(self, valid_payload):
        client = _client_with(
            _success_return(
                chart={
                    "chart_type": "Line",
                    "columns": ["date", "sales", "profit"],
                    "numeric_columns": ["sales", "profit"],
                    "x_col": "date",
                    "y_cols": ["sales", "profit"],
                    "reason": "Datetime column detected",
                }
            )
        )
        resp = client.post("/api/v1/data_visualization", json=valid_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        chart = body["data"]["chart"]
        assert chart["chart_type"] == "Line"
        assert chart["x_col"] == "date"
        assert body["data"]["data_insight"] is None

    def test_returns_with_data_insight(self, valid_payload):
        client = _client_with(
            _success_return(
                chart={
                    "chart_type": "Bar",
                    "columns": ["date", "sales"],
                    "numeric_columns": ["sales"],
                    "x_col": "date",
                    "y_cols": ["sales"],
                    "reason": "ok",
                },
                data_insight={
                    "period": "2024-01-01 ~ 2024-01-02",
                    "filters": ["BP购买"],
                    "insight": "Sales grew.",
                },
            )
        )
        resp = client.post("/api/v1/data_visualization", json=valid_payload)
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["chart"]["chart_type"] == "Bar"
        di = body["data"]["data_insight"]
        assert di["period"] == "2024-01-01 ~ 2024-01-02"
        assert di["filters"] == ["BP购买"]
        assert di["insight"] == "Sales grew."


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


# ═══════════════════════════════════════════════════════════════════
# 3. Service delegation
# ═══════════════════════════════════════════════════════════════════


class TestServiceDelegation:
    def _make_svc_return(self):
        return _success_return(
            chart={
                "chart_type": "Bar",
                "columns": ["date", "sales"],
                "numeric_columns": ["sales"],
                "x_col": "date",
                "y_cols": ["sales"],
                "reason": "ok",
            }
        )

    def test_passes_all_params_to_service(self, valid_payload):
        from datus.api.deps import get_datus_service

        svc = _mock_svc(self._make_svc_return())
        app = _make_app()
        app.dependency_overrides[get_datus_service] = lambda: svc
        client = TestClient(app, raise_server_exceptions=False)

        valid_payload["chart_type"] = "Bar"
        valid_payload["sql"] = "SELECT date, sales FROM t"
        valid_payload["user_question"] = "Show me sales"
        client.post("/api/v1/data_visualization", json=valid_payload)

        kw = svc.visualization.generate.call_args.kwargs
        assert kw["chart_type"] == "Bar"
        assert kw["sql"] == "SELECT date, sales FROM t"
        assert kw["user_question"] == "Show me sales"

    def test_optional_params_default_to_none(self, valid_payload):
        from datus.api.deps import get_datus_service

        svc = _mock_svc(self._make_svc_return())
        app = _make_app()
        app.dependency_overrides[get_datus_service] = lambda: svc
        client = TestClient(app, raise_server_exceptions=False)

        client.post("/api/v1/data_visualization", json=valid_payload)

        kw = svc.visualization.generate.call_args.kwargs
        assert kw["chart_type"] is None
        assert kw["sql"] is None
        assert kw["user_question"] is None
