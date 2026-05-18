# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Grafana BI tool integration tests.

These tests exercise the Datus BI tool layer against a real Grafana compose
environment without involving an LLM.  The adapter's own repository owns the
deeper Grafana contract tests; this suite verifies Datus wires the adapter,
service config, and function-tool envelope correctly.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from datus.configuration.agent_config import DashboardConfig, DatasetDbConfig, DbConfig
from datus.tools.func_tool.bi_tools import BIFuncTool
from tests.conftest import load_acceptance_config

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.environ.get("GRAFANA_USER", "admin")
GRAFANA_PASS = os.environ.get("GRAFANA_PASS", "admin123")
GRAFANA_POSTGRES_HOST = os.environ.get("GRAFANA_POSTGRES_HOST", "127.0.0.1")
GRAFANA_POSTGRES_PORT = os.environ.get("GRAFANA_POSTGRES_PORT", "5434")


def _is_grafana_running() -> bool:
    try:
        return httpx.get(f"{GRAFANA_URL.rstrip('/')}/api/health", timeout=5.0).is_success
    except Exception:
        return False


@pytest.fixture(scope="module")
def grafana_tool() -> BIFuncTool:
    if not _is_grafana_running():
        pytest.skip(f"Grafana not reachable at {GRAFANA_URL}. Run the Grafana compose service first.")
    pytest.importorskip("datus_bi_grafana", reason="datus-bi-grafana package not installed")

    config = load_acceptance_config(datasource="bird_school")
    config.services.datasources["grafana_postgres"] = DbConfig(
        type="postgresql",
        host=GRAFANA_POSTGRES_HOST,
        port=GRAFANA_POSTGRES_PORT,
        username="grafana",
        password="grafana",
        database="grafana_data",
        schema="public",
    )
    config.dashboard_config["grafana"] = DashboardConfig(
        platform="grafana",
        adapter_type="grafana",
        api_base_url=GRAFANA_URL,
        username=GRAFANA_USER,
        password=GRAFANA_PASS,
        dataset_db=DatasetDbConfig(
            datasource_ref="grafana_postgres",
            bi_database_name="datus-test-postgres",
        ),
        default=True,
    )

    tool = BIFuncTool(agent_config=config, bi_service="grafana")
    if hasattr(tool.adapter, "find_or_create_datasource"):
        tool.adapter.find_or_create_datasource(
            name="datus-test-postgres",
            db_type="grafana-postgresql-datasource",
            url="postgres:5432",
            user="grafana",
            password="grafana",
            database="grafana_data",
        )
    return tool


@pytest.mark.nightly
class TestGrafanaBIFuncTool:
    def test_list_datasets_uses_real_grafana_adapter(self, grafana_tool: BIFuncTool):
        result = grafana_tool.list_datasets(limit=10)

        assert result.success == 1, result.error
        assert isinstance(result.result, dict)
        assert isinstance(result.result["items"], list)
        names = [item["name"] for item in result.result["items"]]
        assert "datus-test-postgres" in names

    def test_dashboard_crud_uses_real_grafana_adapter(self, grafana_tool: BIFuncTool):
        title = f"[Datus-Nightly] Grafana Tool {uuid.uuid4().hex[:8]}"
        created_id = ""

        try:
            created = grafana_tool.create_dashboard(title=title, description="Created by Datus nightly")
            assert created.success == 1, created.error
            assert isinstance(created.result, dict)
            assert isinstance(created.result["id"], str)
            created_id = str(created.result["id"])
            assert created.result["name"] == title
            assert created.result["deliverable_target"]["platform"] == "grafana"
            assert created.result["deliverable_target"]["dashboard_id"] == created_id

            fetched = grafana_tool.get_dashboard(created_id)
            assert fetched.success == 1, fetched.error
            assert isinstance(fetched.result, dict)
            assert fetched.result["name"] == title

            listed = grafana_tool.list_dashboards(search=title, limit=10)
            assert listed.success == 1, listed.error
            assert isinstance(listed.result, dict)
            assert isinstance(listed.result["items"], list)
            assert any(str(item["id"]) == created_id for item in listed.result["items"])

            deleted = grafana_tool.delete_dashboard(created_id)
            created_id = ""
            assert deleted.success == 1, deleted.error
            assert deleted.result == {"deleted": True, "dashboard_id": str(created.result["id"])}
        finally:
            if created_id:
                grafana_tool.delete_dashboard(created_id)
