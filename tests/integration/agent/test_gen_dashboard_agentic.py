# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for GenDashboardAgenticNode.

Tests the BI dashboard subagent with real Superset (Docker).
Initialization and tool wiring are covered by the deterministic component suite.
Real Superset/model execution is available with ``-m product_e2e``.
Requires:
- Superset running on localhost:8088 (docker compose -f ../datus-bi-adapters/datus-bi-superset/tests/integration/docker-compose.yml up -d)
- datus-bi-superset package installed
- LLM API key (DEEPSEEK_API_KEY) for manual real-model evaluation only
"""

import copy
import os

import pytest
import requests

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

SUPERSET_URL = os.environ.get("SUPERSET_URL", "http://localhost:8088")
SUPERSET_USER = os.environ.get("SUPERSET_USER", "admin")
SUPERSET_PASS = os.environ.get("SUPERSET_PASS", "admin")


def _is_superset_running() -> bool:
    """Check if Superset is reachable."""
    try:
        resp = requests.get(f"{SUPERSET_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def dashboard_agent_config():
    """Load acceptance config with gen_dashboard agentic node configured."""
    if not _is_superset_running():
        pytest.skip(f"Superset not reachable at {SUPERSET_URL}. Run docker compose up -d")
    pytest.importorskip("datus_bi_superset", reason="datus-bi-superset package not installed")

    from tests.conftest import load_acceptance_config

    config = load_acceptance_config(datasource="bird_school")
    config.rag_base_path = "tests/data"
    config.agentic_nodes = copy.deepcopy(config.agentic_nodes)

    # Ensure gen_dashboard agentic node is configured
    config.agentic_nodes["gen_dashboard"] = {
        "system_prompt": "gen_dashboard",
        "bi_platform": "superset",
        "max_turns": 30,
    }

    # Ensure dashboard config points to the running Superset
    from datus.configuration.agent_config import DashboardConfig

    config.dashboard_config["superset"] = DashboardConfig(
        platform="superset",
        api_base_url=SUPERSET_URL,
        username=SUPERSET_USER,
        password=SUPERSET_PASS,
        extra={"provider": "db"},
    )

    return config


@pytest.mark.nightly
@pytest.mark.product_e2e
class TestGenDashboardAgenticExecution:
    """Manual evaluation of execute_stream with real Superset + LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_list_dashboards(self, dashboard_agent_config):
        """execute_stream should list dashboards from real Superset."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

        node = GenDashboardAgenticNode(
            agent_config=dashboard_agent_config,
            execution_mode="workflow",
        )
        node.input = GenDashboardNodeInput(user_message="List all dashboards available in Superset.")

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"
        assert actions[0].role == ActionRole.USER
        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
        assert actions[-1].action_type == "gen_dashboard_response"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_creates_a_persisted_queryable_dashboard(self, dashboard_agent_config):
        """Create through the public node, then reload and independently verify BI data."""
        import uuid

        from rich.console import Console

        from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode
        from datus.cli.bootstrap_bi_picker import BootstrapBiPicker
        from datus.schemas.action_history import ActionHistoryManager
        from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput
        from datus.tools.db_tools.db_manager import DBManager
        from tests.integration.tools.test_bi_dashboard import (
            _create_adapter,
            _ensure_superset_test_database,
            _get_or_create_sales_dataset,
            _items_from_adapter_result,
        )

        assert os.environ.get("DEEPSEEK_API_KEY"), "P0 BI creation requires DEEPSEEK_API_KEY"
        config = dashboard_agent_config
        adapter = _create_adapter(
            BootstrapBiPicker(config, Console(log_path=False, force_terminal=False)),
            config,
            {"platform": "superset", "dashboard_url": SUPERSET_URL, "api_base_url": SUPERSET_URL},
        )
        history = ActionHistoryManager()
        node = GenDashboardAgenticNode(agent_config=config, execution_mode="workflow")
        title = f"P0 Revenue {uuid.uuid4().hex[:8]}"
        chart_title = f"{title} Chart"
        database_manager = None

        def saved_dashboards():
            return [
                item for item in _items_from_adapter_result(adapter.list_dashboards(search=title)) if item.name == title
            ]

        try:
            dataset = _get_or_create_sales_dataset(adapter, _ensure_superset_test_database(adapter))
            node.input = GenDashboardNodeInput(
                user_message=(
                    f"Create exactly one new Superset dashboard titled '{title}'. "
                    f"Use existing dataset id {dataset.id}, datus_nightly_bi_sales. "
                    f"Create one table chart titled '{chart_title}' showing SUM(revenue) grouped by region, "
                    "attach it to the dashboard, "
                    "and verify its data. Use the existing dataset; do not create a new dataset."
                )
            )
            actions = [action async for action in node.execute_stream(history)]
            assert node.result.success, node.result.error
            assert actions[-1].status == "success"
            successful = {action.action_type for action in actions if action.status == "success"}
            assert {"create_dashboard", "create_chart"} <= successful, successful

            # These fresh API reads prove persistence, separate from the model output.
            dashboards = saved_dashboards()
            assert len(dashboards) == 1
            saved = adapter.get_dashboard_info(dashboards[0].id)
            assert saved.name == title
            assert len(saved.chart_ids) == 1
            chart_id = str(saved.chart_ids[0])
            assert adapter.get_chart(chart_id).name == chart_title
            chart_data = adapter.get_chart_data(chart_id, limit=100)
            assert "region" in chart_data.columns
            values = [name for name in chart_data.columns if name != "region"]
            assert len(values) == 1, chart_data.columns
            actual = sorted((str(row["region"]), float(row[values[0]])) for row in chart_data.rows)

            database = copy.deepcopy(config.services.datasources["superset"])
            for name, value in {
                "host": os.getenv("SUPERSET_POSTGRES_HOST", "127.0.0.1"),
                "port": os.getenv("SUPERSET_POSTGRES_PORT", "5433"),
                "username": "superset",
                "password": "superset",
                "database": "superset_examples",
                "schema": "public",
            }.items():
                setattr(database, name, value)
            database_manager = DBManager({"superset": database})
            oracle = database_manager.get_conn("superset").execute_query(
                "SELECT region, SUM(revenue) AS revenue FROM public.datus_nightly_bi_sales GROUP BY region",
                result_format="list",
            )
            assert oracle.success, oracle.error
            expected = sorted((str(row["region"]), float(row["revenue"])) for row in oracle.sql_return)
            assert actual == expected
        finally:
            for dashboard in saved_dashboards():
                saved = adapter.get_dashboard_info(dashboard.id)
                for chart_id in saved.chart_ids:
                    if adapter.get_chart(chart_id).name == chart_title:
                        adapter.delete_chart(chart_id)
                adapter.delete_dashboard(dashboard.id)
            adapter.close()
            if database_manager is not None:
                database_manager.close()
