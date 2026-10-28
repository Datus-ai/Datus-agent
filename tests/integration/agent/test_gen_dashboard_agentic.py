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
