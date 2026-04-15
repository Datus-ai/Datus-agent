# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for GenDashboardAgenticNode.

Tests the BI dashboard subagent with real Superset (Docker) + real LLM.
Requires:
- Superset running on localhost:8088 (docker compose -f ../datus-bi-adapters/datus-bi-superset/tests/integration/docker-compose.yml up -d)
- datus-bi-superset package installed
- LLM API key (DEEPSEEK_API_KEY)
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
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    if not _is_superset_running():
        pytest.skip(f"Superset not reachable at {SUPERSET_URL}. Run docker compose up -d")
    try:
        import datus_bi_superset  # noqa: F401
    except ImportError:
        pytest.skip("datus-bi-superset package not installed")

    from tests.conftest import load_acceptance_config

    config = load_acceptance_config(namespace="bird_school")
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
        api_url=SUPERSET_URL,
        username=SUPERSET_USER,
        password=SUPERSET_PASS,
        extra={"provider": "db"},
    )

    return config


@pytest.mark.nightly
class TestGenDashboardAgenticInit:
    """Integration tests for GenDashboardAgenticNode initialization with real Superset."""

    def test_node_initialization_with_bi_tools(self, dashboard_agent_config):
        """Node initializes with BI tools connected to real Superset."""
        from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

        node = GenDashboardAgenticNode(
            agent_config=dashboard_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_dashboard"
        assert node.execution_mode == "workflow"
        assert node.bi_func_tool is not None, "BI func tool should be initialized with real Superset"

        tool_names = [tool.name for tool in node.tools]
        logger.info(f"GenDashboard node initialized with {len(node.tools)} tools: {tool_names}")

        # Read tools should always be present
        assert "list_dashboards" in tool_names, f"Missing list_dashboards, got: {tool_names}"
        assert "get_dashboard" in tool_names, f"Missing get_dashboard, got: {tool_names}"
        assert "list_charts" in tool_names, f"Missing list_charts, got: {tool_names}"
        assert "list_datasets" in tool_names, f"Missing list_datasets, got: {tool_names}"

        # Write tools should be present (Superset adapter supports all mixins)
        assert "create_dashboard" in tool_names, f"Missing create_dashboard, got: {tool_names}"
        assert "create_chart" in tool_names, f"Missing create_chart, got: {tool_names}"
        assert "create_dataset" in tool_names, f"Missing create_dataset, got: {tool_names}"

    def test_no_db_or_filesystem_tools(self, dashboard_agent_config):
        """GenDashboard node should NOT expose DB or filesystem tools."""
        from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

        node = GenDashboardAgenticNode(
            agent_config=dashboard_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" not in tool_names
        assert "describe_table" not in tool_names
        assert "read_query" not in tool_names
        assert "read_file" not in tool_names
        assert "write_file" not in tool_names


@pytest.mark.nightly
class TestGenDashboardAgenticExecution:
    """Integration tests for GenDashboardAgenticNode execute_stream with real Superset + LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_list_dashboards(self, dashboard_agent_config):
        """execute_stream should list dashboards from real Superset."""
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
