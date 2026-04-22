# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for SchedulerAgenticNode.

Tests the scheduler subagent with real Airflow (Docker) + real LLM.
Requires:
- Airflow running on localhost:8080 (docker compose -f ../datus-scheduler-adapters/datus-scheduler-airflow/docker-compose.yml up -d)
- datus-scheduler-core package installed
- LLM API key (DEEPSEEK_API_KEY)
"""

import copy
import os

import pytest
import requests

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080/api/v1")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASSWORD", "admin123")


def _is_airflow_running() -> bool:
    """Check if Airflow is reachable."""
    try:
        resp = requests.get(f"{AIRFLOW_URL}/health", auth=(AIRFLOW_USER, AIRFLOW_PASS), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def scheduler_agent_config():
    """Load acceptance config with scheduler agentic node configured."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    if not _is_airflow_running():
        pytest.skip(f"Airflow not reachable at {AIRFLOW_URL}. Run docker compose up -d")
    pytest.importorskip("datus_scheduler_core")
    pytest.importorskip("datus_scheduler_airflow")

    from tests.conftest import load_acceptance_config

    config = load_acceptance_config(datasource="bird_school")
    config.rag_base_path = "tests/data"
    config.agentic_nodes = copy.deepcopy(config.agentic_nodes)

    # Ensure scheduler agentic node is configured
    config.agentic_nodes["scheduler"] = {
        "system_prompt": "scheduler",
        "max_turns": 30,
        "scheduler_service": "airflow_local",
    }

    # Ensure scheduler service config points to the running Airflow
    config.services.schedulers = {
        "airflow_local": {
            "name": "airflow_local",
            "type": "airflow",
            "api_base_url": AIRFLOW_URL,
            "username": AIRFLOW_USER,
            "password": AIRFLOW_PASS,
            "dags_folder": "/tmp/dags",
            "dag_discovery_timeout": 60,
            "dag_discovery_poll_interval": 5,
        }
    }
    config.init_scheduler_services(config.services.schedulers)

    return config


@pytest.mark.nightly
class TestSchedulerAgenticInit:
    """Integration tests for SchedulerAgenticNode initialization with real Airflow."""

    def test_node_initialization_with_scheduler_tools(self, scheduler_agent_config):
        """Node initializes with scheduler tools connected to real Airflow."""
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        node = SchedulerAgenticNode(
            agent_config=scheduler_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "scheduler"
        assert node.execution_mode == "workflow"
        assert node.scheduler_tools is not None, "Scheduler tools should be initialized with real Airflow"

        tool_names = [tool.name for tool in node.tools]
        logger.info(f"Scheduler node initialized with {len(node.tools)} tools: {tool_names}")

        # Core scheduler tools should be present
        assert "submit_sql_job" in tool_names, f"Missing submit_sql_job, got: {tool_names}"
        assert "list_scheduler_jobs" in tool_names, f"Missing list_scheduler_jobs, got: {tool_names}"
        assert "get_scheduler_job" in tool_names, f"Missing get_scheduler_job, got: {tool_names}"
        assert "trigger_scheduler_job" in tool_names, f"Missing trigger_scheduler_job, got: {tool_names}"

    def test_no_db_bi_filesystem_tools(self, scheduler_agent_config):
        """Scheduler node should NOT expose DB, BI, or filesystem tools."""
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        node = SchedulerAgenticNode(
            agent_config=scheduler_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" not in tool_names
        assert "read_query" not in tool_names
        assert "list_dashboards" not in tool_names
        assert "read_file" not in tool_names


@pytest.mark.nightly
class TestSchedulerAgenticExecution:
    """Integration tests for SchedulerAgenticNode execute_stream with real Airflow + LLM."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_list_jobs(self, scheduler_agent_config):
        """execute_stream should list scheduler jobs from real Airflow."""
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode
        from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
        from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

        node = SchedulerAgenticNode(
            agent_config=scheduler_agent_config,
            execution_mode="workflow",
        )
        node.input = SchedulerNodeInput(user_message="List all scheduled jobs currently in Airflow.")

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
        assert actions[-1].action_type == "scheduler_response"
