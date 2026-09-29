# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for AskMetricsAgenticNode (subagent ``subagent/ask_metrics.md``).

These tests drive the metric-QA loop against the bundled Dosi engine. The
deterministic test queries the fixture database; the second test also uses a
real LLM to answer a question.

The committed OSI model defines metrics over the ``frpm`` table of
california_schools.sqlite. Only this function-scoped fixture points Dosi at
the committed model directory, so other nightly suites can use their own
workspace models.
"""

import csv
import io
from pathlib import Path

import pytest

from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Committed fixture semantic model dir (tests/data/semantic_models/bird_school).
FIXTURE_SEMANTIC_MODELS_DIR = Path(__file__).parents[2] / "data" / "semantic_models" / "bird_school"


@pytest.fixture
def ask_metrics_agent_config(nightly_agent_config, monkeypatch):
    """Point the function-scoped Dosi config at the queryable FRPM model."""
    assert (FIXTURE_SEMANTIC_MODELS_DIR / "frpm.yml").is_file(), (
        f"Missing committed fixture semantic model: {FIXTURE_SEMANTIC_MODELS_DIR / 'frpm.yml'}"
    )
    monkeypatch.setattr(
        nightly_agent_config.path_manager,
        "semantic_model_path",
        lambda datasource: FIXTURE_SEMANTIC_MODELS_DIR,
    )
    return nightly_agent_config


def _build_node(agent_config) -> AskMetricsAgenticNode:
    return AskMetricsAgenticNode(
        node_id="ask_metrics_itest",
        description="Ask metrics integration test",
        node_type=NodeType.TYPE_ASK_METRICS,
        agent_config=agent_config,
        node_name="ask_metrics",
        execution_mode="workflow",
    )


@pytest.mark.nightly
@pytest.mark.product_e2e
class TestAskMetricsAgentic:
    """Ask Metrics against a real Dosi catalog and SQLite database."""

    def test_metric_catalog_is_available(self, ask_metrics_agent_config):
        """Dosi loads the fixture and executes a real metric query."""
        node = _build_node(ask_metrics_agent_config)

        assert node.startup_error is None, f"AskMetrics failed to start: {node.startup_error}"
        tool_names = {tool.name for tool in node.tools}
        assert "list_metrics" in tool_names, f"Missing list_metrics tool, got: {sorted(tool_names)}"
        assert "query_metrics" in tool_names, f"Missing query_metrics tool, got: {sorted(tool_names)}"

        result = node.semantic_tools.list_metrics(limit=50, offset=0)
        assert result.success == 1, f"list_metrics failed: {getattr(result, 'error', None)}"
        items = (result.result or {}).get("items", []) if isinstance(result.result, dict) else []
        metric_names = {item.get("name") for item in items}
        assert "school_count" in metric_names, f"Fixture metrics not loaded, got: {sorted(metric_names)}"

        queried = node.semantic_tools.query_metrics(metrics=["school_count", "total_enrollment_k12"])
        assert queried.success == 1, f"query_metrics failed: {queried.error}"
        cached = node.semantic_tools.get_cached_query_metrics_result(queried.result["result_id"])
        assert cached["columns"] == ["school_count", "total_enrollment_k12"]
        assert cached["row_count"] == 1
        rows = list(csv.DictReader(io.StringIO(cached["csv"])))
        assert len(rows) == 1
        assert int(rows[0]["school_count"]) == 9986
        assert float(rows[0]["total_enrollment_k12"]) == 6199569.0

    @pytest.mark.asyncio
    async def test_answers_metric_question_end_to_end(self, ask_metrics_agent_config):
        """The agent lists metrics, queries one, and answers with a real number."""
        node = _build_node(ask_metrics_agent_config)
        node.input = AskMetricsNodeInput(
            user_message=(
                "Using the available metrics, how many distinct schools are in the FRPM dataset, "
                "and what is the total K-12 enrollment? List the metrics first, then query them."
            ),
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info("Action: role=%s status=%s type=%s", action.role, action.status, action.action_type)

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        # First action is the USER request entering the loop.
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        # The agent must have actually used the metric tools (not answered from
        # thin air): at least one successful query_metrics / list_metrics call.
        metric_tool_calls = [
            a
            for a in actions
            if a.role == ActionRole.TOOL
            and a.action_type in ("query_metrics", "list_metrics")
            and a.status == ActionStatus.SUCCESS
        ]
        assert metric_tool_calls, (
            f"ask_metrics should call list_metrics/query_metrics, got tool actions: "
            f"{[a.action_type for a in actions if a.role == ActionRole.TOOL]}"
        )

        # Terminal action is a successful completion.
        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )

        # The final answer should carry a real number from the query (the metric
        # values are in the millions / thousands), not just an acknowledgement.
        final_text = ""
        for action in reversed(actions):
            if action.role == ActionRole.ASSISTANT and action.output:
                final_text = str(action.output)
                break
        assert any(ch.isdigit() for ch in final_text), (
            f"Answer should contain a numeric metric value, got: {final_text[:500]}"
        )
