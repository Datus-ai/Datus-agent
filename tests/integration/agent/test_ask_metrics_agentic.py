# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for AskMetricsAgenticNode (subagent ``subagent/ask_metrics.md``).

Ask Metrics is a documented core subagent with only unit coverage before this
suite. These tests drive the real metric-QA loop with a real LLM against a real
MetricFlow semantic adapter: the node lists the available metrics and queries
them to answer a question.

Fixture wiring (deterministic, zero-copy): the committed semantic model
``tests/data/semantic_models/bird_school/frpm.yml`` defines metrics over the
``frpm`` table of california_schools.sqlite. The MetricFlow adapter resolves
``semantic_models_path`` from ``semantic_layer.metricflow`` config when present
(``build_semantic_adapter_config`` uses ``setdefault``), so the test points that
key at the committed fixture dir on the *function-scoped* ``nightly_agent_config``
only — never the shared YAML, which would redirect gen_metrics /
gen_semantic_model away from their runtime project_root models.
"""

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
def ask_metrics_agent_config(nightly_agent_config):
    """nightly_agent_config with the MetricFlow adapter pointed at the committed
    frpm semantic model, so ask_metrics has a real, queryable metric catalog.

    Function-scoped: mutating ``semantic_layer_configs`` here cannot leak into
    other suites (nightly_agent_config is itself function-scoped)."""
    models_dir = str(FIXTURE_SEMANTIC_MODELS_DIR.resolve())
    assert (FIXTURE_SEMANTIC_MODELS_DIR / "frpm.yml").is_file(), (
        f"Missing committed fixture semantic model: {FIXTURE_SEMANTIC_MODELS_DIR / 'frpm.yml'}"
    )

    configs = dict(nightly_agent_config.semantic_layer_configs)
    metricflow_cfg = dict(configs.get("metricflow", {}))
    metricflow_cfg["semantic_models_path"] = models_dir
    configs["metricflow"] = metricflow_cfg
    nightly_agent_config.semantic_layer_configs = configs
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
    """Ask Metrics end-to-end against a real MetricFlow catalog + real LLM."""

    def test_metric_catalog_is_available(self, ask_metrics_agent_config):
        """No-LLM wiring check: the adapter loads the fixture and exposes metrics.

        This fails fast (and deterministically) if the semantic-model fixture or
        the ``semantic_models_path`` wiring regresses, instead of surfacing as a
        confusing LLM-run failure later."""
        node = _build_node(ask_metrics_agent_config)

        assert node.startup_error is None, f"ask_metrics adapter failed to start: {node.startup_error}"
        tool_names = {tool.name for tool in node.tools}
        assert "list_metrics" in tool_names, f"Missing list_metrics tool, got: {sorted(tool_names)}"
        assert "query_metrics" in tool_names, f"Missing query_metrics tool, got: {sorted(tool_names)}"

        result = node.semantic_tools.list_metrics(limit=50, offset=0)
        assert result.success == 1, f"list_metrics failed: {getattr(result, 'error', None)}"
        items = (result.result or {}).get("items", []) if isinstance(result.result, dict) else []
        metric_names = {item.get("name") for item in items}
        assert "school_count" in metric_names, f"Fixture metrics not loaded, got: {sorted(metric_names)}"

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
