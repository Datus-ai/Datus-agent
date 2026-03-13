# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for GenMetricsAgenticNode.

Tests the full metrics generation workflow with real LLM, real config,
and real tools (filesystem, generation, semantic).
"""

import pytest

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.mark.nightly
class TestGenMetricsAgentic:
    """Integration tests for GenMetricsAgenticNode with real LLM."""

    def test_node_initialization(self, nightly_agent_config):
        """N6-01: Node initializes with correct tools and configuration."""
        node = GenMetricsAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_metrics"
        assert node.execution_mode == "workflow"
        assert node.hooks is None  # No hooks in workflow mode

        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names, f"Missing read_file tool, got: {tool_names}"
        assert "write_file" in tool_names, f"Missing write_file tool, got: {tool_names}"
        assert "list_directory" in tool_names, f"Missing list_directory tool, got: {tool_names}"
        assert "check_semantic_object_exists" in tool_names, f"Missing check_semantic_object_exists, got: {tool_names}"
        assert "end_metric_generation" in tool_names, f"Missing end_metric_generation, got: {tool_names}"

        logger.info(f"Node initialized with {len(node.tools)} tools: {tool_names}")

    def test_interactive_mode_has_hooks(self, nightly_agent_config):
        """N6-02: Interactive mode initializes hooks."""
        node = GenMetricsAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="interactive",
        )

        assert node.hooks is not None, "Interactive mode should have hooks"

    @pytest.mark.asyncio
    async def test_execute_stream_generates_metric(self, nightly_agent_config):
        """N6-03: execute_stream generates a metric from existing data source."""
        node = GenMetricsAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message=(
                "Generate a derived metric called 'free_meal_rate' that calculates "
                "the ratio of total_free_meal_count_k12 to total_enrollment_k12 from the frpm data source. "
                "The frpm.yml semantic model already exists with measures: "
                "total_enrollment_k12, total_free_meal_count_k12. "
                "Read frpm.yml first to understand the existing data source, then create the metric. "
                "Use the end_metric_generation tool when done."
            ),
            max_turns=5,
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        # First action should be USER/PROCESSING
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        # Last action should be SUCCESS
        assert (
            actions[-1].status == ActionStatus.SUCCESS
        ), f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"

    @pytest.mark.asyncio
    async def test_execute_stream_uses_tools(self, nightly_agent_config):
        """N6-04: LLM invokes tools during metrics generation."""
        node = GenMetricsAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message=(
                "Read the frpm.yml data source file and generate a metric for total FRPM count. "
                "The frpm data source has a measure called total_frpm_count_k12 with SUM aggregation. "
                "Use the end_metric_generation tool when done."
            ),
            max_turns=5,
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Check that tool actions exist
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 1, (
            f"LLM should invoke at least one tool, got {len(tool_actions)} tool actions. "
            f"All action types: {[a.action_type for a in actions]}"
        )

        assert actions[-1].status == ActionStatus.SUCCESS
