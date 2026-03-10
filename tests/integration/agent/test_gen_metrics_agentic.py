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
    async def test_execute_stream_produces_actions(self, nightly_agent_config):
        """N6-03: execute_stream produces valid action sequence with real LLM."""
        node = GenMetricsAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message="List the files in the current directory to see existing metrics.",
            max_turns=3,
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
                "Please list the directory to check existing metric files, "
                "then generate a simple revenue metric YAML file."
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
