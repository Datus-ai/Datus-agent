# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for GenExtKnowledgeAgenticNode.

Tests the full external knowledge generation workflow with real LLM,
real database tools, filesystem tools, and verification logic.
"""

import pytest

from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.mark.nightly
class TestGenExtKnowledgeAgentic:
    """Integration tests for GenExtKnowledgeAgenticNode with real LLM."""

    def test_node_initialization(self, nightly_agent_config):
        """N8-01: Node initializes with correct tools including DB and context search."""
        node = GenExtKnowledgeAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_ext_knowledge"
        assert node.execution_mode == "workflow"
        assert node.hooks is None

        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names, f"Missing read_file, got: {tool_names}"
        assert "write_file" in tool_names, f"Missing write_file, got: {tool_names}"

        # Should have DB tools
        assert node.db_func_tool is not None, "Database func tool should be initialized"

        # Should have context search tools
        assert node.context_search_tools is not None, "Context search tools should be initialized"

        # verify_sql tool
        assert "verify_sql" in tool_names, f"Missing verify_sql tool, got: {tool_names}"

        logger.info(f"Node initialized with {len(node.tools)} tools: {tool_names}")

    def test_verification_state_initialization(self, nightly_agent_config):
        """N8-02: Verification state is properly initialized."""
        node = GenExtKnowledgeAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node._verification_passed is False
        assert node._last_verification_result is None
        assert node._verification_attempt_count == 0
        assert node.max_verification_retries == 3

    def test_interactive_mode_has_hooks(self, nightly_agent_config):
        """N8-03: Interactive mode initializes hooks."""
        node = GenExtKnowledgeAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=nightly_agent_config,
            execution_mode="interactive",
        )

        assert node.hooks is not None, "Interactive mode should have hooks"

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_execute_stream_generates_ext_knowledge(self, nightly_agent_config):
        """N8-04: execute_stream generates external knowledge for a business concept."""
        node = GenExtKnowledgeAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = ExtKnowledgeNodeInput(
            user_message=(
                "Generate external knowledge for the concept 'Eligible Free Rate' in California schools. "
                "This is the ratio of Free Meal Count (K-12) to total Enrollment (K-12) from the frpm table. "
                "The SQL to compute it is: SELECT `Free Meal Count (K-12)` * 1.0 / `Enrollment (K-12)` FROM frpm. "
                "Write the knowledge YAML file with appropriate search_text and explanation."
            ),
            question="What is the Eligible Free Rate for California schools?",
            search_text="Eligible Free Rate",
            gold_sql="SELECT `Free Meal Count (K-12)` * 1.0 / `Enrollment (K-12)` AS eligible_free_rate FROM frpm",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        assert (
            actions[-1].status == ActionStatus.SUCCESS
        ), f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_execute_stream_with_question(self, nightly_agent_config):
        """N8-05: execute_stream works with a business question input."""
        node = GenExtKnowledgeAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = ExtKnowledgeNodeInput(
            user_message="Define external knowledge for: what does 'Eligible Free Rate' mean for California schools?",
            question="What does Eligible Free Rate mean?",
            search_text="Eligible Free Rate",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS
