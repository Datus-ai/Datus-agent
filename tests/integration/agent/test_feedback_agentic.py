# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for FeedbackAgenticNode.

Tests the feedback node wiring with real prompts, real tools, and a real LLM —
verifying the conversation-feedback analysis path end-to-end. Archival is
delegated via the task() tool to gen_* subagents, so this suite focuses on the
feedback node's own orchestration rather than the downstream gen_* nodes
(covered by their own integration tests).
"""

import pytest

from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput, FeedbackNodeResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.mark.nightly
class TestFeedbackAgentic:
    """Integration tests for FeedbackAgenticNode with real LLM."""

    def test_node_initialization(self, nightly_agent_config):
        """Feedback node initializes with filesystem + task delegation tools."""
        node = FeedbackAgenticNode(agent_config=nightly_agent_config, execution_mode="workflow")

        assert node.get_node_name() == "feedback"
        assert node.execution_mode == "workflow"
        assert node.memory_enabled is False

        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names, f"Missing read_file, got: {tool_names}"
        assert "write_file" in tool_names, f"Missing write_file, got: {tool_names}"
        assert "task" in tool_names, f"Missing task tool, got: {tool_names}"
        assert "ask_user" not in tool_names, "workflow mode must not expose ask_user"

        assert node.sub_agent_task_tool is not None
        assert node.filesystem_func_tool is not None

    def test_factory_creates_feedback_node(self, nightly_agent_config):
        """Node factory wires feedback -> FeedbackAgenticNode."""
        from datus.agent.node import Node
        from datus.configuration.node_type import NodeType

        node = Node.new_instance(
            node_id="nightly_feedback",
            description="Nightly feedback factory check",
            node_type=NodeType.TYPE_FEEDBACK,
            input_data=None,
            agent_config=nightly_agent_config,
            tools=[],
        )
        assert isinstance(node, FeedbackAgenticNode)
        assert node.execution_mode == "workflow"

    @pytest.mark.asyncio
    async def test_execute_stream_completes(self, nightly_agent_config):
        """Running execute_stream with a short prompt returns a successful result.

        Uses a deliberately simple ask so the LLM is unlikely to fan out to
        gen_* subagents — the goal here is to exercise the real prompt/session
        wiring of the feedback node itself, not the archival subagents.
        """
        node = FeedbackAgenticNode(agent_config=nightly_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(
            user_message=(
                "The conversation history is empty. Reply with a single short "
                "sentence that nothing is worth archiving and do not call any tools."
            )
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert actions, "execute_stream must yield at least one action"
        assert node.result is not None
        assert isinstance(node.result, FeedbackNodeResult)
        assert node.result.success is True, f"feedback run failed: {node.result.error}"
        assert actions[-1].status != ActionStatus.FAILED
