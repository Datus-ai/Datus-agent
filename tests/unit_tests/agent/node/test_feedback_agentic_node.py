# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for FeedbackAgenticNode.

Tests cover:
- Node initialization and tool setup
- Session copy mechanism
- Streaming execution with MockLLMModel
- Input validation
- Storage info extraction
- Node factory integration

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real Tools (FilesystemFuncTool)
- Real PromptManager (using built-in templates)
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput, FeedbackNodeResult
from tests.unit_tests.mock_llm_model import build_simple_response

# ---------------------------------------------------------------------------
# Schema Model Tests
# ---------------------------------------------------------------------------


class TestFeedbackNodeModels:
    """Tests for FeedbackNodeInput and FeedbackNodeResult."""

    def test_input_minimal(self):
        inp = FeedbackNodeInput(user_message="analyze and archive")
        assert inp.user_message == "analyze and archive"
        assert inp.source_session_id is None
        assert inp.database is None

    def test_input_full(self):
        inp = FeedbackNodeInput(
            user_message="analyze",
            source_session_id="chat_session_abc123",
            database="test_db",
        )
        assert inp.source_session_id == "chat_session_abc123"
        assert inp.database == "test_db"

    def test_result_minimal(self):
        result = FeedbackNodeResult(success=True, response="Done")
        assert result.items_saved == 0
        assert result.storage_summary is None
        assert result.tokens_used == 0

    def test_result_full(self):
        result = FeedbackNodeResult(
            success=True,
            response="Archived 3 items",
            items_saved=3,
            storage_summary={"ext_knowledge": 2, "sql_summary": 1},
            tokens_used=1500,
        )
        assert result.items_saved == 3
        assert result.storage_summary["ext_knowledge"] == 2

    def test_result_error(self):
        result = FeedbackNodeResult(
            success=False,
            error="Template not found",
            response="Sorry, error occurred.",
        )
        assert result.success is False
        assert result.error == "Template not found"


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestFeedbackAgenticNodeInit:
    """Tests for FeedbackAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.get_node_name() == "feedback"
        assert node.configured_node_name == "feedback"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.id == "feedback_node"

    def test_node_type(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.configuration.node_type import NodeType

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.type == NodeType.TYPE_FEEDBACK

    def test_setup_tools_includes_filesystem(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_setup_tools_includes_task(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "task" in tool_names

    def test_workflow_mode_no_ask_user(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" not in tool_names

    def test_interactive_mode_has_ask_user(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="interactive")
        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" in tool_names

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 30

    def test_execution_mode_stored(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.execution_mode == "workflow"


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestFeedbackAgenticNodeExecution:
    """Tests for FeedbackAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_simple_response(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("I analyzed the conversation and found nothing worth archiving."),
            ]
        )

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze and archive this conversation")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_input_not_set_raises(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = None

        action_manager = ActionHistoryManager()
        with pytest.raises(ValueError, match="Feedback input not set"):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            raise ExecutionInterrupted("User pressed ESC")
            yield  # noqa: makes this an async generator

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_execution_error_yields_error_action(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"

    @pytest.mark.asyncio
    async def test_result_set_on_success(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Feedback analysis complete."),
            ]
        )

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert node.result is not None
        assert isinstance(node.result, FeedbackNodeResult)
        assert node.result.success is True

    @pytest.mark.asyncio
    async def test_result_set_on_error(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        async def _raise(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # noqa

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _raise

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert node.result is not None
        assert node.result.success is False
        assert "boom" in node.result.error


# ---------------------------------------------------------------------------
# Storage Info Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractStorageInfo:
    """Tests for _extract_storage_info method."""

    def test_no_actions_returns_zero(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        items_saved, summary = node._extract_storage_info("response text")
        assert items_saved == 0
        assert summary is None

    def test_counts_successful_task_actions(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

        # Simulate task tool actions in node.actions
        node.actions = [
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="task",
                messages="Tool call: task",
                input_data={"arguments": json.dumps({"type": "gen_ext_knowledge", "prompt": "test"})},
                status=ActionStatus.SUCCESS,
            ),
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="task",
                messages="Tool call: task",
                input_data={"arguments": json.dumps({"type": "gen_sql_summary", "prompt": "test"})},
                status=ActionStatus.SUCCESS,
            ),
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="read_file",
                messages="Tool call: read_file",
                input_data={},
                status=ActionStatus.SUCCESS,
            ),
        ]

        items_saved, summary = node._extract_storage_info("")
        assert items_saved == 2
        assert summary == {"ext_knowledge": 1, "sql_summary": 1}


# ---------------------------------------------------------------------------
# NodeType and Node Factory Tests
# ---------------------------------------------------------------------------


class TestFeedbackNodeType:
    """Tests for NodeType integration with feedback."""

    def test_type_input_feedback(self):
        from datus.configuration.node_type import NodeType

        inp = NodeType.type_input(
            NodeType.TYPE_FEEDBACK,
            {"user_message": "analyze conversation"},
        )
        assert isinstance(inp, FeedbackNodeInput)
        assert inp.user_message == "analyze conversation"

    def test_feedback_in_action_types(self):
        from datus.configuration.node_type import NodeType

        assert NodeType.TYPE_FEEDBACK in NodeType.ACTION_TYPES

    def test_feedback_in_descriptions(self):
        from datus.configuration.node_type import NodeType

        assert NodeType.TYPE_FEEDBACK in NodeType.NODE_TYPE_DESCRIPTIONS
        desc = NodeType.get_description(NodeType.TYPE_FEEDBACK)
        assert "feedback" in desc.lower() or "archival" in desc.lower()

    def test_node_factory_creates_feedback(self, real_agent_config, mock_llm_create):
        from datus.agent.node import Node
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.configuration.node_type import NodeType

        node = Node.new_instance(
            node_id="test_feedback",
            description="Test feedback factory",
            node_type=NodeType.TYPE_FEEDBACK,
            input_data=None,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, FeedbackAgenticNode)
        assert node.execution_mode == "workflow"

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        from datus.agent.node import Node
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.configuration.node_type import NodeType

        input_data = FeedbackNodeInput(user_message="test input")
        node = Node.new_instance(
            node_id="test_feedback",
            description="Test feedback factory",
            node_type=NodeType.TYPE_FEEDBACK,
            input_data=input_data,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, FeedbackAgenticNode)
        assert node.input is not None
        assert node.input.user_message == "test input"


# ---------------------------------------------------------------------------
# Constants and Memory Tests
# ---------------------------------------------------------------------------


class TestFeedbackConstants:
    """Tests for feedback registration in constants and memory."""

    def test_feedback_in_sys_sub_agents(self):
        """feedback is a reserved system name (in SYS_SUB_AGENTS) even though
        it is a top-level node and not a task()-delegatable subagent."""
        from datus.utils.constants import SYS_SUB_AGENTS

        assert "feedback" in SYS_SUB_AGENTS

    def test_feedback_has_no_own_memory(self):
        """The feedback node updates the caller's memory, not its own."""
        from datus.utils.memory_loader import has_memory

        assert has_memory("feedback") is False

    def test_feedback_in_init_all(self):
        from datus.agent.node import __all__

        assert "FeedbackAgenticNode" in __all__
