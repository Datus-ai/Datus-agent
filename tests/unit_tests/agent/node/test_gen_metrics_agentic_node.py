# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenMetricsAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (FilesystemFuncTool, GenerationTools, SemanticTools)
- Max turns configuration
- Streaming execution with MockLLMModel
- Filesystem tool invocation
- Thinking content in responses
- Input validation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real Storage/RAG (LanceDB in tmp_path)
- Real Tools (FilesystemFuncTool, GenerationTools, SemanticTools)
- Real PromptManager (using built-in templates)
- Real PathManager
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenMetricsAgenticNodeInit:
    """Tests for GenMetricsAgenticNode initialization."""

    def test_metrics_init(self, real_agent_config, mock_llm_create):
        """Test that GenMetricsAgenticNode initializes with real config."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_metrics"
        assert node.id == "gen_metrics_node"
        assert node.execution_mode == "workflow"
        assert node.hooks is None  # No hooks in workflow mode

    def test_metrics_has_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has filesystem and generation tools."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert len(node.tools) > 0
        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names
        assert "edit_file" in tool_names
        assert "read_multiple_files" in tool_names

        # Generation tools
        assert "check_semantic_object_exists" in tool_names
        assert "end_metric_generation" in tool_names

        # Tool instances should be initialized
        assert node.filesystem_func_tool is not None
        assert node.generation_tools is not None

    def test_metrics_max_turns(self, real_agent_config, mock_llm_create):
        """Test max_turns is read from agentic_nodes config."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        # The real_agent_config has gen_metrics.max_turns = 5
        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.max_turns == 5

    def test_metrics_max_turns_default(self, real_agent_config, mock_llm_create):
        """Test default max_turns is 30 when not configured."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        # Remove gen_metrics from agentic_nodes to test default
        original = real_agent_config.agentic_nodes.pop("gen_metrics", None)
        try:
            node = GenMetricsAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
            assert node.max_turns == 30
        finally:
            if original is not None:
                real_agent_config.agentic_nodes["gen_metrics"] = original


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestGenMetricsAgenticNodeExecution:
    """Tests for GenMetricsAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_metrics_simple_response(self, real_agent_config, mock_llm_create):
        """Test execute_stream with a simple text response."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Metrics generation completed successfully."),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have at least: USER action + LLM response + final action
        assert len(actions) >= 2

        # First action should be USER/PROCESSING
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        # Last action should be SUCCESS
        last_action = actions[-1]
        assert last_action.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_metrics_with_filesystem_tool(self, real_agent_config, mock_llm_create):
        """Test execute_stream where LLM calls write_file tool."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="write_file",
                            arguments=json.dumps(
                                {
                                    "path": "revenue_metrics.yml",
                                    "content": "metric:\n  name: revenue\n  type: simple",
                                }
                            ),
                        ),
                    ],
                    content="I have generated the revenue metrics file.",
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have: USER + TOOL start + TOOL complete + ASSISTANT response + final action
        assert len(actions) >= 4

        # Check tool actions exist
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 2  # 1 tool call x 2 (start + complete)

        tool_processing = [a for a in tool_actions if a.status == ActionStatus.PROCESSING]
        assert any(a.action_type == "write_file" for a in tool_processing)

        # Check the tool was actually executed
        tool_results = mock_llm_create.tool_results
        assert len(tool_results) >= 1
        assert tool_results[0]["tool"] == "write_file"
        assert tool_results[0]["executed"] is True

    @pytest.mark.asyncio
    async def test_metrics_workflow_mode(self, real_agent_config, mock_llm_create):
        """Test node in workflow mode has no hooks."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Done generating metrics."),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.hooks is None
        assert node.execution_mode == "workflow"

        node.input = SemanticNodeInput(user_message="Generate metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Execution should succeed in workflow mode
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_metrics_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Test that execute_stream raises when input is not set."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = None

        action_manager = ActionHistoryManager()
        with pytest.raises(ValueError, match="Metrics input not set"):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_metrics_with_thinking(self, real_agent_config, mock_llm_create):
        """Test response with thinking content yields a thinking action."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response(
                    content="Generated revenue metrics.",
                    thinking="I need to analyze the revenue data and create appropriate metrics.",
                ),
            ]
        )

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate revenue metrics")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have thinking action among the assistant actions
        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) >= 2  # thinking + response + final

        # Check that at least one action contains thinking text
        thinking_found = False
        for a in assistant_actions:
            if a.output and isinstance(a.output, dict):
                raw = a.output.get("raw_output", "")
                if "analyze the revenue data" in str(raw):
                    thinking_found = True
                    break
            if a.messages and "Thinking" in str(a.messages):
                thinking_found = True
                break
        assert thinking_found, "Expected thinking content in actions"
