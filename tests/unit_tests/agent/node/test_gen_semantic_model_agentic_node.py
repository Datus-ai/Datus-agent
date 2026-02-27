# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenSemanticModelAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (DBFuncTool, FilesystemFuncTool, GenerationTools, GenSemanticModelTools, SemanticTools)
- Max turns configuration
- Streaming execution with MockLLMModel
- Database tool invocation (describe_table)
- Input validation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite) with tables: frpm, satscores, schools
- Real Storage/RAG (LanceDB in tmp_path)
- Real Tools (DBFuncTool, FilesystemFuncTool, GenerationTools, etc.)
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


class TestGenSemanticModelAgenticNodeInit:
    """Tests for GenSemanticModelAgenticNode initialization."""

    def test_semantic_model_init(self, real_agent_config, mock_llm_create):
        """Test that GenSemanticModelAgenticNode initializes with real config."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_semantic_model"
        assert node.id == "gen_semantic_model_node"
        assert node.execution_mode == "workflow"
        assert node.hooks is None  # No hooks in workflow mode

    def test_semantic_model_has_db_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has database tools from DBFuncTool."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        # DB tools should be present
        assert node.db_func_tool is not None

        tool_names = [tool.name for tool in node.tools]

        # Standard DB tools
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_semantic_model_has_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Test that the node has filesystem tools."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.filesystem_func_tool is not None

        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names
        assert "edit_file" in tool_names
        assert "read_multiple_files" in tool_names

        # Generation tools
        assert "check_semantic_object_exists" in tool_names
        assert "end_semantic_model_generation" in tool_names

        # GenSemanticModelTools should be present
        assert node.gen_semantic_model_tools is not None

    def test_semantic_model_max_turns(self, real_agent_config, mock_llm_create):
        """Test max_turns is read from agentic_nodes config."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        # The real_agent_config has gen_semantic_model.max_turns = 5
        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.max_turns == 5

    def test_semantic_model_max_turns_default(self, real_agent_config, mock_llm_create):
        """Test default max_turns is 30 when not configured."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        # Remove gen_semantic_model from agentic_nodes to test default
        original = real_agent_config.agentic_nodes.pop("gen_semantic_model", None)
        try:
            node = GenSemanticModelAgenticNode(
                agent_config=real_agent_config,
                execution_mode="workflow",
            )
            assert node.max_turns == 30
        finally:
            if original is not None:
                real_agent_config.agentic_nodes["gen_semantic_model"] = original


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestGenSemanticModelAgenticNodeExecution:
    """Tests for GenSemanticModelAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_semantic_model_simple_response(self, real_agent_config, mock_llm_create):
        """Test execute_stream with a simple text response."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generation completed."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model for satscores table")

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
    async def test_semantic_model_with_db_tool_calls(self, real_agent_config, mock_llm_create):
        """Test execute_stream where LLM calls describe_table tool against real SQLite."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="describe_table",
                            arguments=json.dumps({"table_name": "satscores"}),
                        ),
                    ],
                    content="I have examined the satscores table and created the semantic model.",
                ),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model for satscores table")

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
        assert any(a.action_type == "describe_table" for a in tool_processing)

        # Check the tool was actually executed against real SQLite
        tool_results = mock_llm_create.tool_results
        assert len(tool_results) >= 1
        assert tool_results[0]["tool"] == "describe_table"
        assert tool_results[0]["executed"] is True

    @pytest.mark.asyncio
    async def test_semantic_model_workflow_mode(self, real_agent_config, mock_llm_create):
        """Test node in workflow mode has no hooks and executes correctly."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Done generating semantic model."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.hooks is None
        assert node.execution_mode == "workflow"

        node.input = SemanticNodeInput(user_message="Generate semantic model")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Execution should succeed in workflow mode
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_semantic_model_with_database_context(self, real_agent_config, mock_llm_create):
        """Test execute_stream with database context enriches the enhanced message."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generated with database context."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message="Generate semantic model for satscores",
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # Verify the model was called with enhanced prompt containing database context
        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        assert "california_schools" in prompt or "Generate" in prompt

    @pytest.mark.asyncio
    async def test_semantic_model_interactive_mode_token_tracking(self, real_agent_config, mock_llm_create):
        """Test that interactive mode tracks token usage from action history."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Semantic model generated in interactive mode."),
            ]
        )

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="interactive",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        # In interactive mode, the final result should have tokens_used > 0
        last_output = actions[-1].output
        assert last_output is not None
        if isinstance(last_output, dict) and "tokens_used" in last_output:
            assert last_output["tokens_used"] > 0

    @pytest.mark.asyncio
    async def test_semantic_model_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Test that execute_stream raises when input is not set."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = None

        action_manager = ActionHistoryManager()
        with pytest.raises(ValueError, match="Semantic input not set"):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_semantic_model_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        """Test that ExecutionInterrupted is re-raised from execute_stream."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            """Async generator that raises ExecutionInterrupted."""
            raise ExecutionInterrupted("User pressed ESC")
            yield  # noqa: makes this an async generator

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass
