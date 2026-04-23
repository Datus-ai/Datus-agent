# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenTableAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (DBFuncTool + execute_ddl)
- Max turns configuration
- Streaming execution with MockLLMModel
- Input validation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite)
- Real Tools (DBFuncTool)
- Real PromptManager (using built-in templates)
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import MockToolCall, build_simple_response, build_tool_then_response

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenTableAgenticNodeInit:
    """Tests for GenTableAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.NODE_NAME == "gen_table"
        assert node.get_node_name() == "gen_table"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.id == "gen_table_node"

    def test_setup_tools_includes_ddl(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "execute_ddl" in tool_names

    def test_setup_tools_includes_standard_db_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names
        assert "get_table_ddl" in tool_names

    def test_setup_tools_includes_filesystem_tools(self, real_agent_config, mock_llm_create):
        """gen_table node should include filesystem tools for SQL artifact handling."""
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "check_semantic_object_exists" not in tool_names

    def test_max_turns_from_config(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        # real_agent_config has gen_table.max_turns = 10
        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 10

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        original = real_agent_config.agentic_nodes.pop("gen_table", None)
        try:
            node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
            assert node.max_turns == 20
        finally:
            if original is not None:
                real_agent_config.agentic_nodes["gen_table"] = original

    def test_workflow_mode_has_no_hooks(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.execution_mode == "workflow"

    def test_tool_category_map_buckets_db_and_filesystem(self, real_agent_config, mock_llm_create):
        """DB helpers (incl. ``execute_ddl``) land in ``db_tools`` and FS tools
        in ``filesystem_tools`` so profile rules gate them correctly."""
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        mapping = node._tool_category_map()
        db_names = {t.name for t in mapping.get("db_tools", [])}
        assert "execute_ddl" in db_names
        assert "read_query" in db_names
        assert "filesystem_tools" in mapping

    def test_interactive_mode(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="interactive")
        assert node.execution_mode == "interactive"

    def test_db_func_tool_initialized(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.db_func_tool is not None


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly
class TestGenTableAgenticNodeExecution:
    """Tests for GenTableAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_simple_response(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Table creation completed."),
            ]
        )

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Create a wide table from orders and customers")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_with_db_tool_calls(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="describe_table",
                            arguments=json.dumps({"table_name": "satscores"}),
                        ),
                    ],
                    content="I have examined the satscores table and proposed the wide table.",
                ),
            ]
        )

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Create wide table from satscores")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 4
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 2

        tool_results = mock_llm_create.tool_results
        assert len(tool_results) >= 1
        assert tool_results[0]["tool"] == "describe_table"
        assert tool_results[0]["executed"] is True

    @pytest.mark.asyncio
    async def test_input_not_set_raises(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = None

        action_manager = ActionHistoryManager()
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            raise ExecutionInterrupted("User pressed ESC")
            yield  # noqa: makes this an async generator

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Create table")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_execution_error_yields_error_action(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Create table")
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
    async def test_with_database_context(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Table created with database context."),
            ]
        )

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(
            user_message="Create wide table",
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

        assert len(mock_llm_create.call_history) >= 1
        call = mock_llm_create.call_history[0]
        prompt = call.get("prompt", "")
        assert "california_schools" in prompt


# ---------------------------------------------------------------------------
# Template Context Tests
# ---------------------------------------------------------------------------


class TestPrepareTemplateContextGenTable:
    def test_template_context_contains_required_keys(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        user_input = SemanticNodeInput(user_message="Create table")
        context = node._prepare_template_context(user_input)

        assert "native_tools" in context
        assert "mcp_tools" in context
        assert "has_ask_user_tool" in context

    def test_template_context_shows_ddl_tool(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        user_input = SemanticNodeInput(user_message="Create table")
        context = node._prepare_template_context(user_input)

        assert "execute_ddl" in context["native_tools"]


# ---------------------------------------------------------------------------
# NodeType and Node Factory Tests
# ---------------------------------------------------------------------------


class TestGenTableNodeType:
    """Tests for NodeType integration with gen_table."""

    def test_type_input_gen_table(self):
        """NodeType.type_input should handle TYPE_GEN_TABLE and return SemanticNodeInput."""
        from datus.configuration.node_type import NodeType

        inp = NodeType.type_input(
            NodeType.TYPE_GEN_TABLE,
            {"user_message": "create wide table"},
        )
        assert isinstance(inp, SemanticNodeInput)
        assert inp.user_message == "create wide table"

    def test_node_factory_creates_gen_table(self, real_agent_config, mock_llm_create):
        """Node.new_instance should create GenTableAgenticNode for TYPE_GEN_TABLE."""
        from datus.agent.node import Node
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.configuration.node_type import NodeType

        node = Node.new_instance(
            node_id="test_gen_table",
            description="Test gen_table factory",
            node_type=NodeType.TYPE_GEN_TABLE,
            input_data=None,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, GenTableAgenticNode)
        assert node.execution_mode == "workflow"

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        """Node.new_instance should set input when input_data is provided."""
        from datus.agent.node import Node
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.configuration.node_type import NodeType

        input_data = SemanticNodeInput(user_message="test input")
        node = Node.new_instance(
            node_id="test_gen_table",
            description="Test gen_table factory",
            node_type=NodeType.TYPE_GEN_TABLE,
            input_data=input_data,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, GenTableAgenticNode)
        assert node.input is not None
        assert node.input.user_message == "test input"
