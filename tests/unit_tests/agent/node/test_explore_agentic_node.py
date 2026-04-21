# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ExploreAgenticNode.

Tests cover node initialization, tool setup (read-only tools only),
execute_stream flow, and integration with SubAgentTaskTool.

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
"""

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.explore_agentic_node_models import ExploreNodeInput
from tests.unit_tests.mock_llm_model import MockLLMModel, MockToolCall, build_simple_response, build_tool_then_response


class TestExploreAgenticNodeInit:
    """Tests for ExploreAgenticNode initialization."""

    def test_explore_inherits_from_agentic_node(self, real_agent_config, mock_llm_create):
        """ExploreAgenticNode should inherit from AgenticNode."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_1",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert isinstance(node, AgenticNode)

    def test_explore_is_not_gen_sql(self, real_agent_config, mock_llm_create):
        """ExploreAgenticNode should NOT be a GenSQLAgenticNode."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_2",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert not isinstance(node, GenSQLAgenticNode)

    def test_explore_node_name(self, real_agent_config, mock_llm_create):
        """Node name should be 'explore'."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_3",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert node.get_node_name() == "explore"

    def test_explore_default_max_turns(self, real_agent_config, mock_llm_create):
        """Default max_turns should be 15."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_4",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert node.max_turns == 15

    def test_explore_model_is_mock(self, real_agent_config, mock_llm_create):
        """Model should be the mock model."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_5",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert isinstance(node.model, MockLLMModel)


class TestExploreAgenticNodeTools:
    """Tests for ExploreAgenticNode tool setup (read-only enforcement)."""

    def test_explore_has_db_tools(self, real_agent_config, mock_llm_create):
        """After init, node should have database tools."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_tools_1",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert node.db_func_tool is not None
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names

    def test_explore_has_readonly_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Node should have read-only filesystem tools."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_tools_2",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        tool_names = [t.name for t in node.tools]
        # Read-only tools should be present
        assert "read_file" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names

    def test_explore_excludes_write_tools(self, real_agent_config, mock_llm_create):
        """Node should NOT have write/edit/create/move filesystem tools."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_tools_3",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        tool_names = [t.name for t in node.tools]
        # Write tools should NOT be present
        assert "write_file" not in tool_names
        assert "edit_file" not in tool_names

    def test_explore_has_date_parsing_tools(self, real_agent_config, mock_llm_create):
        """Node should have date parsing tools."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_tools_4",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        tool_names = [t.name for t in node.tools]
        assert "parse_temporal_expressions" in tool_names

    def test_explore_has_no_mcp_servers(self, real_agent_config, mock_llm_create):
        """Node should have no MCP servers (lightweight)."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_tools_5",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert node.mcp_servers == {}


@pytest.mark.acceptance
class TestExploreAgenticNodeExecution:
    """Tests for ExploreAgenticNode execute_stream."""

    @pytest.mark.asyncio
    async def test_explore_simple_response(self, real_agent_config, mock_llm_create):
        """execute_stream with simple text response produces USER and ASSISTANT actions."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Found 3 tables: orders, customers, products."),
            ]
        )

        node = ExploreAgenticNode(
            node_id="test_explore_exec_1",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )

        node.input = ExploreNodeInput(
            user_message="Find tables related to orders",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Should have at least USER + final ASSISTANT actions
        assert len(actions) >= 2
        # First action should be USER/PROCESSING
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        # Last action should be ASSISTANT/SUCCESS
        assert actions[-1].role == ActionRole.ASSISTANT
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_explore_with_tool_calls(self, real_agent_config, mock_llm_create):
        """execute_stream where LLM calls list_tables then responds."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                    ],
                    content="Found tables: satscores, schools, frpm.",
                ),
            ]
        )

        node = ExploreAgenticNode(
            node_id="test_explore_exec_2",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )

        node.input = ExploreNodeInput(
            user_message="List all available tables",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        roles = [a.role for a in actions]
        assert ActionRole.TOOL in roles
        assert ActionRole.USER in roles
        assert ActionRole.ASSISTANT in roles

        # Verify tool was actually called
        assert len(mock_llm_create.tool_results) >= 1
        tool_result = mock_llm_create.tool_results[0]
        assert tool_result["tool"] == "list_tables"
        assert tool_result["executed"] is True

    @pytest.mark.asyncio
    async def test_explore_result_has_response(self, real_agent_config, mock_llm_create):
        """Final result should contain the response text."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Schema summary: table satscores has columns cds, AvgScrRead."),
            ]
        )

        node = ExploreAgenticNode(
            node_id="test_explore_exec_3",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )

        node.input = ExploreNodeInput(
            user_message="Describe satscores table",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        async for _ in node.execute_stream(ahm):
            pass

        assert node.result is not None
        assert node.result.success is True
        assert "satscores" in node.result.response

    @pytest.mark.asyncio
    async def test_explore_no_input_raises(self, real_agent_config, mock_llm_create):
        """execute_stream should raise DatusException if no input is set."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.utils.exceptions import DatusException

        node = ExploreAgenticNode(
            node_id="test_explore_exec_4",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )

        ahm = ActionHistoryManager()
        with pytest.raises(DatusException):
            async for _ in node.execute_stream(ahm):
                pass


class TestExploreNodeTypeRegistration:
    """Tests for ExploreAgenticNode type registration."""

    def test_type_explore_exists(self):
        """TYPE_EXPLORE should be defined in NodeType."""
        assert hasattr(NodeType, "TYPE_EXPLORE")
        assert NodeType.TYPE_EXPLORE == "explore"

    def test_type_explore_in_action_types(self):
        """TYPE_EXPLORE should be in ACTION_TYPES list."""
        assert NodeType.TYPE_EXPLORE in NodeType.ACTION_TYPES

    def test_type_explore_has_description(self):
        """TYPE_EXPLORE should have a description."""
        desc = NodeType.get_description(NodeType.TYPE_EXPLORE)
        assert "explore" in desc.lower() or "exploration" in desc.lower()

    def test_type_input_explore(self):
        """NodeType.type_input should handle explore type."""
        inp = NodeType.type_input(
            NodeType.TYPE_EXPLORE,
            {"user_message": "test explore"},
        )
        assert isinstance(inp, ExploreNodeInput)
        assert inp.user_message == "test explore"

    def test_node_factory_creates_explore(self, real_agent_config, mock_llm_create):
        """Node.new_instance should create ExploreAgenticNode for TYPE_EXPLORE."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.agent.node.node import Node

        node = Node.new_instance(
            node_id="factory_test",
            description="Factory test",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        assert isinstance(node, ExploreAgenticNode)


class TestExploreUpdateContext:
    """Tests for ExploreAgenticNode update_context (should be no-op)."""

    def test_update_context_is_noop(self, real_agent_config, mock_llm_create):
        """update_context should return success but not modify workflow."""
        from unittest.mock import MagicMock

        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_ctx",
            description="Test Explore node",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )

        mock_workflow = MagicMock()
        result = node.update_context(mock_workflow)
        assert result["success"] is True
        assert "read-only" in result["message"]


class TestExploreSystemPromptCurrentDate:
    """Verify current_date is injected into the system prompt."""

    def test_system_prompt_contains_current_date(self, real_agent_config, mock_llm_create):
        from unittest.mock import patch

        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        node = ExploreAgenticNode(
            node_id="test_explore_date",
            description="Test current_date",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )

        with patch(
            "datus.utils.time_utils.get_default_current_date",
            return_value="2025-06-15",
        ):
            prompt = node._get_system_prompt()
        assert "2025-06-15" in prompt
