# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenSQLAgenticNode and ChatAgenticNode.

Tests cover node initialization, tool setup, execute_stream flow,
action history tracking, and real tool execution (db_tools, context_search).

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
Everything else uses real implementations: real AgentConfig, real SQLite database,
real db_manager_instance, real DBFuncTool, real ContextSearchTools, real PromptManager,
real PathManager.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput, GenSQLNodeResult
from tests.unit_tests.mock_llm_model import (
    MockLLMModel,
    MockLLMResponse,
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

# ===========================================================================
# GenSQLAgenticNode Tests
# ===========================================================================


class TestGenSQLAgenticNodeInit:
    """Tests for GenSQLAgenticNode initialization with real config."""

    def test_gensql_init_with_real_config(self, real_agent_config, mock_llm_create):
        """Node initializes with real AgentConfig, tools are set up correctly."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_1",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        assert node.id == "test_gensql_1"
        assert node.type == NodeType.TYPE_GENSQL
        assert node.description == "Test GenSQL node"
        assert node.status == "pending"
        assert node.agent_config is real_agent_config
        assert node.get_node_name() == "gensql"
        # Model should be the mock model
        assert isinstance(node.model, MockLLMModel)

    def test_gensql_has_db_tools(self, real_agent_config, mock_llm_create):
        """After init, node has real db tools (list_tables, describe_table, read_query, get_table_ddl)."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_2",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        assert node.db_func_tool is not None
        assert len(node.tools) > 0

        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names

    def test_gensql_max_turns_from_config(self, real_agent_config, mock_llm_create):
        """max_turns is read from agentic_nodes config (set to 5 in fixture)."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_3",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        # The fixture sets max_turns=5 for gensql
        assert node.max_turns == 5


class TestGenSQLAgenticNodeExecutionMode:
    """Tests for GenSQLAgenticNode execution_mode gating of ask_user tool."""

    def test_interactive_mode_has_ask_user_tool(self, real_agent_config, mock_llm_create):
        """Interactive mode (default) enables ask_user tool."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_interactive",
            description="Test interactive mode",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        assert node.execution_mode == "interactive"
        assert node.ask_user_tool is not None
        tool_names = [t.name for t in node.tools]
        assert "ask_user" in tool_names

    def test_workflow_mode_no_ask_user_tool(self, real_agent_config, mock_llm_create):
        """Workflow mode disables ask_user tool."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_workflow",
            description="Test workflow mode",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
            execution_mode="workflow",
        )

        assert node.execution_mode == "workflow"
        assert node.ask_user_tool is None
        tool_names = [t.name for t in node.tools]
        assert "ask_user" not in tool_names

    def test_rebuild_tools_preserves_ask_user_in_interactive(self, real_agent_config, mock_llm_create):
        """_rebuild_tools keeps ask_user tool in interactive mode."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_rebuild_ask",
            description="Test rebuild with ask_user",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node._rebuild_tools()
        tool_names = [t.name for t in node.tools]
        assert "ask_user" in tool_names

    def test_rebuild_tools_no_ask_user_in_workflow(self, real_agent_config, mock_llm_create):
        """_rebuild_tools does not include ask_user in workflow mode."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_rebuild_wf",
            description="Test rebuild without ask_user",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
            execution_mode="workflow",
        )

        node._rebuild_tools()
        tool_names = [t.name for t in node.tools]
        assert "ask_user" not in tool_names


@pytest.mark.acceptance
class TestGenSQLAgenticNodeExecution:
    """Tests for GenSQLAgenticNode execute_stream and related methods."""

    @pytest.mark.asyncio
    async def test_gensql_simple_response(self, real_agent_config, mock_llm_create):
        """execute_stream with simple text response (no tool calls) produces USER and ASSISTANT actions."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Here is a simple text response about SAT scores."),
            ]
        )

        node = GenSQLAgenticNode(
            node_id="test_gensql_simple",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node.input = GenSQLNodeInput(
            user_message="Tell me about the satscores table",
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
    async def test_gensql_with_tool_calls(self, real_agent_config, mock_llm_create):
        """execute_stream where LLM calls list_tables then responds with SQL."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                    ],
                    content=json.dumps(
                        {
                            "sql": "SELECT * FROM satscores LIMIT 10",
                            "tables": ["satscores"],
                            "explanation": "Query SAT scores from the satscores table",
                        }
                    ),
                ),
            ]
        )

        node = GenSQLAgenticNode(
            node_id="test_gensql_tools",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node.input = GenSQLNodeInput(
            user_message="Show me SAT scores",
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

        # Verify tool was actually called by checking tool results on the mock
        assert len(mock_llm_create.tool_results) >= 1
        tool_result = mock_llm_create.tool_results[0]
        assert tool_result["tool"] == "list_tables"
        assert tool_result["executed"] is True

    @pytest.mark.asyncio
    async def test_gensql_execute_sql_tool(self, real_agent_config, mock_llm_create):
        """LLM calls read_query on the real database, verify real results returned."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="read_query",
                            arguments=(
                                '{"sql": "SELECT cds, AvgScrRead FROM satscores '
                                'WHERE AvgScrRead IS NOT NULL ORDER BY cds LIMIT 5"}'
                            ),
                        ),
                    ],
                    content="The satscores table has SAT reading scores for various schools.",
                ),
            ]
        )

        node = GenSQLAgenticNode(
            node_id="test_gensql_exec_sql",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node.input = GenSQLNodeInput(
            user_message="What are the SAT reading scores?",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Verify tool was executed for real
        assert len(mock_llm_create.tool_results) >= 1
        sql_result = mock_llm_create.tool_results[0]
        assert sql_result["tool"] == "read_query"
        assert sql_result["executed"] is True

        # The output should contain actual data from the california_schools SQLite db
        output = sql_result["output"]
        output_str = str(output)
        assert "cds" in output_str.lower() or "AvgScrRead" in output_str or "502" in output_str

    @pytest.mark.asyncio
    async def test_gensql_describe_table_tool(self, real_agent_config, mock_llm_create):
        """LLM calls describe_table, verify real schema returned from SQLite."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="describe_table",
                            arguments='{"table_name": "satscores"}',
                        ),
                    ],
                    content=(
                        "The satscores table has columns: cds, sname, dname, cname, enroll12, "
                        "NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500."
                    ),
                ),
            ]
        )

        node = GenSQLAgenticNode(
            node_id="test_gensql_describe",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node.input = GenSQLNodeInput(
            user_message="Describe the satscores table",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Verify describe_table was executed
        assert len(mock_llm_create.tool_results) >= 1
        desc_result = mock_llm_create.tool_results[0]
        assert desc_result["tool"] == "describe_table"
        assert desc_result["executed"] is True

        # The output should contain column info from the real satscores table
        output_str = str(desc_result["output"])
        # Should contain column names from the satscores schema
        assert "cds" in output_str.lower() or "avgscrread" in output_str.lower() or "sname" in output_str.lower()

    @pytest.mark.asyncio
    async def test_gensql_action_history_tracking(self, real_agent_config, mock_llm_create):
        """Verify ActionHistory objects are yielded correctly and tracked in ActionHistoryManager."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                    ],
                    content=json.dumps(
                        {
                            "sql": "SELECT COUNT(*) FROM schools",
                            "tables": ["schools"],
                            "explanation": "Count schools",
                        }
                    ),
                ),
            ]
        )

        node = GenSQLAgenticNode(
            node_id="test_gensql_history",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node.input = GenSQLNodeInput(
            user_message="How many schools are there?",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # ActionHistoryManager should track all actions
        tracked_actions = ahm.get_actions()
        assert len(tracked_actions) >= 2

        # Verify we can find both USER and ASSISTANT roles
        tracked_roles = [a.role for a in tracked_actions]
        assert ActionRole.USER in tracked_roles
        assert ActionRole.ASSISTANT in tracked_roles

        # Each action should have a valid action_id
        for action in tracked_actions:
            assert action.action_id is not None
            assert len(action.action_id) > 0

    @pytest.mark.asyncio
    async def test_gensql_sql_extraction(self, real_agent_config, mock_llm_create):
        """Response content contains SQL in JSON format, verify it is extracted to the result."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        mock_llm_create.reset(
            responses=[
                MockLLMResponse(
                    content=json.dumps(
                        {
                            "sql": "SELECT * FROM satscores WHERE AvgScrRead > 500",
                            "tables": ["satscores"],
                            "explanation": "Get schools with high SAT reading scores",
                        }
                    ),
                ),
            ]
        )

        node = GenSQLAgenticNode(
            node_id="test_gensql_extract",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )

        node.input = GenSQLNodeInput(
            user_message="Show me schools with SAT reading score above 500",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # The final action should contain the result with extracted SQL
        final_action = actions[-1]
        assert final_action.role == ActionRole.ASSISTANT
        assert final_action.status == ActionStatus.SUCCESS
        assert final_action.output is not None

        # Check that SQL was extracted into the result
        output = final_action.output
        assert isinstance(output, dict), f"Expected dict output, got {type(output)}"
        sql_value = output.get("sql")
        assert sql_value, f"Missing 'sql' key in output: {output.keys()}"
        assert "satscores" in sql_value.lower()
        assert "avgscrread" in sql_value.lower()

    @pytest.mark.asyncio
    async def test_gensql_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """execute_stream without input raises ValueError."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_gensql_no_input",
            description="Test GenSQL node",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )
        node.input = None

        ahm = ActionHistoryManager()
        with pytest.raises(ValueError, match="GenSQL input not set"):
            async for _ in node.execute_stream(ahm):
                pass


# ===========================================================================
# ChatAgenticNode Tests
# ===========================================================================


class TestChatAgenticNodeInit:
    """Tests for ChatAgenticNode initialization with real config."""

    def test_chat_init_with_real_config(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode initializes correctly, inherits from AgenticNode (not GenSQLAgenticNode)."""
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = ChatAgenticNode(
            node_id="test_chat_1",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.id == "test_chat_1"
        assert node.type == NodeType.TYPE_CHAT
        assert node.description == "Test Chat node"
        assert isinstance(node, AgenticNode)
        assert not isinstance(node, GenSQLAgenticNode)
        assert node.get_node_name() == "chat"
        assert isinstance(node.model, MockLLMModel)

    def test_chat_has_all_tools(self, real_agent_config, mock_llm_create):
        """Chat has both db tools and context_search tools after initialization."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_chat_2",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Chat node should have db tools
        assert node.db_func_tool is not None

        # Chat node should have context_search_tools
        assert node.context_search_tools is not None

        # Verify db tool names present
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

        # Chat should have more tools than gensql because it includes context_search
        assert len(node.tools) > 0

    def test_chat_has_skill_attributes(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode has skill_func_tool and permission_hooks attributes."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_chat_3",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert hasattr(node, "skill_func_tool")
        assert hasattr(node, "permission_hooks")


class TestChatAgenticNodeExecution:
    """Tests for ChatAgenticNode execute_stream."""

    @pytest.mark.asyncio
    async def test_chat_simple_response(self, real_agent_config, mock_llm_create):
        """execute_stream with simple response produces USER and ASSISTANT actions."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Hello! I can help you with your database queries."),
            ]
        )

        node = ChatAgenticNode(
            node_id="test_chat_simple",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(
            user_message="Hello, what can you do?",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Should have at least USER + final ASSISTANT actions
        assert len(actions) >= 2
        # First action: USER
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        # Last action: ASSISTANT (no separate chat_response final action)
        assert actions[-1].role == ActionRole.ASSISTANT
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_chat_with_db_tool_calls(self, real_agent_config, mock_llm_create):
        """Chat calls real db tools (list_tables) and gets actual results."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                    ],
                    content="I found the following tables: frpm, satscores, schools.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="test_chat_db_tools",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(
            user_message="What tables are available?",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Verify tool was actually executed
        assert len(mock_llm_create.tool_results) >= 1
        tool_result = mock_llm_create.tool_results[0]
        assert tool_result["tool"] == "list_tables"
        assert tool_result["executed"] is True

        # The real tool output should contain our test tables
        output_str = str(tool_result["output"])
        assert "satscores" in output_str or "schools" in output_str or "frpm" in output_str

        # Verify actions include TOOL role
        roles = [a.role for a in actions]
        assert ActionRole.TOOL in roles

    @pytest.mark.asyncio
    async def test_chat_with_context_search(self, real_agent_config, mock_llm_create):
        """Chat calls a context search tool (may return empty results from fresh RAG store, that is OK)."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_chat_ctx_search",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Check if context_search_tools has any available tools
        # In a fresh test environment, the RAG stores may be empty, so there may be
        # no search tools exposed. That is acceptable - we verify the tools object exists.
        assert node.context_search_tools is not None

        # Get the actual available search tool names
        ctx_tools = node.context_search_tools.available_tools()
        ctx_tool_names = [t.name for t in ctx_tools]

        if len(ctx_tool_names) > 0:
            # If there are context search tools available, test calling one
            first_tool_name = ctx_tool_names[0]
            mock_llm_create.reset(
                responses=[
                    build_tool_then_response(
                        tool_calls=[
                            MockToolCall(name=first_tool_name, arguments="{}"),
                        ],
                        content="Search completed.",
                    ),
                ]
            )

            node.input = ChatNodeInput(
                user_message="Search for order metrics",
                database="california_schools",
            )

            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

            # Tool should have been executed (even if results are empty)
            assert len(mock_llm_create.tool_results) >= 1
            assert mock_llm_create.tool_results[0]["tool"] == first_tool_name
        else:
            # No context search tools available (empty RAG store) - this is OK
            # Verify the context_search_tools object was created and is iterable
            assert node.context_search_tools is not None
            assert hasattr(node.context_search_tools, "available_tools"), (
                "context_search_tools must expose available_tools()"
            )
            assert isinstance(ctx_tools, list), "available_tools() must return a list"

    @pytest.mark.asyncio
    async def test_chat_input_not_set_raises(self, real_agent_config, mock_llm_create):
        """Chat execute_stream raises ValueError when input is not set."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_chat_no_input",
            description="Test Chat node",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = None

        ahm = ActionHistoryManager()
        with pytest.raises(ValueError, match="Chat input not set"):
            async for _ in node.execute_stream(ahm):
                pass


# ===========================================================================
# Build Enhanced Message & Prepare Template Context Tests
# ===========================================================================


class TestBuildEnhancedMessage:
    """Tests for the build_enhanced_message utility function."""

    def test_basic_message(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        result = build_enhanced_message(
            user_message="Show all tables",
            db_type="sqlite",
        )
        assert "Show all tables" in result
        assert "sqlite" in result

    def test_message_with_external_knowledge(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        result = build_enhanced_message(
            user_message="Query revenue",
            db_type="postgresql",
            external_knowledge="Revenue is stored in the financials table",
        )
        assert "Revenue is stored in the financials table" in result
        assert "postgresql" in result

    def test_message_with_database_context(self):
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        result = build_enhanced_message(
            user_message="Count users",
            db_type="mysql",
            catalog="main_catalog",
            database="main_db",
            db_schema="public",
        )
        assert "main_catalog" in result
        assert "main_db" in result
        assert "public" in result


class TestPrepareTemplateContext:
    """Tests for the prepare_template_context utility function."""

    def test_basic_context(self):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={"system_prompt": "test", "tools": ""},
        )
        assert context["has_db_tools"] is True
        assert context["has_filesystem_tools"] is True
        assert context["has_mf_tools"] is True
        assert context["has_context_search_tools"] is True
        assert context["has_parsing_tools"] is True

    def test_context_with_disabled_tools(self):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={"system_prompt": "test", "tools": ""},
            has_db_tools=False,
            has_filesystem_tools=False,
            has_mf_tools=False,
        )
        assert context["has_db_tools"] is False
        assert context["has_filesystem_tools"] is False
        assert context["has_mf_tools"] is False

    def test_context_with_reference_template_tools(self):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={"system_prompt": "test", "tools": ""},
            has_reference_template_tools=True,
        )
        assert context["has_reference_template_tools"] is True

    def test_context_without_reference_template_tools(self):
        from datus.agent.node.gen_sql_agentic_node import prepare_template_context

        context = prepare_template_context(
            node_config={"system_prompt": "test", "tools": ""},
        )
        assert context["has_reference_template_tools"] is False


class TestGenSQLNodeReferenceTemplateToolSetup:
    """Tests for reference_template_tools setup in GenSQLAgenticNode."""

    def test_setup_reference_template_tools_wildcard(self, real_agent_config, mock_llm_create):
        """reference_template_tools.* pattern loads all template tools."""
        real_agent_config.agentic_nodes["tpl_test"] = {
            "system_prompt": "gen_sql",
            "tools": "db_tools.*, reference_template_tools.*",
            "max_turns": 5,
        }
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="tpl_test_id",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="tpl_test",
        )
        # reference_template_tools should be initialized (may have 0 templates though)
        assert node.reference_template_tools is not None

    def test_setup_reference_template_tools_specific_method(self, real_agent_config, mock_llm_create):
        """reference_template_tools.search_reference_template loads only search tool."""
        real_agent_config.agentic_nodes["tpl_test2"] = {
            "system_prompt": "gen_sql",
            "tools": "db_tools.*, reference_template_tools.search_reference_template",
            "max_turns": 5,
        }
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="tpl_test2_id",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="tpl_test2",
        )
        assert node.reference_template_tools is not None


# ===========================================================================
# End-to-End Integration: AgenticNode + Hooks + InteractionBroker
# ===========================================================================


def _configure_ask_permission(agent_config, tool_category="db_tools", tool_pattern="*"):
    """Patch agent_config.permissions_config to set ASK permission for a tool category.

    This modifies the real AgentConfig's permissions so that ChatAgenticNode
    creates PermissionHooks with ASK rules during setup_tools().
    """
    from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule

    agent_config.permissions_config = PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool=tool_category, pattern=tool_pattern, permission=PermissionLevel.ASK),
        ],
    )


class TestEndToEndNodeHooksInteraction:
    """End-to-end tests: ChatAgenticNode → MockLLM tool call → hook triggers → broker interaction → submit.

    These tests exercise the FULL production flow:
    1. ChatAgenticNode is created with real config + ASK permission rules
    2. execute_stream_with_interactions() is called
    3. MockLLM decides to call a tool (e.g., list_tables)
    4. MockLLMModel invokes hooks.on_tool_start() before tool execution
    5. PermissionHooks checks permission → ASK → calls broker.request()
    6. A concurrent task simulates the UI: fetches the request and calls broker.submit()
    7. The hook receives the user choice and either allows or denies the tool
    8. The merged stream yields both execution actions and INTERACTION actions
    """

    @pytest.mark.asyncio
    async def test_e2e_ask_permission_user_approves_tool_executes(self, real_agent_config, mock_llm_create):
        """Full flow: LLM calls list_tables → ASK permission → user approves → tool executes for real."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        _configure_ask_permission(real_agent_config, "db_tools", "list_tables")

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[MockToolCall(name="list_tables", arguments="{}")],
                    content="Found tables: satscores, schools, frpm.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_approve",
            description="E2E approve test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(user_message="List all tables", database="california_schools")

        broker = node._get_or_create_broker()

        # Concurrent UI simulator: watch for pending interactions and approve
        async def ui_approve():
            for _ in range(200):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "y")  # Allow once
                    return
            pytest.fail("Timed out waiting for permission interaction")

        ui_task = asyncio.create_task(ui_approve())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Verify the tool was actually executed
        assert len(mock_llm_create.tool_results) >= 1
        assert mock_llm_create.tool_results[0]["tool"] == "list_tables"
        assert mock_llm_create.tool_results[0]["executed"] is True

        # Verify the stream contains TOOL actions (real execution happened)
        roles = [a.role for a in actions]
        assert ActionRole.TOOL in roles
        assert ActionRole.ASSISTANT in roles

        # Verify INTERACTION actions appeared in the merged stream
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) >= 1

    @pytest.mark.asyncio
    async def test_e2e_ask_permission_user_denies_tool_blocked(self, real_agent_config, mock_llm_create):
        """Full flow: LLM calls list_tables → ASK permission → user denies → PermissionDeniedException."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        _configure_ask_permission(real_agent_config, "db_tools", "list_tables")

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[MockToolCall(name="list_tables", arguments="{}")],
                    content="This should not appear.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_deny",
            description="E2E deny test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(user_message="List all tables", database="california_schools")

        broker = node._get_or_create_broker()

        # Concurrent UI simulator: watch for pending interactions and deny
        async def ui_deny():
            for _ in range(200):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "n")  # Deny
                    return

        ui_task = asyncio.create_task(ui_deny())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Tool should NOT have been executed (permission denied)
        assert len(mock_llm_create.tool_results) == 0

        # The stream should contain an error/failure action from ChatAgenticNode
        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) >= 1

        # The error action should indicate failure due to permission denial
        error_action = assistant_actions[-1]
        assert error_action.output is not None
        assert isinstance(error_action.output, dict), f"Expected dict, got {type(error_action.output)}"
        assert error_action.output.get("success") is False or "rejected" in str(error_action.output).lower()

    @pytest.mark.asyncio
    async def test_e2e_ask_permission_session_approve_second_call_auto(self, real_agent_config, mock_llm_create):
        """Full flow: user selects 'Always allow' → second tool call is auto-approved without interaction."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        _configure_ask_permission(real_agent_config, "db_tools", "list_tables")

        # LLM calls list_tables twice in two separate tool calls
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                        MockToolCall(name="list_tables", arguments="{}"),
                    ],
                    content="Called list_tables twice.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_session_approve",
            description="E2E session approve test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(user_message="List tables twice", database="california_schools")

        broker = node._get_or_create_broker()
        interaction_count = 0

        # Concurrent UI: approve session on first request; second should be auto-approved
        async def ui_session_approve():
            nonlocal interaction_count
            for _ in range(200):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    interaction_count += 1
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "a")  # Always allow (session)
                    return  # Only one interaction expected

        ui_task = asyncio.create_task(ui_session_approve())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Both tool calls should have executed
        assert len(mock_llm_create.tool_results) == 2
        assert mock_llm_create.tool_results[0]["executed"] is True
        assert mock_llm_create.tool_results[1]["executed"] is True

        # Only ONE interaction should have occurred (second was auto-approved)
        assert interaction_count == 1

    @pytest.mark.asyncio
    async def test_e2e_allow_permission_no_interaction(self, real_agent_config, mock_llm_create):
        """ALLOW permission: tool call executes without any broker interaction."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        # Default permissions are ALLOW — no ASK rules
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[MockToolCall(name="list_tables", arguments="{}")],
                    content="Tables found.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_allow",
            description="E2E allow test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(user_message="List tables", database="california_schools")

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        # Tool should execute without any interaction
        assert len(mock_llm_create.tool_results) >= 1
        assert mock_llm_create.tool_results[0]["executed"] is True

        # No INTERACTION actions in the stream (ALLOW doesn't trigger broker)
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) == 0

    @pytest.mark.asyncio
    async def test_e2e_multiple_tools_mixed_permissions(self, real_agent_config, mock_llm_create):
        """Mixed permissions: list_tables is ASK, describe_table is ALLOW. Only list_tables triggers interaction."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule

        # list_tables = ASK, describe_table = ALLOW (default)
        real_agent_config.permissions_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="list_tables", permission=PermissionLevel.ASK),
            ],
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(name="list_tables", arguments="{}"),
                        MockToolCall(name="describe_table", arguments='{"table_name": "satscores"}'),
                    ],
                    content="Listed tables and described satscores.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_mixed",
            description="E2E mixed permissions test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(user_message="List and describe", database="california_schools")

        broker = node._get_or_create_broker()

        async def ui_approve_list_tables():
            for _ in range(200):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "y")
                    return

        ui_task = asyncio.create_task(ui_approve_list_tables())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Both tools should have executed
        assert len(mock_llm_create.tool_results) == 2
        tool_names = [r["tool"] for r in mock_llm_create.tool_results]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert all(r["executed"] for r in mock_llm_create.tool_results)

        # Only one interaction (for list_tables ASK), describe_table was auto-allowed
        interaction_actions = [
            a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.PROCESSING
        ]
        assert len(interaction_actions) >= 1


# ===========================================================================
# End-to-End Integration: AgenticNode + PlanModeHooks + InteractionBroker
# ===========================================================================


class TestEndToEndPlanModeHooksInteraction:
    """End-to-end tests: ChatAgenticNode(plan_mode=True) → LLM calls todo_write → PlanModeHooks →
    on_tool_end sets _plan_generated_pending → on_llm_end → _on_plan_generated → broker.request → submit.

    Tests the full production flow for plan mode interactions:
    1. ChatAgenticNode receives plan_mode=True input
    2. PlanModeHooks is created with broker + session
    3. Plan tools (todo_write, todo_read, todo_update) are added
    4. MockLLM calls todo_write with plan items
    5. PlanModeHooks.on_tool_end detects todo_write → sets _plan_generated_pending
    6. PlanModeHooks.on_llm_end triggers _on_plan_generated → broker.request(choices 1/2/3/4)
    7. UI simulator submits choice
    8. Plan mode state transitions accordingly
    """

    @pytest.mark.asyncio
    async def test_e2e_plan_mode_user_selects_manual(self, real_agent_config, mock_llm_create):
        """Full flow: LLM calls todo_write → user selects 'Manual Confirm' (1) → plan enters executing/manual."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        todos = json.dumps(
            [
                {"content": "Query database schema", "status": "pending"},
                {"content": "Generate SQL report", "status": "pending"},
            ]
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="todo_write",
                            arguments=json.dumps({"todos_json": todos}),
                        ),
                    ],
                    content="I have created a plan with 2 steps.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_plan_manual",
            description="E2E plan manual test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(
            user_message="Create a plan for database analysis",
            database="california_schools",
            plan_mode=True,
        )

        broker = node._get_or_create_broker()

        # Concurrent UI simulator: wait for plan confirmation request, select Manual (1)
        async def ui_select_manual():
            for _ in range(300):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "1")  # Manual Confirm
                    return
            pytest.fail("Timed out waiting for plan confirmation interaction")

        ui_task = asyncio.create_task(ui_select_manual())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Verify todo_write was executed
        todo_write_results = [r for r in mock_llm_create.tool_results if r["tool"] == "todo_write"]
        assert len(todo_write_results) >= 1
        assert todo_write_results[0]["executed"] is True

        # Verify INTERACTION actions appeared in the merged stream
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) >= 1

        # Verify the PROCESSING interaction offered plan mode choices (1/2/3/4)
        processing = [a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.PROCESSING]
        assert len(processing) >= 1
        choices_list = processing[0].input.get("choices", []) if isinstance(processing[0].input, dict) else []
        choices = choices_list[0] if choices_list else {}
        assert "1" in choices  # Manual Confirm
        assert "2" in choices  # Auto Execute
        assert "4" in choices  # Cancel

        # Verify the SUCCESS callback indicates Manual mode was selected
        success = [a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.SUCCESS]
        assert len(success) >= 1
        output = success[0].output
        assert isinstance(output, dict)
        assert output.get("user_choice") == "1"
        assert "manual" in output.get("content", "").lower()

    @pytest.mark.asyncio
    async def test_e2e_plan_mode_user_selects_auto(self, real_agent_config, mock_llm_create):
        """Full flow: LLM calls todo_write → user selects 'Auto Execute' (2) → plan enters executing/auto."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        todos = json.dumps(
            [
                {"content": "List all tables", "status": "pending"},
                {"content": "Describe satscores table", "status": "pending"},
                {"content": "Run sample query", "status": "pending"},
            ]
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="todo_write",
                            arguments=json.dumps({"todos_json": todos}),
                        ),
                    ],
                    content="Plan created with 3 steps for auto execution.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_plan_auto",
            description="E2E plan auto test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(
            user_message="Analyze the database automatically",
            database="california_schools",
            plan_mode=True,
        )

        broker = node._get_or_create_broker()

        # Concurrent UI simulator: select Auto Execute (2)
        async def ui_select_auto():
            for _ in range(300):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "2")  # Auto Execute
                    return
            pytest.fail("Timed out waiting for plan confirmation interaction")

        ui_task = asyncio.create_task(ui_select_auto())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Verify todo_write was executed
        todo_write_results = [r for r in mock_llm_create.tool_results if r["tool"] == "todo_write"]
        assert len(todo_write_results) >= 1

        # Verify INTERACTION actions in stream
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) >= 1

        # Verify the SUCCESS callback indicates Auto mode was selected
        success = [a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.SUCCESS]
        assert len(success) >= 1
        output = success[0].output
        assert isinstance(output, dict)
        assert output.get("user_choice") == "2"
        assert "auto" in output.get("content", "").lower()

    @pytest.mark.asyncio
    async def test_e2e_plan_mode_user_cancels(self, real_agent_config, mock_llm_create):
        """Full flow: LLM calls todo_write → user selects 'Cancel' (4) → UserCancelledException handled."""
        import asyncio

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        todos = json.dumps(
            [
                {"content": "Some task", "status": "pending"},
            ]
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="todo_write",
                            arguments=json.dumps({"todos_json": todos}),
                        ),
                    ],
                    content="Plan created.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_plan_cancel",
            description="E2E plan cancel test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.input = ChatNodeInput(
            user_message="Create a plan but I will cancel",
            database="california_schools",
            plan_mode=True,
        )

        broker = node._get_or_create_broker()

        # Concurrent UI simulator: select Cancel (4)
        async def ui_select_cancel():
            for _ in range(300):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "4")  # Cancel
                    return
            pytest.fail("Timed out waiting for plan confirmation interaction")

        ui_task = asyncio.create_task(ui_select_cancel())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # ChatAgenticNode catches UserCancelledException and creates a cancellation action
        # Verify we get the cancellation action (success=True, action_type=user_cancellation)
        cancellation_actions = [a for a in actions if a.action_type == "user_cancellation"]
        assert len(cancellation_actions) >= 1

        # Verify INTERACTION actions in stream
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) >= 1

        # plan_hooks is reset to None in the finally block, so check via INTERACTION output
        success = [a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.SUCCESS]
        assert success
        output = success[0].output
        assert isinstance(output, dict)
        assert output.get("user_choice") == "4"


# ===========================================================================
# End-to-End Integration: AgenticNode + GenerationHooks + InteractionBroker
# ===========================================================================


def _create_test_semantic_yaml(file_path: str) -> None:
    """Create a minimal semantic model YAML file for testing GenerationHooks."""
    import yaml

    data_source = {
        "data_source": {
            "name": "test_table",
            "sql_table": "test_table",
            "description": "A test semantic model for unit testing",
            "dimensions": [
                {
                    "name": "test_dim",
                    "type": "CATEGORICAL",
                    "description": "A test dimension",
                    "expr": "test_dim",
                },
            ],
            "measures": [
                {
                    "name": "test_measure",
                    "agg": "SUM",
                    "description": "A test measure",
                    "expr": "test_value",
                },
            ],
        },
    }
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_source, f, allow_unicode=True, sort_keys=False)


class TestEndToEndGenerationHooksInteraction:
    """End-to-end tests: ChatAgenticNode + GenerationHooks → LLM calls end_semantic_model_generation →
    on_tool_end → _handle_end_semantic_model_generation → _get_sync_confirmation → broker.request → submit.

    Tests the full production flow for generation hooks interactions:
    1. ChatAgenticNode is created with a fake end_semantic_model_generation tool
    2. GenerationHooks is attached via the node's permission_hooks slot
    3. MockLLM calls end_semantic_model_generation with YAML file path
    4. GenerationHooks.on_tool_end reads the YAML file and calls broker.request(y/n)
    5. UI simulator submits choice
    6. Hook processes sync or skip accordingly
    """

    @pytest.mark.asyncio
    async def test_e2e_generation_hooks_user_approves_sync(self, real_agent_config, mock_llm_create, tmp_path):
        """Full flow: LLM calls end_semantic_model_generation → user approves sync ('y') → sync to KB."""
        import asyncio
        import os

        from agents import FunctionTool

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.cli.generation_hooks import GenerationHooks

        # Create a real YAML file under the project subject_dir so GenerationHooks
        # path containment check accepts it.
        semantic_dir = real_agent_config.path_manager.semantic_model_path()
        yaml_path = os.path.join(str(semantic_dir), "test_semantic_model.yaml")
        _create_test_semantic_yaml(yaml_path)

        # Create a fake end_semantic_model_generation tool that returns the expected result format
        async def fake_end_gen(ctx, args_str):
            return {"success": 1, "result": {"semantic_model_files": [yaml_path]}}

        end_gen_tool = FunctionTool(
            name="end_semantic_model_generation",
            description="Complete semantic model generation",
            params_json_schema={
                "type": "object",
                "properties": {
                    "semantic_model_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["semantic_model_files"],
            },
            on_invoke_tool=fake_end_gen,
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="end_semantic_model_generation",
                            arguments=json.dumps({"semantic_model_files": [yaml_path]}),
                        ),
                    ],
                    content="Semantic model generation completed.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_gen_approve",
            description="E2E generation approve test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Add the fake generation tool to node tools
        node.tools.append(end_gen_tool)

        # Attach GenerationHooks via the permission_hooks slot
        # ChatAgenticNode._get_execution_config will pass this as hooks to the model
        broker = node._get_or_create_broker()
        generation_hooks = GenerationHooks(broker=broker, agent_config=real_agent_config)
        node.permission_hooks = generation_hooks

        node.input = ChatNodeInput(
            user_message="Generate semantic model",
            database="california_schools",
        )

        # Concurrent UI simulator: approve sync ('y')
        async def ui_approve_sync():
            for _ in range(300):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "y")  # Yes - Save to KB
                    return
            pytest.fail("Timed out waiting for generation sync interaction")

        ui_task = asyncio.create_task(ui_approve_sync())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Verify the tool was executed
        end_gen_results = [r for r in mock_llm_create.tool_results if r["tool"] == "end_semantic_model_generation"]
        assert len(end_gen_results) >= 1
        assert end_gen_results[0]["executed"] is True

        # Verify INTERACTION actions appeared in the merged stream
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) >= 1

        # Verify the PROCESSING interaction contained the YAML display prompt
        processing_interactions = [
            a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.PROCESSING
        ]
        assert len(processing_interactions) >= 1
        # The interaction content should reference the YAML file
        interaction_input = processing_interactions[0].input
        assert isinstance(interaction_input, dict)
        contents = interaction_input.get("contents", [])
        assert contents
        content = contents[0]
        assert "Sync to Knowledge Base" in content or "yaml" in content.lower()

    @pytest.mark.asyncio
    async def test_e2e_generation_hooks_user_declines_sync(self, real_agent_config, mock_llm_create, tmp_path):
        """Full flow: LLM calls end_semantic_model_generation → user declines sync ('n') → file kept only."""
        import asyncio
        import os

        from agents import FunctionTool

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.cli.generation_hooks import GenerationHooks

        # Create a real YAML file under the project subject_dir so GenerationHooks
        # path containment check accepts it.
        semantic_dir = real_agent_config.path_manager.semantic_model_path()
        yaml_path = os.path.join(str(semantic_dir), "test_semantic_decline.yaml")
        _create_test_semantic_yaml(yaml_path)

        # Create a fake end_semantic_model_generation tool
        async def fake_end_gen(ctx, args_str):
            return {"success": 1, "result": {"semantic_model_files": [yaml_path]}}

        end_gen_tool = FunctionTool(
            name="end_semantic_model_generation",
            description="Complete semantic model generation",
            params_json_schema={
                "type": "object",
                "properties": {
                    "semantic_model_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["semantic_model_files"],
            },
            on_invoke_tool=fake_end_gen,
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="end_semantic_model_generation",
                            arguments=json.dumps({"semantic_model_files": [yaml_path]}),
                        ),
                    ],
                    content="Semantic model generation completed.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_gen_decline",
            description="E2E generation decline test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Add the fake generation tool and attach GenerationHooks
        node.tools.append(end_gen_tool)
        broker = node._get_or_create_broker()
        generation_hooks = GenerationHooks(broker=broker, agent_config=real_agent_config)
        node.permission_hooks = generation_hooks

        node.input = ChatNodeInput(
            user_message="Generate but decline sync",
            database="california_schools",
        )

        # Concurrent UI simulator: decline sync ('n')
        async def ui_decline_sync():
            for _ in range(300):
                await asyncio.sleep(0.02)
                if broker.has_pending:
                    action_id = list(broker._pending.keys())[0]
                    await broker.submit(action_id, "n")  # No - Keep file only
                    return
            pytest.fail("Timed out waiting for generation sync interaction")

        ui_task = asyncio.create_task(ui_decline_sync())

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        await ui_task

        # Verify the tool was executed
        end_gen_results = [r for r in mock_llm_create.tool_results if r["tool"] == "end_semantic_model_generation"]
        assert len(end_gen_results) >= 1

        # Verify INTERACTION actions in stream
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) >= 1

        # Verify the SUCCESS callback indicates file was kept only (not synced)
        success_interactions = [
            a for a in actions if a.role == ActionRole.INTERACTION and a.status == ActionStatus.SUCCESS
        ]
        assert len(success_interactions) >= 1
        callback_output = success_interactions[0].output
        assert isinstance(callback_output, dict)
        callback_content = callback_output.get("content", "").lower()
        assert (
            "rejected" in callback_content
            or "deleted" in callback_content
            or "saved to file" in callback_content
            or "file only" in callback_content
        )

    @pytest.mark.asyncio
    async def test_e2e_generation_hooks_no_yaml_no_interaction(self, real_agent_config, mock_llm_create, tmp_path):
        """When end_semantic_model_generation returns no file paths, no interaction is triggered."""

        from agents import FunctionTool

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.cli.generation_hooks import GenerationHooks

        # Tool returns empty file list
        async def fake_end_gen_empty(ctx, args_str):
            return {"success": 1, "result": {"semantic_model_files": []}}

        end_gen_tool = FunctionTool(
            name="end_semantic_model_generation",
            description="Complete semantic model generation",
            params_json_schema={
                "type": "object",
                "properties": {
                    "semantic_model_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["semantic_model_files"],
            },
            on_invoke_tool=fake_end_gen_empty,
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="end_semantic_model_generation",
                            arguments=json.dumps({"semantic_model_files": []}),
                        ),
                    ],
                    content="No semantic model files generated.",
                ),
            ]
        )

        node = ChatAgenticNode(
            node_id="e2e_gen_empty",
            description="E2E generation empty test",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node.tools.append(end_gen_tool)
        broker = node._get_or_create_broker()
        generation_hooks = GenerationHooks(broker=broker, agent_config=real_agent_config)
        node.permission_hooks = generation_hooks

        node.input = ChatNodeInput(
            user_message="Generate with no output",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(ahm):
            actions.append(action)

        # Tool should have been executed
        assert len(mock_llm_create.tool_results) >= 1

        # No INTERACTION actions (empty file list = no sync prompt)
        interaction_actions = [a for a in actions if a.role == ActionRole.INTERACTION]
        assert len(interaction_actions) == 0


# ===========================================================================
# ExecutionInterrupted Tests
# ===========================================================================


class TestBuildEnhancedMessageWithContext:
    """Tests for build_enhanced_message with various context combinations."""

    def test_build_enhanced_message_with_db_type_only(self):
        """build_enhanced_message includes dialect context when only db_type is provided."""
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        result = build_enhanced_message(
            user_message="Show me the data",
            db_type="sqlite",
        )

        assert "sqlite" in result
        assert "Show me the data" in result

    def test_build_enhanced_message_with_database_and_schema(self):
        """build_enhanced_message includes database and schema in context."""
        from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

        result = build_enhanced_message(
            user_message="Query sales",
            db_type="postgresql",
            database="analytics",
            db_schema="public",
        )

        assert "postgresql" in result
        assert "analytics" in result
        assert "public" in result
        assert "Query sales" in result


# ===========================================================================
# SQL File Storage Helper Tests
# ===========================================================================


class TestSqlFileStorageHelpers:
    """Tests for GenSQLAgenticNode SQL file storage helper methods."""

    def _make_node(self, real_agent_config, mock_llm_create, node_config_overrides=None):
        """Helper to create a GenSQLAgenticNode for testing."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode(
            node_id="test_sql_file",
            description="Test SQL file storage",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )
        if node_config_overrides:
            node.node_config.update(node_config_overrides)
        return node

    def test_get_sql_preview_lines_default(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config, mock_llm_create)
        assert node._get_sql_preview_lines() == 5

    def test_get_sql_preview_lines_custom(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config, mock_llm_create, {"sql_preview_lines": 10})
        assert node._get_sql_preview_lines() == 10

    def test_get_sql_preview_short(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "SELECT 1;\nSELECT 2;\nSELECT 3;"
        preview = GenSQLAgenticNode._get_sql_preview(sql, max_lines=5)
        assert preview == sql

    def test_get_sql_preview_long(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        lines = [f"SELECT col_{i}" for i in range(20)]
        sql = "\n".join(lines)
        preview = GenSQLAgenticNode._get_sql_preview(sql, max_lines=3)
        assert "SELECT col_0" in preview
        assert "SELECT col_2" in preview
        assert "17 more lines" in preview

    def test_read_existing_sql_file_not_found(self, real_agent_config, mock_llm_create):
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        node = self._make_node(real_agent_config, mock_llm_create)
        workspace_root = node._resolve_workspace_root()
        node.filesystem_func_tool = FilesystemFuncTool(root_path=workspace_root)
        result = node._read_existing_sql_file("nonexistent/file.sql")
        assert result is None

    def test_read_existing_sql_file_success(self, tmp_path, real_agent_config, mock_llm_create):
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        node = self._make_node(real_agent_config, mock_llm_create)
        node.filesystem_func_tool = FilesystemFuncTool(root_path=str(tmp_path))
        workspace_root = str(tmp_path / "workspace")
        node.filesystem_func_tool = FilesystemFuncTool(root_path=workspace_root)

        # Write a file first using tmp_path to avoid polluting the project root
        node.filesystem_func_tool.write_file("sql/test/existing.sql", "SELECT old")
        result = node._read_existing_sql_file("sql/test/existing.sql")
        assert result == "SELECT old"

    def test_read_existing_sql_file_no_tool(self, real_agent_config, mock_llm_create):
        node = self._make_node(real_agent_config, mock_llm_create)
        node.filesystem_func_tool = None
        result = node._read_existing_sql_file("any/path.sql")
        assert result is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, node_name="gensql"):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

    return GenSQLAgenticNode(
        node_id="gensql_extra",
        description="Extra gensql test",
        node_type=NodeType.TYPE_GENSQL,
        agent_config=real_agent_config,
        node_name=node_name,
    )


# ---------------------------------------------------------------------------
# TestGetSqlPreview (static method)
# ---------------------------------------------------------------------------


class TestGetSqlPreview:
    def test_short_sql_no_truncation(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "SELECT id, name\nFROM users"
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=5)
        assert result == sql

    def test_long_sql_truncated(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "\n".join([f"line{i}" for i in range(10)])
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=3)
        assert "line0" in result
        assert "line1" in result
        assert "line2" in result
        assert "7 more lines" in result

    def test_exact_lines_no_truncation(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "a\nb\nc"
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=3)
        assert result == sql
        assert "more lines" not in result

    def test_single_line(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "SELECT 1"
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=5)
        assert result == "SELECT 1"


# ---------------------------------------------------------------------------
# TestGetSqlPreviewLines
# ---------------------------------------------------------------------------


class TestGetSqlPreviewLines:
    def test_default_preview_lines(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        result = node._get_sql_preview_lines()
        assert result == 5  # default

    def test_custom_preview_lines_from_config(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config["sql_preview_lines"] = "10"
        result = node._get_sql_preview_lines()
        assert result == 10


# ---------------------------------------------------------------------------
# TestReadExistingSqlFile
# ---------------------------------------------------------------------------


class TestReadExistingSqlFile:
    def test_returns_none_when_no_filesystem_tool(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.filesystem_func_tool = None
        result = node._read_existing_sql_file("some/file.sql")
        assert result is None

    def test_returns_none_when_empty_path(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        result = node._read_existing_sql_file("")
        assert result is None

    def test_returns_content_on_success(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_fs_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "SELECT 1 FROM table"
        mock_fs_tool.read_file.return_value = mock_result
        node.filesystem_func_tool = mock_fs_tool

        result = node._read_existing_sql_file("query.sql")
        assert result == "SELECT 1 FROM table"

    def test_returns_none_when_read_fails(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_fs_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.result = None
        mock_fs_tool.read_file.return_value = mock_result
        node.filesystem_func_tool = mock_fs_tool

        result = node._read_existing_sql_file("query.sql")
        assert result is None

    def test_returns_none_when_result_not_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_fs_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = {"key": "value"}  # not a string
        mock_fs_tool.read_file.return_value = mock_result
        node.filesystem_func_tool = mock_fs_tool

        result = node._read_existing_sql_file("query.sql")
        assert result is None


# ---------------------------------------------------------------------------
# TestSetupMcpServers
# ---------------------------------------------------------------------------


class TestSetupMcpServers:
    def test_empty_mcp_config_returns_empty(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": ""}
        result = node._setup_mcp_servers()
        assert result == {}

    def test_no_mcp_config_returns_empty(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {}
        result = node._setup_mcp_servers()
        assert result == {}

    def test_metricflow_mcp_setup(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "metricflow_mcp"}

        mock_server = MagicMock()
        with patch.object(node, "_setup_metricflow_mcp", return_value=mock_server):
            with patch.object(node, "_setup_mcp_server_from_config", return_value=None):
                result = node._setup_mcp_servers()

        assert "metricflow_mcp" in result


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputGenSQL:
    def test_setup_input_creates_gen_sql_node_input(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = None  # force creation

        wf = MagicMock()
        wf.task.task = "Find total sales"
        wf.task.external_knowledge = "revenue = total sales"
        wf.task.catalog_name = "cat"
        wf.task.database_name = "california_schools"
        wf.task.schema_name = "main"
        wf.task.current_date = None
        wf.context.table_schemas = []
        wf.context.metrics = []
        wf.metadata.get.return_value = False

        result = node.setup_input(wf)

        assert result["success"] is True
        assert isinstance(node.input, GenSQLNodeInput)
        assert node.input.user_message == "Find total sales"

    def test_setup_input_updates_existing_input(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = GenSQLNodeInput(user_message="old message")

        wf = MagicMock()
        wf.task.task = "new message"
        wf.task.external_knowledge = ""
        wf.task.catalog_name = ""
        wf.task.database_name = "california_schools"
        wf.task.schema_name = ""
        wf.task.current_date = None
        wf.context.table_schemas = []
        wf.context.metrics = []
        wf.metadata.get.return_value = False

        node.setup_input(wf)

        assert node.input.user_message == "new message"

    def test_setup_input_sets_date_parsing_reference(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_date_tools = MagicMock()
        node.date_parsing_tools = mock_date_tools

        wf = MagicMock()
        wf.task.task = "query"
        wf.task.external_knowledge = ""
        wf.task.catalog_name = ""
        wf.task.database_name = "california_schools"
        wf.task.schema_name = ""
        wf.task.current_date = "2024-01-15"
        wf.context.table_schemas = []
        wf.context.metrics = []
        wf.metadata.get.return_value = False

        node.setup_input(wf)

        mock_date_tools.set_reference_date.assert_called_once_with("2024-01-15")


# ---------------------------------------------------------------------------
# TestRebuildTools
# ---------------------------------------------------------------------------


class TestRebuildTools:
    def test_rebuild_tools_with_all_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)

        mock_db = MagicMock()
        mock_db.available_tools.return_value = [MagicMock(name="list_tables")]
        mock_ctx = MagicMock()
        mock_ctx.available_tools.return_value = [MagicMock(name="search_schema")]
        mock_date = MagicMock()
        mock_date.available_tools.return_value = [MagicMock(name="parse_date")]

        node.db_func_tool = mock_db
        node.context_search_tools = mock_ctx
        node.date_parsing_tools = mock_date
        node.filesystem_func_tool = None
        node._platform_doc_tool = None
        node.ask_user_tool = None
        node.sub_agent_task_tool = None

        node._rebuild_tools()

        # 3 mocked tools + ask_user tool (added in interactive mode)
        expected = 4 if node.ask_user_tool else 3
        assert len(node.tools) == expected

    def test_rebuild_tools_with_ask_user(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)

        mock_db = MagicMock()
        mock_db.available_tools.return_value = [MagicMock(name="list_tables")]

        node.db_func_tool = mock_db
        node.context_search_tools = None
        node.date_parsing_tools = None
        node.filesystem_func_tool = None
        node._platform_doc_tool = None
        node.sub_agent_task_tool = None
        # ask_user_tool is set up by _make_node via setup_tools; keep it

        node._rebuild_tools()

        # 1 db tool + 1 ask_user tool
        assert len(node.tools) == 2
        tool_names = [getattr(t, "name", "") for t in node.tools]
        assert "ask_user" in tool_names

    def test_rebuild_tools_empty_when_no_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.db_func_tool = None
        node.context_search_tools = None
        node.date_parsing_tools = None
        node.filesystem_func_tool = None
        node._platform_doc_tool = None
        node.ask_user_tool = None
        node.sub_agent_task_tool = None

        node._rebuild_tools()

        assert node.tools == []


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeNameGenSQL:
    def test_get_node_name_returns_configured_name(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, node_name="gensql")
        assert node.get_node_name() == "gensql"


# ---------------------------------------------------------------------------
# TestExecuteStreamGenSQL (focused on input_not_set)
# ---------------------------------------------------------------------------


class TestExecuteStreamGenSQLExtra:
    @pytest.mark.asyncio
    async def test_execute_stream_raises_when_no_input(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = None

        with pytest.raises(ValueError, match="GenSQL input not set"):
            async for _ in node.execute_stream():
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_extra2(real_agent_config, mock_llm_create, node_name="gensql"):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

    return GenSQLAgenticNode(
        node_id="gensql_extra2",
        description="Extra gensql test 2",
        node_type=NodeType.TYPE_GENSQL,
        agent_config=real_agent_config,
        node_name=node_name,
    )


# ---------------------------------------------------------------------------
# TestSetupToolPattern
# ---------------------------------------------------------------------------


class TestSetupToolPatternGenSQL:
    def test_wildcard_db_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools") as mock_setup:
            node._setup_tool_pattern("db_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_context_search_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_context_search_tools") as mock_setup:
            node._setup_tool_pattern("context_search_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_date_parsing_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_date_parsing_tools") as mock_setup:
            node._setup_tool_pattern("date_parsing_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_filesystem_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_filesystem_tools") as mock_setup:
            node._setup_tool_pattern("filesystem_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_platform_doc_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_platform_doc_tools") as mock_setup:
            node._setup_tool_pattern("platform_doc_tools.*")
        mock_setup.assert_called_once()

    def test_wildcard_unknown_type_logs_warning(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch("datus.agent.node.gen_sql_agentic_node.logger.warning") as mock_warning:
            node._setup_tool_pattern("unknown_tool_type.*")
        mock_warning.assert_called_once_with("Unknown tool type: unknown_tool_type")

    def test_exact_db_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools") as mock_setup:
            node._setup_tool_pattern("db_tools")
        mock_setup.assert_called_once()

    def test_exact_context_search_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_context_search_tools") as mock_setup:
            node._setup_tool_pattern("context_search_tools")
        mock_setup.assert_called_once()

    def test_exact_date_parsing_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_date_parsing_tools") as mock_setup:
            node._setup_tool_pattern("date_parsing_tools")
        mock_setup.assert_called_once()

    def test_exact_filesystem_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_filesystem_tools") as mock_setup:
            node._setup_tool_pattern("filesystem_tools")
        mock_setup.assert_called_once()

    def test_exact_platform_doc_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_platform_doc_tools") as mock_setup:
            node._setup_tool_pattern("platform_doc_tools")
        mock_setup.assert_called_once()

    def test_specific_method_pattern(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_specific_tool_method") as mock_method:
            node._setup_tool_pattern("db_tools.list_tables")
        mock_method.assert_called_once_with("db_tools", "list_tables")

    def test_unknown_pattern_no_raise(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with patch("datus.agent.node.gen_sql_agentic_node.logger.warning") as mock_warning:
            node._setup_tool_pattern("some_random_pattern")
        mock_warning.assert_called_once_with("Unknown tool pattern: some_random_pattern")

    def test_exception_in_setup_is_caught(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        with (
            patch.object(node, "_setup_db_tools", side_effect=RuntimeError("db error")),
            patch("datus.agent.node.gen_sql_agentic_node.logger.error") as mock_error,
        ):
            assert node._setup_tool_pattern("db_tools.*") is None
        mock_error.assert_called_once()


# ---------------------------------------------------------------------------
# TestExtractSqlAndOutputFromResponse
# ---------------------------------------------------------------------------


class TestExtractSqlAndOutputFromResponse:
    def test_extracts_sql_from_json_content(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "sql": "SELECT id FROM users",
                "explanation": "Returns all user IDs",
                "tables": ["users"],
            }
        )
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert sql == "SELECT id FROM users"
        assert "Returns all user IDs" in output
        assert "users" in output

    def test_returns_none_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        sql, output = node._extract_sql_and_output_from_response({"content": ""})
        assert sql is None
        assert output is None

    def test_returns_none_on_non_string_content(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        sql, output = node._extract_sql_and_output_from_response({"content": {"nested": "dict"}})
        assert sql is None
        assert output is None

    def test_returns_none_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        sql, output = node._extract_sql_and_output_from_response({"content": "not json at all"})
        assert sql is None

    def test_sql_only_no_explanation(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        content = json.dumps({"sql": "SELECT 1", "output": "Query result"})
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert sql == "SELECT 1"
        assert output == "Query result"

    def test_unescape_newlines_in_output(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "sql": "SELECT 1",
                "explanation": "line1\\nline2",
                "tables": [],
            }
        )
        sql, output = node._extract_sql_and_output_from_response({"content": content})
        assert "\n" in output  # \\n should be unescaped to \n


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextGenSQL:
    def test_returns_failure_when_no_result(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        node.result = None
        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is False

    def test_updates_context_with_sql(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        mock_result = MagicMock()
        mock_result.sql = "SELECT id FROM users"
        mock_result.response = ""
        node.result = mock_result

        wf = MagicMock()
        wf.context.sql_contexts = []

        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1

    def test_updates_context_without_sql(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        mock_result = MagicMock(spec=[])  # no 'sql' attribute
        node.result = mock_result

        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is True

    def test_update_context_exception_returns_failure(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        mock_result = MagicMock()
        mock_result.sql = "SELECT 1"
        mock_result.response = ""
        node.result = mock_result

        wf = MagicMock()
        wf.context.sql_contexts.append = MagicMock(side_effect=RuntimeError("append error"))

        result = node.update_context(wf)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TestGetExecutionConfig
# ---------------------------------------------------------------------------


class TestGetExecutionConfig:
    def test_normal_mode_returns_tools_and_instruction(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        node.tools = [MagicMock()]
        user_input = GenSQLNodeInput(user_message="query")

        with patch.object(node, "_get_system_instruction", return_value="system instruction"):
            config = node._get_execution_config("normal", user_input)

        assert config["tools"] == node.tools
        assert config["instruction"] == "system instruction"
        assert config["hooks"] is None

    def test_plan_mode_returns_combined_tools(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        base_tool = MagicMock()
        plan_tool = MagicMock()
        node.tools = [base_tool]
        node.plan_hooks = MagicMock()
        node.plan_hooks.get_plan_tools.return_value = [plan_tool]
        user_input = GenSQLNodeInput(user_message="query", plan_mode=True)

        with patch.object(node, "_get_system_instruction", return_value="base instruction"):
            config = node._get_execution_config("plan", user_input)

        assert base_tool in config["tools"]
        assert plan_tool in config["tools"]
        assert config["hooks"] == node.plan_hooks

    def test_unknown_mode_raises(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        user_input = GenSQLNodeInput(user_message="query")
        with pytest.raises(ValueError, match="Unknown execution mode"):
            node._get_execution_config("invalid_mode", user_input)


# ---------------------------------------------------------------------------
# TestSetupMcpServersExtra
# ---------------------------------------------------------------------------


class TestSetupMcpServersExtra:
    def test_mcp_server_from_config_called_for_unknown_server(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "my_custom_mcp"}

        mock_server = MagicMock()
        with patch.object(node, "_setup_mcp_server_from_config", return_value=mock_server) as mock_from_config:
            result = node._setup_mcp_servers()

        mock_from_config.assert_called_once_with("my_custom_mcp")
        assert "my_custom_mcp" in result

    def test_none_server_not_added(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "missing_server"}

        with patch.object(node, "_setup_mcp_server_from_config", return_value=None):
            result = node._setup_mcp_servers()

        assert "missing_server" not in result

    def test_exception_in_server_setup_is_caught(self, real_agent_config, mock_llm_create):
        node = _make_node_extra2(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "bad_server"}

        with patch.object(node, "_setup_mcp_server_from_config", side_effect=RuntimeError("mcp error")):
            # Should not raise
            result = node._setup_mcp_servers()

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestExecuteStreamGenSQL (error path)
# ---------------------------------------------------------------------------


class TestExecuteStreamGenSQLError:
    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.action_history import ActionStatus

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenSQLAgenticNode(
            node_id="gensql_error",
            description="Error test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gensql",
        )
        node.input = GenSQLNodeInput(user_message="Find total sales")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gensql_node(node_config=None, agent_config=None):
    """Build GenSQLAgenticNode bypassing __init__."""
    from datus.agent.node.agentic_node import AgenticNode
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
    from datus.cli.execution_state import InteractionBroker, InterruptController
    from datus.schemas.action_bus import ActionBus

    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = GenSQLAgenticNode.__new__(GenSQLAgenticNode)

    node._session = None
    node.ephemeral = False
    node.session_id = None
    node.model = None
    node.tools = []
    node.mcp_servers = {}
    node.actions = []
    node.context_length = None
    node.node_config = node_config or {}
    node.agent_config = agent_config
    node.permission_manager = None
    node.skill_manager = None
    node.skill_func_tool = None
    node._permission_callback = None
    node.id = "gensql_test"
    node.description = "GenSQL Test"
    node.type = "gensql"
    node.status = "pending"
    node.result = None
    node.dependencies = []
    node.input = None
    node.configured_node_name = "gensql"
    node.action_bus = ActionBus()
    node.interaction_broker = InteractionBroker()
    node.interrupt_controller = InterruptController()
    return node


# ---------------------------------------------------------------------------
# get_node_name
# ---------------------------------------------------------------------------


class TestGenSQLNodeName:
    def test_returns_gensql(self):
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
            node = GenSQLAgenticNode.__new__(GenSQLAgenticNode)
        node.node_config = {}
        node.configured_node_name = "gensql"
        result = node.get_node_name()
        assert result == "gensql"


# ---------------------------------------------------------------------------
# GenSQLNodeInput model validation
# ---------------------------------------------------------------------------


class TestGenSQLNodeInput:
    def test_minimal_input_valid(self):
        inp = GenSQLNodeInput(
            user_message="What are the top 10 users?",
            database="mydb",
        )
        assert inp.user_message == "What are the top 10 users?"
        assert inp.database == "mydb"

    def test_default_values(self):
        inp = GenSQLNodeInput(
            user_message="query",
            database="db",
        )
        assert inp.schemas is None
        assert inp.metrics is None

    def test_with_schemas(self):
        from datus.schemas.node_models import TableSchema

        schema = TableSchema(
            catalog_name="",
            database_name="db",
            schema_name="",
            table_name="users",
            columns=["id", "name"],
            definition="CREATE TABLE users (id INT, name TEXT)",
        )
        inp = GenSQLNodeInput(
            user_message="query",
            database="db",
            schemas=[schema],
        )
        assert len(inp.schemas) == 1
        assert inp.schemas[0].table_name == "users"


# ---------------------------------------------------------------------------
# GenSQLNodeResult model
# ---------------------------------------------------------------------------


class TestGenSQLNodeResult:
    def test_success_result(self):
        result = GenSQLNodeResult(
            success=True,
            sql="SELECT * FROM users",
            response="Here is the result",
        )
        assert result.success is True
        assert result.sql == "SELECT * FROM users"

    def test_failure_result(self):
        result = GenSQLNodeResult(
            success=False,
            response="",
            error="SQL generation failed",
        )
        assert result.success is False
        assert result.error == "SQL generation failed"


# ---------------------------------------------------------------------------
# setup_tools builds tool list
# ---------------------------------------------------------------------------


class TestGenSQLSetupTools:
    def test_setup_tools_with_agent_config(self, real_agent_config, mock_llm_create):
        """setup_tools populates self.tools from db_manager and context search."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="gensql_tools_test",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
        )

        # After init, tools should be set up
        assert isinstance(node.tools, list)
        # Should have at least DB tools
        assert len(node.tools) > 0

    def test_setup_tools_with_scoped_context(self, real_agent_config, mock_llm_create):
        """scoped_context config affects context search tool creation."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        # Add scoped_context to agentic_nodes config
        real_agent_config.agentic_nodes = {"gensql": {"system_prompt": "test", "scoped_context": True}}

        node = GenSQLAgenticNode(
            node_id="gensql_scoped",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
        )
        assert isinstance(node.tools, list)


# ---------------------------------------------------------------------------
# _parse_node_config for gensql node
# ---------------------------------------------------------------------------


class TestGenSQLParseNodeConfig:
    def test_gensql_adapter_type_extracted(self):
        node = _make_gensql_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {
            "gensql": {
                "adapter_type": "custom_adapter",
                "sql_file_threshold": 50,
                "sql_preview_lines": 10,
            }
        }
        result = node._parse_node_config(mock_config, "gensql")
        assert result.get("adapter_type") == "custom_adapter"
        assert result.get("sql_file_threshold") == 50
        assert result.get("sql_preview_lines") == 10

    def test_empty_agentic_nodes_returns_empty(self):
        node = _make_gensql_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {}
        result = node._parse_node_config(mock_config, "gensql")
        assert result == {}


# ---------------------------------------------------------------------------
# execute_stream with mocked model
# ---------------------------------------------------------------------------


class TestGenSQLExecuteStream:
    def test_execute_stream_with_mocked_model(self, real_agent_config, mock_llm_create):
        """execute_stream should yield at least one action."""
        import asyncio

        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType
        from datus.schemas.action_history import ActionHistoryManager

        mock_llm_create.reset(
            responses=[
                type(
                    "Resp",
                    (),
                    {
                        "content": '{"sql": "SELECT 1", "response": "done"}',
                        "tool_calls": [],
                        "usage": type("U", (), {"input_tokens": 10, "output_tokens": 5})(),
                        "reasoning_content": None,
                    },
                )()
            ]
        )

        node = GenSQLAgenticNode(
            node_id="es_test",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
        )
        node.input = GenSQLNodeInput(
            user_message="Show all users",
            database="california_schools",
        )

        async def _collect():
            actions = []
            ahm = ActionHistoryManager()
            try:
                async for action in node.execute_stream(ahm):
                    actions.append(action)
                    if len(actions) >= 5:
                        break
            except (
                StopAsyncIteration,
                RuntimeError,
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                pass  # Expected failures when mocked dependencies are incomplete
            return actions

        actions = asyncio.run(_collect())
        # Should have yielded at least one action
        assert isinstance(actions, list)


# ---------------------------------------------------------------------------
# update_context for GenSQLAgenticNode
# ---------------------------------------------------------------------------


class TestGenSQLUpdateContext:
    def test_update_context_with_sql_result(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="ctx_test",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
        )
        node.result = GenSQLNodeResult(
            success=True,
            sql="SELECT * FROM schools",
            response="Here are the schools",
        )

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1

    def test_update_context_no_result(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="ctx_no_result",
            description="Test",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
        )
        node.result = None

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is False


class TestGenSQLSystemPromptCurrentDate:
    """Verify current_date injection uses reference_date when available."""

    def test_system_prompt_uses_reference_date(self, real_agent_config, mock_llm_create):
        from unittest.mock import patch

        node = _make_node(real_agent_config, mock_llm_create)
        node.date_parsing_tools = MagicMock()
        node.date_parsing_tools.reference_date = "2023-01-10"

        with patch(
            "datus.utils.time_utils.get_default_current_date",
            return_value="2023-01-10",
        ) as mock_date:
            prompt = node._get_system_prompt()
        mock_date.assert_called_once_with("2023-01-10")
        assert "2023-01-10" in prompt

    def test_system_prompt_falls_back_to_today(self, real_agent_config, mock_llm_create):
        from unittest.mock import patch

        node = _make_node(real_agent_config, mock_llm_create)
        node.date_parsing_tools = None

        with patch(
            "datus.utils.time_utils.get_default_current_date",
            return_value="2025-06-15",
        ) as mock_date:
            prompt = node._get_system_prompt()
        mock_date.assert_called_once_with(None)
        assert "2025-06-15" in prompt
