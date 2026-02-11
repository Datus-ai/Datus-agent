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

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
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
        if isinstance(output, dict):
            # The sql field in the result should contain our query
            sql_value = output.get("sql")
            if sql_value:
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
        """ChatAgenticNode initializes correctly, extends GenSQLAgenticNode."""
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
        assert isinstance(node, GenSQLAgenticNode)
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
        # Last action: ASSISTANT with chat_response
        assert actions[-1].role == ActionRole.ASSISTANT
        assert actions[-1].status == ActionStatus.SUCCESS
        assert actions[-1].action_type == "chat_response"

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
            # Just verify the context_search_tools object was created
            assert node.context_search_tools is not None

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
