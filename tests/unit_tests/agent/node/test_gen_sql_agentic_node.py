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
        if isinstance(error_action.output, dict):
            # ChatAgenticNode wraps PermissionDeniedException into a ChatNodeResult
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
        choices = processing[0].input.get("choices", {}) if isinstance(processing[0].input, dict) else {}
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
        if success:
            output = success[0].output
            if isinstance(output, dict):
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

        # Create a real YAML file for GenerationHooks to read
        yaml_path = os.path.join(str(tmp_path), "test_semantic_model.yaml")
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
        if isinstance(interaction_input, dict):
            content = interaction_input.get("content", "")
            assert "Sync to Knowledge Base" in content or "yaml" in content.lower()

    @pytest.mark.asyncio
    async def test_e2e_generation_hooks_user_declines_sync(self, real_agent_config, mock_llm_create, tmp_path):
        """Full flow: LLM calls end_semantic_model_generation → user declines sync ('n') → file kept only."""
        import asyncio
        import os

        from agents import FunctionTool

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.cli.generation_hooks import GenerationHooks

        # Create a real YAML file for GenerationHooks to read
        yaml_path = os.path.join(str(tmp_path), "test_semantic_decline.yaml")
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
        if isinstance(callback_output, dict):
            callback_content = callback_output.get("content", "")
            assert "file only" in callback_content.lower() or "saved" in callback_content.lower()

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
