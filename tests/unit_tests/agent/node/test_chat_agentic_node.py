# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ChatAgenticNode independence from GenSQLAgenticNode.

Tests verify:
- ChatAgenticNode inherits from AgenticNode, NOT GenSQLAgenticNode
- ChatNodeResult has no sql field
- ChatAgenticNode produces markdown output without SQL/JSON parsing
- ChatAgenticNode has skills and permissions support
- MCP server setup logic
- setup_input / update_context workflow integration
- execute_stream error handling (cancellation, general exceptions, ExecutionInterrupted)
- _get_system_prompt fallback and error paths
- _build_plan_prompt structured / non-structured content branches
- _update_database_connection
- Summary report fallback logic in execute_stream

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from tests.unit_tests.mock_llm_model import MockToolCall, build_simple_response, build_tool_then_response

# ===========================================================================
# ChatAgenticNode Inheritance Tests
# ===========================================================================


class TestChatAgenticNodeInheritance:
    """Verify ChatAgenticNode is independent from GenSQLAgenticNode."""

    def test_inherits_from_agentic_node(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode inherits from AgenticNode."""
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_inherit",
            description="Test inheritance",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node, AgenticNode)

    def test_not_instance_of_gen_sql(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode is NOT a subclass of GenSQLAgenticNode."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = ChatAgenticNode(
            node_id="test_no_gen_sql",
            description="Test not gen_sql",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert not isinstance(node, GenSQLAgenticNode)

    def test_node_name_is_chat(self, real_agent_config, mock_llm_create):
        """get_node_name() returns 'chat'."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_name",
            description="Test name",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.get_node_name() == "chat"


# ===========================================================================
# ChatNodeResult Tests
# ===========================================================================


class TestChatNodeResult:
    """Verify ChatNodeResult has no sql field."""

    def test_no_sql_field(self):
        """ChatNodeResult does not have a sql field."""
        result = ChatNodeResult(
            success=True,
            response="Hello, how can I help?",
            tokens_used=100,
        )

        assert not hasattr(result, "sql") or "sql" not in result.model_fields
        assert result.response == "Hello, how can I help?"
        assert result.tokens_used == 100

    def test_rejects_sql_kwarg(self):
        """ChatNodeResult raises ValidationError if sql is passed (extra='forbid')."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatNodeResult(
                success=True,
                response="Test",
                sql="SELECT 1",  # type: ignore[call-arg]
                tokens_used=0,
            )

    def test_model_dump_no_sql(self):
        """model_dump() output does not contain 'sql' key."""
        result = ChatNodeResult(
            success=True,
            response="Test response",
            tokens_used=50,
        )

        dumped = result.model_dump()
        assert "sql" not in dumped
        assert dumped["response"] == "Test response"


# ===========================================================================
# ChatAgenticNode Tool Setup Tests
# ===========================================================================


class TestChatAgenticNodeToolSetup:
    """Verify ChatAgenticNode has all expected tools."""

    def test_has_db_tools(self, real_agent_config, mock_llm_create):
        """Chat node has database tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.database import DBFuncTool

        node = ChatAgenticNode(
            node_id="test_db",
            description="Test db tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node.db_func_tool, DBFuncTool)
        tool_names = [t.name for t in node.tools]
        # Should have at least some db tools
        assert any("table" in name or "query" in name or "sql" in name for name in tool_names)

    def test_has_context_search_tools(self, real_agent_config, mock_llm_create):
        """Chat node has context search tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.context_search import ContextSearchTools

        node = ChatAgenticNode(
            node_id="test_ctx",
            description="Test context search",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node.context_search_tools, ContextSearchTools)

    def test_has_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Chat node has filesystem tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        node = ChatAgenticNode(
            node_id="test_fs",
            description="Test filesystem tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node.filesystem_func_tool, FilesystemFuncTool)

    def test_filesystem_strict_from_agent_config(self, real_agent_config, mock_llm_create):
        """agent_config.filesystem_strict = True propagates into the tool
        and the permission-hook policy. This is how API / gateway bootstraps
        force strict mode for every node they spawn."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        previous_strict = getattr(real_agent_config, "filesystem_strict", False)
        real_agent_config.filesystem_strict = True
        try:
            node = ChatAgenticNode(
                node_id="test_fs_strict",
                description="Test fs strict",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )
            assert node.filesystem_func_tool.strict is True
            # Contract is mandatory: strict mode must reach the hook policy,
            # otherwise EXTERNAL paths would still trigger broker prompts in
            # non-interactive bootstraps (the whole point of strict mode).
            # PermissionHooks is built lazily by ``_ensure_permission_hooks``
            # (called from ``_compose_hooks`` on the first LLM turn); trigger
            # it explicitly here to inspect the constructed policy.
            node._ensure_permission_hooks()
            assert node.permission_hooks.fs_policy.strict is True
        finally:
            real_agent_config.filesystem_strict = previous_strict

    def test_filesystem_root_from_node_config_workspace(self, real_agent_config, mock_llm_create, tmp_path):
        """``node_config.workspace_root`` replaces the fs tool root so a node
        can be scoped to a directory outside the project root. Verified via
        ``_resolve_workspace_root`` -> ``_make_filesystem_tool``."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        custom_workspace = tmp_path / "node_ws"
        custom_workspace.mkdir()

        # Snapshot the whole container, not just the "chat" key — if the
        # fixture started with ``agentic_nodes = None`` we must restore that
        # exact state, otherwise downstream tests that assert ``is None`` see
        # a stray ``{}``.
        prev_container = getattr(real_agent_config, "agentic_nodes", None)
        if not prev_container:
            real_agent_config.agentic_nodes = {}
        prev = real_agent_config.agentic_nodes.get("chat")
        real_agent_config.agentic_nodes["chat"] = {"workspace_root": str(custom_workspace)}
        try:
            node = ChatAgenticNode(
                node_id="test_node_ws",
                description="Test node workspace",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )
            assert node.filesystem_func_tool.root_path == str(custom_workspace)
        finally:
            if prev is not None:
                real_agent_config.agentic_nodes["chat"] = prev
            else:
                real_agent_config.agentic_nodes.pop("chat", None)
            if not prev_container:
                real_agent_config.agentic_nodes = prev_container

    def test_has_date_parsing_tools(self, real_agent_config, mock_llm_create):
        """Chat node has date parsing tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.date_parsing_tools import DateParsingTools

        node = ChatAgenticNode(
            node_id="test_date",
            description="Test date parsing",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node.date_parsing_tools, DateParsingTools)

    def test_has_bash_tool_registered_under_bash_tools_category(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode owns a general-purpose BashTool registered to the
        ``bash_tools`` permission category and exposed via ``node.tools``."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.bash_tool import BashTool

        node = ChatAgenticNode(
            node_id="test_bash",
            description="Test bash tool",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node.bash_tool, BashTool)

        # ``["*"]`` pattern: tool is exposed; per-call gating is handled by
        # the PermissionManager ``bash_tools.bash`` ASK rule.
        tool_names = [t.name for t in node.tools]
        assert "bash" in tool_names

        # Permission category mapping is mandatory — without it, the ASK rule
        # added in ``profiles._NORMAL_RULES`` would never fire.
        assert node.tool_registry.get("bash") == "bash_tools"

    def test_context_search_failure_does_not_remove_db_tools(self, real_agent_config, mock_llm_create):
        """Embedding/context setup failures should only remove context tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.database import DBFuncTool

        with patch("datus.agent.node.chat_agentic_node.ContextSearchTools", side_effect=RuntimeError("hf offline")):
            node = ChatAgenticNode(
                node_id="test_context_degraded",
                description="Test context degradation",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )

        tool_names = {tool.name for tool in node.tools}
        registry = node.tool_registry.to_dict()

        assert isinstance(node.db_func_tool, DBFuncTool)
        assert node.context_search_tools is None
        assert "list_tables" in tool_names
        assert "context_search_tools" not in set(registry.values())
        assert registry["list_tables"] == "db_tools"
        assert "context_search_tools" in node.degraded_capabilities
        assert "hf offline" in node.degraded_capabilities["context_search_tools"]

    def test_reference_template_failure_does_not_remove_db_tools(self, real_agent_config, mock_llm_create):
        """Embedding/template setup failures should only remove template tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.database import DBFuncTool

        with patch("datus.agent.node.chat_agentic_node.ReferenceTemplateTools", side_effect=RuntimeError("hf offline")):
            node = ChatAgenticNode(
                node_id="test_reference_template_degraded",
                description="Test reference template degradation",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )

        tool_names = {tool.name for tool in node.tools}
        registry = node.tool_registry.to_dict()

        assert isinstance(node.db_func_tool, DBFuncTool)
        assert node.reference_template_tools is None
        assert "list_tables" in tool_names
        assert "reference_template_tools" not in set(registry.values())
        assert registry["list_tables"] == "db_tools"
        assert "reference_template_tools" in node.degraded_capabilities
        assert "hf offline" in node.degraded_capabilities["reference_template_tools"]

    def test_has_ask_user_tools(self, real_agent_config, mock_llm_create):
        """Chat node has ask_user tool set up via _setup_ask_user_tool."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.func_tool.ask_user_tools import AskUserTool

        node = ChatAgenticNode(
            node_id="test_ask_user",
            description="Test ask user tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert isinstance(node.ask_user_tool, AskUserTool)
        tool_names = [t.name for t in node.ask_user_tool.available_tools()]
        assert "ask_user" in tool_names

    def test_workflow_mode_excludes_ask_user_tool(self, real_agent_config, mock_llm_create):
        """In workflow mode, ask_user tool is not registered to avoid blocking pipelines."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_workflow",
            description="Test workflow mode",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.execution_mode == "workflow"
        assert node.ask_user_tool is None
        tool_names = [t.name for t in node.tools]
        assert "ask_user" not in tool_names

    def test_interactive_mode_is_default(self, real_agent_config, mock_llm_create):
        """Default execution_mode is 'interactive' with ask_user tool registered."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_interactive_default",
            description="Test interactive default",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.execution_mode == "interactive"
        assert [t.name for t in node.ask_user_tool.available_tools()] == ["ask_user"]


@pytest.mark.acceptance
@pytest.mark.llm_harness
class TestChatMemoryFlowAcceptance:
    """Deterministic chain-level coverage for chat memory write and later use."""

    @pytest.mark.asyncio
    async def test_chat_turn_writes_memory_and_later_turn_receives_it(self, real_agent_config, mock_llm_create):
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        memory_text = "Net revenue excludes refunds."
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall(
                            name="add_memory",
                            arguments={"content": memory_text},
                        )
                    ],
                    content="Saved the revenue convention to memory.",
                ),
                build_simple_response("I will apply the saved net revenue convention."),
            ]
        )

        first_turn = ChatAgenticNode(
            node_id="chat_memory_writer",
            description="Write chat memory",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        first_turn.input = ChatNodeInput(user_message="Remember that net revenue excludes refunds.")

        action_manager = ActionHistoryManager()
        async for _ in first_turn.execute_stream(action_manager):
            pass

        memory_file = Path(real_agent_config.project_root) / ".datus" / "memory" / "chat" / "MEMORY.md"
        assert memory_file.read_text(encoding="utf-8") == memory_text

        second_turn = ChatAgenticNode(
            node_id="chat_memory_reader",
            description="Read chat memory",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        second_turn.input = ChatNodeInput(user_message="How should I define net revenue?")

        action_manager = ActionHistoryManager()
        async for _ in second_turn.execute_stream(action_manager):
            pass

        tool_calls = [item for item in mock_llm_create.tool_results if item["tool"] == "add_memory"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["executed"] is True

        second_model_call = mock_llm_create.call_history[-1]
        assert second_model_call["method"] == "generate_with_tools_stream"
        assert "Net revenue excludes refunds" in second_model_call["instruction"]


# ===========================================================================
# ChatAgenticNode execute_stream Tests
# ===========================================================================


@pytest.mark.component
@pytest.mark.llm_harness
class TestChatAgenticNodeExecuteStream:
    """Verify execute_stream produces markdown output without SQL extraction."""

    @pytest.mark.asyncio
    async def test_execute_stream_produces_chat_response(self, real_agent_config, mock_llm_create):
        """execute_stream yields a final ASSISTANT action (no separate chat_response)."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_stream",
            description="Test stream",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(responses=[build_simple_response("Here is a helpful answer in **markdown**.")])

        node.input = ChatNodeInput(
            user_message="How can I help?",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Should have at least user action + final action
        assert len(actions) >= 2

        final_action = actions[-1]
        assert final_action.role == ActionRole.ASSISTANT
        assert final_action.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_result_has_no_sql(self, real_agent_config, mock_llm_create):
        """Final result in action output does not contain sql field."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_no_sql",
            description="Test no sql",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(responses=[build_simple_response("Just a text response.")])

        node.input = ChatNodeInput(
            user_message="Tell me about the database",
            database="california_schools",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        final_action = actions[-1]
        assert isinstance(final_action.output, dict)
        assert "sql" not in final_action.output

    @pytest.mark.asyncio
    async def test_execute_stream_raises_when_no_input(self, real_agent_config, mock_llm_create):
        """execute_stream raises ValueError when input is not set."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_no_input",
            description="Test no input",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = None

        ahm = ActionHistoryManager()
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException, match="Missing required field"):
            async for _ in node.execute_stream(ahm):
                pass


# ===========================================================================
# ChatAgenticNode update_context Tests
# ===========================================================================


class TestChatAgenticNodeUpdateContext:
    """Verify update_context does not add SQL to workflow context."""

    def test_update_context_no_sql(self, real_agent_config, mock_llm_create):
        """update_context returns success without adding SQL to workflow."""
        from unittest.mock import MagicMock

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_ctx_update",
            description="Test context update",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Set a result
        node.result = ChatNodeResult(
            success=True,
            response="Here is some analysis.",
            tokens_used=50,
        )

        # Mock workflow
        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)

        assert result["success"] is True
        # Should NOT add any SQL context
        assert len(workflow.context.sql_contexts) == 0

    def test_update_context_returns_failure_when_no_result(self, real_agent_config, mock_llm_create):
        """update_context returns failure dict when self.result is None."""
        from unittest.mock import MagicMock

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_ctx_no_result",
            description="Test no result update",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.result = None

        workflow = MagicMock()
        result = node.update_context(workflow)

        assert result["success"] is False
        assert "No result" in result["message"]


# ===========================================================================
# _update_database_connection Tests
# ===========================================================================


class TestChatAgenticNodeUpdateDatabaseConnection:
    """Verify _update_database_connection rebuilds the DB tool bound to the target database."""

    def test_update_database_connection_rebuilds_tools(self, real_agent_config, mock_llm_create):
        """_update_database_connection creates a new DBFuncTool (bound to db) and rebuilds tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_update_db",
            description="Test update db conn",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        original_db_tool = node.db_func_tool

        # Update to the same database available in the fixture.
        node._update_database_connection("california_schools")

        # db_func_tool should be a new instance carrying the requested default_database.
        assert node.db_func_tool is not original_db_tool
        assert node.db_func_tool._default_database == "california_schools"
        # Tools should be rebuilt and contain core db tools.
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names


# ===========================================================================
# _setup_mcp_servers Tests
# ===========================================================================


class TestChatAgenticNodeMCPSetup:
    """Verify MCP server setup handles various configurations."""

    def test_mcp_servers_empty_when_no_config(self, real_agent_config, mock_llm_create):
        """MCP servers dict is empty when no mcp config is set."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_mcp_empty",
            description="Test empty MCP",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # The default fixture has no MCP config, so mcp_servers should be empty
        assert isinstance(node.mcp_servers, dict)
        assert len(node.mcp_servers) == 0

    def test_mcp_setup_uses_configured_servers_only(self, real_agent_config, mock_llm_create):
        """MCP server setup delegates configured names to MCPManager."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_mcp_configured",
            description="Test configured MCP",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.node_config = {"mcp": "custom_server"}

        mock_server = MagicMock()
        with patch.object(node, "_setup_mcp_server_from_config", return_value=mock_server) as mock_setup:
            result = node._setup_mcp_servers()
        mock_setup.assert_called_once_with("custom_server")
        assert result == {"custom_server": mock_server}

    def test_setup_mcp_server_from_config_returns_none_for_unknown_server(self, real_agent_config, mock_llm_create):
        """_setup_mcp_server_from_config returns None for non-existent server name."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_mcp_unknown",
            description="Test unknown MCP server",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        result = node._setup_mcp_server_from_config("non_existent_server_xyz")
        assert result is None


# ===========================================================================
# _get_system_prompt Tests
# ===========================================================================


class TestChatAgenticNodeSystemPrompt:
    """Verify system prompt generation and error handling."""

    def test_get_system_prompt_returns_string(self, real_agent_config, mock_llm_create):
        """_get_system_prompt returns a non-empty string for valid template."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_prompt",
            description="Test system prompt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        prompt = node._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) >= 100

    def test_get_system_prompt_excludes_permission_profile(self, real_agent_config, mock_llm_create):
        """The permission profile is enforced by hooks at tool-call time, never prompted.

        Keeping it out of the system prompt also keeps the frozen per-session
        snapshot byte-stable across a runtime /profile switch.
        """
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        real_agent_config.active_profile_name = "dangerous"
        node = ChatAgenticNode(
            node_id="test_prompt_profile",
            description="Test permission profile absent from prompt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        prompt = node._get_system_prompt()

        assert "Current permission profile" not in prompt
        assert "dangerous" not in prompt

    def test_workflow_prompt_does_not_advertise_ask_user(self, real_agent_config, mock_llm_create):
        """Workflow chat has no ask_user tool, so the prompt must not route to it."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_prompt_workflow",
            description="Test workflow prompt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        prompt = node._get_system_prompt()

        assert "Ask user tool (`ask_user`)" not in prompt
        assert "call `ask_user` FIRST" not in prompt
        assert "No ask_user tool is available" in prompt
        assert "stop with a concise missing-information response" in prompt

    def test_interactive_prompt_advertises_ask_user(self, real_agent_config, mock_llm_create):
        """Interactive chat keeps ask_user routing guidance."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_prompt_interactive",
            description="Test interactive prompt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
            execution_mode="interactive",
        )

        prompt = node._get_system_prompt()

        assert "Ask user tool (`ask_user`)" in prompt
        assert "call `ask_user` FIRST" in prompt

    def test_get_system_prompt_fallback_on_missing_template(self, real_agent_config, mock_llm_create):
        """_get_system_prompt falls back to chat_system when configured template is missing."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_prompt_fallback",
            description="Test prompt fallback",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Override the system_prompt config to use a non-existent template
        node.node_config["system_prompt"] = "nonexistent_template_xyz"

        # Should fall back to chat_system template without raising
        prompt = node._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) >= 100

    def test_get_system_prompt_raises_on_template_error(self, real_agent_config, mock_llm_create):
        """_get_system_prompt raises DatusException when both primary and fallback templates fail."""
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.utils.exceptions import DatusException

        node = ChatAgenticNode(
            node_id="test_prompt_error",
            description="Test prompt error",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Patch render_template to raise a non-FileNotFoundError exception
        with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_gpm:
            mock_gpm.return_value.render_template.side_effect = RuntimeError("broken")
            with pytest.raises(DatusException):
                node._get_system_prompt()


# ===========================================================================
# setup_input Tests
# ===========================================================================


class TestChatAgenticNodeSetupInput:
    """Verify setup_input creates and updates ChatNodeInput from Workflow."""

    def test_setup_input_creates_new_input(self, real_agent_config, mock_llm_create):
        """setup_input creates ChatNodeInput when self.input is None."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.workflow import Workflow
        from datus.schemas.node_models import SqlTask

        node = ChatAgenticNode(
            node_id="test_setup_new",
            description="Test setup input new",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = None

        task = SqlTask(
            task="Tell me about the schools",
            database_name="california_schools",
            catalog_name="test_catalog",
            schema_name="public",
        )
        workflow = Workflow(name="test_workflow", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)

        assert result["success"] is True
        assert node.input.user_message == "Tell me about the schools"
        assert node.input.database == "california_schools"
        assert node.input.catalog == "test_catalog"
        assert node.input.db_schema == "public"

    def test_setup_input_updates_existing_input(self, real_agent_config, mock_llm_create):
        """setup_input updates existing ChatNodeInput fields from workflow."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.workflow import Workflow
        from datus.schemas.node_models import SqlTask

        node = ChatAgenticNode(
            node_id="test_setup_update",
            description="Test setup input update",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Pre-set an input
        node.input = ChatNodeInput(user_message="old message", database="old_db")

        task = SqlTask(
            task="New question about data",
            database_name="california_schools",
            catalog_name="new_catalog",
            schema_name="new_schema",
        )
        workflow = Workflow(name="test_workflow", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)

        assert result["success"] is True
        assert node.input.user_message == "New question about data"
        assert node.input.database == "california_schools"
        assert node.input.catalog == "new_catalog"
        assert node.input.db_schema == "new_schema"

    def test_setup_input_with_plan_mode_metadata(self, real_agent_config, mock_llm_create):
        """setup_input reads plan_mode and auto_execute_plan from workflow metadata."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.workflow import Workflow
        from datus.schemas.node_models import SqlTask

        node = ChatAgenticNode(
            node_id="test_setup_plan",
            description="Test setup plan mode",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = None

        task = SqlTask(task="Plan mode task", database_name="california_schools")
        workflow = Workflow(name="test_wf", task=task, agent_config=real_agent_config)
        workflow.metadata["plan_mode"] = True
        workflow.metadata["auto_execute_plan"] = True

        result = node.setup_input(workflow)

        assert result["success"] is True
        assert node.input.plan_mode is True
        assert node.input.auto_execute_plan is True


# ===========================================================================
# execute_stream Error Handling Tests
# ===========================================================================


class TestChatAgenticNodeExecuteStreamErrors:
    """Verify execute_stream error handling for cancellation and general exceptions."""

    @pytest.mark.asyncio
    async def test_execute_stream_handles_general_exception(self, real_agent_config, mock_llm_create):
        """General exceptions yield a FAILED action with error message."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_error",
            description="Test error handling",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Configure mock to raise an exception
        mock_llm_create.reset(responses=[])  # No responses will cause empty response

        # Patch generate_with_tools_stream to raise an error
        original_method = mock_llm_create.generate_with_tools_stream

        async def raising_stream(*args, **kwargs):
            raise RuntimeError("Simulated LLM failure")
            yield  # unreachable - makes this an async generator

        mock_llm_create.generate_with_tools_stream = raising_stream

        node.input = ChatNodeInput(user_message="Test error", database="california_schools")
        ahm = ActionHistoryManager()

        try:
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

            # Should have yielded at least the initial user action and a failure action
            assert len(actions) >= 2
            final_action = actions[-1]
            assert final_action.status == ActionStatus.FAILED
            assert "Simulated LLM failure" in str(final_action.output.get("error", ""))
        finally:
            mock_llm_create.generate_with_tools_stream = original_method

    @pytest.mark.asyncio
    async def test_execute_stream_user_cancellation_via_execution_interrupted(self, real_agent_config, mock_llm_create):
        """User cancellations propagate as ``ExecutionInterrupted``.

        Cancellation is the single, typed channel used everywhere else
        (``execute_stream_with_interactions`` converts it into the
        ``interrupted`` SUCCESS action). Generic ``Exception("User
        cancelled ...")`` strings raised by upstream SDK plumbing fall
        through to the regular error branch — they're real failures, not
        user-initiated cancels.
        """
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        node = ChatAgenticNode(
            node_id="test_cancel",
            description="Test cancellation",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        original_method = mock_llm_create.generate_with_tools_stream

        async def cancel_stream(*args, **kwargs):
            raise ExecutionInterrupted("Ctrl+C")
            yield  # unreachable - makes this an async generator

        mock_llm_create.generate_with_tools_stream = cancel_stream

        node.input = ChatNodeInput(user_message="Cancel me", database="california_schools")
        ahm = ActionHistoryManager()

        try:
            with pytest.raises(ExecutionInterrupted):
                async for _ in node.execute_stream(ahm):
                    pass
        finally:
            mock_llm_create.generate_with_tools_stream = original_method

    @pytest.mark.asyncio
    async def test_execute_stream_propagates_execution_interrupted(self, real_agent_config, mock_llm_create):
        """ExecutionInterrupted is re-raised without being caught."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        node = ChatAgenticNode(
            node_id="test_interrupt",
            description="Test interrupt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        original_method = mock_llm_create.generate_with_tools_stream

        async def interrupt_stream(*args, **kwargs):
            raise ExecutionInterrupted("Ctrl+C pressed")
            yield  # unreachable - makes this an async generator

        mock_llm_create.generate_with_tools_stream = interrupt_stream

        node.input = ChatNodeInput(user_message="Interrupt me", database="california_schools")
        ahm = ActionHistoryManager()

        try:
            with pytest.raises(ExecutionInterrupted):
                async for _ in node.execute_stream(ahm):
                    pass
        finally:
            mock_llm_create.generate_with_tools_stream = original_method

    @pytest.mark.asyncio
    async def test_execute_stream_creates_default_action_history_manager(self, real_agent_config, mock_llm_create):
        """execute_stream creates a default ActionHistoryManager when None is passed."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_default_ahm",
            description="Test default ahm",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(responses=[build_simple_response("Default AHM test response.")])

        node.input = ChatNodeInput(user_message="Test default", database="california_schools")

        # Pass None as action_history_manager - should create one internally
        actions = []
        async for action in node.execute_stream(None):
            actions.append(action)

        assert len(actions) >= 2
        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_passes_node_name_as_agent_name(self, real_agent_config, mock_llm_create):
        """execute_stream passes the chat node name through to the model trace metadata."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_trace_agent_name",
            description="Test trace agent name",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(responses=[build_simple_response("Trace name test response.")])
        node.input = ChatNodeInput(user_message="Test trace naming", database="california_schools")

        actions = []
        async for action in node.execute_stream(ActionHistoryManager()):
            actions.append(action)

        assert len(actions) >= 2
        assert mock_llm_create.call_history[-1]["method"] == "generate_with_tools_stream"
        assert mock_llm_create.call_history[-1]["kwargs"]["agent_name"] == "chat"


# ===========================================================================
# execute_stream with Tool Calls Tests
# ===========================================================================


@pytest.mark.component
@pytest.mark.llm_harness
class TestChatAgenticNodeExecuteStreamWithTools:
    """Verify execute_stream correctly handles tool calls and content extraction."""

    @pytest.mark.asyncio
    async def test_execute_stream_with_tool_call(self, real_agent_config, mock_llm_create):
        """execute_stream correctly processes tool calls followed by final response."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_tool_call",
            description="Test tool call",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[MockToolCall(name="list_tables", arguments="{}")],
                    content="Here are the tables in your database.",
                ),
            ]
        )

        node.input = ChatNodeInput(user_message="What tables are available?", database="california_schools")
        ahm = ActionHistoryManager()

        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Should have: user action + tool processing + tool complete + assistant response + final chat_response
        assert len(actions) >= 4

        # Check tool actions
        tool_actions = [a for a in actions if a.role == ActionRole.TOOL]
        assert len(tool_actions) >= 1

        # Final action should be successful
        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS
        assert final_action.role == ActionRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_execute_stream_collects_token_usage(self, real_agent_config, mock_llm_create):
        """execute_stream extracts token usage from action history into result."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_tokens",
            description="Test token usage",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(responses=[build_simple_response("Token usage test.")])

        node.input = ChatNodeInput(user_message="Count tokens", database="california_schools")
        ahm = ActionHistoryManager()

        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        final_action = actions[-1]
        result_data = final_action.output
        # tokens_used should be extracted from mock usage (700 per _mock_usage)
        assert result_data.get("tokens_used", 0) == 700

    @pytest.mark.asyncio
    async def test_execute_stream_execution_stats(self, real_agent_config, mock_llm_create):
        """execute_stream builds execution_stats with tool call counts."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_exec_stats",
            description="Test exec stats",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[MockToolCall(name="list_tables", arguments="{}")],
                    content="Found the tables.",
                ),
            ]
        )

        node.input = ChatNodeInput(user_message="List tables", database="california_schools")
        ahm = ActionHistoryManager()

        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        final_action = actions[-1]
        stats = final_action.output.get("execution_stats", {})
        assert stats.get("total_actions", 0) > 0
        assert stats.get("tool_calls_count", 0) >= 1
        assert "list_tables" in stats.get("tools_used", [])

    @pytest.mark.asyncio
    async def test_execute_stream_dict_response_value_does_not_crash(self, real_agent_config, mock_llm_create):
        """execute_stream converts dict response values to string, preventing Pydantic ValidationError.

        Regression test: when a tool result dict (e.g. from execute_sql) is stored under the
        "response" key in an action output, the or-chain extraction must not pass the raw dict
        to ChatNodeResult(response=...) which expects a str.
        """
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = ChatAgenticNode(
            node_id="test_dict_response",
            description="Test dict response handling",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="Show me data", database="california_schools")

        # Simulate the problematic scenario: the last successful action's output
        # has "response" as a dict (e.g. DB tool result) and no string "content".
        async def mock_execute(*args, action_history_manager, **kwargs):
            action = ActionHistory(
                action_id="msg_dict",
                role=ActionRole.ASSISTANT,
                messages="Query result",
                action_type="message",
                input={},
                output={
                    "content": "",
                    "response": {"success": 1, "error": None, "expression_type": "rows"},
                },
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(action)
            yield action

        with patch.object(mock_llm_create, "generate_with_tools_stream", mock_execute):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS
        assert final_action.action_type == "chat_response"
        # Key assertion: response must be a string, not a dict
        assert isinstance(final_action.output["response"], str)

    @pytest.mark.asyncio
    async def test_execute_stream_uses_tool_summary_when_model_gives_no_response(
        self, real_agent_config, mock_llm_create
    ):
        """Tool raw_output stays out of the final response, but its summary can be used."""
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = ChatAgenticNode(
            node_id="test_tool_raw_output_not_response",
            description="Test tool raw output is ignored",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="List tables", database="california_schools")

        async def mock_execute(*args, action_history_manager, **kwargs):
            action = ActionHistory(
                action_id="complete_tool",
                role=ActionRole.TOOL,
                messages="Tool call: list_tables",
                action_type="list_tables",
                input={"function_name": "list_tables", "arguments": "{}"},
                output={
                    "success": True,
                    "raw_output": {
                        "success": 1,
                        "error": None,
                        "result": [{"type": "table", "name": "orders"}],
                    },
                    "summary": "1 table: orders",
                },
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(action)
            yield action

        with patch.object(mock_llm_create, "generate_with_tools_stream", mock_execute):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        final_action = actions[-1]
        assert final_action.action_type == "chat_response"
        assert final_action.output["response"] == "1 table: orders"

    @pytest.mark.asyncio
    async def test_execute_stream_filters_all_thinking_text(self, real_agent_config, mock_llm_create):
        """Provider-marked thinking text never lands in the final response.

        Behavior change (template refactor): the unified ``_stream_once``
        helper drops any assistant chunk with ``is_thinking=True`` regardless
        of whether a tool result already arrived. Pre-refactor ChatNode had a
        ``tool_result_seen`` gate that promoted post-tool thinking to the
        final response — this distinction is gone now. The tool's summary
        becomes the response fallback (see ``last_tool_summary`` path).
        """
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = ChatAgenticNode(
            node_id="test_final_thinking_after_tool",
            description="Test final thinking text after tool is preserved",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="List tables", database="california_schools")

        async def mock_execute(*args, action_history_manager, **kwargs):
            pre_tool_thinking = ActionHistory(
                action_id="thinking_before_tool",
                role=ActionRole.ASSISTANT,
                messages="thinking",
                action_type="response",
                input={},
                output={"content": "I should inspect the database first.", "is_thinking": True},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(pre_tool_thinking)
            yield pre_tool_thinking

            tool_action = ActionHistory(
                action_id="complete_tool",
                role=ActionRole.TOOL,
                messages="Tool call: list_tables",
                action_type="list_tables",
                input={"function_name": "list_tables", "arguments": "{}"},
                output={"success": True, "summary": "1 table: orders"},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(tool_action)
            yield tool_action

            final_thinking = ActionHistory(
                action_id="thinking_after_tool",
                role=ActionRole.ASSISTANT,
                messages="final",
                action_type="response",
                input={},
                output={"content": "The database has one table: orders.", "is_thinking": True},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_thinking)
            yield final_thinking

        with patch.object(mock_llm_create, "generate_with_tools_stream", mock_execute):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        final_action = actions[-1]
        assert final_action.action_type == "chat_response"
        # Both thinking chunks are filtered; tool summary fills the response.
        assert final_action.output["response"] == "1 table: orders"

    @pytest.mark.asyncio
    async def test_execute_stream_extracts_string_content_from_action(self, real_agent_config, mock_llm_create):
        """execute_stream correctly extracts string content from action output's content key.

        Covers the isinstance(candidate, str) branch in the stream loop extraction.
        """
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = ChatAgenticNode(
            node_id="test_str_content",
            description="Test string content extraction",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="Hello", database="california_schools")

        async def mock_execute(*args, action_history_manager, **kwargs):
            action = ActionHistory(
                action_id="msg_str",
                role=ActionRole.ASSISTANT,
                messages="Text response",
                action_type="message",
                input={},
                output={"content": "Here are your results in markdown."},
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(action)
            yield action

        with patch.object(mock_llm_create, "generate_with_tools_stream", mock_execute):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS
        assert final_action.output["response"] == "Here are your results in markdown."

    @pytest.mark.asyncio
    async def test_execute_stream_fallback_dict_in_text_key(self, real_agent_config, mock_llm_create):
        """Fallback extraction stringifies non-string candidate from last_successful_output.

        When the stream loop finds no content but last_successful_output has a dict
        in the "text" key (only checked in fallback, not in stream loop), the fallback
        must convert it to string rather than skipping it.
        """
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = ChatAgenticNode(
            node_id="test_fallback_dict_text",
            description="Test fallback dict text",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="Query", database="california_schools")

        async def mock_execute(*args, action_history_manager, **kwargs):
            # Action with "text" as dict — stream loop doesn't check "text",
            # so response_content stays empty. Fallback checks "text" and finds the dict.
            action = ActionHistory(
                action_id="tool_result",
                role=ActionRole.ASSISTANT,
                messages="Result",
                action_type="tool_output",
                input={},
                output={
                    "content": "",
                    "response": "",
                    "text": {"rows": [1, 2, 3], "total": 3},
                    "raw_output": "",
                },
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(action)
            yield action

        with patch.object(mock_llm_create, "generate_with_tools_stream", mock_execute):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS
        response = final_action.output["response"]
        assert isinstance(response, str)
        assert "rows" in response

    @pytest.mark.asyncio
    async def test_execute_stream_summary_report_dict_does_not_crash(self, real_agent_config, mock_llm_create):
        """execute_stream handles dict values in summary_report action outputs.

        Regression test: when a summary_report action has "markdown" or "content"
        as a dict, the fallback extraction must convert it to string.
        The summary_report is added to action_history_manager without being yielded
        through the stream so that earlier extraction points don't intercept it.
        """
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = ChatAgenticNode(
            node_id="test_summary_dict",
            description="Test summary report dict handling",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="Summarize", database="california_schools")

        async def mock_execute(*args, action_history_manager, **kwargs):
            # Add summary_report directly to action_history_manager (simulates sub-component adding it).
            # Do NOT yield it, so last_successful_output stays None and the summary_report
            # fallback loop is actually reached.
            summary_action = ActionHistory(
                action_id="summary_1",
                role=ActionRole.ASSISTANT,
                messages="Summary report",
                action_type="summary_report",
                input={},
                output={
                    "markdown": {"title": "Report", "sections": ["a", "b"]},
                    "content": "",
                },
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(summary_action)

            # Yield a non-dict output action so the stream has at least one item
            empty_action = ActionHistory(
                action_id="empty_1",
                role=ActionRole.ASSISTANT,
                messages="Processing",
                action_type="thinking",
                input={},
                output="",
                status=ActionStatus.SUCCESS,
            )
            yield empty_action

        with patch.object(mock_llm_create, "generate_with_tools_stream", mock_execute):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS
        assert isinstance(final_action.output["response"], str)
        assert "Report" in final_action.output["response"]


# ===========================================================================
# Plan Mode Tests
# ===========================================================================


class TestChatAgenticNodePlanMode:
    """Verify plan-mode lifecycle on ChatAgenticNode (state lives on AgenticNode)."""

    @pytest.mark.asyncio
    async def test_plan_mode_persists_across_turns_until_toggled_off(
        self, real_agent_config, mock_llm_create, tmp_path
    ):
        """plan_mode stays active across execute_stream runs; toggling off clears it."""
        import os

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            node = ChatAgenticNode(
                node_id="test_plan_persist",
                description="Test plan persistence",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )

            mock_llm_create.reset(
                responses=[
                    build_simple_response("Drafting the plan."),
                    build_simple_response("Continuing on the same plan."),
                ]
            )

            node.input = ChatNodeInput(
                user_message="Create a plan",
                database="california_schools",
                plan_mode=True,
            )
            ahm = ActionHistoryManager()
            async for _action in node.execute_stream(ahm):
                pass

            assert node.plan_mode_active is True
            first_plan_path = node.plan_file_path
            assert isinstance(first_plan_path, str) and first_plan_path
            assert os.path.exists(first_plan_path)

            # Second turn with plan_mode=True must reuse the same file path.
            node.input = ChatNodeInput(
                user_message="Refine the plan",
                database="california_schools",
                plan_mode=True,
            )
            ahm = ActionHistoryManager()
            async for _action in node.execute_stream(ahm):
                pass

            assert node.plan_mode_active is True
            assert node.plan_file_path == first_plan_path

            saved_plan_path = node.plan_file_path

            # User toggles plan mode off — only the active flag flips; the
            # plan_file_path is preserved for the lifetime of the session.
            node.input = ChatNodeInput(
                user_message="Just answer me",
                database="california_schools",
                plan_mode=False,
            )
            mock_llm_create.reset(responses=[build_simple_response("Normal answer.")])
            ahm = ActionHistoryManager()
            async for _action in node.execute_stream(ahm):
                pass

            assert node.plan_mode_active is False
            assert node.plan_file_path == saved_plan_path
            assert node.workflow_prompt_sent is False
        finally:
            os.chdir(cwd)

    def test_plan_file_path_is_never_reset_within_session(self, real_agent_config, mock_llm_create, tmp_path):
        """plan_file_path is allocated once per session and survives toggle/confirm cycles."""
        import os

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            node = ChatAgenticNode(
                node_id="test_plan_reuse",
                description="Reuse plan path",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )

            first_path = node.activate_plan_mode()
            assert os.path.exists(first_path)

            # confirm_plan exit shape: flip active flag, keep path.
            node.plan_mode_active = False
            node.workflow_prompt_sent = False

            # Re-activation reuses the same file.
            assert node.activate_plan_mode() == first_path
            assert node.plan_mode_active is True

            # Explicit Shift+Tab off also keeps the path (session-scoped).
            node.deactivate_plan_mode()
            assert node.plan_file_path == first_path
            assert node.plan_mode_active is False

            # Re-activation after toggle off still reuses the same file.
            assert node.activate_plan_mode() == first_path
        finally:
            os.chdir(cwd)

    def test_build_plan_mode_prompt_passes_auto_execute_flag(self, real_agent_config, mock_llm_create, tmp_path):
        """build_plan_mode_enhanced_prompt forwards input.auto_execute_plan to the template."""
        import os

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            node = ChatAgenticNode(
                node_id="test_auto_exec_prompt",
                description="Plan prompt auto-execute flag",
                node_type=NodeType.TYPE_CHAT,
                agent_config=real_agent_config,
            )
            node.activate_plan_mode()
            node.input = ChatNodeInput(user_message="hi", plan_mode=True, auto_execute_plan=True)

            with patch("datus.agent.node.agentic_node.get_prompt_manager") as mock_pm:
                mock_pm.return_value.render_template.return_value = "RENDERED"
                rendered = node.build_plan_mode_enhanced_prompt()

            assert rendered == "RENDERED"
            assert mock_pm.return_value.render_template.call_args.kwargs["auto_execute_plan"] is True

            # Interactive default: the flag resolves False.
            node.workflow_prompt_sent = False
            node.input = ChatNodeInput(user_message="hi", plan_mode=True)
            with patch("datus.agent.node.agentic_node.get_prompt_manager") as mock_pm:
                mock_pm.return_value.render_template.return_value = "RENDERED"
                node.build_plan_mode_enhanced_prompt()
            assert mock_pm.return_value.render_template.call_args.kwargs["auto_execute_plan"] is False
        finally:
            os.chdir(cwd)


# ===========================================================================
# _rebuild_tools Tests
# ===========================================================================


class TestChatAgenticNodeRebuildTools:
    """Verify _rebuild_tools correctly assembles tools from all sources."""

    def test_rebuild_tools_with_all_components(self, real_agent_config, mock_llm_create):
        """_rebuild_tools includes tools from db, context, date, filesystem, skills, sub_agent, ask_user."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_rebuild",
            description="Test rebuild tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # _rebuild_tools assembles core tools (db, context, date, fs, skills, sub_agent, ask_user)
        # but NOT platform_doc_tools (which are added separately in setup_tools)
        node._rebuild_tools()
        rebuilt_count = len(node.tools)

        # Should have tools from all core components
        assert rebuilt_count > 0
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_rebuild_tools_includes_ask_user(self, real_agent_config, mock_llm_create):
        """_rebuild_tools includes ask_user tool when available."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_rebuild_ask",
            description="Test rebuild with ask_user",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # ask_user_tool should be set up during __init__
        assert [tool.name for tool in node.ask_user_tool.available_tools()] == ["ask_user"]

        node._rebuild_tools()
        tool_names = [t.name for t in node.tools]
        assert "ask_user" in tool_names

    def test_orchestrator_origin_survives_rebuild(self, real_agent_config, mock_llm_create):
        """setup_tools() ends in _rebuild_tools(), which clears the list — so a tool
        only appended by its own _setup_* helper never reaches the model. That is
        exactly how submit_task_result shipped inert."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        real_agent_config._request_origin = "orchestrator"
        node = ChatAgenticNode(
            node_id="test_rebuild_task_result",
            description="Test rebuild with submit_task_result",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert "submit_task_result" in [t.name for t in node.tools]

        # A mid-session rebuild (task-database switch, skill reload) must keep it.
        node._rebuild_tools()
        assert "submit_task_result" in [t.name for t in node.tools]

    def test_submit_task_result_does_not_prompt(self, real_agent_config, mock_llm_create):
        """It is the channel the run reports through, so it must never ASK.

        The ``tools`` bucket has no ALLOW rule matching it, so without the
        injected one it falls to ``default=ASK`` and a finished dispatch stops
        to ask permission to say it finished. Asserting the resolved level, not
        the rule's presence: rules are last-match-wins while the injection
        inserts at the front, so "the rule is there" does not imply it decides.
        """
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.tools.permission.permission_config import PermissionLevel

        real_agent_config._request_origin = "orchestrator"
        node = ChatAgenticNode(
            node_id="test_task_result_permission",
            description="Test submit_task_result permission",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert (
            node.permission_manager.check_permission("tools", "submit_task_result", node.get_node_name())
            == PermissionLevel.ALLOW
        )

    def test_ordinary_chat_never_sees_submit_task_result(self, real_agent_config, mock_llm_create):
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        real_agent_config._request_origin = None
        node = ChatAgenticNode(
            node_id="test_rebuild_no_task_result",
            description="Test rebuild without submit_task_result",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        node._rebuild_tools()
        assert "submit_task_result" not in [t.name for t in node.tools]

    def test_rebuild_tools_resets_transformer_flag(self, real_agent_config, mock_llm_create):
        """Rebuilding replaces wrapped FunctionTools with fresh unwrapped ones,
        so plugin tool transformers must re-apply on the next hook composition
        (e.g. after a mid-session task-database switch)."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_rebuild_flag",
            description="Test rebuild resets transformer flag",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node._tool_transformers_applied = True

        node._rebuild_tools()

        assert node._tool_transformers_applied is False

    def test_rebuild_tools_with_no_optional_components(self, real_agent_config, mock_llm_create):
        """_rebuild_tools works when optional tool components are None."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_rebuild_empty",
            description="Test rebuild empty",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Clear all optional tools
        node.context_search_tools = None
        node.date_parsing_tools = None
        node.filesystem_func_tool = None
        node.sub_agent_task_tool = None
        node.ask_user_tool = None

        # Rebuild should still work with just db tools
        node._rebuild_tools()

        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "ask_user" not in tool_names


# ===========================================================================
# _get_node_permission_overrides Tests
# ===========================================================================


class TestChatAgenticNodePermissionOverrides:
    """Verify _get_node_permission_overrides extracts config correctly."""

    def test_returns_empty_dict_when_no_permissions_config(self, real_agent_config, mock_llm_create):
        """Returns empty dict when chat config has no 'permissions' key."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_perm_empty",
            description="Test empty permissions",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        result = node._get_node_permission_overrides()
        assert result == {}

    def test_returns_empty_dict_when_no_agent_config(self, real_agent_config, mock_llm_create):
        """Returns empty dict when agent_config is None."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_perm_no_config",
            description="Test no config permissions",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        original = node.agent_config
        node.agent_config = None

        result = node._get_node_permission_overrides()
        assert result == {}

        node.agent_config = original


class TestChatSystemPromptCurrentDate:
    """Verify current_date is injected into the system prompt."""

    def test_get_system_prompt_contains_current_date(self, real_agent_config, mock_llm_create):
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_prompt_date",
            description="Test current_date in prompt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        with patch(
            "datus.utils.time_utils.get_default_current_date",
            return_value="2025-06-15",
        ):
            prompt = node._get_system_prompt()
        assert "2025-06-15" in prompt


class TestChatAgenticNodeExecutionMode:
    """Verify the `execution_mode` constructor parameter controls ask_user_tool setup."""

    def _build(self, real_agent_config, execution_mode):
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        return ChatAgenticNode(
            node_id="test_execution_mode",
            description="Test execution_mode flag",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
            execution_mode=execution_mode,
        )

    def test_execution_mode_default_is_interactive(self, real_agent_config, mock_llm_create):
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_default_execution_mode",
            description="Default",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        assert node.execution_mode == "interactive"
        assert [tool.name for tool in node.ask_user_tool.available_tools()] == ["ask_user"]

    def test_workflow_mode_disables_ask_user_tool(self, real_agent_config, mock_llm_create):
        node = self._build(real_agent_config, execution_mode="workflow")
        assert node.execution_mode == "workflow"
        assert node.ask_user_tool is None

    def test_interactive_mode_keeps_ask_user_tool(self, real_agent_config, mock_llm_create):
        node = self._build(real_agent_config, execution_mode="interactive")
        assert node.execution_mode == "interactive"
        assert [tool.name for tool in node.ask_user_tool.available_tools()] == ["ask_user"]


# ===========================================================================
# BI Tools Removed from Chat Node Tests
# ===========================================================================


class TestChatAgenticNodeNoBITools:
    """Verify ChatAgenticNode no longer has BI tools (moved to GenDashboardAgenticNode)."""

    def test_no_bi_tool_names_in_tools_list(self, real_agent_config, mock_llm_create):
        """Chat node tools list should not contain any BI tool names."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.configuration.node_type import NodeType

        node = ChatAgenticNode(
            node_id="test_no_bi_tools",
            description="Test no BI tool names",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        bi_tool_names = {
            "list_dashboards",
            "get_dashboard",
            "list_charts",
            "list_datasets",
            "create_dashboard",
            "update_dashboard",
            "delete_dashboard",
            "create_chart",
            "update_chart",
            "add_chart_to_dashboard",
            "delete_chart",
            "create_dataset",
            "list_bi_databases",
            "delete_dataset",
            "write_query",
        }
        tool_names = {tool.name for tool in node.tools}
        assert tool_names.isdisjoint(bi_tool_names), f"Chat node still has BI tools: {tool_names & bi_tool_names}"


# ===========================================================================
# Scheduler Tools Removed from Chat Node Tests
# ===========================================================================


class TestChatAgenticNodeNoSchedulerTools:
    """Verify ChatAgenticNode no longer has scheduler tools (moved to SchedulerAgenticNode)."""

    def test_no_scheduler_tool_names_in_tools_list(self, real_agent_config, mock_llm_create):
        """Chat node tools list should not contain any scheduler tool names."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_no_scheduler_tools",
            description="Test no scheduler tool names",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        scheduler_tool_names = {
            "submit_sql_job",
            "submit_sparksql_job",
            "trigger_scheduler_job",
            "pause_job",
            "resume_job",
            "delete_job",
            "update_job",
            "get_scheduler_job",
            "list_scheduler_jobs",
            "list_scheduler_connections",
            "list_job_runs",
            "get_run_log",
        }
        tool_names = {tool.name for tool in node.tools}
        assert tool_names.isdisjoint(scheduler_tool_names), (
            f"Chat node still has scheduler tools: {tool_names & scheduler_tool_names}"
        )


# ===========================================================================
# agentic_nodes[<name>].tools
# ===========================================================================


class TestChatAgenticNodeHonoursConfiguredTools:
    """`ChatAgenticNode` was the one agentic node class that ignored `tools`.

    The five others (gen_sql, gen_report, ask_metrics, and the two artifact
    bases) read `node_config.get("tools")` and build only what it names. Chat
    mounted every family unconditionally, so a host that publishes a chat
    sub-agent and strips write-capable tools from that list got no effect from
    the strip: the model was still shown tools the permission layer would then
    deny.
    """

    @staticmethod
    def _node(real_agent_config, tools=None):
        """Build a chat node with `tools` set to *tools*, or removed entirely.

        The shared fixture's `chat` entry declares a narrow `tools` list, so a
        test for the no-key default has to delete the key rather than just leave
        the argument out.
        """
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        chat_entry = dict((real_agent_config.agentic_nodes or {}).get("chat") or {})
        if tools is None:
            chat_entry.pop("tools", None)
        else:
            chat_entry["tools"] = tools
        real_agent_config.agentic_nodes = {
            **(real_agent_config.agentic_nodes or {}),
            "chat": chat_entry,
        }
        return ChatAgenticNode(
            node_id="test_tools",
            description="Test configured tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

    @staticmethod
    def _families(node):
        """The tool families actually mounted, by instance attribute."""
        return {
            "db_tools": node.db_func_tool is not None,
            "context_search_tools": node.context_search_tools is not None,
            "reference_template_tools": node.reference_template_tools is not None,
            "date_parsing_tools": node.date_parsing_tools is not None,
            "filesystem_tools": node.filesystem_func_tool is not None,
            "memory_tools": node.memory_func_tool is not None,
            "bash_tools": node.bash_tool is not None,
        }

    def test_no_tools_key_mounts_everything(self, real_agent_config, mock_llm_create):
        """The default must not change. Chat nodes without a `tools` key are the
        overwhelming majority, so narrowing this would be a silent capability
        regression rather than a policy change anyone asked for."""
        node = self._node(real_agent_config)

        families = self._families(node)
        assert families["db_tools"] is True
        assert families["filesystem_tools"] is True
        assert families["bash_tools"] is True

    def test_an_empty_tools_value_is_treated_as_absent(self, real_agent_config, mock_llm_create):
        """`tools: ""` is a missing value, not a request for no tools — reading
        it as "none" would strip a node its operator never meant to strip."""
        node = self._node(real_agent_config, tools="")

        assert self._families(node) == self._families(self._node(real_agent_config))

    def test_a_narrow_list_excludes_the_other_families(self, real_agent_config, mock_llm_create):
        """The acceptance case: a db-only chat node must not carry filesystem or
        bash tools."""
        node = self._node(real_agent_config, tools="db_tools.*")

        families = self._families(node)
        assert families["db_tools"] is True
        assert families["filesystem_tools"] is False
        assert families["bash_tools"] is False
        assert families["context_search_tools"] is False

    def test_excluded_families_stay_out_of_available_tools(self, real_agent_config, mock_llm_create):
        """Instance attributes are the mechanism; the tool list the LLM sees is
        the thing that actually matters."""
        node = self._node(real_agent_config, tools="db_tools.*")

        names = {tool.name for tool in node.tools}
        assert names, "a db-only node should still expose the db tools"
        for excluded in ("read_file", "write_file", "glob", "bash"):
            assert excluded not in names

    def test_bash_is_dropped_even_though_init_built_it(self, real_agent_config, mock_llm_create):
        """`bash_tool` is created in `AgenticNode.__init__`, not in
        `setup_tools`, and `_rebuild_tools` re-adds it from the attribute on
        every rebuild. Clearing the attribute is the only thing that keeps it
        out — leaving it set would let a later rebuild resurrect it."""
        node = self._node(real_agent_config, tools="db_tools.*")

        assert node.bash_tool is None

        node._rebuild_tools()

        assert all(tool.name != "bash" for tool in node.tools)

    def test_several_families_can_be_selected(self, real_agent_config, mock_llm_create):
        node = self._node(real_agent_config, tools="db_tools.*, filesystem_tools.*")

        families = self._families(node)
        assert families["db_tools"] is True
        assert families["filesystem_tools"] is True
        assert families["bash_tools"] is False

    def test_a_per_tool_pattern_still_enables_its_family(self, real_agent_config, mock_llm_create):
        """Other node classes accept `context_search_tools.list_subject_tree`.
        This node acts at family granularity, so the suffix selects the family
        rather than being dropped — dropping it would silently remove a tool the
        operator explicitly asked for."""
        node = self._node(real_agent_config, tools="db_tools.execute_sql")

        families = self._families(node)
        assert families["db_tools"] is True
        assert families["filesystem_tools"] is False

    def test_an_unknown_family_selects_nothing_rather_than_everything(self, real_agent_config, mock_llm_create):
        """Fail closed. A typo that fell back to "all tools" would hand a
        published sub-agent the full surface its author meant to narrow."""
        node = self._node(real_agent_config, tools="no_such_tools.*")

        families = self._families(node)
        assert families["db_tools"] is False
        assert families["filesystem_tools"] is False
        assert families["bash_tools"] is False

    def test_the_ask_subclasses_opt_out(self):
        """`BaseArtifactAskAgenticNode` inherits this setup but runs its own
        whitelist over `self.tools` *after* it, and that design needs every tool
        instance to exist — the artifact-anchored filesystem instance is
        infrastructure it uses whether or not `filesystem_tools` is exposed.

        Letting both mechanisms narrow on the same field left those attributes
        None and broke the subclass, which is why the opt-out is explicit rather
        than implied by overriding `setup_tools`.
        """
        from datus.agent.node.base_artifact_ask_agentic_node import BaseArtifactAskAgenticNode
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        assert ChatAgenticNode.HONOURS_CONFIGURED_TOOLS is True
        assert BaseArtifactAskAgenticNode.HONOURS_CONFIGURED_TOOLS is False

    def test_opting_out_ignores_a_configured_list(self, real_agent_config, mock_llm_create):
        """The flag, not the class name, is what disables the narrowing."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = self._node(real_agent_config, tools="db_tools.*")
        assert node.filesystem_func_tool is None

        with patch.object(ChatAgenticNode, "HONOURS_CONFIGURED_TOOLS", False):
            unrestricted = self._node(real_agent_config, tools="db_tools.*")

        # The list is ignored entirely, not partly: same surface as no key at all.
        assert self._families(unrestricted) == self._families(self._node(real_agent_config))

    def test_exclusions_survive_the_lazy_prompt_build_mount(self, real_agent_config, mock_llm_create):
        """`AgenticNode._ensure_lazy_tools_mounted` re-adds bash, skills, memory
        and web on every prompt build and on a snapshot-cache hit — after
        `setup_tools` already decided what this node may have.

        Without gating those too, the exclusion holds only until the first
        prompt is built. Memory is the worst case: the base *re-creates* the
        instance when it finds None, so clearing it in `setup_tools` does not
        survive on its own.
        """
        node = self._node(real_agent_config, tools="db_tools.*")

        node._ensure_lazy_tools_mounted()

        names = {tool.name for tool in node.tools}
        for resurrected in ("add_memory", "edit_memory", "web_fetch", "bash", "load_skill"):
            assert resurrected not in names

    def test_the_default_still_gets_the_lazy_tools(self, real_agent_config, mock_llm_create):
        """The gates must not cost an ungated node its lazily mounted tools —
        they are how a normal chat node gets bash, skills, memory and web at
        all."""
        node = self._node(real_agent_config)

        node._ensure_lazy_tools_mounted()

        names = {tool.name for tool in node.tools}
        # `web_fetch` included deliberately: web is gated here too, so a
        # regression that stops mounting it for an unconfigured node would
        # otherwise pass — the exclusion test only proves configured nodes do
        # not receive it.
        for expected in ("add_memory", "bash", "load_skill", "web_fetch"):
            assert expected in names

    def test_a_selected_family_survives_the_lazy_mount(self, real_agent_config, mock_llm_create):
        """Gating is per family, not a blanket "narrowed nodes get nothing lazy"."""
        node = self._node(real_agent_config, tools="db_tools.*, memory_tools.*")

        node._ensure_lazy_tools_mounted()

        names = {tool.name for tool in node.tools}
        assert "add_memory" in names
        assert "bash" not in names

    #: The genuine `PlatformDocSearchTool` builds fine in tests but exposes
    #: nothing — there is no indexed docstore for the fixture project — so
    #: assertions about its tool names pass whether or not the code under test
    #: mounts it. These tests patch the constructor so the group has a name to
    #: be wrong about, which is also what a deployment with indexed docs looks
    #: like.
    _DOC_TOOL_NAME = "search_document"

    @classmethod
    def _doc_tool_stub(cls, *_args, **_kwargs):
        stub = MagicMock()
        tool = MagicMock()
        tool.name = cls._DOC_TOOL_NAME
        stub.available_tools.return_value = [tool]
        return stub

    @classmethod
    def _patch_platform_doc(cls):
        """Patch the constructor, so a node built inside this context has a
        platform-doc group with a real name — what a deployment with indexed
        docs looks like."""
        return patch(
            "datus.agent.node.chat_agentic_node.PlatformDocSearchTool",
            side_effect=cls._doc_tool_stub,
        )

    @classmethod
    def _stub_platform_doc(cls, node):
        """Attach a non-empty platform-doc group to an already-built node."""
        node._platform_doc_tool = cls._doc_tool_stub()
        return cls._DOC_TOOL_NAME

    def test_platform_doc_tools_survive_a_rebuild(self, real_agent_config, mock_llm_create):
        """`_setup_platform_doc_tools` mounts *after* the initial
        `_rebuild_tools`, and the rebuild never re-added it — so a task-database
        switch, which rebuilds, silently dropped platform docs from the LLM's
        surface mid-session.
        """
        node = self._node(real_agent_config)
        name = self._stub_platform_doc(node)

        node._rebuild_tools()

        assert name in {tool.name for tool in node.tools}

    def test_the_rebuild_does_not_duplicate_platform_doc_tools(self, real_agent_config, mock_llm_create):
        """Setup mounts it once after the first rebuild; later rebuilds mount it
        from the attribute. Neither path may double-count."""
        node = self._node(real_agent_config)
        name = self._stub_platform_doc(node)

        node._rebuild_tools()
        node._rebuild_tools()

        assert [tool.name for tool in node.tools].count(name) == 1

    def test_an_excluded_platform_doc_family_stays_out_across_a_rebuild(self, real_agent_config, mock_llm_create):
        """The re-add is gated like everything else — otherwise the rebuild
        would hand the family back to a node that excluded it."""
        node = self._node(real_agent_config, tools="db_tools.*")
        assert node._platform_doc_tool is None
        # Even if something later populated the attribute, the gate holds.
        name = self._stub_platform_doc(node)

        node._rebuild_tools()

        assert name not in {tool.name for tool in node.tools}

    @pytest.mark.parametrize(
        ("attribute", "family", "tool_name"),
        [
            ("skill_func_tool", "skills", "load_skill"),
            ("memory_func_tool", "memory_tools", "add_memory"),
            ("filesystem_func_tool", "filesystem_tools", "read_file"),
            ("bash_tool", "bash_tools", "bash"),
        ],
    )
    def test_a_populated_instance_cannot_reinstate_an_excluded_family(
        self, real_agent_config, mock_llm_create, attribute, family, tool_name
    ):
        """`_rebuild_tools` asks the config, not just whether the attribute is set.

        Testing "did setup_tools leave it None" would make the exclusion depend
        on nothing else ever populating the attribute — an assumption that has
        already failed twice here: the base re-creates the memory tool when it
        finds None, and platform docs were mounted after the rebuild meant to
        own them. This pins the stronger property: whatever populates the
        instance, an excluded family stays out.
        """
        node = self._node(real_agent_config, tools="db_tools.*")

        stub = MagicMock()
        tool = MagicMock()
        tool.name = tool_name
        stub.available_tools.return_value = [tool]
        setattr(node, attribute, stub)

        node._rebuild_tools()

        assert tool_name not in {t.name for t in node.tools}

    def test_a_second_setup_tools_does_not_duplicate_platform_doc_tools(self, real_agent_config, mock_llm_create):
        """The CLI calls `setup_tools()` again after a datasource change.

        Once the rebuild started mounting this group from the attribute,
        building it *after* the rebuild appended a second copy of every name on
        that second call. The fixture's real doc tool exposes nothing, so the
        duplication is invisible without patching the constructor — the same
        emptiness that made an earlier version of these tests vacuous.
        """
        with self._patch_platform_doc():
            node = self._node(real_agent_config)
            assert [t.name for t in node.tools].count(self._DOC_TOOL_NAME) == 1

            node.setup_tools()

            assert [t.name for t in node.tools].count(self._DOC_TOOL_NAME) == 1
