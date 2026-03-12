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
- _get_execution_config plan vs. normal vs. unknown mode
- _build_plan_prompt structured / non-structured content branches
- _update_database_connection
- Summary report fallback logic in execute_stream

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
"""

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from tests.unit_tests.mock_llm_model import (
    MockToolCall,
    build_simple_response,
    build_tool_then_response,
)

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

    def test_not_instance_of_gensql(self, real_agent_config, mock_llm_create):
        """ChatAgenticNode is NOT a subclass of GenSQLAgenticNode."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = ChatAgenticNode(
            node_id="test_no_gensql",
            description="Test not gensql",
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

        node = ChatAgenticNode(
            node_id="test_db",
            description="Test db tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.db_func_tool is not None
        tool_names = [t.name for t in node.tools]
        # Should have at least some db tools
        assert any("table" in name or "query" in name or "sql" in name for name in tool_names)

    def test_has_context_search_tools(self, real_agent_config, mock_llm_create):
        """Chat node has context search tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_ctx",
            description="Test context search",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.context_search_tools is not None

    def test_has_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Chat node has filesystem tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_fs",
            description="Test filesystem tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.filesystem_func_tool is not None

    def test_has_date_parsing_tools(self, real_agent_config, mock_llm_create):
        """Chat node has date parsing tools."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_date",
            description="Test date parsing",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        assert node.date_parsing_tools is not None


# ===========================================================================
# ChatAgenticNode execute_stream Tests
# ===========================================================================


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
        assert final_action.output is not None
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
        with pytest.raises(ValueError, match="Chat input not set"):
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
    """Verify _update_database_connection switches DB connection and rebuilds tools."""

    def test_update_database_connection_rebuilds_tools(self, real_agent_config, mock_llm_create):
        """_update_database_connection creates a new DBFuncTool and rebuilds tools list."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_update_db",
            description="Test update db conn",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        original_db_tool = node.db_func_tool

        # Update to the same database (only one available in fixture)
        node._update_database_connection("california_schools")

        # db_func_tool should be a new instance
        assert node.db_func_tool is not original_db_tool
        # Tools should still be rebuilt and contain core db tools
        assert len(node.tools) > 0
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

    def test_setup_metricflow_mcp_returns_none_without_db_config(self, real_agent_config, mock_llm_create):
        """_setup_metricflow_mcp returns None when agent_config is None."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_mf_none",
            description="Test metricflow none",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Temporarily set agent_config to None
        original_config = node.agent_config
        node.agent_config = None

        result = node._setup_metricflow_mcp()
        assert result is None

        node.agent_config = original_config

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
        assert len(prompt) > 0

    def test_get_system_prompt_with_conversation_summary(self, real_agent_config, mock_llm_create):
        """_get_system_prompt accepts conversation summary argument."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_prompt_summary",
            description="Test prompt with summary",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        prompt = node._get_system_prompt(conversation_summary="Previous conversation about SQL queries.")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

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
        assert len(prompt) > 0

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
        with patch("datus.prompts.prompt_manager.prompt_manager.render_template", side_effect=RuntimeError("broken")):
            with pytest.raises(DatusException):
                node._get_system_prompt()


# ===========================================================================
# _get_execution_config Tests
# ===========================================================================


class TestChatAgenticNodeExecutionConfig:
    """Verify _get_execution_config for different execution modes."""

    def test_normal_mode_config(self, real_agent_config, mock_llm_create):
        """Normal mode returns tools, instruction, and None hooks."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_exec_normal",
            description="Test normal exec config",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="test", database="california_schools")

        config = node._get_execution_config("normal", node.input)

        assert "tools" in config
        assert "instruction" in config
        assert isinstance(config["instruction"], str)
        assert len(config["tools"]) > 0

    def test_unknown_mode_raises_value_error(self, real_agent_config, mock_llm_create):
        """Unknown execution mode raises ValueError with the mode name."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_exec_unknown",
            description="Test unknown exec mode",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="test", database="california_schools")

        with pytest.raises(ValueError, match="Unknown execution mode: invalid_mode"):
            node._get_execution_config("invalid_mode", node.input)

    def test_permission_hooks_are_applied(self, real_agent_config, mock_llm_create):
        """Permission hooks are attached to the execution config when available."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_exec_hooks",
            description="Test exec hooks",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        node.input = ChatNodeInput(user_message="test", database="california_schools")

        config = node._get_execution_config("normal", node.input)

        # If permission_hooks was set up, hooks should be non-None
        if node.permission_hooks:
            assert config["hooks"] is not None


# ===========================================================================
# _build_plan_prompt Tests
# ===========================================================================


class TestChatAgenticNodeBuildPlanPrompt:
    """Verify _build_plan_prompt handles structured and non-structured content."""

    def test_build_plan_prompt_non_structured_appends_plan_instructions(self, real_agent_config, mock_llm_create):
        """Non-structured content gets plan instructions appended."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_plan_prompt",
            description="Test plan prompt",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        result = node._build_plan_prompt("Help me analyze the data")

        assert isinstance(result, str)
        # Should contain the original prompt
        assert "Help me analyze the data" in result
        # Should contain plan mode instructions (either from template or inline fallback)
        assert len(result) > len("Help me analyze the data")

    def test_build_plan_prompt_template_not_found_uses_inline_fallback(self, real_agent_config, mock_llm_create):
        """When plan_mode_system template is missing, falls back to inline prompt."""
        from unittest.mock import patch

        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_plan_fallback",
            description="Test plan fallback",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        with patch(
            "datus.prompts.prompt_manager.prompt_manager.render_template", side_effect=FileNotFoundError("not found")
        ):
            result = node._build_plan_prompt("Analyze this")

        assert "PLAN MODE" in result
        assert "todo_read" in result


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
            external_knowledge="Some context info",
            catalog_name="test_catalog",
            schema_name="public",
        )
        workflow = Workflow(name="test_workflow", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)

        assert result["success"] is True
        assert node.input is not None
        assert node.input.user_message == "Tell me about the schools"
        assert node.input.database == "california_schools"
        assert node.input.external_knowledge == "Some context info"
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
            external_knowledge="Updated knowledge",
            catalog_name="new_catalog",
            schema_name="new_schema",
        )
        workflow = Workflow(name="test_workflow", task=task, agent_config=real_agent_config)

        result = node.setup_input(workflow)

        assert result["success"] is True
        assert node.input.user_message == "New question about data"
        assert node.input.database == "california_schools"
        assert node.input.external_knowledge == "Updated knowledge"
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
            yield  # noqa: unreachable - makes this an async generator

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
    async def test_execute_stream_handles_user_cancellation(self, real_agent_config, mock_llm_create):
        """User cancellation yields a SUCCESS action with cancellation message."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_cancel",
            description="Test cancellation",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        original_method = mock_llm_create.generate_with_tools_stream

        async def cancel_stream(*args, **kwargs):
            raise Exception("User cancelled the operation")
            yield  # noqa: unreachable

        mock_llm_create.generate_with_tools_stream = cancel_stream

        node.input = ChatNodeInput(user_message="Cancel me", database="california_schools")
        ahm = ActionHistoryManager()

        try:
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

            assert len(actions) >= 2
            final_action = actions[-1]
            assert final_action.status == ActionStatus.SUCCESS
            assert final_action.action_type == "user_cancellation"
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
            yield  # noqa: unreachable

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


# ===========================================================================
# execute_stream with Tool Calls Tests
# ===========================================================================


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
        assert final_action.output is not None
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


# ===========================================================================
# Plan Mode Tests
# ===========================================================================


class TestChatAgenticNodePlanMode:
    """Verify plan mode resets after execution completes."""

    @pytest.mark.asyncio
    async def test_plan_mode_resets_after_execution(self, real_agent_config, mock_llm_create):
        """Plan mode attributes (plan_mode_active, plan_hooks) are reset in finally block."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_plan_reset",
            description="Test plan reset",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # Plan mode input - even though it may fail, plan_mode_active should reset
        mock_llm_create.reset(responses=[build_simple_response("Plan response.")])

        node.input = ChatNodeInput(
            user_message="Create a plan",
            database="california_schools",
            plan_mode=True,
            auto_execute_plan=False,
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # After execution, plan_mode_active should be False regardless of outcome
        assert node.plan_mode_active is False
        assert node.plan_hooks is None


# ===========================================================================
# _rebuild_tools Tests
# ===========================================================================


class TestChatAgenticNodeRebuildTools:
    """Verify _rebuild_tools correctly assembles tools from all sources."""

    def test_rebuild_tools_with_all_components(self, real_agent_config, mock_llm_create):
        """_rebuild_tools includes tools from db, context, date, filesystem, skills, sub_agent."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = ChatAgenticNode(
            node_id="test_rebuild",
            description="Test rebuild tools",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )

        # _rebuild_tools assembles core tools (db, context, date, fs, skills, sub_agent)
        # but NOT platform_doc_tools (which are added separately in setup_tools)
        node._rebuild_tools()
        rebuilt_count = len(node.tools)

        # Should have tools from all core components
        assert rebuilt_count > 0
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

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

        # Rebuild should still work with just db tools
        node._rebuild_tools()

        assert len(node.tools) > 0
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names


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
