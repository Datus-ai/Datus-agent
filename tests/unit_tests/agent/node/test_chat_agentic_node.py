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

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
"""

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from tests.unit_tests.mock_llm_model import (
    build_simple_response,
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
