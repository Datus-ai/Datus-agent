# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Nightly smoke test for the documented quickstart chat path.

Exercises the core promise of docs/getting_started/Quickstart.md and
docs/cli/chat_command.md: a fresh user asks a natural-language data question
through the chat/agent path and gets a successful answer backed by SQL, on a
local SQLite datasource.

Covers the coverage-map flow ``onboarding.quickstart`` (GitHub issue #921).

The chat path is exercised via ChatAgenticNode (the node behind the CLI ``/``
chat command), mirroring the real-LLM invocation pattern used in
tests/integration/agent/test_chat_agentic.py and
tests/integration/agent/test_sql_summary_agentic.py. Tests reuse the
``nightly_agent_config`` fixture (datasource ``bird_school``) from
tests/integration/conftest.py.
"""

import os

import pytest

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Database (read) tools the quickstart chat path must expose so a fresh user's
# natural-language question can be answered with SQL against a local datasource.
EXPECTED_DB_TOOLS = ("execute_sql", "list_tables", "describe_table")


@pytest.mark.nightly
class TestQuickstartChatInitialization:
    """Deterministic checks for the quickstart chat node (no LLM)."""

    def test_chat_node_exposes_sql_tools_on_quickstart_datasource(self, nightly_agent_config):
        """The chat node initializes with the SQL read tools the quickstart relies on."""
        node = ChatAgenticNode(
            node_id="quickstart_smoke_init",
            description="Quickstart smoke initialization check",
            node_type="chat",
            agent_config=nightly_agent_config,
        )

        assert node.get_node_name() == "chat", f"Expected node name 'chat', got '{node.get_node_name()}'"

        tool_names = [tool.name for tool in node.tools]
        for expected in EXPECTED_DB_TOOLS:
            assert expected in tool_names, f"Missing quickstart SQL tool '{expected}', got: {tool_names}"

        logger.info(f"Quickstart chat node initialized with {len(tool_names)} tools: {tool_names}")


@pytest.mark.nightly
@pytest.mark.product_e2e
class TestQuickstartChatSmoke:
    """Real-LLM end-to-end smoke for the documented quickstart chat path."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    @pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
    async def test_quickstart_natural_language_question_answered_with_sql(self, nightly_agent_config):
        """A fresh NL question is answered successfully and is backed by a SQL read."""
        node = ChatAgenticNode(
            node_id="quickstart_smoke_e2e",
            description="Quickstart smoke end-to-end check",
            node_type="chat",
            agent_config=nightly_agent_config,
        )

        node.input = ChatNodeInput(
            user_message="How many schools are there?",
            max_turns=15,
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Expected at least 2 actions (user + result), got {len(actions)}"

        assert actions[0].role == ActionRole.USER, f"First action should be USER, got {actions[0].role}"

        final_action = actions[-1]
        assert final_action.status == ActionStatus.SUCCESS, (
            f"Final action should be SUCCESS, got {final_action.status}: {final_action.output}"
        )

        successful_tool_actions = [
            action for action in actions if action.role == ActionRole.TOOL and action.status == ActionStatus.SUCCESS
        ]
        assert len(successful_tool_actions) >= 1, (
            f"Quickstart answer must be backed by at least one successful TOOL action, "
            f"got tool actions: {[(a.action_type, a.status) for a in actions if a.role == ActionRole.TOOL]}"
        )

        sql_read_actions = [
            a for a in successful_tool_actions if "query" in a.action_type.lower() or "sql" in a.action_type.lower()
        ]
        assert len(sql_read_actions) >= 1, (
            f"Quickstart answer must execute a SQL read tool, "
            f"successful tool action types: {[a.action_type for a in successful_tool_actions]}"
        )
