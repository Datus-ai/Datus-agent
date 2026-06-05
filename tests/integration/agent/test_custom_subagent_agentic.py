# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Nightly coverage for the extension flow ``subagents.custom_subagents``.

A custom (config-defined) subagent is declared in ``agent_config.agentic_nodes``
under a non-builtin name, backed by a chosen ``node_class`` (here ``gen_sql``),
with its own ``system_prompt`` and ``agent_description``. These tests verify:

1. The node factory resolves the custom subagent name to the correct node class
   (driven by ``node_class``), threads the custom name through as ``node_name``,
   and the resolved node exposes the tools configured for it.
2. ``_resolve_node_class_type`` returns the configured ``node_class`` so the
   ``task`` tool path (``NODE_CLASS_MAP``) and the factory agree on the runtime
   class.
3. A real-LLM end-to-end run routes a simple data question through the custom
   subagent and reports ``ActionStatus.SUCCESS``.

The deterministic tests run on every nightly; the real-LLM execute test is
additionally gated behind ``product_e2e`` and ``DEEPSEEK_API_KEY``.
"""

import copy
import os

import pytest

from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.agent.node.node_factory import (
    _resolve_node_class_type,
    create_interactive_node,
    resolve_node_name,
)
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

CUSTOM_SUBAGENT_NAME = "nightly_custom_sql"
CUSTOM_SYSTEM_PROMPT = "gen_sql"
CUSTOM_DESCRIPTION = "Nightly custom SQL subagent for california_schools"


def _config_with_custom_subagent(base_config):
    """Return a config copy with a custom subagent injected into agentic_nodes.

    The config and its agentic_nodes mapping are deep-copied first so the injected
    definition never leaks into the shared configuration_manager cache used by
    other tests.
    """
    config = copy.deepcopy(base_config)
    config.agentic_nodes = copy.deepcopy(config.agentic_nodes)
    config.agentic_nodes[CUSTOM_SUBAGENT_NAME] = {
        "model": "deepseek",
        "node_class": "gen_sql",
        "system_prompt": CUSTOM_SYSTEM_PROMPT,
        "prompt_language": "en",
        "max_turns": 30,
        "tools": "db_tools.*, context_search_tools.*, filesystem_tools.*",
        "agent_description": CUSTOM_DESCRIPTION,
    }
    return config


@pytest.mark.nightly
class TestCustomSubagentResolution:
    """Deterministic resolution of a config-defined custom subagent."""

    def test_resolve_node_class_type_returns_configured_class(self, nightly_agent_config):
        config = _config_with_custom_subagent(nightly_agent_config)
        resolved = _resolve_node_class_type(CUSTOM_SUBAGENT_NAME, config)
        assert resolved == "gen_sql", f"Expected node_class 'gen_sql', got {resolved!r}"

    def test_resolve_node_name_matches_custom_name(self):
        assert resolve_node_name(CUSTOM_SUBAGENT_NAME) == CUSTOM_SUBAGENT_NAME

    def test_factory_builds_gen_sql_node_for_custom_subagent(self, nightly_agent_config):
        config = _config_with_custom_subagent(nightly_agent_config)
        node = create_interactive_node(
            CUSTOM_SUBAGENT_NAME,
            config,
            execution_mode="workflow",
        )

        assert isinstance(node, GenSQLAgenticNode), f"Expected GenSQLAgenticNode, got {type(node).__name__}"
        assert node.get_node_name() == CUSTOM_SUBAGENT_NAME, (
            f"Custom subagent must keep its alias as node name, got {node.get_node_name()!r}"
        )
        assert node.execution_mode == "workflow"

    def test_custom_subagent_exposes_configured_tools(self, nightly_agent_config):
        config = _config_with_custom_subagent(nightly_agent_config)
        node = create_interactive_node(
            CUSTOM_SUBAGENT_NAME,
            config,
            execution_mode="workflow",
        )

        tool_names = {tool.name for tool in node.tools}
        # db_tools.* expands to at least read_query (SQL execution against the DB).
        assert "read_query" in tool_names, f"Expected read_query from db_tools.*, got: {sorted(tool_names)}"
        # filesystem_tools.* expands to file reads used for reference SQL discovery.
        assert "read_file" in tool_names, f"Expected read_file from filesystem_tools.*, got: {sorted(tool_names)}"


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestCustomSubagentRealLLM:
    """Real-LLM execution routed through the config-defined custom subagent."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_custom_subagent_executes_query(self, nightly_agent_config):
        """N924-CS: A simple data question routed through the custom subagent succeeds."""
        config = _config_with_custom_subagent(nightly_agent_config)
        node = create_interactive_node(
            CUSTOM_SUBAGENT_NAME,
            config,
            execution_mode="workflow",
        )
        assert isinstance(node, GenSQLAgenticNode), f"Expected GenSQLAgenticNode, got {type(node).__name__}"
        assert node.get_node_name() == CUSTOM_SUBAGENT_NAME

        node.input = GenSQLNodeInput(
            user_message="How many schools are there in Alameda county?",
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2, f"Expected at least 2 actions, got {len(actions)}"
        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Custom subagent run should end SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
