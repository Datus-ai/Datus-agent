# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for GenSemanticModelAgenticNode.

Tests the full semantic model generation workflow with real LLM, real config,
real database tools, and real filesystem tools.
"""

import pytest

from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.mark.nightly
class TestGenSemanticModelAgentic:
    """Integration tests for GenSemanticModelAgenticNode with real LLM."""

    def test_node_initialization(self, nightly_agent_config):
        """N7-01: Node initializes with correct tools including database tools."""
        node = GenSemanticModelAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_semantic_model"
        assert node.execution_mode == "workflow"
        assert node.hooks is None

        tool_names = [tool.name for tool in node.tools]

        # Filesystem tools
        assert "read_file" in tool_names, f"Missing read_file, got: {tool_names}"
        assert "write_file" in tool_names, f"Missing write_file, got: {tool_names}"
        assert "list_directory" in tool_names, f"Missing list_directory, got: {tool_names}"

        # Generation tools
        assert "check_semantic_object_exists" in tool_names, f"Missing check_semantic_object_exists, got: {tool_names}"
        assert "end_semantic_model_generation" in tool_names, (
            f"Missing end_semantic_model_generation, got: {tool_names}"
        )

        logger.info(f"Node initialized with {len(node.tools)} tools: {tool_names}")

    def test_node_has_db_tools(self, nightly_agent_config):
        """N7-02: Node initializes with database tools for schema exploration."""
        node = GenSemanticModelAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.db_func_tool is not None, "Database func tool should be initialized"

        tool_names = [tool.name for tool in node.tools]
        # DB tools should be present (exact names depend on DBFuncTool implementation)
        assert len(tool_names) > 7, f"Should have DB tools + filesystem + generation tools, got {len(tool_names)}"

        logger.info(f"DB tools initialized, total tools: {len(node.tools)}")

    def test_interactive_mode_has_hooks(self, nightly_agent_config):
        """N7-03: Interactive mode initializes hooks."""
        node = GenSemanticModelAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="interactive",
        )

        assert node.hooks is not None, "Interactive mode should have hooks"

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_execute_stream_generates_semantic_model(self, nightly_agent_config):
        """N7-04: execute_stream generates a semantic model for the satscores table."""
        node = GenSemanticModelAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        node.input = SemanticNodeInput(
            user_message=(
                "Generate a semantic model YAML for the `satscores` table. "
                "Use get_table_ddl to get the schema, then write the YAML file and call end_semantic_model_generation."
            ),
            database="california_schools",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
