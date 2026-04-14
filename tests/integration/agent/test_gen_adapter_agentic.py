# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for GenAdapterAgenticNode.

Tests node initialization, tool registration, and execution modes
with real AgentConfig. LLM-driven tests are marked nightly.
"""

import pytest

from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.mark.nightly
class TestGenAdapterAgenticNodeInit:
    """Integration tests for GenAdapterAgenticNode initialization with real config."""

    def test_node_initialization_workflow_mode(self, nightly_agent_config):
        """N-GA-01: Node initializes in workflow mode with correct name and tools."""
        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_adapter"
        assert node.execution_mode == "workflow"

        tool_names = [tool.name for tool in node.tools]
        logger.info(f"Node initialized with {len(node.tools)} tools: {tool_names}")

        # Scaffold tools
        assert "scaffold_adapter" in tool_names, f"Missing scaffold_adapter, got: {tool_names}"
        assert "validate_adapter" in tool_names, f"Missing validate_adapter, got: {tool_names}"
        assert "list_adapter_types" in tool_names, f"Missing list_adapter_types, got: {tool_names}"

        # Filesystem tools
        assert "read_file" in tool_names, f"Missing read_file, got: {tool_names}"
        assert "write_file" in tool_names, f"Missing write_file, got: {tool_names}"
        assert "list_directory" in tool_names, f"Missing list_directory, got: {tool_names}"

        # Test runner
        assert "run_adapter_pytest" in tool_names, f"Missing run_adapter_pytest, got: {tool_names}"

    def test_node_initialization_interactive_mode(self, nightly_agent_config):
        """N-GA-02: Interactive mode has ask_user tool."""
        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="interactive",
        )

        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" in tool_names, f"Missing ask_user in interactive mode, got: {tool_names}"

    def test_workflow_mode_excludes_ask_user(self, nightly_agent_config):
        """N-GA-03: Workflow mode should NOT have ask_user tool."""
        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" not in tool_names, "Workflow mode should not have ask_user"

    def test_node_excludes_db_tools(self, nightly_agent_config):
        """N-GA-04: gen_adapter should not mount database tools."""
        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        tool_names = [tool.name for tool in node.tools]
        db_tools = {"list_tables", "describe_table", "read_query", "get_table_ddl"}
        present_db_tools = db_tools & set(tool_names)
        assert len(present_db_tools) == 0, f"gen_adapter should not have DB tools, found: {present_db_tools}"

    def test_list_adapter_types_returns_all_types(self, nightly_agent_config):
        """N-GA-05: list_adapter_types via the node's scaffold tool returns all 4 types."""
        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.adapter_scaffold_tool is not None
        result = node.adapter_scaffold_tool.list_adapter_types()
        assert result.success == 1
        types = {r["type"] for r in result.result}
        assert types == {"semantic", "bi", "db", "scheduler"}


@pytest.mark.nightly
class TestGenAdapterAgenticNodeExecution:
    """Integration tests for GenAdapterAgenticNode execution."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, nightly_agent_config):
        """N-GA-06: execute_stream raises DatusException when input is not set."""
        from datus.utils.exceptions import DatusException

        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        action_manager = ActionHistoryManager()
        with pytest.raises(DatusException):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_execute_stream_scaffolds_adapter(self, nightly_agent_config, tmp_path):
        """N-GA-07: execute_stream with scaffold instruction produces actions."""
        node = GenAdapterAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        output_dir = str(tmp_path / "gen_adapter_output")
        node.input = SemanticNodeInput(
            user_message=(
                f"Scaffold a semantic adapter for platform 'cube' in output directory "
                f"'{output_dir}' with display name 'Cube'."
            ),
            max_turns=20,
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
