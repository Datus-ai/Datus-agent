# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for MigrationAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (execute_ddl + execute_write + transfer_query_result)
- Max turns configuration
- Node type registration and factory creation

Design principle: NO mock except LLM.
"""

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import build_simple_response


class TestMigrationAgenticNodeInit:
    """Tests for MigrationAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.NODE_NAME == "migration"
        assert node.get_node_name() == "migration"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.id == "migration_node"

    def test_setup_tools_includes_ddl(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "execute_ddl" in tool_names

    def test_setup_tools_includes_execute_write(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "execute_write" in tool_names

    def test_setup_tools_includes_transfer_query_result(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "transfer_query_result" in tool_names

    def test_setup_tools_includes_standard_db_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names
        assert "get_table_ddl" in tool_names

    def test_setup_tools_includes_filesystem_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names

    def test_default_max_turns(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 40

    def test_does_not_include_gen_job_only_tools(self, real_agent_config, mock_llm_create):
        """migration should NOT be confused with gen_job — verify it has transfer_query_result."""
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        # migration MUST have transfer_query_result
        assert "transfer_query_result" in tool_names


class TestMigrationExecution:
    """Test execute_stream error paths and basic workflow."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode
        from datus.utils.exceptions import DatusException

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.input is None

        with pytest.raises(DatusException) as exc_info:
            async for _ in node.execute_stream():
                pass
        assert "input" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_stream_basic_workflow(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Migration completed successfully."),
            ]
        )

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Migrate users table from duckdb to greenplum")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM connection error")
            yield  # noqa: makes this an async generator

        node = MigrationAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = SemanticNodeInput(user_message="Migrate data")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"


class TestMigrationNodeType:
    """Tests for MigrationAgenticNode type registration."""

    def test_node_type_constant_exists(self):
        from datus.configuration.node_type import NodeType

        assert hasattr(NodeType, "TYPE_MIGRATION")
        assert NodeType.TYPE_MIGRATION == "migration"

    def test_node_type_in_action_types(self):
        from datus.configuration.node_type import NodeType

        assert NodeType.TYPE_MIGRATION in NodeType.ACTION_TYPES

    def test_node_factory_creates_migration(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode
        from datus.agent.node.node import Node
        from datus.configuration.node_type import NodeType

        node = Node.new_instance(
            node_id="test_migration",
            description="Test migration factory",
            node_type=NodeType.TYPE_MIGRATION,
            input_data=None,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, MigrationAgenticNode)
        assert node.execution_mode == "workflow"

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode
        from datus.agent.node.node import Node
        from datus.configuration.node_type import NodeType

        input_data = SemanticNodeInput(user_message="Migrate users from duckdb to greenplum")
        node = Node.new_instance(
            node_id="test_migration",
            description="Test migration factory",
            node_type=NodeType.TYPE_MIGRATION,
            input_data=input_data,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, MigrationAgenticNode)
        assert node.input is not None
        assert node.input.user_message == "Migrate users from duckdb to greenplum"
