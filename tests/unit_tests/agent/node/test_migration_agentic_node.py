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

from tests.unit_tests.agent.node._builtin_node_test_helpers import (
    check_execute_stream_basic_workflow,
    check_execute_stream_error_handling,
    check_execute_stream_raises_without_input,
    check_filesystem_tools,
    check_inherits_agentic_node,
    check_max_turns,
    check_node_factory,
    check_node_factory_with_input,
    check_node_id,
    check_node_name,
    check_node_type_constant,
    check_node_type_in_action_types,
    check_standard_db_tools,
    check_tools_include,
)


class TestMigrationAgenticNodeInit:
    """Tests for MigrationAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_node_name(MigrationAgenticNode, real_agent_config, "migration")

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_inherits_agentic_node(MigrationAgenticNode, real_agent_config)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_node_id(MigrationAgenticNode, real_agent_config, "migration_node")

    def test_setup_tools_includes_ddl(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_tools_include(MigrationAgenticNode, real_agent_config, "execute_ddl")

    def test_setup_tools_includes_execute_write(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_tools_include(MigrationAgenticNode, real_agent_config, "execute_write")

    def test_setup_tools_includes_transfer_query_result(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_tools_include(MigrationAgenticNode, real_agent_config, "transfer_query_result")

    def test_setup_tools_includes_standard_db_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_standard_db_tools(MigrationAgenticNode, real_agent_config)

    def test_setup_tools_includes_filesystem_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_filesystem_tools(MigrationAgenticNode, real_agent_config)

    def test_default_max_turns(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_max_turns(MigrationAgenticNode, real_agent_config, 40)

    def test_does_not_include_gen_job_only_tools(self, real_agent_config, mock_llm_create):
        """migration should NOT be confused with gen_job — verify it has transfer_query_result."""
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        check_tools_include(MigrationAgenticNode, real_agent_config, "transfer_query_result")


class TestMigrationExecution:
    """Test execute_stream error paths and basic workflow."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        await check_execute_stream_raises_without_input(MigrationAgenticNode, real_agent_config)

    @pytest.mark.asyncio
    async def test_execute_stream_basic_workflow(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        await check_execute_stream_basic_workflow(
            MigrationAgenticNode,
            real_agent_config,
            mock_llm_create,
            "Migrate users table from duckdb to greenplum",
        )

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode

        await check_execute_stream_error_handling(
            MigrationAgenticNode,
            real_agent_config,
            mock_llm_create,
            "Migrate data",
        )


class TestMigrationNodeType:
    """Tests for MigrationAgenticNode type registration."""

    def test_node_type_constant_exists(self):
        check_node_type_constant("TYPE_MIGRATION", "migration")

    def test_node_type_in_action_types(self):
        check_node_type_in_action_types("TYPE_MIGRATION")

    def test_node_factory_creates_migration(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode
        from datus.configuration.node_type import NodeType

        check_node_factory(MigrationAgenticNode, NodeType.TYPE_MIGRATION, real_agent_config)

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        from datus.agent.node.migration_agentic_node import MigrationAgenticNode
        from datus.configuration.node_type import NodeType

        check_node_factory_with_input(
            MigrationAgenticNode,
            NodeType.TYPE_MIGRATION,
            real_agent_config,
            "Migrate users from duckdb to greenplum",
        )
