# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenJobAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (DBFuncTool + execute_ddl + execute_write, no transfer_query_result)
- Max turns configuration
- Node type registration and factory creation

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite)
- Real Tools (DBFuncTool)
- Real PromptManager (using built-in templates)
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import pytest

from tests.unit_tests.agent.node._builtin_node_test_helpers import (
    check_dynamic_db_func_tool,
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
    check_tools_exclude,
    check_tools_include,
)

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenJobAgenticNodeInit:
    """Tests for GenJobAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_node_name(GenJobAgenticNode, real_agent_config, "gen_job")

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_inherits_agentic_node(GenJobAgenticNode, real_agent_config)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_node_id(GenJobAgenticNode, real_agent_config, "gen_job_node")

    def test_setup_tools_includes_ddl(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_tools_include(GenJobAgenticNode, real_agent_config, "execute_ddl")

    def test_setup_tools_includes_execute_write(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_tools_include(GenJobAgenticNode, real_agent_config, "execute_write")

    def test_setup_tools_excludes_transfer_query_result(self, real_agent_config, mock_llm_create):
        """gen_job is single-database ETL — transfer_query_result belongs to migration subagent."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_tools_exclude(GenJobAgenticNode, real_agent_config, "transfer_query_result")

    def test_setup_tools_includes_standard_db_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_standard_db_tools(GenJobAgenticNode, real_agent_config)

    def test_setup_tools_includes_filesystem_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_filesystem_tools(GenJobAgenticNode, real_agent_config)

    def test_default_max_turns(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_max_turns(GenJobAgenticNode, real_agent_config, 30)

    def test_uses_dynamic_db_func_tool(self, real_agent_config, mock_llm_create):
        """gen_job should use create_dynamic for multi-connector support."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        check_dynamic_db_func_tool(GenJobAgenticNode, real_agent_config)


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestGenJobExecution:
    """Test execute_stream error paths and basic workflow."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        await check_execute_stream_raises_without_input(GenJobAgenticNode, real_agent_config)

    @pytest.mark.asyncio
    async def test_execute_stream_basic_workflow(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        await check_execute_stream_basic_workflow(
            GenJobAgenticNode,
            real_agent_config,
            mock_llm_create,
            "Create an ETL job to load data into summary table",
        )

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        await check_execute_stream_error_handling(
            GenJobAgenticNode,
            real_agent_config,
            mock_llm_create,
            "Build ETL job",
        )


# ---------------------------------------------------------------------------
# Node Type Integration Tests
# ---------------------------------------------------------------------------


class TestGenJobNodeType:
    """Tests for GenJobAgenticNode type registration."""

    def test_node_type_constant_exists(self):
        check_node_type_constant("TYPE_GEN_JOB", "gen_job")

    def test_node_type_in_action_types(self):
        check_node_type_in_action_types("TYPE_GEN_JOB")

    def test_node_factory_creates_gen_job(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.configuration.node_type import NodeType

        check_node_factory(GenJobAgenticNode, NodeType.TYPE_GEN_JOB, real_agent_config)

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.configuration.node_type import NodeType

        check_node_factory_with_input(
            GenJobAgenticNode,
            NodeType.TYPE_GEN_JOB,
            real_agent_config,
            "Build a summary table",
        )
