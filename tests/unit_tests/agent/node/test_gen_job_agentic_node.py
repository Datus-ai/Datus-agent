# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenJobAgenticNode.

Tests cover:
- Node creation in workflow and interactive modes
- Tools setup (DBFuncTool + execute_ddl + execute_write + transfer_query_result)
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

from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestGenJobAgenticNodeInit:
    """Tests for GenJobAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.NODE_NAME == "gen_job"
        assert node.get_node_name() == "gen_job"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.id == "gen_job_node"

    def test_setup_tools_includes_ddl(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "execute_ddl" in tool_names

    def test_setup_tools_includes_execute_write(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "execute_write" in tool_names

    def test_setup_tools_excludes_transfer_query_result(self, real_agent_config, mock_llm_create):
        """gen_job is single-database ETL — transfer_query_result belongs to migration subagent."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "transfer_query_result" not in tool_names

    def test_setup_tools_includes_standard_db_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names
        assert "get_table_ddl" in tool_names

    def test_setup_tools_includes_filesystem_tools(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names

    def test_default_max_turns(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 30

    def test_uses_dynamic_db_func_tool(self, real_agent_config, mock_llm_create):
        """gen_job should use create_dynamic for multi-connector support."""
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.db_func_tool is not None


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestGenJobExecution:
    """Test execute_stream error paths."""

    @pytest.mark.asyncio
    async def test_execute_stream_raises_without_input(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.utils.exceptions import DatusException

        node = GenJobAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        # input is None by default
        assert node.input is None

        with pytest.raises(DatusException) as exc_info:
            async for _ in node.execute_stream():
                pass
        assert "input" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Node Type Integration Tests
# ---------------------------------------------------------------------------


class TestGenJobNodeType:
    """Tests for GenJobAgenticNode type registration."""

    def test_node_type_constant_exists(self):
        from datus.configuration.node_type import NodeType

        assert hasattr(NodeType, "TYPE_GEN_JOB")
        assert NodeType.TYPE_GEN_JOB == "gen_job"

    def test_node_type_in_action_types(self):
        from datus.configuration.node_type import NodeType

        assert NodeType.TYPE_GEN_JOB in NodeType.ACTION_TYPES

    def test_node_factory_creates_gen_job(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.agent.node.node import Node
        from datus.configuration.node_type import NodeType

        node = Node.new_instance(
            node_id="test_gen_job",
            description="Test gen_job factory",
            node_type=NodeType.TYPE_GEN_JOB,
            input_data=None,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, GenJobAgenticNode)
        assert node.execution_mode == "workflow"

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.agent.node.node import Node
        from datus.configuration.node_type import NodeType

        input_data = SemanticNodeInput(user_message="Build a summary table")
        node = Node.new_instance(
            node_id="test_gen_job",
            description="Test gen_job factory",
            node_type=NodeType.TYPE_GEN_JOB,
            input_data=input_data,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, GenJobAgenticNode)
        assert node.input is not None
        assert node.input.user_message == "Build a summary table"
