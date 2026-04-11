# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for GenAdapterAgenticNode (TDD — written before implementation).

Tests cover:
- Node creation and initialization
- NODE_NAME and node_id
- Tools registered (scaffold + filesystem + platform_doc + ask_user)
"""

import pytest
import pytest_asyncio  # noqa: F401

from datus.agent.node.agentic_node import AgenticNode


class TestGenAdapterAgenticNodeInit:
    """Tests for GenAdapterAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.NODE_NAME == "gen_adapter"
        assert node.get_node_name() == "gen_adapter"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.id == "gen_adapter_node"

    def test_tools_include_scaffold(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "scaffold_adapter" in tool_names
        assert "validate_adapter" in tool_names
        assert "list_adapter_types" in tool_names

    def test_tools_include_filesystem(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_tools_include_platform_doc_search(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        # PlatformDocSearchTool exposes these tools
        assert "web_search_document" in tool_names

    def test_tools_include_adapter_test_runner(self, real_agent_config, mock_llm_create):
        """gen_adapter node must mount run_adapter_pytest to close the loop
        scaffold -> implement -> run tests -> fix."""
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "run_adapter_pytest" in tool_names
        assert node.adapter_test_runner_tool is not None

    def test_tools_include_ask_user_in_interactive_mode(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="interactive")
        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" in tool_names

    def test_tools_exclude_ask_user_in_workflow_mode(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" not in tool_names

    def test_tools_exclude_db_tools(self, real_agent_config, mock_llm_create):
        """gen_adapter node should NOT have DB or semantic tools."""
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "list_tables" not in tool_names
        assert "describe_table" not in tool_names
        assert "read_query" not in tool_names

    def test_get_system_prompt_returns_string(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        prompt = node._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prepare_template_context(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        user_input = SemanticNodeInput(user_message="test")
        ctx = node._prepare_template_context(user_input)
        assert "native_tools" in ctx
        assert "mcp_tools" in ctx
        assert "has_ask_user_tool" in ctx
        assert ctx["has_ask_user_tool"] is False

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 30

    def test_max_turns_from_config(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode

        real_agent_config.agentic_nodes["gen_adapter"] = {"max_turns": 15}
        node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 15


@pytest.mark.asyncio
async def test_execute_stream_raises_when_no_input(real_agent_config, mock_llm_create):
    from datus.agent.node.gen_adapter_agentic_node import GenAdapterAgenticNode
    from datus.utils.exceptions import DatusException

    node = GenAdapterAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    with pytest.raises(DatusException):
        async for _ in node.execute_stream():
            pass
