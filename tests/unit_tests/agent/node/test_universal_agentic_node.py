# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for UniversalAgenticNode.

Tests cover:
- Node creation with different node_name values
- Tool completeness (all tool types registered)
- Max turns configuration
- System prompt generation (template fallback and skill injection)
- Workspace directory resolution

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real SQLite database (california_schools.sqlite)
- Real Tools (DBFuncTool, FilesystemFuncTool, GenerationTools, etc.)
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import pytest

from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import build_simple_response

# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestUniversalAgenticNodeInit:
    """Tests for UniversalAgenticNode initialization."""

    def test_init_gen_semantic_model(self, real_agent_config, mock_llm_create):
        """Test initialization with gen_semantic_model node_name."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_semantic_model"
        assert node.configured_node_name == "gen_semantic_model"
        assert node.execution_mode == "workflow"
        assert node.hooks is None  # No hooks in workflow mode

    def test_init_gen_metrics(self, real_agent_config, mock_llm_create):
        """Test initialization with gen_metrics node_name."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_metrics",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_metrics"
        assert node.execution_mode == "workflow"

    def test_init_gen_ext_knowledge(self, real_agent_config, mock_llm_create):
        """Test initialization with gen_ext_knowledge node_name."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_ext_knowledge"

    def test_init_custom_node_name(self, real_agent_config, mock_llm_create):
        """Test initialization with a custom node_name not in agentic_nodes."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="custom_workflow",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "custom_workflow"
        # Default max_turns when not configured
        assert node.max_turns == 30


# ---------------------------------------------------------------------------
# Tool Completeness Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestUniversalAgenticNodeTools:
    """Tests for tool registration completeness."""

    def test_has_db_tools(self, real_agent_config, mock_llm_create):
        """Test that DB tools are registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.db_func_tool is not None
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_has_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Test that filesystem tools are registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.filesystem_func_tool is not None
        tool_names = [t.name for t in node.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "list_directory" in tool_names
        assert "read_multiple_files" in tool_names

    def test_has_generation_tools(self, real_agent_config, mock_llm_create):
        """Test that generation tools are registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.generation_tools is not None
        tool_names = [t.name for t in node.tools]
        assert "check_semantic_object_exists" in tool_names
        assert "end_semantic_model_generation" in tool_names
        assert "end_metric_generation" in tool_names

    def test_has_verify_sql_tool(self, real_agent_config, mock_llm_create):
        """Test that verify_sql tool is registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        tool_names = [t.name for t in node.tools]
        assert "verify_sql" in tool_names

    def test_has_ask_user_tool(self, real_agent_config, mock_llm_create):
        """Test that ask_user tool is registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        tool_names = [t.name for t in node.tools]
        assert "ask_user" in tool_names

    def test_has_context_search_tools(self, real_agent_config, mock_llm_create):
        """Test that context search tools are registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.context_search_tools is not None

    def test_tool_count_reasonable(self, real_agent_config, mock_llm_create):
        """Test that a reasonable number of tools are registered."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        # Should have at minimum: DB tools + filesystem + generation + verify_sql + ask_user
        assert len(node.tools) >= 10


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestUniversalAgenticNodeConfig:
    """Tests for configuration handling."""

    def test_max_turns_from_config(self, real_agent_config, mock_llm_create):
        """Test max_turns is read from agentic_nodes config."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        # real_agent_config has gen_semantic_model.max_turns = 5
        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.max_turns == 5

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        """Test default max_turns is 30 when not configured."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="unconfigured_node",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.max_turns == 30

    def test_workspace_dir_semantic_model(self, real_agent_config, mock_llm_create):
        """Test workspace_dir resolves to semantic_model_path for gen_semantic_model."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode
        from datus.utils.path_manager import get_path_manager

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        path_manager = get_path_manager()
        expected = str(path_manager.semantic_model_path(real_agent_config.current_namespace))
        assert node.workspace_dir == expected

    def test_workspace_dir_ext_knowledge(self, real_agent_config, mock_llm_create):
        """Test workspace_dir resolves to ext_knowledge_path for gen_ext_knowledge."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode
        from datus.utils.path_manager import get_path_manager

        node = UniversalAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        path_manager = get_path_manager()
        expected = str(path_manager.ext_knowledge_path(real_agent_config.current_namespace))
        assert node.workspace_dir == expected

    def test_workspace_dir_custom_fallback(self, real_agent_config, mock_llm_create):
        """Test workspace_dir falls back to home for unknown node_name."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="custom_workflow",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        assert node.workspace_dir == str(real_agent_config.home)


# ---------------------------------------------------------------------------
# System Prompt Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestUniversalAgenticNodeSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_with_template(self, real_agent_config, mock_llm_create):
        """Test system prompt renders from Jinja2 template when available."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        template_context = node._prepare_template_context(SemanticNodeInput(user_message="test"))
        prompt = node._get_system_prompt(template_context=template_context)

        # Should contain content from the gen_semantic_model_system template
        assert "MetricFlow" in prompt
        assert len(prompt) > 100

    def test_system_prompt_uses_universal_template(self, real_agent_config, mock_llm_create):
        """Test system prompt always uses universal_system template regardless of node_name."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="custom_workflow",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        prompt = node._get_system_prompt()
        assert "AI assistant" in prompt
        assert "output" in prompt.lower()

    def test_template_context_has_tools(self, real_agent_config, mock_llm_create):
        """Test _prepare_template_context includes tool names."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        ctx = node._prepare_template_context(SemanticNodeInput(user_message="test"))
        assert "native_tools" in ctx
        assert "list_tables" in ctx["native_tools"]

    def test_template_context_semantic_model_dir(self, real_agent_config, mock_llm_create):
        """Test _prepare_template_context includes semantic_model_dir."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        ctx = node._prepare_template_context(SemanticNodeInput(user_message="test"))
        assert "semantic_model_dir" in ctx
        assert ctx["semantic_model_dir"] == node.workspace_dir


# ---------------------------------------------------------------------------
# verify_sql Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestUniversalAgenticNodeVerifySql:
    """Tests for verify_sql functionality."""

    def test_verify_sql_no_reference(self, real_agent_config, mock_llm_create):
        """Test verify_sql returns success when no reference SQL is set."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        result = node.verify_sql("SELECT 1")
        assert result.success == 1
        assert "No reference available" in str(result.result)

    def test_verify_sql_with_matching_reference(self, real_agent_config, mock_llm_create):
        """Test verify_sql returns success when SQL matches reference."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        # Set a reference SQL that we can match
        node._gold_sql = "SELECT COUNT(*) FROM frpm"
        result = node.verify_sql("SELECT COUNT(*) FROM frpm")
        assert result.success == 1

    def test_verify_sql_with_mismatching_reference(self, real_agent_config, mock_llm_create):
        """Test verify_sql returns failure when SQL doesn't match reference."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        node._gold_sql = "SELECT COUNT(*) FROM frpm"
        result = node.verify_sql("SELECT 1")
        assert result.success == 0


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestUniversalAgenticNodeExecution:
    """Tests for execute_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_basic(self, real_agent_config, mock_llm_create):
        """Test basic execute_stream produces expected action sequence."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        mock_llm_create.reset(
            responses=[
                build_simple_response('{"semantic_model_files": ["test.yml"], "output": "done"}'),
            ]
        )

        node.input = SemanticNodeInput(user_message="Generate semantic model for frpm table")

        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        # Should have at least: USER action, model actions, final ASSISTANT action
        assert len(actions) >= 2

        # First action should be USER
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        # Last action should be SUCCESS
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_requires_input(self, real_agent_config, mock_llm_create):
        """Test execute_stream raises ValueError when input is not set."""
        from datus.agent.node.universal_agentic_node import UniversalAgenticNode

        node = UniversalAgenticNode(
            node_name="gen_semantic_model",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )

        with pytest.raises(ValueError, match="Input not set"):
            async for _ in node.execute_stream():
                pass
