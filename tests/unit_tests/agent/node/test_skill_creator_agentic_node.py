# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillCreatorAgenticNode.

Tests cover node initialization, tool setup (full filesystem + ask_user + skill loading),
execute_stream flow, and system prompt rendering.

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput
from tests.unit_tests.mock_llm_model import MockLLMModel, build_simple_response


class TestSkillCreatorAgenticNodeInit:
    """Tests for SkillCreatorAgenticNode initialization."""

    def test_inherits_from_agentic_node(self, real_agent_config, mock_llm_create):
        """SkillCreatorAgenticNode should inherit from AgenticNode."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_1",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert isinstance(node, AgenticNode)

    def test_node_name_returns_gen_skill(self, real_agent_config, mock_llm_create):
        """Node name should be 'gen_skill'."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_2",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert node.get_node_name() == "gen_skill"

    def test_default_max_turns_30(self, real_agent_config, mock_llm_create):
        """Default max_turns should be 30."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_3",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert node.max_turns == 30

    def test_model_is_mock(self, real_agent_config, mock_llm_create):
        """Model should be the mock model."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_4",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert isinstance(node.model, MockLLMModel)

    def test_no_mcp_servers(self, real_agent_config, mock_llm_create):
        """Node should have no MCP servers."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_5",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert node.mcp_servers == {}


class TestSkillCreatorAgenticNodeTools:
    """Tests for SkillCreatorAgenticNode tool setup."""

    def test_has_skills_write_tools(self, real_agent_config, mock_llm_create):
        """Node should have skill_* prefixed write tools for skills directory."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_1",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "skill_write_file" in tool_names
        assert "skill_edit_file" in tool_names
        assert "skill_create_directory" in tool_names
        assert "skill_move_file" in tool_names

    def test_has_workspace_read_tools(self, real_agent_config, mock_llm_create):
        """Node should have workspace read-only tools (unprefixed)."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_2",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "read_file" in tool_names
        assert "read_multiple_files" in tool_names
        assert "list_directory" in tool_names
        assert "directory_tree" in tool_names
        assert "search_files" in tool_names

    def test_has_skills_read_tools(self, real_agent_config, mock_llm_create):
        """Node should have skill_* prefixed read tools for skills directory."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_2b",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "skill_read_file" in tool_names
        assert "skill_list_directory" in tool_names
        assert "skill_directory_tree" in tool_names

    def test_has_db_tools(self, real_agent_config, mock_llm_create):
        """Node should have database tools."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_3",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert node.db_func_tool is not None
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_has_ask_user_tool(self, real_agent_config, mock_llm_create):
        """Node should have ask_user tool."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_4",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "ask_user" in tool_names

    def test_has_skill_loading_tools(self, real_agent_config, mock_llm_create):
        """Node should have skill loading tools (load_skill)."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_5",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "load_skill" in tool_names

    def test_has_validate_skill_tool(self, real_agent_config, mock_llm_create):
        """Node should have validate_skill tool."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_6",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "validate_skill" in tool_names

    def test_has_session_search_tool(self, real_agent_config, mock_llm_create):
        """Node should have search_skill_usage tool."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_7",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert "search_skill_usage" in tool_names


class TestSkillCreatorSystemPrompt:
    """Tests for SkillCreatorAgenticNode system prompt."""

    def test_system_prompt_renders(self, real_agent_config, mock_llm_create):
        """System prompt template should render without error."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_prompt_1",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        prompt = node._get_system_prompt()
        assert prompt is not None
        assert len(prompt) > 100

    def test_system_prompt_contains_routing(self, real_agent_config, mock_llm_create):
        """System prompt should contain routing logic for create vs optimize."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_prompt_2",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        prompt = node._get_system_prompt()
        assert "create-skill" in prompt
        assert "optimize-skill" in prompt
        assert "Routing" in prompt

    def test_system_prompt_contains_critical_rules(self, real_agent_config, mock_llm_create):
        """System prompt should contain critical rules."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_prompt_3",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        prompt = node._get_system_prompt()
        assert "Critical Rules" in prompt
        assert "validate_skill" in prompt
        assert "ask_user" in prompt


class TestSkillCreatorExecution:
    """Tests for SkillCreatorAgenticNode execute_stream."""

    @pytest.mark.asyncio
    async def test_simple_response(self, real_agent_config, mock_llm_create):
        """execute_stream with simple text response produces USER and ASSISTANT actions."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Created skill 'sql-analyzer' at ./skills/sql-analyzer/SKILL.md"),
            ]
        )

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_exec_1",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        node.input = SkillCreatorNodeInput(
            user_message="Create a skill for SQL analysis",
        )

        ahm = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(ahm):
            actions.append(action)

        # Should have at least USER + final ASSISTANT actions
        assert len(actions) >= 2
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[-1].role == ActionRole.ASSISTANT
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_result_has_response(self, real_agent_config, mock_llm_create):
        """Final result should contain the response text."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Created skill 'data-profiler' successfully."),
            ]
        )

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_exec_2",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        node.input = SkillCreatorNodeInput(
            user_message="Create a data profiling skill",
        )

        ahm = ActionHistoryManager()
        async for _ in node.execute_stream(ahm):
            pass

        assert node.result is not None
        assert node.result.success is True
        assert "data-profiler" in node.result.response

    @pytest.mark.asyncio
    async def test_no_input_raises(self, real_agent_config, mock_llm_create):
        """execute_stream should raise DatusException if no input is set."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
        from datus.utils.exceptions import DatusException

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_exec_3",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        with pytest.raises(DatusException):
            ahm = ActionHistoryManager()
            async for _ in node.execute_stream(ahm):
                pass

    @pytest.mark.asyncio
    async def test_execute_stream_without_ahm(self, real_agent_config, mock_llm_create):
        """execute_stream should create ActionHistoryManager if None is passed."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        mock_llm_create.reset(responses=[build_simple_response("Done.")])

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_exec_4",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        node.input = SkillCreatorNodeInput(user_message="test")

        actions = []
        async for action in node.execute_stream(None):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_error_handling(self, real_agent_config, mock_llm_create):
        """execute_stream should handle LLM errors gracefully and yield error action."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        mock_llm_create.reset(responses=[build_simple_response("ok")])

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_exec_5",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        node.input = SkillCreatorNodeInput(user_message="test")

        # Patch the model's generate method to raise an error
        async def _raise(*args, **kwargs):
            raise RuntimeError("LLM connection failed")
            yield  # noqa: unreachable — makes this an async generator

        with patch.object(node.model, "generate_with_tools_stream", _raise):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED
        assert node.result is not None
        assert node.result.success is False
        assert "LLM connection failed" in node.result.error

    @pytest.mark.asyncio
    async def test_execute_stream_datus_exception_handling(self, real_agent_config, mock_llm_create):
        """execute_stream should handle DatusException with error code."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
        from datus.utils.exceptions import DatusException, ErrorCode

        mock_llm_create.reset(responses=[build_simple_response("ok")])

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_exec_6",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        node.input = SkillCreatorNodeInput(user_message="test")

        async def _raise_datus(*args, **kwargs):
            raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message_args={"config_error": "bad config"})
            yield  # noqa: unreachable

        with patch.object(node.model, "generate_with_tools_stream", _raise_datus):
            ahm = ActionHistoryManager()
            actions = []
            async for action in node.execute_stream(ahm):
                actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED
        assert node.result is not None
        assert node.result.success is False


class TestSkillCreatorConfigOverrides:
    """Tests for config-driven behavior overrides."""

    def test_max_turns_override_from_config(self, real_agent_config, mock_llm_create):
        """max_turns should be overridden from agentic_nodes config."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        # Inject config for gen_skill node
        real_agent_config.agentic_nodes["gen_skill"] = {"max_turns": 50}

        node = SkillCreatorAgenticNode(
            node_id="test_config_1",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert node.max_turns == 50

        # Cleanup
        del real_agent_config.agentic_nodes["gen_skill"]

    def test_node_name_default_without_name(self, real_agent_config, mock_llm_create):
        """get_node_name should return 'gen_skill' when node_name is None."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_config_2",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name=None,
        )
        assert node.get_node_name() == "gen_skill"

    def test_setup_tools_no_agent_config(self, mock_llm_create):
        """setup_tools should return early if agent_config is None."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_config_3",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=None,
            node_name="gen_skill",
        )
        # Should have empty tools since agent_config is None
        assert node.tools == []


class TestSkillCreatorSetupInputUpdateContext:
    """Tests for setup_input and update_context methods."""

    def test_setup_input_from_workflow(self, real_agent_config, mock_llm_create):
        """setup_input should create input from workflow task."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_setup_1",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        mock_workflow = MagicMock()
        mock_workflow.task.task = "Create a skill for data profiling"
        result = node.setup_input(mock_workflow)

        assert result["success"] is True
        assert isinstance(node.input, SkillCreatorNodeInput)
        assert node.input.user_message == "Create a skill for data profiling"

    def test_setup_input_preserves_existing(self, real_agent_config, mock_llm_create):
        """setup_input should not overwrite existing SkillCreatorNodeInput."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_setup_2",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        existing_input = SkillCreatorNodeInput(user_message="existing message")
        node.input = existing_input

        mock_workflow = MagicMock()
        mock_workflow.task.task = "new message"
        node.setup_input(mock_workflow)

        # Should preserve existing input
        assert node.input.user_message == "existing message"

    def test_update_context(self, real_agent_config, mock_llm_create):
        """update_context should return success without modifying workflow."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_setup_3",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        mock_workflow = MagicMock()
        result = node.update_context(mock_workflow)
        assert result["success"] is True


class TestSkillCreatorSystemPromptEdgeCases:
    """Tests for system prompt edge cases."""

    def test_system_prompt_with_existing_skills(self, real_agent_config, mock_llm_create):
        """System prompt should include existing skills when available."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_prompt_edge_1",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        prompt = node._get_system_prompt()
        # The prompt should render successfully even if no skills found
        assert prompt is not None
        assert len(prompt) > 0

    def test_system_prompt_template_error(self, real_agent_config, mock_llm_create):
        """System prompt should raise DatusException on template error."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
        from datus.utils.exceptions import DatusException

        node = SkillCreatorAgenticNode(
            node_id="test_prompt_edge_2",
            description="Test",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )

        with patch(
            "datus.prompts.prompt_manager.prompt_manager.render_template", side_effect=Exception("bad template")
        ):
            with pytest.raises(DatusException):
                node._get_system_prompt()
