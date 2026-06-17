# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillCreatorAgenticNode.

Tests cover node initialization, tool setup (full filesystem + ask_user + skill loading),
execute_stream flow, and system prompt rendering.

NO MOCK EXCEPT LLM: The only mock is LLMBaseModel.create_model -> MockLLMModel.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput, SkillCreatorNodeResult
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from tests.unit_tests.mock_llm_model import MockLLMModel, MockToolCall, build_simple_response, build_tool_then_response


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

    def test_default_max_turns_50(self, real_agent_config, mock_llm_create):
        """Default max_turns should be 50."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_3",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        assert node.max_turns == 50

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

    def test_aliased_node_wires_class_name_and_authoring_mode(self, real_agent_config, mock_llm_create):
        """A custom-aliased ``gen_skill`` subagent (``my_skill_editor:
        { node_class: gen_skill }``) must construct its ``SkillFuncTool`` with
        ``node_class="gen_skill"`` (so class-level ``allowed_agents`` match)
        AND ``authoring_mode=True`` (so scoped skills can still be loaded for
        editing). Regression guard for the alias path identified in review.
        """
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_alias",
            description="Aliased skill editor",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="my_skill_editor",
        )

        # Alias flows through to get_node_name(), class name still resolves.
        assert node.get_node_name() == "my_skill_editor"
        assert node.get_node_class_name() == "gen_skill"

        # The SkillFuncTool instance inherits both dimensions.
        assert isinstance(node.skill_func_tool_instance, SkillFuncTool)
        assert node.skill_func_tool_instance.node_name == "my_skill_editor"
        assert node.skill_func_tool_instance.node_class == "gen_skill"
        assert node.skill_func_tool_instance.authoring_mode is True


class TestSkillCreatorAgenticNodeTools:
    """Tests for SkillCreatorAgenticNode tool setup."""

    def test_has_unified_filesystem_tools(self, real_agent_config, mock_llm_create):
        """Node exposes a single unified filesystem tool — write scope is
        enforced by GenerationHooks (``.datus/skills/**``), not by a separate
        prefixed instance.
        """
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_tools_1",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        tool_names = [t.name for t in node.tools]
        assert {"read_file", "write_file", "edit_file", "glob", "grep"}.issubset(tool_names)
        # No skill_* prefixed duplicates anymore.
        assert not any(name.startswith("skill_") and name.endswith(("_file", "glob", "grep")) for name in tool_names)

    def test_has_workspace_read_tools(self, real_agent_config, mock_llm_create):
        """Node has the unified read tools rooted at project workspace."""
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
        assert "glob" in tool_names
        assert "grep" in tool_names

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
        assert isinstance(node.db_func_tool, DBFuncTool)
        tool_names = [t.name for t in node.tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names

    def test_tool_registry_splits_filesystem_db_and_skills(self, real_agent_config, mock_llm_create):
        """Filesystem, db, and skill-loading tools must each land in their own
        permission category so profile rules route correctly."""
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode(
            node_id="test_skill_creator_category_map",
            description="Test gen_skill node",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
        )
        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry.get("write_file") == "filesystem_tools"
        assert registry.get("read_query") == "db_tools"
        assert registry.get("load_skill") == "skills"

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
        assert isinstance(prompt, str)
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


@pytest.mark.component
@pytest.mark.llm_harness
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

        assert isinstance(node.result, SkillCreatorNodeResult)
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
        assert isinstance(node.result, SkillCreatorNodeResult)
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
        assert isinstance(node.result, SkillCreatorNodeResult)
        assert node.result.success is False


@pytest.mark.acceptance
@pytest.mark.llm_harness
class TestSkillCreatorLifecycleAcceptance:
    """Deterministic create/optimize skill lifecycle coverage with real tools."""

    @pytest.mark.asyncio
    async def test_create_skill_writes_validates_and_discovers_skill(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
        from datus.tools.skill_tools import SkillConfig, SkillManager

        skill_name = "school-quality-skill"
        skill_path = Path(real_agent_config.project_root) / ".datus" / "skills" / skill_name / "SKILL.md"
        skill_content = """---
name: school-quality-skill
description: Analyze school quality indicators from education datasets.
allowed_agents:
  - chat
  - gen_skill
---
# School Quality Skill

Use this skill when the user asks for school quality analysis.
"""
        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall("load_skill", {"skill_name": "create-skill"}),
                        MockToolCall(
                            "write_file",
                            {
                                "path": f".datus/skills/{skill_name}/SKILL.md",
                                "content": skill_content,
                            },
                        ),
                        MockToolCall("validate_skill", {"skill_path": str(skill_path)}),
                    ],
                    content=f"Created and validated {skill_name}.",
                )
            ]
        )

        node = SkillCreatorAgenticNode(
            node_id="skill_creator_create_acceptance",
            description="Create skill acceptance",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
            execution_mode="workflow",
        )
        node.input = SkillCreatorNodeInput(user_message=f"Create a {skill_name} skill.")

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert skill_path.read_text(encoding="utf-8") == skill_content
        tool_results = {item["tool"]: item for item in mock_llm_create.tool_results}
        assert tool_results["load_skill"]["executed"] is True
        assert tool_results["write_file"]["executed"] is True
        assert tool_results["validate_skill"]["executed"] is True
        assert "PASS" in str(tool_results["validate_skill"]["output"])

        manager = SkillManager(config=SkillConfig(directories=[str(skill_path.parent)]))
        discovered = manager.get_skill(skill_name)
        assert discovered.name == skill_name
        ok, message, loaded_content = manager.load_skill(skill_name, "chat")
        assert ok is True
        assert message == f"Skill '{skill_name}' loaded successfully"
        assert "School Quality Skill" in loaded_content
        assert isinstance(node.result, SkillCreatorNodeResult)
        assert node.result.success is True

    @pytest.mark.asyncio
    async def test_optimize_skill_reads_updates_validates_and_reloads_skill(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode
        from datus.tools.skill_tools import SkillConfig, SkillManager

        skill_name = "district-profile-skill"
        skill_dir = Path(real_agent_config.project_root) / ".datus" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
        original_content = """---
name: district-profile-skill
description: Build concise district profile summaries from education datasets.
allowed_agents:
  - chat
  - gen_skill
---
# District Profile Skill

Summarize district-level facts.
"""
        updated_content = """---
name: district-profile-skill
description: Build district profile summaries with enrollment and performance checks.
allowed_agents:
  - chat
  - gen_skill
---
# District Profile Skill

Summarize district-level facts and verify enrollment and performance metrics.
"""
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(original_content, encoding="utf-8")

        mock_llm_create.reset(
            responses=[
                build_tool_then_response(
                    tool_calls=[
                        MockToolCall("load_skill", {"skill_name": "optimize-skill"}),
                        MockToolCall("read_file", {"path": f".datus/skills/{skill_name}/SKILL.md"}),
                        MockToolCall(
                            "write_file",
                            {
                                "path": f".datus/skills/{skill_name}/SKILL.md",
                                "content": updated_content,
                            },
                        ),
                        MockToolCall("validate_skill", {"skill_path": str(skill_path)}),
                    ],
                    content=f"Optimized and validated {skill_name}.",
                )
            ]
        )

        node = SkillCreatorAgenticNode(
            node_id="skill_creator_optimize_acceptance",
            description="Optimize skill acceptance",
            node_type=NodeType.TYPE_GEN_SKILL,
            agent_config=real_agent_config,
            node_name="gen_skill",
            execution_mode="workflow",
        )
        node.input = SkillCreatorNodeInput(user_message=f"Optimize {skill_name}.")

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert skill_path.read_text(encoding="utf-8") == updated_content
        tool_results = {item["tool"]: item for item in mock_llm_create.tool_results}
        assert tool_results["load_skill"]["executed"] is True
        assert tool_results["read_file"]["executed"] is True
        assert "Summarize district-level facts." in str(tool_results["read_file"]["output"])
        assert tool_results["write_file"]["executed"] is True
        assert tool_results["validate_skill"]["executed"] is True
        assert "PASS" in str(tool_results["validate_skill"]["output"])

        manager = SkillManager(config=SkillConfig(directories=[str(skill_dir)]))
        ok, message, loaded_content = manager.load_skill(skill_name, "chat")
        assert ok is True
        assert message == f"Skill '{skill_name}' loaded successfully"
        assert "verify enrollment and performance metrics" in loaded_content
        assert isinstance(node.result, SkillCreatorNodeResult)
        assert node.result.success is True


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
        assert isinstance(prompt, str)
        assert "skill" in prompt.lower()

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

        with patch("datus.prompts.prompt_manager.get_prompt_manager") as mock_gpm:
            mock_gpm.return_value.render_template.side_effect = Exception("bad template")
            with pytest.raises(DatusException):
                node._get_system_prompt()
