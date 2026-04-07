# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SubAgentTaskTool skill_creator integration.

Tests that the create_skill subagent type is properly registered
and wired through the SubAgentTaskTool dispatch machinery.
"""

from datus.configuration.node_type import NodeType
from datus.tools.func_tool.sub_agent_task_tool import BUILTIN_SUBAGENT_DESCRIPTIONS, NODE_CLASS_MAP, SubAgentTaskTool
from datus.utils.constants import SYS_SUB_AGENTS


class TestSkillCreatorRegistration:
    """Tests for create_skill registration in SubAgentTaskTool."""

    def test_create_skill_in_node_class_map(self):
        """create_skill should be in NODE_CLASS_MAP."""
        assert "create_skill" in NODE_CLASS_MAP
        assert NODE_CLASS_MAP["create_skill"] == NodeType.TYPE_SKILL_CREATOR

    def test_create_skill_in_builtin_descriptions(self):
        """create_skill should have a builtin description."""
        assert "create_skill" in BUILTIN_SUBAGENT_DESCRIPTIONS
        desc = BUILTIN_SUBAGENT_DESCRIPTIONS["create_skill"]
        assert "skill" in desc.lower()
        assert "create" in desc.lower() or "Create" in desc

    def test_create_skill_in_sys_sub_agents(self):
        """create_skill should be in SYS_SUB_AGENTS constant."""
        assert "create_skill" in SYS_SUB_AGENTS

    def test_create_skill_in_available_types(self, real_agent_config, mock_llm_create):
        """create_skill should appear in _get_available_types()."""
        tool = SubAgentTaskTool(agent_config=real_agent_config)
        available = tool._get_available_types()
        assert "create_skill" in available

    def test_create_builtin_node_returns_correct_type(self, real_agent_config, mock_llm_create):
        """_create_builtin_node('create_skill') should return SkillCreatorAgenticNode."""
        from datus.agent.node.skill_creator_agentic_node import SkillCreatorAgenticNode

        tool = SubAgentTaskTool(agent_config=real_agent_config)
        node = tool._create_builtin_node("create_skill")
        assert isinstance(node, SkillCreatorAgenticNode)
        assert node.get_node_name() == "create_skill"

    def test_build_node_input_creates_skill_creator_input(self, real_agent_config, mock_llm_create):
        """_build_node_input should create SkillCreatorNodeInput for SkillCreatorAgenticNode."""
        from datus.agent.node.skill_creator_agentic_node import SkillCreatorAgenticNode
        from datus.schemas.skill_creator_agentic_node_models import SkillCreatorNodeInput

        tool = SubAgentTaskTool(agent_config=real_agent_config)
        node = SkillCreatorAgenticNode(
            node_id="test_input_build",
            description="Test",
            node_type=NodeType.TYPE_SKILL_CREATOR,
            agent_config=real_agent_config,
            node_name="create_skill",
        )
        input_obj = tool._build_node_input(node, "Create a SQL analysis skill")
        assert isinstance(input_obj, SkillCreatorNodeInput)
        assert input_obj.user_message == "Create a SQL analysis skill"
