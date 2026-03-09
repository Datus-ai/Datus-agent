# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for AgenticNode skill integration.

Tests the skill-related methods in the AgenticNode base class:
- _setup_skill_func_tools()
- _ensure_skill_tools_in_tools()
- _finalize_system_prompt()
- _get_available_skills_context()
"""

from unittest.mock import Mock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_manager import SkillManager

# Globally patch LLMBaseModel.create_model to return None for all tests
pytestmark = pytest.mark.usefixtures("mock_llm_model")


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create test skills
    for skill_name, desc, tags in [
        ("sql-analysis", "SQL analysis techniques", ["sql", "analysis"]),
        ("report-generator", "Generate reports", ["report", "output"]),
        ("data-profiler", "Profile data quality", ["data", "quality"]),
    ]:
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {skill_name}
description: {desc}
tags: {tags}
---

# {skill_name.replace('-', ' ').title()}

{desc}

## Instructions

1. Do this
2. Do that
"""
        )

    return skills_dir


@pytest.fixture
def skill_config(temp_skills_dir):
    """Create SkillConfig for testing."""
    return SkillConfig(directories=[str(temp_skills_dir)])


@pytest.fixture
def permission_manager():
    """Create PermissionManager for testing."""
    config = PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
        ],
    )
    return PermissionManager(global_config=config)


@pytest.fixture
def skill_manager(skill_config, permission_manager):
    """Create SkillManager for testing."""
    return SkillManager(config=skill_config, permission_manager=permission_manager)


@pytest.fixture
def mock_llm_model():
    """Patch LLMBaseModel.create_model to prevent model instantiation."""
    with patch("datus.models.base.LLMBaseModel.create_model", return_value=None):
        yield


@pytest.fixture
def mock_agent_config():
    """Create a mock AgentConfig for testing."""
    config = Mock(spec=AgentConfig)
    config.agentic_nodes = {}
    config.permissions_config = None
    config.skills_config = None
    config.prompt_version = None
    config.workspace_root = "."

    # Prevent model creation by returning None for active_model
    config.active_model.return_value = None

    return config


class MinimalAgenticNode(AgenticNode):
    """Minimal AgenticNode subclass for testing."""

    def get_node_name(self) -> str:
        return "test_node"

    def setup_input(self, workflow):
        pass

    def update_context(self, result, workflow):
        pass

    def run(self):
        pass

    async def execute_stream(self, action_history_manager=None):
        """Minimal implementation of abstract method."""
        # Return empty async generator for testing
        if False:
            yield


def create_test_node(node_id, mock_agent_config):
    """Helper function to create test nodes with valid node_type."""
    return MinimalAgenticNode(
        node_id=node_id,
        description="Test node",
        node_type="chat",  # Use valid node type from NodeType.ACTION_TYPES
        agent_config=mock_agent_config,
    )


class TestFinalizeSystemPrompt:
    """Test suite for _finalize_system_prompt method."""

    def test_no_skill_func_tool_returns_prompt_unchanged(self, mock_agent_config):
        """When skill_func_tool is None, returns base_prompt as-is."""
        node = MinimalAgenticNode(
            node_id="test1",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_func_tool = None

        base_prompt = "This is the base system prompt."
        result = node._finalize_system_prompt(base_prompt)

        assert result == base_prompt

    def test_with_skill_func_tool_appends_xml(self, mock_agent_config, skill_manager):
        """When skill_func_tool exists and _get_available_skills_context() returns XML, it's appended."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*"}}

        node = MinimalAgenticNode(
            node_id="test2",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")

        base_prompt = "This is the base system prompt."
        result = node._finalize_system_prompt(base_prompt)

        # Should append skills XML
        assert result.startswith(base_prompt)
        assert "<available_skills>" in result
        assert "sql-analysis" in result

    def test_with_skill_func_tool_empty_xml_returns_prompt_unchanged(self, mock_agent_config, skill_manager):
        """When skills context returns empty string, prompt unchanged."""
        # Configure node with pattern that matches no skills
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "nonexistent-*"}}

        node = MinimalAgenticNode(
            node_id="test3",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")

        base_prompt = "This is the base system prompt."
        result = node._finalize_system_prompt(base_prompt)

        # When no skills match, XML generation returns empty string
        # So result should just be base_prompt
        assert result == base_prompt

    def test_calls_ensure_skill_tools(self, mock_agent_config, skill_manager):
        """Verify _ensure_skill_tools_in_tools() is called."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*"}}

        node = MinimalAgenticNode(
            node_id="test4",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")
        node.tools = []

        # Patch _ensure_skill_tools_in_tools to verify it's called
        with patch.object(node, "_ensure_skill_tools_in_tools", wraps=node._ensure_skill_tools_in_tools) as mock_ensure:
            node._finalize_system_prompt("Base prompt")
            mock_ensure.assert_called_once()


class TestEnsureSkillToolsInTools:
    """Test suite for _ensure_skill_tools_in_tools method."""

    def test_no_skill_func_tool_does_nothing(self, mock_agent_config):
        """When skill_func_tool is None, self.tools unchanged."""
        node = MinimalAgenticNode(
            node_id="test5",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_func_tool = None
        node.tools = []

        initial_tools = node.tools.copy()
        node._ensure_skill_tools_in_tools()

        assert node.tools == initial_tools

    def test_adds_skill_tools_when_missing(self, mock_agent_config, skill_manager):
        """When skill tools not in self.tools, adds them."""
        node = MinimalAgenticNode(
            node_id="test6",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")
        node.tools = []

        node._ensure_skill_tools_in_tools()

        # Should have added skill tools
        assert len(node.tools) == 2  # load_skill and skill_execute_command
        tool_names = [t.name for t in node.tools]
        assert "load_skill" in tool_names
        assert "skill_execute_command" in tool_names

    def test_idempotent_does_not_duplicate(self, mock_agent_config, skill_manager):
        """Calling twice doesn't add duplicate tools."""
        node = MinimalAgenticNode(
            node_id="test7",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")
        node.tools = []

        # First call
        node._ensure_skill_tools_in_tools()
        first_count = len(node.tools)
        first_tool_names = [t.name for t in node.tools]

        # Second call
        node._ensure_skill_tools_in_tools()
        second_count = len(node.tools)
        second_tool_names = [t.name for t in node.tools]

        # Should be the same
        assert first_count == second_count
        assert first_tool_names == second_tool_names

    def test_handles_none_tools_list(self, mock_agent_config, skill_manager):
        """When self.tools is None, creates list and adds tools."""
        node = MinimalAgenticNode(
            node_id="test8",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")
        node.tools = None

        node._ensure_skill_tools_in_tools()

        # Should have created tools list
        assert node.tools is not None
        assert len(node.tools) == 2
        tool_names = [t.name for t in node.tools]
        assert "load_skill" in tool_names
        assert "skill_execute_command" in tool_names

    def test_preserves_existing_tools(self, mock_agent_config, skill_manager):
        """When self.tools has existing tools, they are preserved."""
        node = MinimalAgenticNode(
            node_id="test9",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")

        # Add existing tool (using Mock)
        existing_tool = Mock()
        existing_tool.name = "existing_tool"
        node.tools = [existing_tool]

        node._ensure_skill_tools_in_tools()

        # Should have both existing and skill tools
        assert len(node.tools) == 3
        tool_names = [t.name for t in node.tools]
        assert "existing_tool" in tool_names
        assert "load_skill" in tool_names
        assert "skill_execute_command" in tool_names


class TestSetupSkillFuncTools:
    """Test suite for _setup_skill_func_tools method."""

    def test_no_skills_in_config_does_nothing(self, mock_agent_config):
        """When node_config has no 'skills' key, skill_func_tool stays None."""
        mock_agent_config.agentic_nodes = {"test_node": {}}

        node = MinimalAgenticNode(
            node_id="test10",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        assert node.skill_func_tool is None

    def test_creates_skill_manager_when_none(self, mock_agent_config):
        """When skills is set but skill_manager is None, creates a new SkillManager."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*"}}

        node = MinimalAgenticNode(
            node_id="test11",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        # Should have created skill_manager with defaults
        assert node.skill_manager is not None
        assert isinstance(node.skill_manager, SkillManager)

    def test_uses_existing_skill_manager(self, mock_agent_config, skill_manager):
        """When skill_manager already exists, doesn't create a new one."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "report-*"}}
        mock_agent_config.skills_config = skill_manager.config
        mock_agent_config.permissions_config = PermissionConfig(default_permission=PermissionLevel.ALLOW)

        node = MinimalAgenticNode(
            node_id="test12",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        # Should use the skill manager created during _setup_skill_manager()
        assert node.skill_manager is not None
        assert isinstance(node.skill_manager, SkillManager)

    def test_creates_skill_func_tool(self, mock_agent_config, skill_manager):
        """After setup, skill_func_tool is a SkillFuncTool instance."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "data-*"}}
        mock_agent_config.skills_config = skill_manager.config
        mock_agent_config.permissions_config = PermissionConfig(default_permission=PermissionLevel.ALLOW)

        node = MinimalAgenticNode(
            node_id="test13",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        assert node.skill_func_tool is not None
        assert isinstance(node.skill_func_tool, SkillFuncTool)
        assert node.skill_func_tool.node_name == "test_node"


class TestAgenticNodeSkillDefaults:
    """Test suite for default skill behavior in AgenticNode."""

    def test_base_node_no_skills_by_default(self, mock_agent_config):
        """AgenticNode with no skills config has skill_func_tool = None."""
        mock_agent_config.agentic_nodes = {"test_node": {}}

        node = MinimalAgenticNode(
            node_id="test14",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        assert node.skill_func_tool is None

    def test_node_with_skills_config_activates(self, mock_agent_config):
        """AgenticNode with skills: "report-*" in node_config has skill_func_tool set."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "report-*"}}

        node = MinimalAgenticNode(
            node_id="test15",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        assert node.skill_func_tool is not None
        assert isinstance(node.skill_func_tool, SkillFuncTool)


class TestGetAvailableSkillsContext:
    """Test suite for _get_available_skills_context method."""

    def test_no_skill_manager_returns_empty(self, mock_agent_config):
        """When skill_manager is None, returns empty string."""
        node = MinimalAgenticNode(
            node_id="test16",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = None

        result = node._get_available_skills_context()
        assert result == ""

    def test_with_skill_manager_returns_xml(self, mock_agent_config, skill_manager):
        """When skill_manager exists, returns XML context."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*"}}

        node = MinimalAgenticNode(
            node_id="test17",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager

        result = node._get_available_skills_context()

        assert "<available_skills>" in result
        assert "sql-analysis" in result
        assert "load_skill" in result

    def test_respects_skill_patterns(self, mock_agent_config, skill_manager):
        """XML context respects skill patterns from node config."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "report-*"}}

        node = MinimalAgenticNode(
            node_id="test18",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager

        result = node._get_available_skills_context()

        # Should only include report-* skills
        assert "report-generator" in result
        assert "sql-analysis" not in result
        assert "data-profiler" not in result

    def test_empty_pattern_returns_all_skills(self, mock_agent_config, skill_manager):
        """When skills config is empty string, returns all skills (pattern=None)."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": ""}}

        node = MinimalAgenticNode(
            node_id="test19",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager

        result = node._get_available_skills_context()
        # Empty pattern string means patterns=None, which returns all skills
        assert "<available_skills>" in result
        assert "sql-analysis" in result
        assert "report-generator" in result
        assert "data-profiler" in result


class TestSkillManagerSetup:
    """Test suite for _setup_skill_manager method."""

    def test_no_agent_config_does_nothing(self):
        """When agent_config is None, skill_manager stays None."""
        node = MinimalAgenticNode(
            node_id="test20",
            description="Test node",
            node_type="chat",
            agent_config=None,
        )

        assert node.skill_manager is None

    def test_no_skills_config_does_nothing(self, mock_agent_config):
        """When agent_config has no skills_config, skill_manager stays None."""
        mock_agent_config.skills_config = None

        node = MinimalAgenticNode(
            node_id="test21",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        assert node.skill_manager is None

    def test_creates_skill_manager_with_config(self, mock_agent_config, skill_config, permission_manager):
        """When skills_config is present, creates SkillManager."""
        mock_agent_config.skills_config = skill_config
        mock_agent_config.permissions_config = PermissionConfig(default_permission=PermissionLevel.ALLOW)

        node = MinimalAgenticNode(
            node_id="test22",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )

        assert node.skill_manager is not None
        assert isinstance(node.skill_manager, SkillManager)
        assert node.skill_manager.get_skill_count() > 0


class TestSkillIntegrationEdgeCases:
    """Edge case tests for skill integration."""

    def test_skill_func_tool_with_multiple_patterns(self, mock_agent_config, skill_manager):
        """Test multiple skill patterns in node config."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*, report-*"}}

        node = MinimalAgenticNode(
            node_id="test23",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager

        xml = node._get_available_skills_context()

        # Should include both patterns
        assert "sql-analysis" in xml
        assert "report-generator" in xml
        assert "data-profiler" not in xml

    def test_finalize_prompt_with_existing_tools(self, mock_agent_config, skill_manager):
        """Test that finalize_system_prompt preserves existing tools."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*"}}

        node = MinimalAgenticNode(
            node_id="test24",
            description="Test node",
            node_type="chat",
            agent_config=mock_agent_config,
        )
        node.skill_manager = skill_manager
        node.skill_func_tool = SkillFuncTool(manager=skill_manager, node_name="test_node")

        # Add existing tools (using Mock)
        existing_tool1 = Mock()
        existing_tool1.name = "tool1"
        existing_tool2 = Mock()
        existing_tool2.name = "tool2"
        node.tools = [existing_tool1, existing_tool2]

        base_prompt = "Base prompt"
        node._finalize_system_prompt(base_prompt)

        # Should have all tools
        assert len(node.tools) == 4  # 2 existing + 2 skill tools
        tool_names = [t.name for t in node.tools]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
        assert "load_skill" in tool_names
        assert "skill_execute_command" in tool_names

    def test_setup_exception_handling(self, mock_agent_config):
        """Test that setup exceptions are handled gracefully."""
        mock_agent_config.agentic_nodes = {"test_node": {"skills": "sql-*"}}

        # Patch SkillFuncTool import to raise exception
        with patch("datus.tools.skill_tools.skill_func_tool.SkillFuncTool", side_effect=Exception("Test error")):
            node = MinimalAgenticNode(
                node_id="test25",
                description="Test node",
                node_type="chat",
                agent_config=mock_agent_config,
            )

            # Should handle exception gracefully
            assert node.skill_func_tool is None
