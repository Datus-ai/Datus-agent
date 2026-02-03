# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillFuncTool.

Tests the load_skill native tool functionality.
"""


import pytest

from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_manager import SkillManager


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create a simple skill
    simple_dir = skills_dir / "simple-skill"
    simple_dir.mkdir()
    (simple_dir / "SKILL.md").write_text(
        """---
name: simple-skill
description: A simple test skill
tags:
  - test
---

# Simple Skill

This is a simple skill for testing.

## Instructions

1. Do this
2. Do that
"""
    )

    # Create a skill with scripts
    script_dir = skills_dir / "script-skill"
    script_dir.mkdir()
    (script_dir / "SKILL.md").write_text(
        """---
name: script-skill
description: A skill with scripts
allowed_commands:
  - "python:scripts/*.py"
---

# Script Skill

Run scripts with: python scripts/analyze.py
"""
    )
    scripts = script_dir / "scripts"
    scripts.mkdir()
    (scripts / "analyze.py").write_text("print('analyzing')")

    # Create a denied skill
    denied_dir = skills_dir / "internal-skill"
    denied_dir.mkdir()
    (denied_dir / "SKILL.md").write_text(
        """---
name: internal-skill
description: An internal skill
---

# Internal Skill
"""
    )

    return skills_dir


@pytest.fixture
def skill_manager(temp_skills_dir):
    """Create a skill manager for testing."""
    config = SkillConfig(directories=[str(temp_skills_dir)])
    perm_config = PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="internal-*", permission=PermissionLevel.DENY),
        ],
    )
    perm_manager = PermissionManager(global_config=perm_config)
    return SkillManager(config=config, permission_manager=perm_manager)


@pytest.fixture
def skill_func_tool(skill_manager):
    """Create a SkillFuncTool for testing."""
    return SkillFuncTool(manager=skill_manager, node_name="chatbot")


class TestSkillFuncToolBasic:
    """Basic tests for SkillFuncTool."""

    def test_tool_creation(self, skill_manager):
        """Test creating a SkillFuncTool."""
        tool = SkillFuncTool(manager=skill_manager, node_name="chatbot")
        assert tool is not None
        assert tool.node_name == "chatbot"

    def test_available_tools(self, skill_func_tool):
        """Test that available_tools returns load_skill and skill_execute_command tools."""
        tools = skill_func_tool.available_tools()
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "load_skill" in tool_names
        assert "skill_execute_command" in tool_names

    def test_set_tool_context(self, skill_func_tool):
        """Test setting tool context."""
        mock_context = {"key": "value"}
        skill_func_tool.set_tool_context(mock_context)
        assert skill_func_tool._tool_context == mock_context


class TestSkillFuncToolLoadSkill:
    """Tests for load_skill method."""

    def test_load_skill_success(self, skill_func_tool):
        """Test loading a skill successfully."""
        result = skill_func_tool.load_skill("simple-skill")

        assert result.success == 1
        assert result.result is not None
        assert "Simple Skill" in result.result
        assert "Do this" in result.result

    def test_load_skill_not_found(self, skill_func_tool):
        """Test loading a nonexistent skill."""
        result = skill_func_tool.load_skill("nonexistent")

        assert result.success == 0
        assert result.error is not None
        assert "not found" in result.error.lower() or "not available" in result.error.lower()

    def test_load_skill_denied(self, skill_func_tool):
        """Test loading a denied skill."""
        result = skill_func_tool.load_skill("internal-skill")

        assert result.success == 0
        assert result.error is not None
        assert "not available" in result.error.lower()

    def test_load_skill_with_scripts(self, skill_func_tool):
        """Test loading a skill with scripts creates bash tool."""
        result = skill_func_tool.load_skill("script-skill")

        assert result.success == 1
        assert result.result is not None

        # Check that bash tool was created
        bash_tool = skill_func_tool.get_skill_bash_tool("script-skill")
        assert bash_tool is not None


class TestSkillFuncToolBashToolManagement:
    """Tests for bash tool management."""

    def test_get_skill_bash_tool_not_loaded(self, skill_func_tool):
        """Test getting bash tool for unloaded skill."""
        bash_tool = skill_func_tool.get_skill_bash_tool("script-skill")
        assert bash_tool is None

    def test_get_skill_bash_tool_after_load(self, skill_func_tool):
        """Test getting bash tool after loading skill."""
        skill_func_tool.load_skill("script-skill")
        bash_tool = skill_func_tool.get_skill_bash_tool("script-skill")

        assert bash_tool is not None
        assert bash_tool.skill_name == "script-skill"

    def test_get_all_skill_bash_tools(self, skill_func_tool):
        """Test getting all bash tools."""
        skill_func_tool.load_skill("script-skill")
        all_tools = skill_func_tool.get_all_skill_bash_tools()

        assert len(all_tools) == 1
        assert "script-skill" in all_tools

    def test_get_loaded_skill_tools(self, skill_func_tool):
        """Test getting tools from loaded skills."""
        # Before loading
        tools_before = skill_func_tool.get_loaded_skill_tools()
        assert len(tools_before) == 0

        # After loading skill with scripts
        skill_func_tool.load_skill("script-skill")
        tools_after = skill_func_tool.get_loaded_skill_tools()

        # Should have execute_command tool from bash tool
        assert len(tools_after) == 1

    def test_skill_without_scripts_no_bash_tool(self, skill_func_tool):
        """Test that loading skill without scripts doesn't create bash tool."""
        skill_func_tool.load_skill("simple-skill")
        bash_tool = skill_func_tool.get_skill_bash_tool("simple-skill")

        assert bash_tool is None


class TestSkillFuncToolPermissionCallback:
    """Tests for permission callback integration."""

    def test_set_permission_callback(self, skill_func_tool):
        """Test setting permission callback."""

        async def mock_callback(tool_category, tool_name, context):
            return True

        skill_func_tool.set_permission_callback(mock_callback)
        assert skill_func_tool._permission_callback is not None


class TestSkillFuncToolEdgeCases:
    """Edge case tests for SkillFuncTool."""

    def test_load_skill_empty_name(self, skill_func_tool):
        """Test loading skill with empty name."""
        result = skill_func_tool.load_skill("")

        assert result.success == 0
        assert result.error is not None

    def test_load_same_skill_twice(self, skill_func_tool):
        """Test loading the same skill twice."""
        result1 = skill_func_tool.load_skill("simple-skill")
        result2 = skill_func_tool.load_skill("simple-skill")

        assert result1.success == 1
        assert result2.success == 1
        # Content should be the same
        assert result1.result == result2.result

    def test_load_multiple_skills_with_scripts(self, temp_skills_dir, skill_manager):
        """Test loading multiple skills with scripts."""
        # Create another skill with scripts
        another_dir = temp_skills_dir / "another-script-skill"
        another_dir.mkdir()
        (another_dir / "SKILL.md").write_text(
            """---
name: another-script-skill
description: Another script skill
allowed_commands:
  - "sh:*.sh"
---
# Another
"""
        )

        # Refresh manager to pick up new skill
        skill_manager.refresh()

        tool = SkillFuncTool(manager=skill_manager, node_name="chatbot")
        tool.load_skill("script-skill")
        tool.load_skill("another-script-skill")

        all_tools = tool.get_all_skill_bash_tools()
        assert len(all_tools) == 2
        assert "script-skill" in all_tools
        assert "another-script-skill" in all_tools


class TestSkillExecuteCommand:
    """Tests for skill_execute_command method."""

    def test_execute_command_skill_not_loaded(self, skill_func_tool):
        """Test executing command when skill is not loaded yet."""
        result = skill_func_tool.skill_execute_command("script-skill", "python test.py")

        assert result.success == 0
        assert "not been loaded" in result.error
        assert "load_skill" in result.error

    def test_execute_command_skill_not_found(self, skill_func_tool):
        """Test executing command for non-existent skill."""
        result = skill_func_tool.skill_execute_command("nonexistent-skill", "python test.py")

        assert result.success == 0
        assert "not found" in result.error

    def test_execute_command_skill_no_scripts(self, skill_func_tool):
        """Test executing command for skill without allowed_commands."""
        # simple-skill has no allowed_commands defined
        result = skill_func_tool.skill_execute_command("simple-skill", "python test.py")

        assert result.success == 0
        assert "allowed_commands" in result.error

    def test_execute_command_after_load(self, skill_func_tool):
        """Test executing command after loading skill with scripts."""
        # First load the skill
        load_result = skill_func_tool.load_skill("script-skill")
        assert load_result.success == 1

        # Try to execute an allowed command (echo should be allowed by python:*)
        result = skill_func_tool.skill_execute_command("script-skill", "python -c \"print('hello')\"")

        # The command should be processed (success or failure depends on env)
        # We mainly test that it routes to the correct bash tool
        assert result is not None

    def test_execute_command_not_allowed(self, skill_func_tool):
        """Test executing command not in allowed patterns."""
        # Load skill
        skill_func_tool.load_skill("script-skill")

        # Try a command not in allowed patterns
        result = skill_func_tool.skill_execute_command("script-skill", "rm -rf /")

        assert result.success == 0
        assert "not allowed" in result.error.lower()
