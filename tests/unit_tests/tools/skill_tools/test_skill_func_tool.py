# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillFuncTool.

Tests the load_skill native tool functionality.
"""

import pytest

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
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

    # Create a skill scoped to the ``gen_table`` subagent — used to exercise
    # the ``authoring_mode`` bypass and ``node_class`` alias matching.
    scoped_dir = skills_dir / "scoped-table"
    scoped_dir.mkdir()
    (scoped_dir / "SKILL.md").write_text(
        """---
name: scoped-table
description: Only usable by the gen_table subagent
allowed_agents:
  - gen_table
---

# Scoped Table Skill
Body.
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
        assert tool.node_name == "chatbot"

    def test_available_tools(self, skill_func_tool):
        """Test that available_tools returns only load_skill."""
        tools = skill_func_tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "load_skill"

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
        assert "Simple Skill" in result.result
        assert "Do this" in result.result

    def test_load_skill_not_found(self, skill_func_tool):
        """Test loading a nonexistent skill."""
        result = skill_func_tool.load_skill("nonexistent")

        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_load_skill_denied(self, skill_func_tool):
        """Test loading a denied skill."""
        result = skill_func_tool.load_skill("internal-skill")

        assert result.success == 0
        assert "not available" in result.error.lower()

    def test_aliased_authoring_tool_can_load_scoped_skill(self, skill_manager):
        """End-to-end: an aliased ``gen_skill`` subagent (e.g.
        ``my_skill_editor`` with ``node_class: gen_skill``) instantiates
        ``SkillFuncTool`` with ``node_class="gen_skill"`` and
        ``authoring_mode=True``. It must then be able to ``load_skill`` a
        scoped skill (``allowed_agents: [gen_table]``) — proving that the
        alias-aware scope check and the authoring bypass are wired through
        the whole ``SkillFuncTool -> SkillManager -> registry`` chain.
        """
        tool = SkillFuncTool(
            manager=skill_manager,
            node_name="my_skill_editor",
            node_class="gen_skill",
            authoring_mode=True,
        )
        result = tool.load_skill("scoped-table")

        assert result.success == 1
        assert "Scoped Table Skill" in result.result

    def test_non_authoring_tool_is_still_blocked_by_scope(self, skill_manager):
        """Chat (no authoring mode) cannot bypass ``allowed_agents``."""
        tool = SkillFuncTool(
            manager=skill_manager,
            node_name="chat",
            node_class="chat",
        )
        result = tool.load_skill("scoped-table")

        assert result.success == 0
        assert "not available" in result.error.lower()

    def test_aliased_non_authoring_tool_passes_via_node_class(self, skill_manager):
        """An alias whose ``node_class`` matches the whitelist loads even
        without ``authoring_mode`` — scope check honours the class name."""
        tool = SkillFuncTool(
            manager=skill_manager,
            node_name="my_tables",
            node_class="gen_table",
        )
        result = tool.load_skill("scoped-table")

        assert result.success == 1
        assert "Scoped Table Skill" in result.result


class TestSkillFuncToolPermissionCallback:
    """Tests for permission callback integration."""

    def test_set_permission_callback(self, skill_func_tool):
        """Test setting permission callback."""

        async def mock_callback(tool_category, tool_name, context):
            return True

        skill_func_tool.set_permission_callback(mock_callback)
        assert skill_func_tool._permission_callback is mock_callback


class TestSkillFuncToolEdgeCases:
    """Edge case tests for SkillFuncTool."""

    def test_load_skill_empty_name(self, skill_func_tool):
        """Test loading skill with empty name."""
        result = skill_func_tool.load_skill("")

        assert result.success == 0
        assert result.error == "Skill '' not found"

    def test_load_same_skill_twice(self, skill_func_tool):
        """Test loading the same skill twice."""
        result1 = skill_func_tool.load_skill("simple-skill")
        result2 = skill_func_tool.load_skill("simple-skill")

        assert result1.success == 1
        assert result2.success == 1
        # Content should be the same
        assert result1.result == result2.result
