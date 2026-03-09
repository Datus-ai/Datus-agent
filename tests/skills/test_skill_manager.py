# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillManager.

Tests skill coordination, permission integration, and XML generation.
"""

import pytest

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_manager import SkillManager


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create multiple skills
    for skill_name, desc, tags in [
        ("sql-optimization", "SQL query optimization techniques", ["sql", "performance"]),
        ("data-analysis", "Data analysis workflows", ["data", "analysis"]),
        ("internal-admin", "Internal admin tools", ["internal", "admin"]),
        ("dangerous-script", "Dangerous script skill", ["dangerous"]),
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
            PermissionRule(tool="skills", pattern="internal-*", permission=PermissionLevel.DENY),
            PermissionRule(tool="skills", pattern="dangerous-*", permission=PermissionLevel.ASK),
        ],
    )
    return PermissionManager(global_config=config)


class TestSkillManagerBasic:
    """Basic tests for SkillManager."""

    def test_manager_creation(self, skill_config):
        """Test creating a SkillManager."""
        manager = SkillManager(config=skill_config)
        assert manager is not None
        assert manager.get_skill_count() > 0

    def test_manager_with_permission_manager(self, skill_config, permission_manager):
        """Test creating SkillManager with permission manager."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        assert manager.permission_manager is not None

    def test_get_skill_count(self, skill_config):
        """Test getting skill count."""
        manager = SkillManager(config=skill_config)
        assert manager.get_skill_count() == 4

    def test_get_skill(self, skill_config):
        """Test getting a specific skill."""
        manager = SkillManager(config=skill_config)
        skill = manager.get_skill("sql-optimization")
        assert skill is not None
        assert skill.name == "sql-optimization"

    def test_list_all_skills(self, skill_config):
        """Test listing all skills."""
        manager = SkillManager(config=skill_config)
        skills = manager.list_all_skills()
        assert len(skills) == 4
        skill_names = [s.name for s in skills]
        assert "sql-optimization" in skill_names
        assert "data-analysis" in skill_names


class TestSkillManagerAvailableSkills:
    """Tests for get_available_skills with permissions."""

    def test_available_skills_no_permissions(self, skill_config):
        """Test available skills without permission manager."""
        manager = SkillManager(config=skill_config)
        skills = manager.get_available_skills("chatbot")
        assert len(skills) == 4

    def test_available_skills_with_permissions(self, skill_config, permission_manager):
        """Test available skills filtered by permissions."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        skills = manager.get_available_skills("chatbot")

        # internal-* should be hidden (DENY)
        skill_names = [s.name for s in skills]
        assert "internal-admin" not in skill_names

        # Others should be visible (including ASK ones)
        assert "sql-optimization" in skill_names
        assert "data-analysis" in skill_names
        assert "dangerous-script" in skill_names  # ASK, but visible

    def test_available_skills_with_patterns(self, skill_config):
        """Test available skills filtered by patterns."""
        manager = SkillManager(config=skill_config)
        skills = manager.get_available_skills("chatbot", patterns=["sql-*"])

        assert len(skills) == 1
        assert skills[0].name == "sql-optimization"

    def test_available_skills_multiple_patterns(self, skill_config):
        """Test available skills with multiple patterns."""
        manager = SkillManager(config=skill_config)
        skills = manager.get_available_skills("chatbot", patterns=["sql-*", "data-*"])

        assert len(skills) == 2
        skill_names = [s.name for s in skills]
        assert "sql-optimization" in skill_names
        assert "data-analysis" in skill_names

    def test_available_skills_wildcard_pattern(self, skill_config):
        """Test available skills with wildcard pattern."""
        manager = SkillManager(config=skill_config)
        skills = manager.get_available_skills("chatbot", patterns=["*"])

        assert len(skills) == 4


class TestSkillManagerLoadSkill:
    """Tests for load_skill method."""

    def test_load_skill_success(self, skill_config):
        """Test loading a skill successfully."""
        manager = SkillManager(config=skill_config)
        success, message, content = manager.load_skill("sql-optimization", "chatbot")

        assert success is True
        assert "loaded successfully" in message.lower() or "sql-optimization" in message.lower()
        assert content is not None
        assert "SQL query optimization" in content

    def test_load_skill_not_found(self, skill_config):
        """Test loading a nonexistent skill."""
        manager = SkillManager(config=skill_config)
        success, message, content = manager.load_skill("nonexistent", "chatbot")

        assert success is False
        assert "not found" in message.lower()
        assert content is None

    def test_load_skill_denied(self, skill_config, permission_manager):
        """Test loading a denied skill."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        success, message, content = manager.load_skill("internal-admin", "chatbot")

        assert success is False
        assert "denied" in message.lower() or "permission" in message.lower()
        assert content is None

    def test_load_skill_ask_permission(self, skill_config, permission_manager):
        """Test loading a skill that requires ASK permission."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        success, message, content = manager.load_skill("dangerous-script", "chatbot")

        # Should return special status for ASK
        assert success is False
        assert message == "ASK_PERMISSION"
        assert content is None

    def test_load_skill_skip_permission_check(self, skill_config, permission_manager):
        """Test loading a skill with permission check disabled."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        success, message, content = manager.load_skill("internal-admin", "chatbot", check_permission=False)

        # Should succeed when permission check is skipped
        assert success is True
        assert content is not None


class TestSkillManagerXMLGeneration:
    """Tests for XML context generation."""

    def test_generate_xml_basic(self, skill_config):
        """Test generating XML for available skills."""
        manager = SkillManager(config=skill_config)
        xml = manager.generate_available_skills_xml("chatbot")

        assert "<available_skills>" in xml
        assert "</available_skills>" in xml
        assert "sql-optimization" in xml
        assert "load_skill" in xml

    def test_generate_xml_with_permissions(self, skill_config, permission_manager):
        """Test generating XML respects permissions."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        xml = manager.generate_available_skills_xml("chatbot")

        # DENY skills should not appear
        assert "internal-admin" not in xml

        # Other skills should appear
        assert "sql-optimization" in xml
        assert "data-analysis" in xml

    def test_generate_xml_with_patterns(self, skill_config):
        """Test generating XML with pattern filter."""
        manager = SkillManager(config=skill_config)
        xml = manager.generate_available_skills_xml("chatbot", patterns=["sql-*"])

        assert "sql-optimization" in xml
        assert "data-analysis" not in xml

    def test_generate_xml_empty(self, tmp_path):
        """Test generating XML when no skills available."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = SkillConfig(directories=[str(empty_dir)])
        manager = SkillManager(config=config)
        xml = manager.generate_available_skills_xml("chatbot")

        assert xml == ""

    def test_generate_xml_includes_description(self, skill_config):
        """Test that generated XML includes skill descriptions."""
        manager = SkillManager(config=skill_config)
        xml = manager.generate_available_skills_xml("chatbot")

        assert "<description>" in xml
        assert "SQL query optimization techniques" in xml

    def test_generate_xml_includes_tags(self, skill_config):
        """Test that generated XML includes skill tags."""
        manager = SkillManager(config=skill_config)
        xml = manager.generate_available_skills_xml("chatbot")

        assert "<tags>" in xml


class TestSkillManagerPermissionCheck:
    """Tests for check_skill_permission method."""

    def test_check_permission_allow(self, skill_config, permission_manager):
        """Test checking ALLOW permission."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        permission = manager.check_skill_permission("sql-optimization", "chatbot")

        assert permission == PermissionLevel.ALLOW

    def test_check_permission_deny(self, skill_config, permission_manager):
        """Test checking DENY permission."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        permission = manager.check_skill_permission("internal-admin", "chatbot")

        assert permission == PermissionLevel.DENY

    def test_check_permission_ask(self, skill_config, permission_manager):
        """Test checking ASK permission."""
        manager = SkillManager(config=skill_config, permission_manager=permission_manager)
        permission = manager.check_skill_permission("dangerous-script", "chatbot")

        assert permission == PermissionLevel.ASK

    def test_check_permission_no_manager(self, skill_config):
        """Test checking permission without permission manager."""
        manager = SkillManager(config=skill_config)
        permission = manager.check_skill_permission("any-skill", "chatbot")

        # Should default to ALLOW when no manager
        assert permission == PermissionLevel.ALLOW


class TestSkillManagerPatternParsing:
    """Tests for pattern parsing utility."""

    def test_parse_empty_patterns(self, skill_config):
        """Test parsing empty patterns string."""
        manager = SkillManager(config=skill_config)
        patterns = manager.parse_skill_patterns("")

        assert patterns == []

    def test_parse_single_pattern(self, skill_config):
        """Test parsing single pattern."""
        manager = SkillManager(config=skill_config)
        patterns = manager.parse_skill_patterns("sql-*")

        assert patterns == ["sql-*"]

    def test_parse_multiple_patterns(self, skill_config):
        """Test parsing multiple patterns."""
        manager = SkillManager(config=skill_config)
        patterns = manager.parse_skill_patterns("sql-*, data-*, analysis-*")

        assert patterns == ["sql-*", "data-*", "analysis-*"]

    def test_parse_patterns_with_whitespace(self, skill_config):
        """Test parsing patterns with extra whitespace."""
        manager = SkillManager(config=skill_config)
        patterns = manager.parse_skill_patterns("  sql-*  ,  data-*  ")

        assert patterns == ["sql-*", "data-*"]


class TestSkillManagerRefresh:
    """Tests for refresh functionality."""

    def test_refresh(self, skill_config, temp_skills_dir):
        """Test refreshing skill registry."""
        manager = SkillManager(config=skill_config)
        initial_count = manager.get_skill_count()

        # Add a new skill
        new_skill_dir = temp_skills_dir / "new-skill"
        new_skill_dir.mkdir()
        (new_skill_dir / "SKILL.md").write_text(
            """---
name: new-skill
description: A new skill
---
# New Skill
"""
        )

        manager.refresh()
        assert manager.get_skill_count() == initial_count + 1
        assert manager.get_skill("new-skill") is not None
