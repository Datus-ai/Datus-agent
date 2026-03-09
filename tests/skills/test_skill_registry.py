# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillRegistry.

Tests skill discovery from filesystem and SKILL.md parsing.
"""

from pathlib import Path

import pytest

from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_registry import SkillRegistry


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory with test skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return skills_dir


@pytest.fixture
def simple_skill(temp_skills_dir):
    """Create a simple skill for testing."""
    skill_dir = temp_skills_dir / "simple-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: simple-skill
description: A simple test skill
tags:
  - test
  - simple
version: 1.0.0
---

# Simple Skill

This is a simple skill for testing.

## Usage

Just load and use it.
"""
    )
    return skill_dir


@pytest.fixture
def skill_with_scripts(temp_skills_dir):
    """Create a skill with script execution."""
    skill_dir = temp_skills_dir / "script-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: script-skill
description: A skill with scripts
tags:
  - scripts
  - python
allowed_commands:
  - "python:scripts/*.py"
  - "sh:*.sh"
---

# Script Skill

This skill includes executable scripts.
"""
    )

    # Create scripts directory
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "analyze.py").write_text("print('analyzing')")

    return skill_dir


@pytest.fixture
def skill_disabled_model(temp_skills_dir):
    """Create a skill with model invocation disabled."""
    skill_dir = temp_skills_dir / "user-only-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: user-only-skill
description: A skill that can only be invoked by user
disable_model_invocation: true
---

# User-Only Skill

This skill can only be invoked by user commands.
"""
    )
    return skill_dir


class TestSkillRegistryBasic:
    """Basic tests for SkillRegistry."""

    def test_registry_creation(self, temp_skills_dir):
        """Test creating a SkillRegistry."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        assert registry is not None

    def test_registry_scan_empty_directory(self, temp_skills_dir):
        """Test scanning an empty directory."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0

    def test_registry_scan_nonexistent_directory(self, tmp_path):
        """Test scanning a nonexistent directory."""
        config = SkillConfig(directories=[str(tmp_path / "nonexistent")])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0


class TestSkillRegistryDiscovery:
    """Tests for skill discovery."""

    def test_discover_simple_skill(self, temp_skills_dir, simple_skill):
        """Test discovering a simple skill."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        assert registry.get_skill_count() == 1
        skill = registry.get_skill("simple-skill")
        assert skill is not None
        assert skill.name == "simple-skill"
        assert skill.description == "A simple test skill"
        assert skill.tags == ["test", "simple"]
        assert skill.version == "1.0.0"

    def test_discover_skill_with_scripts(self, temp_skills_dir, skill_with_scripts):
        """Test discovering a skill with script execution."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("script-skill")
        assert skill is not None
        assert skill.has_scripts() is True
        assert "python:scripts/*.py" in skill.allowed_commands
        assert "sh:*.sh" in skill.allowed_commands

    def test_discover_multiple_skills(self, temp_skills_dir, simple_skill, skill_with_scripts):
        """Test discovering multiple skills."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        assert registry.get_skill_count() == 2
        skills = registry.list_skills()
        skill_names = [s.name for s in skills]
        assert "simple-skill" in skill_names
        assert "script-skill" in skill_names

    def test_discover_skill_disabled_model(self, temp_skills_dir, skill_disabled_model):
        """Test discovering a skill with model invocation disabled."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("user-only-skill")
        assert skill is not None
        assert skill.disable_model_invocation is True
        assert skill.is_model_invocable() is False


class TestSkillRegistryContentLoading:
    """Tests for skill content loading."""

    def test_load_skill_content(self, temp_skills_dir, simple_skill):
        """Test loading skill content."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        content = registry.load_skill_content("simple-skill")
        assert content is not None
        assert "# Simple Skill" in content
        assert "This is a simple skill for testing" in content

    def test_load_nonexistent_skill_content(self, temp_skills_dir):
        """Test loading content for nonexistent skill."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        content = registry.load_skill_content("nonexistent")
        assert content is None

    def test_content_cached_on_metadata(self, temp_skills_dir, simple_skill):
        """Test that loaded content is cached on metadata."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("simple-skill")
        assert skill.content is None  # Not loaded yet

        content = registry.load_skill_content("simple-skill")
        skill = registry.get_skill("simple-skill")
        assert skill.content is not None
        assert skill.content == content


class TestSkillRegistryRefresh:
    """Tests for registry refresh functionality."""

    def test_refresh_adds_new_skill(self, temp_skills_dir, simple_skill):
        """Test that refresh discovers new skills."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        assert registry.get_skill_count() == 1

        # Add a new skill
        new_skill_dir = temp_skills_dir / "new-skill"
        new_skill_dir.mkdir()
        (new_skill_dir / "SKILL.md").write_text(
            """---
name: new-skill
description: A newly added skill
---

# New Skill
"""
        )

        registry.refresh()
        assert registry.get_skill_count() == 2
        assert registry.get_skill("new-skill") is not None


class TestSkillRegistryEdgeCases:
    """Edge case tests for SkillRegistry."""

    def test_skill_without_frontmatter(self, temp_skills_dir):
        """Test handling skill without YAML frontmatter."""
        skill_dir = temp_skills_dir / "no-frontmatter"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """
# No Frontmatter Skill

This skill has no YAML frontmatter.
"""
        )

        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        # Should not be discovered or should handle gracefully
        assert registry.get_skill("no-frontmatter") is None

    def test_skill_with_invalid_yaml(self, temp_skills_dir):
        """Test handling skill with invalid YAML frontmatter."""
        skill_dir = temp_skills_dir / "invalid-yaml"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: invalid-yaml
description: [invalid yaml here
---

# Invalid YAML Skill
"""
        )

        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        # Should handle gracefully, not crash
        # Skill may or may not be discovered depending on error handling

    def test_skill_missing_required_fields(self, temp_skills_dir):
        """Test handling skill with missing required fields."""
        skill_dir = temp_skills_dir / "missing-fields"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: missing-fields
---

# Missing Fields Skill

No description field.
"""
        )

        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        # Should handle missing description gracefully
        registry.get_skill("missing-fields")
        # May or may not be discovered, but shouldn't crash

    def test_duplicate_skill_names(self, temp_skills_dir):
        """Test handling duplicate skill names across directories."""
        # Create two directories with same skill name
        dir1 = temp_skills_dir / "dir1" / "dup-skill"
        dir1.mkdir(parents=True)
        (dir1 / "SKILL.md").write_text(
            """---
name: dup-skill
description: First duplicate
---
# Dup 1
"""
        )

        dir2 = temp_skills_dir / "dir2" / "dup-skill"
        dir2.mkdir(parents=True)
        (dir2 / "SKILL.md").write_text(
            """---
name: dup-skill
description: Second duplicate
---
# Dup 2
"""
        )

        config = SkillConfig(
            directories=[str(temp_skills_dir / "dir1"), str(temp_skills_dir / "dir2")],
            warn_duplicates=True,
        )
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        # Should keep one of them (first one discovered)
        skill = registry.get_skill("dup-skill")
        assert skill is not None
        # Total count should be 1 (duplicates merged)
        assert registry.get_skill_count() == 1

    def test_get_nonexistent_skill(self, temp_skills_dir):
        """Test getting a skill that doesn't exist."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("nonexistent")
        assert skill is None

    def test_skill_location_is_path(self, temp_skills_dir, simple_skill):
        """Test that skill location is a Path object."""
        config = SkillConfig(directories=[str(temp_skills_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("simple-skill")
        assert isinstance(skill.location, Path)
        assert skill.location.exists()
