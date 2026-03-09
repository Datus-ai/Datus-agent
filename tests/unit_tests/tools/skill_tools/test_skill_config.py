# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for skill configuration models.

Tests SkillConfig and SkillMetadata classes.
"""

from pathlib import Path

from datus.tools.skill_tools.skill_config import SkillConfig, SkillMetadata


class TestSkillConfig:
    """Tests for SkillConfig model."""

    def test_skill_config_defaults(self):
        """Test SkillConfig with default values."""
        config = SkillConfig()
        assert config.directories == ["~/.datus/skills", "./skills", "~/.claude/skills"]
        assert config.warn_duplicates is True
        assert config.whitelist_from_compaction is True

    def test_skill_config_custom_directories(self):
        """Test SkillConfig with custom directories."""
        config = SkillConfig(directories=["/custom/skills", "./project/skills"])
        assert config.directories == ["/custom/skills", "./project/skills"]

    def test_skill_config_disable_warnings(self):
        """Test SkillConfig with warnings disabled."""
        config = SkillConfig(warn_duplicates=False)
        assert config.warn_duplicates is False

    def test_skill_config_from_dict(self):
        """Test creating SkillConfig from dictionary."""
        config_dict = {
            "directories": ["/my/skills"],
            "warn_duplicates": False,
            "whitelist_from_compaction": False,
        }
        config = SkillConfig.from_dict(config_dict)
        assert config.directories == ["/my/skills"]
        assert config.warn_duplicates is False
        assert config.whitelist_from_compaction is False

    def test_skill_config_from_dict_empty(self):
        """Test creating SkillConfig from empty dictionary."""
        config = SkillConfig.from_dict({})
        assert config.directories == ["~/.datus/skills", "./skills", "~/.claude/skills"]
        assert config.warn_duplicates is True

    def test_skill_config_from_dict_partial(self):
        """Test creating SkillConfig from partial dictionary."""
        config = SkillConfig.from_dict({"warn_duplicates": False})
        assert config.directories == ["~/.datus/skills", "./skills", "~/.claude/skills"]
        assert config.warn_duplicates is False

    def test_skill_config_serialization(self):
        """Test SkillConfig serialization."""
        config = SkillConfig(directories=["/test"])
        data = config.model_dump()
        assert data["directories"] == ["/test"]
        assert data["warn_duplicates"] is True


class TestSkillMetadata:
    """Tests for SkillMetadata model."""

    def test_skill_metadata_required_fields(self):
        """Test SkillMetadata with required fields only."""
        metadata = SkillMetadata(
            name="test-skill",
            description="A test skill",
            location=Path("/skills/test-skill"),
        )
        assert metadata.name == "test-skill"
        assert metadata.description == "A test skill"
        assert metadata.location == Path("/skills/test-skill")

    def test_skill_metadata_all_fields(self):
        """Test SkillMetadata with all fields."""
        metadata = SkillMetadata(
            name="advanced-skill",
            description="An advanced skill",
            location=Path("/skills/advanced"),
            tags=["sql", "optimization"],
            version="1.0.0",
            allowed_commands=["python:scripts/*.py", "sh:*.sh"],
            disable_model_invocation=False,
            user_invocable=True,
            context="fork",
            agent="Explore",
        )
        assert metadata.name == "advanced-skill"
        assert metadata.tags == ["sql", "optimization"]
        assert metadata.version == "1.0.0"
        assert metadata.allowed_commands == ["python:scripts/*.py", "sh:*.sh"]
        assert metadata.context == "fork"
        assert metadata.agent == "Explore"

    def test_skill_metadata_defaults(self):
        """Test SkillMetadata default values."""
        metadata = SkillMetadata(
            name="test",
            description="test",
            location=Path("/test"),
        )
        assert metadata.tags == []
        assert metadata.version is None
        assert metadata.allowed_commands == []
        assert metadata.disable_model_invocation is False
        assert metadata.user_invocable is True
        assert metadata.context is None
        assert metadata.agent is None
        assert metadata.content is None

    def test_skill_metadata_has_scripts(self):
        """Test has_scripts method."""
        # Skill without scripts
        metadata_no_scripts = SkillMetadata(
            name="no-scripts",
            description="No scripts",
            location=Path("/test"),
        )
        assert metadata_no_scripts.has_scripts() is False

        # Skill with scripts
        metadata_with_scripts = SkillMetadata(
            name="with-scripts",
            description="With scripts",
            location=Path("/test"),
            allowed_commands=["python:*.py"],
        )
        assert metadata_with_scripts.has_scripts() is True

    def test_skill_metadata_is_model_invocable(self):
        """Test is_model_invocable method."""
        # Model invocable by default
        metadata_default = SkillMetadata(
            name="default",
            description="Default",
            location=Path("/test"),
        )
        assert metadata_default.is_model_invocable() is True

        # Explicitly disabled
        metadata_disabled = SkillMetadata(
            name="disabled",
            description="Disabled",
            location=Path("/test"),
            disable_model_invocation=True,
        )
        assert metadata_disabled.is_model_invocable() is False

    def test_skill_metadata_from_frontmatter(self):
        """Test creating SkillMetadata from frontmatter dict."""
        frontmatter = {
            "name": "sql-optimization",
            "description": "SQL query optimization techniques",
            "tags": ["sql", "performance"],
            "version": "1.0.0",
            "allowed_commands": ["python:scripts/*.py"],
        }
        metadata = SkillMetadata.from_frontmatter(frontmatter, Path("/skills/sql-optimization"))
        assert metadata.name == "sql-optimization"
        assert metadata.description == "SQL query optimization techniques"
        assert metadata.tags == ["sql", "performance"]
        assert metadata.version == "1.0.0"
        assert metadata.allowed_commands == ["python:scripts/*.py"]
        assert metadata.location == Path("/skills/sql-optimization")

    def test_skill_metadata_from_frontmatter_minimal(self):
        """Test creating SkillMetadata from minimal frontmatter."""
        frontmatter = {
            "name": "simple",
            "description": "Simple skill",
        }
        metadata = SkillMetadata.from_frontmatter(frontmatter, Path("/skills/simple"))
        assert metadata.name == "simple"
        assert metadata.description == "Simple skill"
        assert metadata.tags == []
        assert metadata.allowed_commands == []

    def test_skill_metadata_serialization(self):
        """Test SkillMetadata serialization."""
        metadata = SkillMetadata(
            name="test",
            description="Test skill",
            location=Path("/test"),
            tags=["tag1"],
        )
        data = metadata.model_dump()
        assert data["name"] == "test"
        assert data["description"] == "Test skill"
        assert data["tags"] == ["tag1"]
        # Path should be converted to string
        assert str(data["location"]) == "/test"

    def test_skill_metadata_content_lazy_loaded(self):
        """Test that content is lazy loaded (initially None)."""
        metadata = SkillMetadata(
            name="test",
            description="Test",
            location=Path("/test"),
        )
        assert metadata.content is None

        # Content can be set later
        metadata.content = "# Test Skill\n\nContent here"
        assert metadata.content == "# Test Skill\n\nContent here"
