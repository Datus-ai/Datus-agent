# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CI-level tests for the ext-knowledge-gen skill.

Tests SKILL.md parsing, script existence, and skill discovery.
"""

from pathlib import Path

import pytest
import yaml

from datus.tools.skill_tools.skill_config import SkillMetadata
from datus.tools.skill_tools.skill_registry import SkillRegistry

# ── Fixtures ──


@pytest.fixture
def skill_dir():
    """Return the path to the ext-knowledge-gen skill directory."""
    d = Path(__file__).resolve().parents[4] / "skills" / "ext-knowledge-gen"
    if not d.exists():
        pytest.skip("skills/ext-knowledge-gen directory not found")
    return d


@pytest.fixture
def skill_metadata(skill_dir):
    """Parse and return SkillMetadata from SKILL.md."""
    content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "SKILL.md must have YAML frontmatter"
    frontmatter = yaml.safe_load(parts[1])
    return SkillMetadata.from_frontmatter(frontmatter, skill_dir)


# ── SKILL.md Parsing Tests ──


@pytest.mark.ci
class TestExtKnowledgeGenSkillMd:
    """Test SKILL.md frontmatter parsing and content structure."""

    def test_skill_md_exists(self, skill_dir):
        assert (skill_dir / "SKILL.md").exists()

    def test_frontmatter_name(self, skill_metadata):
        assert skill_metadata.name == "ext-knowledge-gen"

    def test_frontmatter_description(self, skill_metadata):
        assert skill_metadata.description
        assert "knowledge" in skill_metadata.description.lower()

    def test_frontmatter_allowed_commands(self, skill_metadata):
        assert skill_metadata.allowed_commands
        assert "python:scripts/*.py" in skill_metadata.allowed_commands

    def test_frontmatter_tags(self, skill_metadata):
        assert skill_metadata.tags
        assert "ext-knowledge" in skill_metadata.tags

    def test_has_scripts(self, skill_metadata):
        assert skill_metadata.has_scripts()

    def test_is_model_invocable(self, skill_metadata):
        assert skill_metadata.is_model_invocable()

    def test_scripts_exist(self, skill_dir):
        scripts_dir = skill_dir / "scripts"
        assert scripts_dir.exists()
        assert (scripts_dir / "prepare_context.py").exists()
        assert (scripts_dir / "save_to_db.py").exists()


# ── Skill Discovery Tests ──


@pytest.mark.ci
class TestExtKnowledgeGenSkillDiscovery:
    """Test that ext-knowledge-gen skill is discoverable by SkillRegistry."""

    def test_registry_discovers_skill(self, skill_dir):
        registry = SkillRegistry(directories=[str(skill_dir.parent)])
        skill_names = [s.name for s in registry.list_skills()]
        assert "ext-knowledge-gen" in skill_names

    def test_registry_get_skill(self, skill_dir):
        registry = SkillRegistry(directories=[str(skill_dir.parent)])
        skill = registry.get_skill("ext-knowledge-gen")
        assert skill is not None
        assert skill.name == "ext-knowledge-gen"


# ── SKILL.md Content Tests ──


@pytest.mark.ci
class TestExtKnowledgeGenSkillContent:
    """Test that SKILL.md content includes expected workflow sections."""

    def test_has_phases(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "PHASE 1: Blind Test" in content
        assert "PHASE 2: Verify SQL" in content
        assert "PHASE 3: Analyze Gaps" in content
        assert "PHASE 4: Extract and Save Knowledge" in content

    def test_has_output_format(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "ext_knowledge_file" in content
        assert "Output Format" in content

    def test_has_yaml_format(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "search_text" in content
        assert "explanation" in content
        assert "subject_path" in content

    def test_has_subject_classification(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "Subject Classification" in content

    def test_has_verify_sql_rules(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "verify_sql" in content
        assert "success=1" in content
