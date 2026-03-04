# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for skill_registry.py covering diff lines."""

import pytest

from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_registry import SkillRegistry

SKILL_MD = """---
name: test-skill
description: A test skill
tags: [test, sql]
version: "1.0.0"
---

# Test Skill Body
"""


@pytest.fixture
def skill_dir(tmp_path):
    """Create a skill directory with SKILL.md."""
    d = tmp_path / "skills" / "test-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD)
    return tmp_path / "skills"


class TestSkillRegistryScanAndAccess:
    def test_scan_discovers_skills(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 1
        assert registry.get_skill("test-skill") is not None

    def test_scan_skips_nonexistent_dir(self):
        config = SkillConfig(directories=["/nonexistent/path"])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0

    def test_scan_skips_non_directory(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("hello")
        config = SkillConfig(directories=[str(f)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0

    def test_double_scan_is_noop(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        count1 = registry.get_skill_count()
        registry.scan_directories()
        assert registry.get_skill_count() == count1

    def test_list_skills(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        skills = registry.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_skill_exists(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.skill_exists("test-skill") is True
        assert registry.skill_exists("nonexistent") is False

    def test_get_skills_by_tag(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert len(registry.get_skills_by_tag("sql")) == 1
        assert len(registry.get_skills_by_tag("nonexistent")) == 0

    def test_load_skill_content(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        content = registry.load_skill_content("test-skill")
        assert "# Test Skill Body" in content

    def test_load_skill_content_caches(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        c1 = registry.load_skill_content("test-skill")
        c2 = registry.load_skill_content("test-skill")
        assert c1 == c2

    def test_load_skill_content_not_found(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.load_skill_content("nonexistent") is None

    def test_refresh(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 1
        registry.refresh()
        assert registry.get_skill_count() == 1

    def test_remove_skill(self, skill_dir):
        config = SkillConfig(directories=[str(skill_dir)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.remove_skill("test-skill") is True
        assert registry.remove_skill("test-skill") is False
        assert registry.get_skill_count() == 0


class TestSkillRegistryInstall:
    def test_install_skill(self, tmp_path):
        d = tmp_path / "installed"
        d.mkdir()
        (d / "SKILL.md").write_text(SKILL_MD)
        config = SkillConfig(directories=[])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        meta = registry.install_skill("test-skill", d)
        assert meta is not None
        assert meta.name == "test-skill"
        assert meta.source == "marketplace"

    def test_install_skill_missing_skill_md(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        config = SkillConfig(directories=[])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.install_skill("test", d) is None

    def test_install_skill_name_mismatch(self, tmp_path):
        d = tmp_path / "mismatch"
        d.mkdir()
        (d / "SKILL.md").write_text(SKILL_MD)
        config = SkillConfig(directories=[])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.install_skill("wrong-name", d) is None


class TestSkillRegistryDuplicateWarning:
    def test_duplicate_skill_warns(self, tmp_path):
        d1 = tmp_path / "skills1" / "my-skill"
        d1.mkdir(parents=True)
        (d1 / "SKILL.md").write_text(SKILL_MD)
        d2 = tmp_path / "skills2" / "my-skill"
        d2.mkdir(parents=True)
        (d2 / "SKILL.md").write_text(SKILL_MD)

        config = SkillConfig(
            directories=[str(tmp_path / "skills1"), str(tmp_path / "skills2")],
            warn_duplicates=True,
        )
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        # Only one should be registered (first wins)
        assert registry.get_skill_count() == 1


class TestSkillRegistryParseErrors:
    def test_invalid_yaml(self, tmp_path):
        d = tmp_path / "bad-skill"
        d.mkdir()
        (d / "SKILL.md").write_text("---\n: invalid: yaml: [[\n---\nBody")
        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0

    def test_no_frontmatter(self, tmp_path):
        d = tmp_path / "no-fm"
        d.mkdir()
        (d / "SKILL.md").write_text("Just text, no frontmatter")
        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0

    def test_empty_frontmatter(self, tmp_path):
        d = tmp_path / "empty-fm"
        d.mkdir()
        (d / "SKILL.md").write_text("---\n\n---\nBody")
        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0

    def test_missing_required_field(self, tmp_path):
        d = tmp_path / "no-desc"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: test\n---\nBody")
        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()
        assert registry.get_skill_count() == 0
