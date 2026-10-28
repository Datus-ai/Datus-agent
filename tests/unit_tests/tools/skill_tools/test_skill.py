# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for AgentSkills + Permission system.

Tests that only require local test data (tests/data/skills/).
Tests requiring real LLM APIs or agent config loading are in
tests/integration/tools/test_skill.py (marked nightly).
"""

from pathlib import Path

import pytest

from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillConfig, SkillManager

TESTS_ROOT = Path(__file__).resolve().parents[3]  # tests/
SKILLS_DIR = TESTS_ROOT / "data" / "skills"


# ============================================================================
# 1. Skill Discovery
# ============================================================================


class TestSkillDiscovery:
    """Test skill discovery from real filesystem directories."""

    def test_discovers_all_skills_from_data_dir(self, skill_manager):
        """All 5 skills in tests/data/skills/ are discovered."""
        skills = skill_manager.list_all_skills()
        names = {s.name for s in skills}
        assert names == {
            "sql-analysis",
            "sql-optimization",
            "report-generator",
            "admin-tools",
            "data-profiler",
        }

    def test_multi_directory_discovery(self, skill_config_with_extra):
        """Skills from multiple directories are merged."""
        config, extra_dir = skill_config_with_extra

        new_skill_dir = extra_dir / "extra-skill"
        new_skill_dir.mkdir()
        (new_skill_dir / "SKILL.md").write_text("---\nname: extra-skill\ndescription: Extra\n---\n# Extra")

        manager = SkillManager(config=config)
        names = {s.name for s in manager.list_all_skills()}

        assert "extra-skill" in names
        assert "sql-analysis" in names
        assert len(names) == 6

    def test_refresh_picks_up_new_skills(self, skill_config_with_extra):
        """After adding a skill and calling refresh(), it's discovered."""
        config, extra_dir = skill_config_with_extra
        manager = SkillManager(config=config)

        initial_count = manager.get_skill_count()

        runtime_dir = extra_dir / "runtime-skill"
        runtime_dir.mkdir()
        (runtime_dir / "SKILL.md").write_text("---\nname: runtime-skill\ndescription: Added at runtime\n---\n# Runtime")

        manager.refresh()
        assert manager.get_skill_count() == initial_count + 1
        assert manager.get_skill("runtime-skill").name == "runtime-skill"

    def test_nonexistent_directory_gracefully_skipped(self, tmp_path):
        """Mix of valid + invalid directories works without error."""
        config = SkillConfig(
            directories=[
                str(SKILLS_DIR),
                str(tmp_path / "does_not_exist"),
            ]
        )
        manager = SkillManager(config=config)
        assert manager.get_skill_count() == 5

    def test_duplicate_skill_first_directory_wins(self, skill_config_with_extra):
        """When same skill name exists in two dirs, first discovered wins."""
        config, extra_dir = skill_config_with_extra

        dup_dir = extra_dir / "sql-analysis"
        dup_dir.mkdir()
        (dup_dir / "SKILL.md").write_text("---\nname: sql-analysis\ndescription: Override version\n---\n# Override")

        manager = SkillManager(config=config)
        skill = manager.get_skill("sql-analysis")
        assert skill.name == "sql-analysis"
        assert skill.description == "Guided workflow for SQL data analysis using db_tools"
        assert manager.get_skill_count() == 5


# ============================================================================
# 2. Load → Execute Pipeline
# ============================================================================


class TestSkillLoadAndExecute:
    """Test the full load → execute → result pipeline with real scripts."""

    def test_workflow_skill_loads_content(self, skill_func_tool):
        """Workflow skill returns content."""
        result = skill_func_tool.load_skill("sql-analysis")
        assert result.success == 1
        assert "Schema Discovery" in result.result
        assert "db_tools" in result.result

    def test_script_skill_loads_content(self, skill_func_tool):
        """Script skill (with scripts dir) loads content successfully."""
        result = skill_func_tool.load_skill("report-generator")
        assert result.success == 1
        assert "python scripts/generate_report.py" in result.result


# ============================================================================
# 3. Permission Enforcement
# ============================================================================


class TestPermissionEnforcement:
    """Test permission enforcement across SkillManager + PermissionManager layers."""

    def test_deny_hides_skill_from_available_and_xml(self, skill_manager_with_perms):
        """DENY permission hides skill from get_available_skills and XML."""
        available = skill_manager_with_perms.get_available_skills("chatbot")
        names = [s.name for s in available]

        assert "admin-tools" not in names
        assert "sql-analysis" in names
        assert "report-generator" in names

        xml = skill_manager_with_perms.generate_available_skills_xml("chatbot")
        assert "admin-tools" not in xml
        assert "sql-analysis" in xml

    def test_deny_blocks_load(self, skill_manager_with_perms):
        """DENY permission blocks load_skill."""
        success, message, _content = skill_manager_with_perms.load_skill("admin-tools", "chatbot")
        assert success is False
        assert "denied" in message.lower()

    def test_ask_keeps_skill_visible_but_blocks_load(self, skill_config, perm_ask_sql):
        """ASK permission keeps skill visible but returns ASK_PERMISSION on load."""
        perm_manager = PermissionManager(global_config=perm_ask_sql)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]
        assert "sql-analysis" in names

        success, message, _content = manager.load_skill("sql-analysis", "chatbot")
        assert success is False
        assert message == "ASK_PERMISSION"

    def test_node_override_grants_access_to_denied_skill(self, skill_config, perm_deny_admin_with_node_override):
        """Global DENY + node-specific ALLOW → skill accessible for that node."""
        global_config, node_overrides = perm_deny_admin_with_node_override
        perm_manager = PermissionManager(global_config=global_config, node_overrides=node_overrides)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        chatbot_skills = manager.get_available_skills("chatbot")
        chatbot_names = [s.name for s in chatbot_skills]
        assert "admin-tools" not in chatbot_names

        all_skills = manager.get_available_skills("school_all")
        all_names = [s.name for s in all_skills]
        assert "admin-tools" in all_names

        success, message, content = manager.load_skill("admin-tools", "school_all")
        assert success is True
        assert "Administrative" in content

    def test_permission_with_pattern_filtering_combined(self, skill_config, perm_deny_admin):
        """Pattern filter + permission filter work together."""
        perm_manager = PermissionManager(global_config=perm_deny_admin)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        available = manager.get_available_skills("chatbot", patterns=["sql-*"])
        names = [s.name for s in available]

        assert "sql-analysis" in names
        assert "sql-optimization" in names
        assert "report-generator" not in names
        assert "admin-tools" not in names

    def test_disable_model_invocation_hides_from_available(self, tmp_path):
        """disable_model_invocation: true hides skill from get_available_skills."""
        skill_dir = tmp_path / "hidden-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: hidden-skill\ndescription: Hidden\ndisable_model_invocation: true\n---\n# Hidden"
        )

        config = SkillConfig(directories=[str(tmp_path)])
        manager = SkillManager(config=config)

        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]
        assert "hidden-skill" not in names

        assert manager.get_skill("hidden-skill").name == "hidden-skill"


@pytest.mark.acceptance
class TestSkillsAcceptance:
    """Deterministic skill discovery, permission filtering, and safe load coverage."""

    def test_discovery_permission_filter_and_safe_load(self, skill_config, perm_deny_admin):
        perm_manager = PermissionManager(global_config=perm_deny_admin)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        available = manager.get_available_skills("chat")
        names = {skill.name for skill in available}
        assert "sql-analysis" in names
        assert "report-generator" in names
        assert "admin-tools" not in names

        success, message, content = manager.load_skill("sql-analysis", "chat")
        assert success is True
        assert "Schema Discovery" in content
        assert message

        denied, deny_message, deny_content = manager.load_skill("admin-tools", "chat")
        assert denied is False
        assert "denied" in deny_message.lower()
        assert deny_content is None


# ============================================================================
# 4. Multi-Skill Tool Accumulation Lifecycle
# ============================================================================
