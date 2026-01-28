# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for flexible workflow configuration.

Tests multiple skills, pattern filtering, permission overrides, and chained workflows.
"""


import pytest

from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import (
    SkillConfig,
    SkillFuncTool,
    SkillManager,
    SkillRegistry,
)


@pytest.fixture
def multi_skill_setup(tmp_path):
    """Create multiple skills for different workflows."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Skill 1: Quick analysis (no scripts)
    quick_dir = skills_dir / "quick-analysis"
    quick_dir.mkdir()
    (quick_dir / "SKILL.md").write_text(
        """---
name: quick-analysis
description: Fast data exploration workflow
tags:
  - quick
  - exploration
---
# Quick Analysis

## Steps
1. list_tables()
2. execute_sql("SELECT * FROM {table} LIMIT 5")
3. Summarize findings
"""
    )

    # Skill 2: Deep analysis (with scripts)
    deep_dir = skills_dir / "deep-analysis"
    deep_dir.mkdir()
    (deep_dir / "SKILL.md").write_text(
        """---
name: deep-analysis
description: Comprehensive analysis with visualization
tags:
  - deep
  - visualization
allowed_commands:
  - "python:scripts/*.py"
---
# Deep Analysis

## Steps
1. Full schema analysis
2. Statistical profiling via scripts
3. Visualization generation
"""
    )
    scripts_dir = deep_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "profile.py").write_text('print("profiling...")')

    # Skill 3: Report generation
    report_dir = skills_dir / "report-generator"
    report_dir.mkdir()
    (report_dir / "SKILL.md").write_text(
        """---
name: report-generator
description: Generate formatted reports
tags:
  - report
  - export
allowed_commands:
  - "python:scripts/generate_report.py"
---
# Report Generator

Generate reports in various formats (markdown, PDF, HTML)
"""
    )
    report_scripts = report_dir / "scripts"
    report_scripts.mkdir()
    (report_scripts / "generate_report.py").write_text('print("report generated")')

    # Skill 4: Admin skill (for permission testing)
    admin_dir = skills_dir / "admin-tools"
    admin_dir.mkdir()
    (admin_dir / "SKILL.md").write_text(
        """---
name: admin-tools
description: Administrative tools
tags:
  - admin
  - internal
---
# Admin Tools

Internal admin functionality.
"""
    )

    return skills_dir


class TestFlexibleWorkflowConfiguration:
    """Tests for configuring flexible workflows via skills."""

    def test_multiple_skills_discovered(self, multi_skill_setup):
        """Test that multiple workflow skills are discovered."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skills = registry.list_skills()
        skill_names = [s.name for s in skills]

        assert "quick-analysis" in skill_names
        assert "deep-analysis" in skill_names
        assert "report-generator" in skill_names
        assert "admin-tools" in skill_names

    def test_skill_selection_by_pattern(self, multi_skill_setup):
        """Test filtering skills by pattern."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        manager = SkillManager(config=config)

        # Only analysis skills
        available = manager.get_available_skills("chatbot", patterns=["*-analysis"])
        names = [s.name for s in available]

        assert "quick-analysis" in names
        assert "deep-analysis" in names
        assert "report-generator" not in names
        assert "admin-tools" not in names

    def test_skill_selection_multiple_patterns(self, multi_skill_setup):
        """Test filtering skills with multiple patterns."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        manager = SkillManager(config=config)

        # Analysis and report skills
        available = manager.get_available_skills("chatbot", patterns=["*-analysis", "report-*"])
        names = [s.name for s in available]

        assert "quick-analysis" in names
        assert "deep-analysis" in names
        assert "report-generator" in names
        assert "admin-tools" not in names

    def test_skill_permission_deny(self, multi_skill_setup):
        """Test that DENY permission hides skills."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        perm_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
            ],
        )
        perm_manager = PermissionManager(global_config=perm_config)
        manager = SkillManager(config=config, permission_manager=perm_manager)

        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]

        # Admin skill should be hidden
        assert "admin-tools" not in names
        # Others should be visible
        assert "quick-analysis" in names
        assert "deep-analysis" in names

    def test_skill_permission_ask(self, multi_skill_setup):
        """Test that ASK permission keeps skill visible but requires confirmation."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        perm_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="deep-analysis", permission=PermissionLevel.ASK),
            ],
        )
        perm_manager = PermissionManager(global_config=perm_config)
        manager = SkillManager(config=config, permission_manager=perm_manager)

        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]

        # ASK skill should still be visible
        assert "deep-analysis" in names

        # But loading should return ASK_PERMISSION
        success, message, content = manager.load_skill("deep-analysis", "chatbot")
        assert success is False
        assert message == "ASK_PERMISSION"

    def test_skill_permission_override_per_node(self, multi_skill_setup):
        """Test per-node permission overrides for skills."""
        config = SkillConfig(directories=[str(multi_skill_setup)])

        # Global: admin-tools denied
        global_perm = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
            ],
        )

        # Node override: admin-tools allowed for admin_node
        node_overrides = {
            "admin_node": PermissionConfig(
                rules=[
                    PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.ALLOW),
                ],
            ),
        }

        perm_manager = PermissionManager(global_config=global_perm, node_overrides=node_overrides)
        manager = SkillManager(config=config, permission_manager=perm_manager)

        # Regular chatbot - admin-tools denied
        chatbot_skills = manager.get_available_skills("chatbot")
        chatbot_names = [s.name for s in chatbot_skills]
        assert "admin-tools" not in chatbot_names

        # Admin node - admin-tools allowed
        admin_skills = manager.get_available_skills("admin_node")
        admin_names = [s.name for s in admin_skills]
        assert "admin-tools" in admin_names

    def test_chained_skill_workflow(self, multi_skill_setup):
        """Test using multiple skills in sequence for complex workflow."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Workflow: quick-analysis -> deep-analysis -> report-generator

        # Step 1: Quick analysis
        result1 = skill_tool.load_skill("quick-analysis")
        assert result1.success == 1
        assert "list_tables" in result1.result

        # Step 2: Deep analysis with script
        result2 = skill_tool.load_skill("deep-analysis")
        assert result2.success == 1

        bash_tool = skill_tool.get_skill_bash_tool("deep-analysis")
        assert bash_tool is not None
        script_result = bash_tool.execute_command("python scripts/profile.py")
        assert script_result.success == 1

        # Step 3: Generate report
        result3 = skill_tool.load_skill("report-generator")
        assert result3.success == 1

        report_bash = skill_tool.get_skill_bash_tool("report-generator")
        assert report_bash is not None
        report_result = report_bash.execute_command("python scripts/generate_report.py")
        assert report_result.success == 1

    def test_skill_on_demand_script_loading(self, multi_skill_setup):
        """Test that SkillBashTool is only created when skill is loaded."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Initially, no bash tools for skills
        assert skill_tool.get_all_skill_bash_tools() == {}

        # After loading skill without scripts, no bash tool
        skill_tool.load_skill("quick-analysis")
        assert skill_tool.get_skill_bash_tool("quick-analysis") is None

        # After loading skill with scripts, bash tool should be available
        skill_tool.load_skill("deep-analysis")
        assert skill_tool.get_skill_bash_tool("deep-analysis") is not None

    def test_loaded_skill_tools_accumulate(self, multi_skill_setup):
        """Test that loaded skill tools accumulate."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        manager = SkillManager(config=config)

        skill_tool = SkillFuncTool(manager=manager, node_name="chatbot")

        # Load first skill with scripts
        skill_tool.load_skill("deep-analysis")
        tools1 = skill_tool.get_loaded_skill_tools()
        assert len(tools1) == 1

        # Load second skill with scripts
        skill_tool.load_skill("report-generator")
        tools2 = skill_tool.get_loaded_skill_tools()
        assert len(tools2) == 2

    def test_xml_generation_with_permissions(self, multi_skill_setup):
        """Test XML generation respects permissions."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        perm_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
            ],
        )
        perm_manager = PermissionManager(global_config=perm_config)
        manager = SkillManager(config=config, permission_manager=perm_manager)

        xml = manager.generate_available_skills_xml("chatbot")

        # Denied skill should not appear in XML
        assert "admin-tools" not in xml

        # Other skills should appear
        assert "quick-analysis" in xml
        assert "deep-analysis" in xml
        assert "report-generator" in xml

    def test_xml_generation_with_patterns(self, multi_skill_setup):
        """Test XML generation with pattern filtering."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        manager = SkillManager(config=config)

        xml = manager.generate_available_skills_xml("chatbot", patterns=["*-analysis"])

        # Only analysis skills should appear
        assert "quick-analysis" in xml
        assert "deep-analysis" in xml
        assert "report-generator" not in xml
        assert "admin-tools" not in xml


class TestSkillMetadataVariations:
    """Tests for various skill metadata configurations."""

    def test_skill_with_tags(self, multi_skill_setup):
        """Test that skill tags are properly loaded."""
        config = SkillConfig(directories=[str(multi_skill_setup)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("quick-analysis")
        assert "quick" in skill.tags
        assert "exploration" in skill.tags

    def test_skill_with_version(self, tmp_path):
        """Test that skill version is properly loaded."""
        skill_dir = tmp_path / "versioned-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: versioned-skill
description: A versioned skill
version: 2.0.0
---
# Versioned Skill
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("versioned-skill")
        assert skill.version == "2.0.0"

    def test_skill_user_invocable_flag(self, tmp_path):
        """Test user_invocable flag."""
        skill_dir = tmp_path / "hidden-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: hidden-skill
description: A skill hidden from user menu
user_invocable: false
---
# Hidden Skill
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        registry = SkillRegistry(config=config)
        registry.scan_directories()

        skill = registry.get_skill("hidden-skill")
        assert skill.user_invocable is False

    def test_skill_disable_model_invocation(self, tmp_path):
        """Test disable_model_invocation flag."""
        skill_dir = tmp_path / "user-only-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: user-only-skill
description: A skill that only users can invoke
disable_model_invocation: true
---
# User-Only Skill
"""
        )

        config = SkillConfig(directories=[str(tmp_path)])
        manager = SkillManager(config=config)

        # Should not appear in available skills for LLM
        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]
        assert "user-only-skill" not in names

        # But should be accessible directly
        skill = manager.get_skill("user-only-skill")
        assert skill is not None
        assert skill.is_model_invocable() is False
