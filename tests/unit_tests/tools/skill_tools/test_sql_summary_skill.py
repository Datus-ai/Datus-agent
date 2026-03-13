# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CI-level tests for the gen-sql-summary skill.

Tests SKILL.md parsing, extra_env injection, and script execution patterns.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datus.tools.skill_tools.skill_bash_tool import SkillBashTool
from datus.tools.skill_tools.skill_config import SkillMetadata
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_registry import SkillRegistry

# ── Fixtures ──


@pytest.fixture
def sql_summary_skill_dir():
    """Return the path to the gen-sql-summary skill directory."""
    skill_dir = Path(__file__).resolve().parents[4] / "skills" / "gen-sql-summary"
    if not skill_dir.exists():
        pytest.skip("skills/gen-sql-summary directory not found")
    return skill_dir


@pytest.fixture
def sql_summary_metadata(sql_summary_skill_dir):
    """Parse and return SkillMetadata from gen-sql-summary SKILL.md."""
    skill_md = sql_summary_skill_dir / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")

    # Extract frontmatter
    parts = content.split("---", 2)
    assert len(parts) >= 3, "SKILL.md must have YAML frontmatter"

    import yaml

    frontmatter = yaml.safe_load(parts[1])
    return SkillMetadata.from_frontmatter(frontmatter, sql_summary_skill_dir)


# ── SKILL.md Parsing Tests ──


@pytest.mark.ci
class TestSqlSummarySkillMd:
    """Test SKILL.md frontmatter parsing and content structure."""

    def test_skill_md_exists(self, sql_summary_skill_dir):
        """SKILL.md file exists in skill directory."""
        assert (sql_summary_skill_dir / "SKILL.md").exists()

    def test_frontmatter_name(self, sql_summary_metadata):
        """Skill name is 'gen-sql-summary'."""
        assert sql_summary_metadata.name == "gen-sql-summary"

    def test_frontmatter_description(self, sql_summary_metadata):
        """Skill has a description."""
        assert sql_summary_metadata.description
        assert "SQL" in sql_summary_metadata.description

    def test_frontmatter_allowed_commands(self, sql_summary_metadata):
        """Skill allows Python script execution."""
        assert sql_summary_metadata.allowed_commands
        assert "python:scripts/*.py" in sql_summary_metadata.allowed_commands

    def test_frontmatter_tags(self, sql_summary_metadata):
        """Skill has relevant tags."""
        assert sql_summary_metadata.tags
        assert "sql" in sql_summary_metadata.tags
        assert "summary" in sql_summary_metadata.tags

    def test_has_scripts(self, sql_summary_metadata):
        """Skill reports having scripts."""
        assert sql_summary_metadata.has_scripts()

    def test_is_model_invocable(self, sql_summary_metadata):
        """Skill is invocable by model."""
        assert sql_summary_metadata.is_model_invocable()

    def test_scripts_exist(self, sql_summary_skill_dir):
        """All required scripts exist."""
        scripts_dir = sql_summary_skill_dir / "scripts"
        assert scripts_dir.exists()
        assert (scripts_dir / "prepare_context.py").exists()
        assert (scripts_dir / "generate_id.py").exists()
        assert (scripts_dir / "save_to_db.py").exists()


# ── Skill Discovery Tests ──


@pytest.mark.ci
class TestSqlSummarySkillDiscovery:
    """Test that gen-sql-summary skill is discoverable by SkillRegistry."""

    def test_registry_discovers_skill(self, sql_summary_skill_dir):
        """SkillRegistry discovers gen-sql-summary from its parent directory."""
        registry = SkillRegistry(directories=[str(sql_summary_skill_dir.parent)])
        skills = registry.list_skills()
        skill_names = [s.name for s in skills]
        assert "gen-sql-summary" in skill_names

    def test_registry_get_skill(self, sql_summary_skill_dir):
        """SkillRegistry can retrieve gen-sql-summary by name."""
        registry = SkillRegistry(directories=[str(sql_summary_skill_dir.parent)])
        skill = registry.get_skill("gen-sql-summary")
        assert skill is not None
        assert skill.name == "gen-sql-summary"


# ── Extra Env Injection Tests ──


@pytest.mark.ci
class TestExtraEnvInjection:
    """Test that extra_env is properly injected into SkillBashTool."""

    def test_skill_bash_tool_extra_env(self, sql_summary_metadata):
        """SkillBashTool passes extra_env to subprocess environment."""
        extra_env = {
            "DATUS_HOME": "/tmp/test_home",
            "DATUS_NAMESPACE": "test_ns",
            "DATUS_CONFIG_PATH": "/tmp/agent.yml",
        }
        bash_tool = SkillBashTool(
            skill_metadata=sql_summary_metadata,
            workspace_root=str(sql_summary_metadata.location),
            extra_env=extra_env,
        )
        env = bash_tool._get_safe_env()
        assert env["DATUS_HOME"] == "/tmp/test_home"
        assert env["DATUS_NAMESPACE"] == "test_ns"
        assert env["DATUS_CONFIG_PATH"] == "/tmp/agent.yml"
        assert env["SKILL_NAME"] == "gen-sql-summary"

    def test_skill_bash_tool_empty_extra_env(self, sql_summary_metadata):
        """SkillBashTool works with no extra_env."""
        bash_tool = SkillBashTool(
            skill_metadata=sql_summary_metadata,
            workspace_root=str(sql_summary_metadata.location),
        )
        env = bash_tool._get_safe_env()
        assert env["SKILL_NAME"] == "gen-sql-summary"
        # Should not have DATUS_* unless already in os.environ
        assert "DATUS_NAMESPACE" not in env or env["DATUS_NAMESPACE"] == ""

    def test_skill_func_tool_passes_extra_env(self, sql_summary_skill_dir):
        """SkillFuncTool passes extra_env when creating SkillBashTool."""
        extra_env = {"DATUS_HOME": "/tmp/test", "DATUS_NAMESPACE": "ns1", "DATUS_CONFIG_PATH": "/tmp/a.yml"}

        # Create a mock SkillManager
        mock_manager = MagicMock()
        func_tool = SkillFuncTool(manager=mock_manager, node_name="test", extra_env=extra_env)

        assert func_tool.extra_env == extra_env

        # Create a skill metadata for testing _create_skill_bash_tool
        metadata = SkillMetadata(
            name="test-skill",
            description="test",
            allowed_commands=["python:scripts/*.py"],
            location=sql_summary_skill_dir,
        )
        func_tool._create_skill_bash_tool(metadata)

        bash_tool = func_tool.get_skill_bash_tool("test-skill")
        assert bash_tool is not None
        assert bash_tool.extra_env == extra_env


# ── Generate ID Script Tests ──


@pytest.mark.ci
class TestGenerateIdScript:
    """Test generate_id.py script execution."""

    def test_generate_id_script_runs(self, sql_summary_skill_dir):
        """generate_id.py produces a non-empty ID for a SQL query."""
        result = subprocess.run(
            [sys.executable, "scripts/generate_id.py", "--sql-query", "SELECT 1"],
            cwd=str(sql_summary_skill_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert result.stdout.strip()  # Non-empty ID

    def test_generate_id_deterministic(self, sql_summary_skill_dir):
        """Same SQL produces same ID."""
        sql = "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id"
        results = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "scripts/generate_id.py", "--sql-query", sql],
                cwd=str(sql_summary_skill_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            results.append(result.stdout.strip())
        assert results[0] == results[1]

    def test_generate_id_empty_query_fails(self, sql_summary_skill_dir):
        """Empty SQL query produces an error."""
        result = subprocess.run(
            [sys.executable, "scripts/generate_id.py", "--sql-query", "   "],
            cwd=str(sql_summary_skill_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0


# ── Prepare Context Script Tests ──


@pytest.mark.ci
class TestPrepareContextScript:
    """Test prepare_context.py error handling (no real config available in CI)."""

    def test_prepare_context_missing_config_path(self, sql_summary_skill_dir):
        """Script fails gracefully when DATUS_CONFIG_PATH is not set."""
        env = {"PATH": subprocess.os.environ.get("PATH", ""), "PYTHONPATH": subprocess.os.environ.get("PYTHONPATH", "")}
        result = subprocess.run(
            [sys.executable, "scripts/prepare_context.py", "--sql-query", "SELECT 1"],
            cwd=str(sql_summary_skill_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode != 0
        # Should output error JSON
        output = result.stdout.strip()
        if output:
            data = json.loads(output)
            assert "error" in data


# ── Save to DB Script Tests ──


@pytest.mark.ci
class TestSaveToDbScript:
    """Test save_to_db.py error handling."""

    def test_save_to_db_missing_config(self, sql_summary_skill_dir):
        """Script fails gracefully when DATUS_CONFIG_PATH is not set."""
        env = {"PATH": subprocess.os.environ.get("PATH", ""), "PYTHONPATH": subprocess.os.environ.get("PYTHONPATH", "")}
        result = subprocess.run(
            [sys.executable, "scripts/save_to_db.py", "--file-path", "test.yaml"],
            cwd=str(sql_summary_skill_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode != 0
