# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillBashTool.

Tests restricted script execution with pattern-based filtering.
"""

from pathlib import Path

import pytest

from datus.tools.skill_tools.skill_bash_tool import SkillBashTool
from datus.tools.skill_tools.skill_config import SkillMetadata


@pytest.fixture
def temp_skill_dir(tmp_path):
    """Create a temporary skill directory with scripts."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()

    # Create scripts directory
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()

    # Create test scripts
    (scripts_dir / "analyze.py").write_text(
        """
import sys
print("Analysis complete")
print(f"Args: {sys.argv[1:]}")
"""
    )

    (scripts_dir / "process.py").write_text(
        """
import json
print(json.dumps({"status": "processed"}))
"""
    )

    (skill_dir / "run.sh").write_text(
        """
#!/bin/bash
echo "Shell script executed"
"""
    )

    return skill_dir


@pytest.fixture
def python_skill_metadata(temp_skill_dir):
    """Create skill metadata allowing Python scripts."""
    return SkillMetadata(
        name="python-skill",
        description="A skill with Python scripts",
        location=temp_skill_dir,
        allowed_commands=["python:scripts/*.py"],
    )


@pytest.fixture
def multi_pattern_skill_metadata(temp_skill_dir):
    """Create skill metadata allowing multiple command types."""
    return SkillMetadata(
        name="multi-skill",
        description="A skill with multiple command patterns",
        location=temp_skill_dir,
        allowed_commands=["python:scripts/*.py", "sh:*.sh", "python:-c:*"],
    )


@pytest.fixture
def wildcard_skill_metadata(temp_skill_dir):
    """Create skill metadata allowing all Python commands."""
    return SkillMetadata(
        name="wildcard-skill",
        description="A skill with wildcard Python pattern",
        location=temp_skill_dir,
        allowed_commands=["python:*"],
    )


@pytest.fixture
def no_scripts_skill_metadata(temp_skill_dir):
    """Create skill metadata with no allowed commands."""
    return SkillMetadata(
        name="no-scripts-skill",
        description="A skill without script execution",
        location=temp_skill_dir,
        allowed_commands=[],
    )


class TestSkillBashToolBasic:
    """Basic tests for SkillBashTool."""

    def test_tool_creation(self, python_skill_metadata, temp_skill_dir):
        """Test creating a SkillBashTool."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool is not None
        assert tool.skill_name == "python-skill"

    def test_tool_with_custom_timeout(self, python_skill_metadata, temp_skill_dir):
        """Test creating a SkillBashTool with custom timeout."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
            timeout=120,
        )
        assert tool.timeout == 120

    def test_available_tools_with_patterns(self, python_skill_metadata, temp_skill_dir):
        """Test that available_tools returns tools when patterns are defined."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        tools = tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "execute_command"

    def test_available_tools_without_patterns(self, no_scripts_skill_metadata, temp_skill_dir):
        """Test that available_tools returns empty when no patterns defined."""
        tool = SkillBashTool(
            skill_metadata=no_scripts_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        tools = tool.available_tools()
        assert len(tools) == 0

    def test_set_tool_context(self, python_skill_metadata, temp_skill_dir):
        """Test setting tool context."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        mock_context = {"key": "value"}
        tool.set_tool_context(mock_context)
        assert tool._tool_context == mock_context


class TestSkillBashToolPatternMatching:
    """Tests for command pattern matching."""

    def test_command_allowed_exact_match(self, python_skill_metadata, temp_skill_dir):
        """Test that matching command is allowed."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool._is_command_allowed("python scripts/analyze.py") is True

    def test_command_allowed_with_args(self, python_skill_metadata, temp_skill_dir):
        """Test that matching command with arguments is allowed."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool._is_command_allowed("python scripts/analyze.py --input data.json") is True

    def test_command_denied_wrong_prefix(self, python_skill_metadata, temp_skill_dir):
        """Test that command with wrong prefix is denied."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool._is_command_allowed("sh scripts/analyze.py") is False

    def test_command_denied_wrong_pattern(self, python_skill_metadata, temp_skill_dir):
        """Test that command with wrong pattern is denied."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        # Not in scripts/ directory
        assert tool._is_command_allowed("python other/analyze.py") is False

    def test_command_denied_dangerous(self, python_skill_metadata, temp_skill_dir):
        """Test that dangerous commands are denied."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool._is_command_allowed("rm -rf /") is False
        assert tool._is_command_allowed("cat /etc/passwd") is False

    def test_wildcard_pattern_allows_any(self, wildcard_skill_metadata, temp_skill_dir):
        """Test that wildcard pattern allows any matching command."""
        tool = SkillBashTool(
            skill_metadata=wildcard_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool._is_command_allowed("python any_script.py") is True
        assert tool._is_command_allowed("python -c \"print('hello')\"") is True

    def test_multi_pattern_matching(self, multi_pattern_skill_metadata, temp_skill_dir):
        """Test matching against multiple patterns."""
        tool = SkillBashTool(
            skill_metadata=multi_pattern_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool._is_command_allowed("python scripts/analyze.py") is True
        assert tool._is_command_allowed("sh run.sh") is True
        assert tool._is_command_allowed("python -c \"print('hello')\"") is True


class TestSkillBashToolExecution:
    """Tests for command execution."""

    def test_execute_allowed_command(self, python_skill_metadata, temp_skill_dir):
        """Test executing an allowed command."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("python scripts/analyze.py")

        assert result.success == 1
        assert "Analysis complete" in result.result

    def test_execute_command_with_args(self, python_skill_metadata, temp_skill_dir):
        """Test executing command with arguments."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("python scripts/analyze.py --input test.json")

        assert result.success == 1
        assert "--input" in result.result or "test.json" in result.result

    def test_execute_denied_command(self, python_skill_metadata, temp_skill_dir):
        """Test executing a denied command."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("rm -rf /")

        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_execute_empty_command(self, python_skill_metadata, temp_skill_dir):
        """Test executing an empty command."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("")

        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_execute_command_returns_json(self, python_skill_metadata, temp_skill_dir):
        """Test executing command that returns JSON."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("python scripts/process.py")

        assert result.success == 1
        assert "processed" in result.result

    def test_execute_command_failure(self, python_skill_metadata, temp_skill_dir):
        """Test executing command that fails."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        # Script that doesn't exist
        result = tool.execute_command("python scripts/nonexistent.py")

        assert result.success == 0
        # Should have error message


class TestSkillBashToolWorkspaceIsolation:
    """Tests for workspace isolation."""

    def test_workspace_root_set(self, python_skill_metadata, temp_skill_dir):
        """Test that workspace root is set correctly."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool.workspace_root == Path(temp_skill_dir).resolve()

    def test_command_runs_in_skill_directory(self, multi_pattern_skill_metadata, temp_skill_dir):
        """Test that commands run in the skill directory."""
        tool = SkillBashTool(
            skill_metadata=multi_pattern_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        # Create a script that prints cwd
        (temp_skill_dir / "scripts" / "pwd_test.py").write_text(
            """
import os
print(os.getcwd())
"""
        )

        result = tool.execute_command("python scripts/pwd_test.py")
        assert result.success == 1
        assert str(temp_skill_dir) in result.result or temp_skill_dir.name in result.result


class TestSkillBashToolEnvironment:
    """Tests for environment variables."""

    def test_skill_environment_variables(self, python_skill_metadata, temp_skill_dir):
        """Test that skill environment variables are set."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )

        # Create a script that prints env vars
        (temp_skill_dir / "scripts" / "env_test.py").write_text(
            """
import os
print(f"SKILL_NAME={os.environ.get('SKILL_NAME', 'NOT_SET')}")
print(f"SKILL_DIR={os.environ.get('SKILL_DIR', 'NOT_SET')}")
"""
        )

        result = tool.execute_command("python scripts/env_test.py")
        assert result.success == 1
        assert "SKILL_NAME=python-skill" in result.result


class TestSkillBashToolEdgeCases:
    """Edge case tests for SkillBashTool."""

    def test_command_with_quotes(self, wildcard_skill_metadata, temp_skill_dir):
        """Test command with quoted arguments."""
        tool = SkillBashTool(
            skill_metadata=wildcard_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("python -c \"print('hello world')\"")

        assert result.success == 1
        assert "hello world" in result.result

    def test_command_with_special_characters(self, wildcard_skill_metadata, temp_skill_dir):
        """Test command with special characters."""
        tool = SkillBashTool(
            skill_metadata=wildcard_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command('python -c "print(1+2)"')

        assert result.success == 1
        assert "3" in result.result

    def test_whitespace_only_command(self, python_skill_metadata, temp_skill_dir):
        """Test whitespace-only command."""
        tool = SkillBashTool(
            skill_metadata=python_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("   ")

        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_no_patterns_returns_empty_tools(self, no_scripts_skill_metadata, temp_skill_dir):
        """Test that no patterns returns empty tools list."""
        tool = SkillBashTool(
            skill_metadata=no_scripts_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        assert tool.available_tools() == []

    def test_no_patterns_denies_all_commands(self, no_scripts_skill_metadata, temp_skill_dir):
        """Test that no patterns denies all commands."""
        tool = SkillBashTool(
            skill_metadata=no_scripts_skill_metadata,
            workspace_root=str(temp_skill_dir),
        )
        result = tool.execute_command("python anything.py")

        assert result.success == 0
        assert "not allowed" in result.error.lower()


class TestSkillBashToolTimeout:
    """Tests for timeout functionality."""

    def test_command_timeout(self, wildcard_skill_metadata, temp_skill_dir):
        """Test that long-running commands timeout."""
        tool = SkillBashTool(
            skill_metadata=wildcard_skill_metadata,
            workspace_root=str(temp_skill_dir),
            timeout=1,  # 1 second timeout
        )

        # Create a script that sleeps
        (temp_skill_dir / "scripts" / "sleep_test.py").write_text(
            """
import time
time.sleep(10)
print("Done")
"""
        )

        result = tool.execute_command("python scripts/sleep_test.py")

        assert result.success == 0
        assert "timed out" in result.error.lower()
