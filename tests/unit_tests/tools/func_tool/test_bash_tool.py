# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for the general-purpose BashTool.

Covers pattern matching, command execution, workspace isolation, env injection,
timeout, output limits, and the ``allowed_patterns`` semantics that decide
whether the tool is exposed.
"""

from pathlib import Path

import pytest

from datus.tools.func_tool.bash_tool import BashTool


@pytest.fixture
def temp_workspace(tmp_path):
    """Workspace with a few helper scripts."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    scripts_dir = workspace / "scripts"
    scripts_dir.mkdir()

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

    (workspace / "run.sh").write_text('#!/bin/bash\necho "Shell script executed"\n')

    return workspace


@pytest.fixture
def python_tool(temp_workspace):
    return BashTool(
        workspace_root=str(temp_workspace),
        allowed_patterns=["python:scripts/*.py"],
    )


@pytest.fixture
def multi_pattern_tool(temp_workspace):
    return BashTool(
        workspace_root=str(temp_workspace),
        allowed_patterns=["python:scripts/*.py", "sh:*.sh", "python:-c:*"],
    )


@pytest.fixture
def wildcard_tool(temp_workspace):
    return BashTool(
        workspace_root=str(temp_workspace),
        allowed_patterns=["python:*"],
    )


@pytest.fixture
def unrestricted_tool(temp_workspace):
    """Tool with ``["*"]`` — pattern filter passes any command."""
    return BashTool(
        workspace_root=str(temp_workspace),
        allowed_patterns=["*"],
    )


@pytest.fixture
def empty_tool(temp_workspace):
    return BashTool(
        workspace_root=str(temp_workspace),
        allowed_patterns=[],
    )


class TestBashToolConstruction:
    def test_basic_construction(self, python_tool, temp_workspace):
        assert python_tool.workspace_root == Path(temp_workspace).resolve()
        assert python_tool.allowed_patterns == ["python:scripts/*.py"]
        assert python_tool.timeout == 60

    def test_custom_timeout(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["python:*"], timeout=120)
        assert tool.timeout == 120

    def test_identity_label(self, temp_workspace):
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["python:*"],
            identity="my-skill",
        )
        assert tool.identity == "my-skill"

    def test_none_patterns_treated_as_empty(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=None)
        assert tool.allowed_patterns == []

    def test_extra_env_is_copied(self, temp_workspace):
        env = {"FOO": "bar"}
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["python:*"],
            extra_env=env,
        )
        # Mutating the input dict should not affect the tool's stored copy.
        env["FOO"] = "mutated"
        assert tool.extra_env == {"FOO": "bar"}

    def test_set_tool_context(self, python_tool):
        ctx = {"key": "value"}
        python_tool.set_tool_context(ctx)
        assert python_tool._tool_context == ctx


class TestBashToolAvailableTools:
    def test_patterns_present_exposes_tool(self, python_tool):
        tools = python_tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "execute_command"

    def test_empty_patterns_hides_tool(self, empty_tool):
        assert empty_tool.available_tools() == []

    def test_none_patterns_hides_tool(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=None)
        assert tool.available_tools() == []

    def test_wildcard_pattern_exposes_tool(self, unrestricted_tool):
        assert len(unrestricted_tool.available_tools()) == 1


class TestBashToolPatternMatching:
    def test_exact_match(self, python_tool):
        assert python_tool._is_command_allowed("python scripts/analyze.py") is True

    def test_match_with_args(self, python_tool):
        assert python_tool._is_command_allowed("python scripts/analyze.py --input data.json") is True

    def test_wrong_prefix_denied(self, python_tool):
        assert python_tool._is_command_allowed("sh scripts/analyze.py") is False

    def test_wrong_path_denied(self, python_tool):
        assert python_tool._is_command_allowed("python other/analyze.py") is False

    def test_dangerous_commands_denied(self, python_tool):
        assert python_tool._is_command_allowed("rm -rf /") is False
        assert python_tool._is_command_allowed("cat /etc/passwd") is False

    def test_wildcard_pattern_allows_any_python(self, wildcard_tool):
        assert wildcard_tool._is_command_allowed("python any_script.py") is True
        assert wildcard_tool._is_command_allowed("python -c \"print('hello')\"") is True

    def test_multi_pattern_matching(self, multi_pattern_tool):
        assert multi_pattern_tool._is_command_allowed("python scripts/analyze.py") is True
        assert multi_pattern_tool._is_command_allowed("sh run.sh") is True
        assert multi_pattern_tool._is_command_allowed("python -c \"print('hello')\"") is True

    def test_unrestricted_wildcard_allows_anything(self, unrestricted_tool):
        assert unrestricted_tool._is_command_allowed("echo hello") is True
        assert unrestricted_tool._is_command_allowed("ls -la") is True
        assert unrestricted_tool._is_command_allowed("python -c 'print(1)'") is True

    def test_empty_patterns_denies_all(self, empty_tool):
        assert empty_tool._is_command_allowed("python anything.py") is False
        assert empty_tool._is_command_allowed("echo hello") is False

    def test_no_bypass_via_trailing_matching_arg(self, python_tool):
        # ``python:scripts/*.py`` must NOT allow a command that smuggles in a
        # disallowed ``-c "..."`` payload as long as some later argument matches
        # the glob. Only the first positional after the executable counts.
        assert python_tool._is_command_allowed("python -c \"import os; os.system('echo pwn')\" scripts/ok.py") is False

    def test_no_bypass_via_trailing_matching_arg_with_options(self, python_tool):
        # Even when a benign-looking matching path appears after flags, the
        # first positional is still ``-m``, so the command must be rejected.
        assert python_tool._is_command_allowed("python -m http.server scripts/ok.py") is False

    def test_first_arg_match_is_still_allowed(self, python_tool):
        # Sanity check: the legitimate use case (``python scripts/ok.py``)
        # continues to pass after the bypass fix.
        assert python_tool._is_command_allowed("python scripts/ok.py") is True


class TestBashToolExecution:
    def test_execute_allowed_command(self, python_tool):
        result = python_tool.execute_command("python scripts/analyze.py")
        assert result.success == 1
        assert "Analysis complete" in result.result

    def test_execute_command_with_args(self, python_tool):
        result = python_tool.execute_command("python scripts/analyze.py --input test.json")
        assert result.success == 1
        assert "--input" in result.result or "test.json" in result.result

    def test_execute_denied_command(self, python_tool):
        result = python_tool.execute_command("rm -rf /")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_execute_empty_command(self, python_tool):
        result = python_tool.execute_command("")
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_execute_whitespace_only(self, python_tool):
        result = python_tool.execute_command("   ")
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_execute_returns_json_output(self, python_tool):
        result = python_tool.execute_command("python scripts/process.py")
        assert result.success == 1
        assert "processed" in result.result

    def test_execute_failing_command(self, python_tool):
        # Script doesn't exist — Python exits non-zero.
        result = python_tool.execute_command("python scripts/nonexistent.py")
        assert result.success == 0
        assert result.error is not None

    def test_empty_patterns_blocks_execution(self, empty_tool):
        result = empty_tool.execute_command("python anything.py")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_stdin_read_gets_eof_not_hang(self, multi_pattern_tool):
        """A command reading stdin must receive immediate EOF, never block.

        stdin is redirected to DEVNULL; without it the child inherits the
        agent's terminal stdin and hangs until the tool timeout, freezing the
        whole process. Reading stdin here should return an empty string fast.
        """
        result = multi_pattern_tool.execute_command('python -c "import sys; print(len(sys.stdin.read()))"')
        assert result.success == 1
        assert result.result.strip() == "0"


class TestBashToolWorkspaceIsolation:
    def test_workspace_root_resolved(self, python_tool, temp_workspace):
        assert python_tool.workspace_root == Path(temp_workspace).resolve()

    def test_commands_run_in_workspace(self, multi_pattern_tool, temp_workspace):
        (temp_workspace / "scripts" / "pwd_test.py").write_text("import os\nprint(os.getcwd())\n")
        result = multi_pattern_tool.execute_command("python scripts/pwd_test.py")
        assert result.success == 1
        assert str(temp_workspace) in result.result or temp_workspace.name in result.result


class TestBashToolExtraEnv:
    def test_extra_env_injected_into_subprocess(self, temp_workspace):
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["python:*"],
            extra_env={"MY_TOOL_NAME": "demo", "MY_TOOL_DIR": str(temp_workspace)},
        )

        (temp_workspace / "scripts" / "env_test.py").write_text(
            "import os\n"
            "print(f\"NAME={os.environ.get('MY_TOOL_NAME', 'NOT_SET')}\")\n"
            "print(f\"DIR={os.environ.get('MY_TOOL_DIR', 'NOT_SET')}\")\n"
        )

        result = tool.execute_command("python scripts/env_test.py")
        assert result.success == 1
        assert "NAME=demo" in result.result
        assert f"DIR={temp_workspace}" in result.result

    def test_no_extra_env_does_not_leak_skill_keys(self, temp_workspace):
        """A bare BashTool must not pre-populate SKILL_NAME/SKILL_DIR.

        Skill-only env vars must be opt-in via ``extra_env`` so generic
        callers don't surface skill semantics.
        """
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["python:*"])
        (temp_workspace / "scripts" / "env_test.py").write_text(
            "import os\nprint(f\"SKILL_NAME={os.environ.get('SKILL_NAME', 'MISSING')}\")\n"
        )
        result = tool.execute_command("python scripts/env_test.py")
        assert result.success == 1
        assert "SKILL_NAME=MISSING" in result.result


class TestBashToolEdgeCases:
    def test_quoted_command(self, wildcard_tool):
        result = wildcard_tool.execute_command("python -c \"print('hello world')\"")
        assert result.success == 1
        assert "hello world" in result.result

    def test_arithmetic_command(self, wildcard_tool):
        result = wildcard_tool.execute_command('python -c "print(1+2)"')
        assert result.success == 1
        assert "3" in result.result

    def test_invalid_shlex_syntax_returns_error(self, wildcard_tool):
        # Unclosed quote: ``execute_command`` calls ``shlex.split`` and reports
        # the syntax error rather than crashing.
        result = wildcard_tool.execute_command('python -c "unclosed')
        assert result.success == 0
        assert "syntax" in result.error.lower() or "not allowed" in result.error.lower()


class TestBashToolTimeout:
    def test_command_timeout(self, temp_workspace):
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["python:*"],
            timeout=1,
        )
        (temp_workspace / "scripts" / "sleep_test.py").write_text("import time\ntime.sleep(10)\nprint('Done')\n")

        result = tool.execute_command("python scripts/sleep_test.py")
        assert result.success == 0
        assert "timed out" in result.error.lower()


class TestBashToolOutputLimit:
    def test_large_output_truncated(self, temp_workspace, monkeypatch):
        from datus.tools.func_tool import bash_tool as bash_tool_module

        # Shrink the cap so the test stays fast.
        monkeypatch.setattr(bash_tool_module, "MAX_OUTPUT_SIZE", 50)

        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["python:*"])
        result = tool.execute_command("python -c \"print('X' * 500)\"")
        assert result.success == 1
        assert "truncated" in result.result
        # Truncation marker tells us the source was longer than the cap.
        assert "total" in result.result
