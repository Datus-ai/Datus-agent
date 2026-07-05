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
        assert tools[0].name == "bash"

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
        result = python_tool.bash("python scripts/analyze.py")
        assert result.success == 1
        assert "Analysis complete" in result.result

    def test_bash_with_args(self, python_tool):
        result = python_tool.bash("python scripts/analyze.py --input test.json")
        assert result.success == 1
        # analyze.py echoes sys.argv[1:] verbatim.
        assert "Args: ['--input', 'test.json']" in result.result

    def test_execute_denied_command(self, python_tool):
        result = python_tool.bash("rm -rf /")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_execute_empty_command(self, python_tool):
        result = python_tool.bash("")
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_execute_whitespace_only(self, python_tool):
        result = python_tool.bash("   ")
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_execute_returns_json_output(self, python_tool):
        result = python_tool.bash("python scripts/process.py")
        assert result.success == 1
        assert "processed" in result.result

    def test_execute_failing_command(self, python_tool):
        # Script doesn't exist — Python exits non-zero.
        result = python_tool.bash("python scripts/nonexistent.py")
        assert result.success == 0
        assert result.error.startswith("Command exited with code ")
        # Python's stderr is merged into the result and names the missing file.
        assert "nonexistent.py" in result.result

    def test_empty_patterns_blocks_execution(self, empty_tool):
        result = empty_tool.bash("python anything.py")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_stdin_read_gets_eof_not_hang(self, multi_pattern_tool):
        """A command reading stdin must receive immediate EOF, never block.

        stdin is redirected to DEVNULL; without it the child inherits the
        agent's terminal stdin and hangs until the tool timeout, freezing the
        whole process. Reading stdin here should return an empty string fast.
        """
        result = multi_pattern_tool.bash('python -c "import sys; print(len(sys.stdin.read()))"')
        assert result.success == 1
        assert result.result.strip() == "0"


class TestBashToolWorkspaceIsolation:
    def test_workspace_root_resolved(self, python_tool, temp_workspace):
        assert python_tool.workspace_root == Path(temp_workspace).resolve()

    def test_commands_run_in_workspace(self, multi_pattern_tool, temp_workspace):
        (temp_workspace / "scripts" / "pwd_test.py").write_text("import os\nprint(os.getcwd())\n")
        result = multi_pattern_tool.bash("python scripts/pwd_test.py")
        assert result.success == 1
        # cwd is locked to the resolved workspace root (symlinks resolved).
        assert result.result.strip() == str(temp_workspace.resolve())


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

        result = tool.bash("python scripts/env_test.py")
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
        result = tool.bash("python scripts/env_test.py")
        assert result.success == 1
        assert "SKILL_NAME=MISSING" in result.result


class TestBashToolEdgeCases:
    def test_quoted_command(self, wildcard_tool):
        result = wildcard_tool.bash("python -c \"print('hello world')\"")
        assert result.success == 1
        assert "hello world" in result.result

    def test_arithmetic_command(self, wildcard_tool):
        result = wildcard_tool.bash('python -c "print(1+2)"')
        assert result.success == 1
        assert "3" in result.result

    def test_invalid_shlex_syntax_returns_error(self, wildcard_tool):
        # Unclosed quote: the restrictive whitelist can't parse the command
        # (``split_pipeline`` returns None on unbalanced quotes), so it is
        # rejected before spawning rather than crashing.
        result = wildcard_tool.bash('python -c "unclosed')
        assert result.success == 0
        assert result.error.startswith("Command not allowed")


class TestBashToolTimeout:
    def test_command_timeout(self, temp_workspace):
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["python:*"],
            timeout=1,
        )
        (temp_workspace / "scripts" / "sleep_test.py").write_text("import time\ntime.sleep(10)\nprint('Done')\n")

        result = tool.bash("python scripts/sleep_test.py")
        assert result.success == 0
        assert "timed out" in result.error.lower()


class TestBashToolOutputLimit:
    def test_large_output_truncated(self, temp_workspace, monkeypatch):
        from datus.tools.func_tool import bash_tool as bash_tool_module

        # Shrink the cap so the test stays fast.
        monkeypatch.setattr(bash_tool_module, "MAX_OUTPUT_SIZE", 50)

        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["python:*"])
        result = tool.bash("python -c \"print('X' * 500)\"")
        assert result.success == 1
        assert "truncated" in result.result
        # Truncation marker tells us the source was longer than the cap.
        assert "total" in result.result


class TestBashToolOutputOffload:
    """Redirect-to-disk path: output streams to a file, decided by size afterwards."""

    @pytest.fixture
    def offload_dir(self, tmp_path):
        return tmp_path / "session_data"

    @pytest.fixture
    def offload_tool(self, temp_workspace, offload_dir):
        return BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            output_dir_provider=lambda: offload_dir,
        )

    def test_small_output_returned_inline_and_no_residual_file(self, offload_tool, offload_dir):
        result = offload_tool.bash("python -c \"print('hi')\"")
        assert result.success == 1
        assert result.result.strip() == "hi"
        # Small output is read back and the temp file deleted — nothing lingers.
        assert list(offload_dir.glob("*")) == []

    def test_empty_output_leaves_no_file(self, offload_tool, offload_dir):
        result = offload_tool.bash('python -c "pass"')
        assert result.success == 1
        assert (result.result or "") == ""
        assert list(offload_dir.glob("*")) == []

    def test_large_output_archived_to_file_with_marker(self, offload_tool, offload_dir, monkeypatch):
        from datus.tools.func_tool import bash_tool as bash_tool_module
        from datus.utils.tool_archive import build_archived_marker, parse_archived_marker

        monkeypatch.setattr(bash_tool_module, "BASH_ARCHIVE_THRESHOLD", 100)
        result = offload_tool.bash("python -c \"print('Y' * 5000)\"")
        assert result.success == 1
        # The file kept on disk holds the complete output.
        kept = list(offload_dir.glob("*_bash_*.txt"))
        assert len(kept) == 1
        assert kept[0].read_text().count("Y") == 5000
        # Model-facing result is exactly the marker (path + 1000-char preview),
        # NOT the full 5000-char output.
        expected = build_archived_marker(str(kept[0]), "Y" * bash_tool_module.BASH_ARCHIVE_PREVIEW_CHARS)
        assert result.result == expected
        assert parse_archived_marker(result.result)["path"] == str(kept[0])

    def test_large_failure_sets_error_and_marker(self, offload_tool, offload_dir, monkeypatch):
        from datus.tools.func_tool import bash_tool as bash_tool_module
        from datus.utils.tool_archive import is_archived_output

        monkeypatch.setattr(bash_tool_module, "BASH_ARCHIVE_THRESHOLD", 100)
        result = offload_tool.bash("python -c \"import sys; sys.stdout.write('Z'*5000); sys.exit(2)\"")
        assert result.success == 0
        assert "exited with code 2" in result.error
        assert is_archived_output(result.result)
        assert len(list(offload_dir.glob("*_bash_*.txt"))) == 1

    def test_no_provider_falls_back_to_in_memory(self, temp_workspace, offload_dir):
        """Without a provider the tool truncates in memory and writes no file."""
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["*"])
        result = tool.bash("python -c \"print('hello')\"")
        assert result.success == 1
        assert result.result.strip() == "hello"
        assert not offload_dir.exists() or list(offload_dir.glob("*")) == []

    def test_provider_returning_none_uses_in_memory(self, temp_workspace):
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            output_dir_provider=lambda: None,
        )
        result = tool.bash("python -c \"print('ok')\"")
        assert result.success == 1
        assert result.result.strip() == "ok"


class TestPipelineExecution:
    """Real-shell execution: pipelines, operators, pipefail, timeout, gate."""

    def test_pipeline_produces_piped_output(self, unrestricted_tool):
        result = unrestricted_tool.bash("printf 'a\\nb\\nc\\n' | grep b | wc -l")
        assert result.success == 1
        assert result.result.strip() == "1"

    def test_pipeline_stages_chain(self, unrestricted_tool):
        result = unrestricted_tool.bash("echo hello world | tr ' ' '\\n' | sort")
        assert result.success == 1
        assert result.result.split() == ["hello", "world"]

    def test_logical_and_executes_under_real_shell(self, unrestricted_tool):
        result = unrestricted_tool.bash("echo first && echo second")
        assert result.success == 1
        assert "first" in result.result and "second" in result.result

    def test_redirection_works(self, unrestricted_tool, temp_workspace):
        result = unrestricted_tool.bash("echo persisted > out.txt")
        assert result.success == 1
        assert (temp_workspace / "out.txt").read_text().strip() == "persisted"

    def test_pipeline_final_stage_failure_surfaces(self, unrestricted_tool):
        # bash default: the pipeline's exit code is the LAST stage's. grep with
        # no match exits 1 → the pipeline reports failure.
        result = unrestricted_tool.bash("echo hello | grep nomatch")
        assert result.success == 0

    def test_pipeline_exit_zero_when_final_succeeds(self, unrestricted_tool):
        # Upstream failure is masked by a succeeding final stage (bash default,
        # no pipefail) — matches Claude Code semantics.
        result = unrestricted_tool.bash("cat /nonexistent/xyz | cat")
        assert result.success == 1

    def test_pipeline_exit_zero_when_all_succeed(self, unrestricted_tool):
        result = unrestricted_tool.bash("echo ok | cat | cat")
        assert result.success == 1

    def test_quoted_pipe_is_literal(self, unrestricted_tool):
        result = unrestricted_tool.bash("echo 'a|b'")
        assert result.success == 1
        assert result.result.strip() == "a|b"

    def test_sigpipe_upstream_terminates(self, unrestricted_tool):
        # `yes` would run forever; `head -1` closes the pipe → upstream dies.
        # With a short timeout this must still return promptly, not hang.
        tool = BashTool(workspace_root=str(unrestricted_tool.workspace_root), allowed_patterns=["*"], timeout=10)
        result = tool.bash("yes | head -1")
        assert result.success == 1
        assert result.result.strip() == "y"

    def test_timeout_kills_whole_pipeline(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["*"], timeout=1)
        import time

        start = time.monotonic()
        result = tool.bash("sleep 30 | cat")
        elapsed = time.monotonic() - start
        assert result.success == 0
        assert "timed out" in (result.error or "").lower()
        # Must not wait for the full 30s sleep — process group was killed.
        assert elapsed < 10

    def test_extglob_disabled(self, unrestricted_tool):
        # With extglob off, `!(...)` is not special; bash errors on the syntax.
        result = unrestricted_tool.bash("echo !(foo)")
        assert result.success == 0

    def test_restrictive_whitelist_allows_matching_pipeline(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["echo:*", "cat:*"])
        result = tool.bash("echo hi | cat")
        assert result.success == 1
        assert result.result.strip() == "hi"

    def test_restrictive_whitelist_blocks_unmatched_segment(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["echo:*"])
        result = tool.bash("echo hi | rm -rf x")
        assert result.success == 0
        assert "not allowed" in (result.error or "").lower()

    def test_restrictive_whitelist_blocks_operator(self, temp_workspace):
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["echo:*"])
        result = tool.bash("echo hi && echo bye")
        assert result.success == 0
        assert "not allowed" in (result.error or "").lower()

    def test_wildcard_allows_operators(self, unrestricted_tool):
        result = unrestricted_tool.bash("echo a; echo b")
        assert result.success == 1


class TestBashTimeoutParam:
    """Optional per-call timeout parameter."""

    def test_per_call_timeout_overrides_default(self, unrestricted_tool):
        import time

        # Instance default is 60s; a per-call timeout of 1s must win.
        start = time.monotonic()
        result = unrestricted_tool.bash("sleep 30", timeout=1)
        elapsed = time.monotonic() - start
        assert result.success == 0
        assert "timed out" in (result.error or "").lower()
        assert elapsed < 10

    def test_timeout_clamped_to_max(self, unrestricted_tool):
        from datus.tools.func_tool.bash_tool import MAX_BASH_TIMEOUT

        assert unrestricted_tool._resolve_timeout(999999) == MAX_BASH_TIMEOUT

    def test_invalid_timeout_falls_back_to_default(self, unrestricted_tool):
        assert unrestricted_tool._resolve_timeout(None) == unrestricted_tool.timeout
        assert unrestricted_tool._resolve_timeout(0) == unrestricted_tool.timeout
        assert unrestricted_tool._resolve_timeout(-5) == unrestricted_tool.timeout
        assert unrestricted_tool._resolve_timeout(True) == unrestricted_tool.timeout

    def test_default_timeout_used_when_omitted(self, unrestricted_tool):
        # A fast command with no explicit timeout succeeds normally.
        result = unrestricted_tool.bash("echo ok")
        assert result.success == 1
        assert result.result.strip() == "ok"

    def test_timeout_exposed_in_tool_schema(self, unrestricted_tool):
        tool = unrestricted_tool.available_tools()[0]
        props = tool.params_json_schema.get("properties", {})
        assert "command" in props
        assert "timeout" in props
