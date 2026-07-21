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
from unittest.mock import MagicMock

import pytest

from datus.tools.func_tool import bash_sandbox
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

    def test_stdin_read_gets_eof_not_hang(self, unrestricted_tool):
        """A command reading stdin must receive immediate EOF, never block.

        stdin is redirected to DEVNULL; without it the child inherits the
        agent's terminal stdin and hangs until the tool timeout, freezing the
        whole process. Reading stdin here should return an empty string fast.
        Uses the unrestricted tool: a restrictive whitelist rejects the
        quoted ``;`` in the inline code, and stdin handling is
        whitelist-independent.
        """
        result = unrestricted_tool.bash('python -c "import sys; print(len(sys.stdin.read()))"')
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


class TestRestrictedWhitelistHardening:
    """A restrictive whitelist is a hard security boundary.

    Commands run through a real shell (``bash -c``), so chaining, command
    substitution and redirection must be rejected even in unspaced or quoted
    form — per-token inspection cannot be made safe there. Regression tests
    for the bypass where metacharacters embedded inside a token (``x;rm``)
    slipped past the token-equality check. The ``["datus*"]`` whitelist is
    what the API surface hands to web clients (plugin CLIs).
    """

    @pytest.fixture
    def datus_tool(self, temp_workspace):
        return BashTool(workspace_root=str(temp_workspace), allowed_patterns=["datus*"])

    @pytest.mark.parametrize(
        "command",
        [
            "datus plugin list",
            "datus hello greet --name x",
            "datus-api --help",
            "datus a | datus b",
        ],
    )
    def test_datus_prefixed_commands_allowed(self, datus_tool, command):
        assert datus_tool._is_command_allowed(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "echo hi",
            'bash -c "datus x"',
            "datus a | grep x",
        ],
    )
    def test_non_datus_commands_denied(self, datus_tool, command):
        assert datus_tool._is_command_allowed(command) is False

    @pytest.mark.parametrize(
        "command",
        [
            # Spaced operator forms (already caught by the token check).
            "datus x && rm y",
            "datus x ; rm y",
            # Unspaced forms — the historical bypass this hardening closes.
            "datus x&&rm y",
            "datus x;rm -rf ~",
            "datus x>f",
            "datus x<f",
            "datus $(rm y)",
            "datus `rm y`",
            "datus ${HOME}",
            "datus x\nrm y",
            # Quoted metacharacters are rejected too: the raw-string check is
            # deliberately quoting-blind, mirroring the permission safety
            # ceiling's conservative semantics.
            'datus query "a;b"',
        ],
    )
    def test_shell_metacharacters_rejected_in_any_form(self, datus_tool, command):
        assert datus_tool._is_command_allowed(command) is False

    def test_denied_execution_never_spawns_and_reports_patterns(self, datus_tool):
        result = datus_tool.bash("datus x;rm -rf ~")
        assert result.success == 0
        assert "not allowed" in result.error.lower()
        assert "datus*" in result.error

    def test_wildcard_tool_unaffected_by_hardening(self, unrestricted_tool):
        # ``["*"]`` short-circuits before the metachar check: compound
        # commands are gated by the permission hooks, not the execution layer.
        assert unrestricted_tool._is_command_allowed('echo "a;b" && ls') is True

    def test_restricted_description_lists_patterns(self, datus_tool):
        tool = datus_tool.available_tools()[0]
        assert "Restricted mode" in tool.description
        assert "datus*" in tool.description

    def test_unrestricted_description_has_no_restriction_note(self, unrestricted_tool):
        tool = unrestricted_tool.available_tools()[0]
        assert "Restricted mode" not in tool.description


class TestBashToolSandbox:
    """Wiring between BashTool and the OS sandbox (no real sandbox spawned)."""

    @pytest.fixture
    def enabled_settings(self):
        return bash_sandbox.SandboxSettings(enabled=True)

    def test_fail_closed_when_mechanism_unavailable(self, temp_workspace, enabled_settings, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "is_available", lambda: False)
        popen_spy = MagicMock()
        monkeypatch.setattr("datus.tools.func_tool.bash_tool.subprocess.Popen", popen_spy)
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            sandbox_settings=enabled_settings,
        )
        result = tool.bash("echo hi")
        assert result.success == 0
        assert "NOT executed" in result.error
        popen_spy.assert_not_called()

    def test_wraps_argv_when_enabled(self, temp_workspace, enabled_settings, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "is_available", lambda: True)
        captured = {}

        def fake_wrap(argv, policy):
            captured["argv"] = argv
            captured["policy"] = policy
            return argv

        monkeypatch.setattr(bash_sandbox, "wrap_argv", fake_wrap)
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            sandbox_settings=enabled_settings,
            sandbox_read_dirs=[str(temp_workspace / "scripts")],
        )
        result = tool.bash("echo sandboxed")
        assert result.success == 1
        assert "sandboxed" in result.result
        assert captured["argv"][0].endswith("bash")
        assert str(temp_workspace.resolve()) in captured["policy"].writable_roots
        assert str((temp_workspace / "scripts").resolve()) in captured["policy"].readable_roots

    def test_no_settings_never_touches_sandbox(self, temp_workspace, monkeypatch):
        wrap_spy = MagicMock(side_effect=AssertionError("wrap_argv must not be called"))
        availability_spy = MagicMock(side_effect=AssertionError("is_available must not be called"))
        monkeypatch.setattr(bash_sandbox, "wrap_argv", wrap_spy)
        monkeypatch.setattr(bash_sandbox, "is_available", availability_spy)
        tool = BashTool(workspace_root=str(temp_workspace), allowed_patterns=["*"])
        result = tool.bash("echo plain")
        assert result.success == 1
        assert "plain" in result.result

    def test_disabled_settings_skip_wrapping(self, temp_workspace, monkeypatch):
        wrap_spy = MagicMock(side_effect=AssertionError("wrap_argv must not be called"))
        monkeypatch.setattr(bash_sandbox, "wrap_argv", wrap_spy)
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            sandbox_settings=bash_sandbox.SandboxSettings(enabled=False),
        )
        result = tool.bash("echo off")
        assert result.success == 1
        assert "off" in result.result

    def test_runtime_toggle_takes_effect_next_call(self, temp_workspace, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "is_available", lambda: True)
        calls = []

        def fake_wrap(argv, policy):
            calls.append(argv)
            return argv

        monkeypatch.setattr(bash_sandbox, "wrap_argv", fake_wrap)
        shared = bash_sandbox.SandboxSettings(enabled=False)
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            sandbox_settings=shared,
        )
        assert tool.bash("echo one").success == 1
        assert calls == []
        shared.enabled = True  # what /sandbox on does to the shared object
        assert tool.bash("echo two").success == 1
        assert len(calls) == 1
        shared.enabled = False
        assert tool.bash("echo three").success == 1
        assert len(calls) == 1

    def test_wrap_failure_returns_error_without_execution(self, temp_workspace, enabled_settings, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "is_available", lambda: True)

        def raise_unavailable(argv, policy):
            raise bash_sandbox.SandboxUnavailableError("gone mid-flight")

        monkeypatch.setattr(bash_sandbox, "wrap_argv", raise_unavailable)
        popen_spy = MagicMock()
        monkeypatch.setattr("datus.tools.func_tool.bash_tool.subprocess.Popen", popen_spy)
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            sandbox_settings=enabled_settings,
        )
        result = tool.bash("echo hi")
        assert result.success == 0
        assert "NOT executed" in result.error
        assert "gone mid-flight" in result.error
        popen_spy.assert_not_called()


class TestBashToolStrictEnv:
    """Strict mode minimizes the child environment (no real sandbox spawned:
    wrap_argv is stubbed to identity so only the env contract is exercised)."""

    @pytest.fixture(autouse=True)
    def sandbox_pass_through(self, monkeypatch):
        monkeypatch.setattr(bash_sandbox, "is_available", lambda: True)
        monkeypatch.setattr(bash_sandbox, "wrap_argv", lambda argv, policy: argv)

    def _tool(self, workspace, mode, extra_env=None):
        return BashTool(
            workspace_root=str(workspace),
            allowed_patterns=["*"],
            extra_env=extra_env,
            sandbox_settings=bash_sandbox.SandboxSettings(enabled=True, mode=mode),
        )

    def test_strict_hides_process_secrets(self, temp_workspace, monkeypatch):
        monkeypatch.setenv("DATUS_TEST_SECRET", "sk-leak-me")
        tool = self._tool(temp_workspace, bash_sandbox.MODE_STRICT)
        result = tool.bash('echo "[${DATUS_TEST_SECRET:-absent}]"')
        assert result.success == 1
        assert "[absent]" in result.result
        assert "sk-leak-me" not in result.result

    def test_normal_mode_keeps_process_env(self, temp_workspace, monkeypatch):
        monkeypatch.setenv("DATUS_TEST_SECRET", "sk-visible")
        tool = self._tool(temp_workspace, bash_sandbox.MODE_NORMAL)
        result = tool.bash('echo "[${DATUS_TEST_SECRET:-absent}]"')
        assert result.success == 1
        assert "[sk-visible]" in result.result

    def test_strict_keeps_baseline_vars(self, temp_workspace):
        tool = self._tool(temp_workspace, bash_sandbox.MODE_STRICT)
        result = tool.bash('echo "path=${PATH:+set} home=${HOME:+set}"')
        assert result.success == 1
        assert "path=set" in result.result
        assert "home=set" in result.result

    def test_strict_still_applies_extra_env(self, temp_workspace):
        tool = self._tool(temp_workspace, bash_sandbox.MODE_STRICT, extra_env={"SKILL_NAME": "demo"})
        result = tool.bash('echo "skill=${SKILL_NAME:-absent}"')
        assert result.success == 1
        assert "skill=demo" in result.result

    def test_sandbox_off_ignores_strict_mode_for_env(self, temp_workspace, monkeypatch):
        # mode=strict with enabled=False must not change behavior — the env
        # contract is tied to the sandbox being active.
        monkeypatch.setenv("DATUS_TEST_SECRET", "sk-still-here")
        tool = BashTool(
            workspace_root=str(temp_workspace),
            allowed_patterns=["*"],
            sandbox_settings=bash_sandbox.SandboxSettings(enabled=False, mode=bash_sandbox.MODE_STRICT),
        )
        result = tool.bash('echo "[${DATUS_TEST_SECRET:-absent}]"')
        assert result.success == 1
        assert "[sk-still-here]" in result.result
