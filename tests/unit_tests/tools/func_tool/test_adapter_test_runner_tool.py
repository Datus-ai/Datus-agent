# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/func_tool/adapter_test_runner_tool.py."""

import os
import subprocess
from unittest.mock import patch

import pytest

from datus.tools.func_tool.adapter_scaffold_tool import AdapterScaffoldTool
from datus.tools.func_tool.adapter_test_runner_tool import (
    PYTEST_TIMEOUT_SECONDS,
    AdapterTestRunnerTool,
    _truncate_tail,
    _validate_inputs,
)
from datus.tools.func_tool.base import FuncToolResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return AdapterTestRunnerTool(agent_config=None)


@pytest.fixture
def scaffolded_project(tmp_path):
    """Scaffold a semantic adapter so we have a valid project on disk."""
    output_dir = str(tmp_path / "project")
    scaffold = AdapterScaffoldTool(agent_config=None)
    result = scaffold.scaffold_adapter("semantic", "dummy", output_dir)
    assert result.success == 1
    return output_dir


# ---------------------------------------------------------------------------
# _truncate_tail
# ---------------------------------------------------------------------------


class TestTruncateTail:
    def test_short_text_unchanged(self):
        assert _truncate_tail("hello", max_chars=100) == "hello"

    def test_none_becomes_empty(self):
        assert _truncate_tail(None) == ""

    def test_long_text_keeps_tail(self):
        text = "A" * 100 + "TAIL"
        result = _truncate_tail(text, max_chars=10)
        assert result.endswith("TAIL")
        assert "truncated from head" in result

    def test_exact_boundary(self):
        text = "X" * 50
        assert _truncate_tail(text, max_chars=50) == text


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------


class TestValidateInputs:
    def test_accepts_valid_scaffolded_project(self, scaffolded_project):
        assert _validate_inputs(scaffolded_project, "tests/unit") == ""

    def test_rejects_empty_project_dir(self):
        err = _validate_inputs("", "tests/unit")
        assert "non-empty" in err

    def test_rejects_non_string_project_dir(self):
        err = _validate_inputs(None, "tests/unit")  # type: ignore[arg-type]
        assert "non-empty" in err

    def test_rejects_relative_project_dir(self):
        err = _validate_inputs("relative/path", "tests/unit")
        assert "absolute path" in err

    def test_rejects_nonexistent_project_dir(self, tmp_path):
        fake = str(tmp_path / "does_not_exist")
        err = _validate_inputs(fake, "tests/unit")
        assert "does not exist" in err

    def test_rejects_dir_without_pyproject(self, tmp_path):
        bare = str(tmp_path / "bare")
        os.makedirs(bare)
        err = _validate_inputs(bare, "tests/unit")
        assert "pyproject.toml" in err

    def test_rejects_empty_test_subpath(self, scaffolded_project):
        err = _validate_inputs(scaffolded_project, "")
        assert "non-empty" in err

    def test_rejects_absolute_test_subpath(self, scaffolded_project):
        err = _validate_inputs(scaffolded_project, "/etc/passwd")
        assert "relative path" in err

    def test_rejects_traversal_in_test_subpath(self, scaffolded_project):
        err = _validate_inputs(scaffolded_project, "tests/../../../etc")
        assert ".." in err

    def test_rejects_test_subpath_not_starting_with_tests(self, scaffolded_project):
        err = _validate_inputs(scaffolded_project, "src/adapter.py")
        assert "tests/" in err

    def test_rejects_test_subpath_resolving_outside_project(self, scaffolded_project, tmp_path):
        # Create a sibling tests dir and symlink it in
        other = tmp_path / "other_tests"
        other.mkdir()
        link = os.path.join(scaffolded_project, "tests", "link_out")
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.symlink(str(other), link)
        except (OSError, NotImplementedError):
            pytest.skip("symlink not supported")
        err = _validate_inputs(scaffolded_project, "tests/link_out/x.py")
        assert err  # any error message — either "outside" or "does not exist"

    def test_rejects_nonexistent_test_path(self, scaffolded_project):
        err = _validate_inputs(scaffolded_project, "tests/unit/does_not_exist.py")
        assert "does not exist" in err


# ---------------------------------------------------------------------------
# AdapterTestRunnerTool — argument rejection returns success=0
# ---------------------------------------------------------------------------


class TestRunAdapterPytestRejectsInvalidInputs:
    def test_returns_failure_for_invalid_path(self, runner):
        result = runner.run_adapter_pytest("relative/path", "tests/unit")
        assert isinstance(result, FuncToolResult)
        assert result.success == 0
        assert result.error is not None
        assert "absolute" in result.error

    def test_returns_failure_for_missing_pyproject(self, runner, tmp_path):
        bare = str(tmp_path / "bare")
        os.makedirs(bare)
        result = runner.run_adapter_pytest(bare, "tests/unit")
        assert result.success == 0
        assert "pyproject.toml" in result.error

    def test_returns_failure_for_traversal(self, runner, scaffolded_project):
        result = runner.run_adapter_pytest(scaffolded_project, "tests/../../../etc")
        assert result.success == 0
        assert ".." in result.error


# ---------------------------------------------------------------------------
# AdapterTestRunnerTool — mocked subprocess invocation
# ---------------------------------------------------------------------------


class TestRunAdapterPytestSubprocessBehavior:
    def test_returns_result_on_success(self, runner, scaffolded_project):
        fake_completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="8 passed in 0.12s",
            stderr="",
        )
        with patch("subprocess.run", return_value=fake_completed) as mock_run:
            result = runner.run_adapter_pytest(scaffolded_project, "tests/unit")

        assert result.success == 1
        payload = result.result
        assert payload["exit_code"] == 0
        assert payload["passed"] is True
        assert "passed" in payload["stdout"]
        assert payload["timed_out"] is False
        # Verify the command shape
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[1:4] == ["-m", "pytest", "tests/unit"]
        assert "-q" in cmd
        assert "--tb=short" in cmd
        assert "--no-header" in cmd
        # cwd must be the project dir, timeout must be the hard limit
        assert call_args[1]["cwd"] == scaffolded_project
        assert call_args[1]["timeout"] == PYTEST_TIMEOUT_SECONDS
        # PYTHONPATH must include project_dir
        env = call_args[1]["env"]
        assert scaffolded_project in env["PYTHONPATH"]

    def test_returns_failure_result_on_nonzero_exit(self, runner, scaffolded_project):
        fake_completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="1 failed, 7 passed",
            stderr="AssertionError",
        )
        with patch("subprocess.run", return_value=fake_completed):
            result = runner.run_adapter_pytest(scaffolded_project, "tests/unit")

        assert result.success == 1  # the tool itself ran OK
        payload = result.result
        assert payload["exit_code"] == 1
        assert payload["passed"] is False
        assert "failed" in payload["stdout"]
        assert "AssertionError" in payload["stderr"]

    def test_timeout_is_reported(self, runner, scaffolded_project):
        timeout_exc = subprocess.TimeoutExpired(
            cmd="pytest",
            timeout=PYTEST_TIMEOUT_SECONDS,
            output="pending stdout",
            stderr="pending stderr",
        )
        with patch("subprocess.run", side_effect=timeout_exc):
            result = runner.run_adapter_pytest(scaffolded_project, "tests/unit")

        assert result.success == 1
        payload = result.result
        assert payload["timed_out"] is True
        assert payload["passed"] is False
        assert payload["exit_code"] == -1

    def test_timeout_with_bytes_output_is_decoded(self, runner, scaffolded_project):
        """subprocess.TimeoutExpired.output can be bytes or str depending on setup."""
        timeout_exc = subprocess.TimeoutExpired(
            cmd="pytest",
            timeout=PYTEST_TIMEOUT_SECONDS,
            output=b"byte stdout",
            stderr=b"byte stderr",
        )
        with patch("subprocess.run", side_effect=timeout_exc):
            result = runner.run_adapter_pytest(scaffolded_project, "tests/unit")

        payload = result.result
        assert "byte stdout" in payload["stdout"]
        assert "byte stderr" in payload["stderr"]

    def test_pytest_not_found_returns_failure(self, runner, scaffolded_project):
        with patch("subprocess.run", side_effect=FileNotFoundError("uv not found")):
            result = runner.run_adapter_pytest(scaffolded_project, "tests/unit")

        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_long_output_is_truncated(self, runner, scaffolded_project):
        huge = "A" * 50_000
        fake_completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=huge + "TAIL_MARKER",
            stderr="",
        )
        with patch("subprocess.run", return_value=fake_completed):
            result = runner.run_adapter_pytest(scaffolded_project, "tests/unit")

        payload = result.result
        assert len(payload["stdout"]) < 50_000
        assert "TAIL_MARKER" in payload["stdout"]
        assert "truncated from head" in payload["stdout"]

    def test_default_test_subpath(self, runner, scaffolded_project):
        """Default test_subpath should be 'tests/unit'."""
        fake_completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=fake_completed) as mock_run:
            runner.run_adapter_pytest(scaffolded_project)
        cmd = mock_run.call_args[0][0]
        assert "tests/unit" in cmd


# ---------------------------------------------------------------------------
# AvailableTools wiring
# ---------------------------------------------------------------------------


class TestAvailableTools:
    def test_exposes_one_tool(self, runner):
        tools = runner.available_tools()
        assert len(tools) == 1

    def test_tool_name_is_run_adapter_pytest(self, runner):
        tools = runner.available_tools()
        assert tools[0].name == "run_adapter_pytest"


# ---------------------------------------------------------------------------
# End-to-end smoke test (not mocked): actually run pytest on a scaffolded
# adapter. This is the real "does the loop close" check.
# ---------------------------------------------------------------------------


class TestEndToEndPytestRun:
    def test_pytest_runs_on_scaffolded_semantic_adapter(self, runner, scaffolded_project):
        """Actually invoke pytest on a freshly scaffolded semantic adapter.

        Expected outcome: pytest runs successfully (exit 0 or test failures from
        the stub NotImplementedError), NOT a collection error or import error.
        The tool should return a populated result, not success=0.

        Contract tests will fail because the stub adapter raises
        NotImplementedError — that is the expected failure mode and indicates
        the scaffold pipeline is wired up correctly end-to-end.
        """
        result = runner.run_adapter_pytest(scaffolded_project, "tests/unit/test_contract.py")
        assert result.success == 1, f"Tool should execute, got error: {result.error}"
        payload = result.result
        # We don't assert passed=True (stubs raise NotImplementedError). The
        # important things: pytest actually ran, and produced output.
        assert isinstance(payload["exit_code"], int)
        assert payload["timed_out"] is False
        assert isinstance(payload["stdout"], str)
        assert len(payload["stdout"]) > 0
        # If pytest collected the contract class, the output mentions "contract"
        # somewhere — either as a collection line or a failure header.
        combined = payload["stdout"] + payload["stderr"]
        assert "Contract" in combined or "contract" in combined, (
            f"Expected pytest output to reference the contract suite. Got:\n"
            f"stdout: {payload['stdout']}\nstderr: {payload['stderr']}"
        )
