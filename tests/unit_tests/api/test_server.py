# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/api/server.py — server utility functions.

CI-level: zero external dependencies. No network, no process spawning.
"""

import argparse
import os
import signal
from pathlib import Path
from unittest.mock import patch

import pytest

from datus.api.server import (
    _build_agent_args,
    _ensure_parent_dir,
    _is_process_running,
    _read_pid,
    _remove_pid_file,
    _status,
    _stop,
    _write_pid_file,
)

# ---------------------------------------------------------------------------
# _ensure_parent_dir
# ---------------------------------------------------------------------------


class TestEnsureParentDir:
    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "file.txt"
        _ensure_parent_dir(nested)
        assert nested.parent.exists()

    def test_idempotent_when_already_exists(self, tmp_path):
        target = tmp_path / "file.txt"
        _ensure_parent_dir(target)
        _ensure_parent_dir(target)  # second call must not raise


# ---------------------------------------------------------------------------
# _read_pid
# ---------------------------------------------------------------------------


class TestReadPid:
    def test_returns_none_when_file_missing(self, tmp_path):
        assert _read_pid(tmp_path / "missing.pid") is None

    def test_returns_pid_from_file(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("12345")
        assert _read_pid(pid_file) == 12345

    def test_returns_none_for_empty_file(self, tmp_path):
        pid_file = tmp_path / "empty.pid"
        pid_file.write_text("")
        assert _read_pid(pid_file) is None

    def test_returns_none_on_non_int_content(self, tmp_path):
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number")
        assert _read_pid(pid_file) is None


# ---------------------------------------------------------------------------
# _write_pid_file / _remove_pid_file
# ---------------------------------------------------------------------------


class TestWriteRemovePidFile:
    def test_write_creates_file_with_pid(self, tmp_path):
        pid_file = tmp_path / "run" / "test.pid"
        _write_pid_file(pid_file, 9999)
        assert pid_file.exists()
        assert pid_file.read_text() == "9999"

    def test_remove_deletes_existing_file(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("1234")
        _remove_pid_file(pid_file)
        assert not pid_file.exists()

    def test_remove_does_not_raise_when_missing(self, tmp_path):
        pid_file = tmp_path / "nonexistent.pid"
        _remove_pid_file(pid_file)  # must not raise


# ---------------------------------------------------------------------------
# _is_process_running
# ---------------------------------------------------------------------------


class TestIsProcessRunning:
    def test_running_process(self):
        # Current process is definitely running
        assert _is_process_running(os.getpid()) is True

    def test_non_running_process(self):
        # PID 0 is the scheduler on Unix, sending signal 0 raises OSError
        # Use a large PID that is very unlikely to exist
        assert _is_process_running(999999999) is False


# ---------------------------------------------------------------------------
# _status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_stopped_when_no_pid_file(self, tmp_path, capsys):
        exit_code = _status(tmp_path / "missing.pid")
        assert exit_code == 1
        assert "stopped" in capsys.readouterr().out

    def test_stopped_when_process_not_running(self, tmp_path, capsys):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("999999999")  # very unlikely pid
        exit_code = _status(pid_file)
        assert exit_code == 1
        assert "stopped" in capsys.readouterr().out

    def test_running_when_process_alive(self, tmp_path, capsys):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text(str(os.getpid()))
        exit_code = _status(pid_file)
        assert exit_code == 0
        assert "running" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _build_agent_args
# ---------------------------------------------------------------------------


class TestBuildAgentArgs:
    def test_maps_fields_correctly(self):
        args = argparse.Namespace(
            namespace="myns",
            config="/etc/conf.yml",
            max_steps=15,
            workflow="fixed",
            load_cp=None,
            debug=True,
            host="0.0.0.0",
            port=8080,
        )
        agent_args = _build_agent_args(args)
        assert agent_args.namespace == "myns"
        assert agent_args.config == "/etc/conf.yml"
        assert agent_args.max_steps == 15
        assert agent_args.workflow == "fixed"
        assert agent_args.load_cp is None
        assert agent_args.debug is True

    def test_returns_namespace_object(self):
        args = argparse.Namespace(
            namespace="ns",
            config=None,
            max_steps=20,
            workflow="reflection",
            load_cp=None,
            debug=False,
        )
        result = _build_agent_args(args)
        assert isinstance(result, argparse.Namespace)


pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# _stop - no pid file
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_no_pid_file_returns_zero(self, tmp_path):
        result = _stop(tmp_path / "missing.pid", timeout_seconds=0.1)
        assert result == 0

    def test_stop_stale_pid_removes_file(self, tmp_path):
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("999999999")  # Very unlikely pid
        result = _stop(pid_file, timeout_seconds=0.1)
        assert result == 0
        # PID file should be removed after stopping stale process
        assert not pid_file.exists()

    def test_stop_running_process_sends_sigterm(self, tmp_path):
        pid_file = tmp_path / "running.pid"
        own_pid = os.getpid()
        pid_file.write_text(str(own_pid))

        # Intercept os.kill so we don't actually signal ourselves
        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0:
                return  # process exists
            # After SIGTERM, pretend process stopped quickly
            raise ProcessLookupError("fake stopped")

        with patch("datus.api.server.os.kill", side_effect=fake_kill), patch(
            "datus.api.server._is_process_running", side_effect=[True, True, False]
        ):
            result = _stop(pid_file, timeout_seconds=2.0)

        # SIGTERM should have been sent
        sigterm_calls = [(pid, sig) for pid, sig in kill_calls if sig == signal.SIGTERM]
        assert len(sigterm_calls) >= 1
        assert result == 0

    def test_stop_removes_pid_when_process_lookup_error(self, tmp_path):
        pid_file = tmp_path / "proc.pid"
        own_pid = os.getpid()
        pid_file.write_text(str(own_pid))

        def fake_kill(pid, sig):
            if sig == 0:
                return  # process exists check passes
            raise ProcessLookupError("not found")

        with patch("datus.api.server.os.kill", side_effect=fake_kill):
            result = _stop(pid_file, timeout_seconds=0.1)

        assert result == 0
        assert not pid_file.exists()


# ---------------------------------------------------------------------------
# _redirect_stdio
# ---------------------------------------------------------------------------


class TestRedirectStdio:
    def test_redirect_creates_log_dir(self, tmp_path):
        from datus.api.server import _redirect_stdio

        log_dir = tmp_path / "logs" / "subdir"
        log_file = log_dir / "test.log"

        # Patch dup2 and open to avoid actually redirecting stdio in test process
        with patch("datus.api.server.os.dup2") as mock_dup2:
            _redirect_stdio(log_file)

        assert log_dir.exists()
        assert log_file.exists()
        # dup2 should have been called 3 times (stdin, stdout, stderr)
        assert mock_dup2.call_count == 3


# ---------------------------------------------------------------------------
# _default_paths
# ---------------------------------------------------------------------------


class TestDefaultPaths:
    def test_returns_pid_and_log_paths(self):
        from datus.api.server import _default_paths

        pid_file, log_file = _default_paths()
        assert isinstance(pid_file, Path)
        assert isinstance(log_file, Path)
        assert str(pid_file).endswith(".pid") or "pid" in str(pid_file)
        assert str(log_file).endswith(".log") or "log" in str(log_file)


# ---------------------------------------------------------------------------
# main() argument parsing (smoke test)
# ---------------------------------------------------------------------------


class TestMainArgParsing:
    def test_status_action_calls_status(self, tmp_path):
        """main() with --action status calls _status and raises SystemExit."""

        pid_file = tmp_path / "test.pid"
        log_file = tmp_path / "test.log"

        with patch("datus.api.server._default_paths", return_value=(pid_file, log_file)), patch(
            "datus.api.server.configure_logging"
        ), patch("sys.argv", ["server.py", "--action", "status"]):
            from datus.api.server import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            # Exit code 0 (stopped) or 1 (running) - both valid
            assert exc_info.value.code in (0, 1)

    def test_stop_action_calls_stop(self, tmp_path):
        """main() with --action stop calls _stop and raises SystemExit."""
        pid_file = tmp_path / "test.pid"
        log_file = tmp_path / "test.log"

        with patch("datus.api.server._default_paths", return_value=(pid_file, log_file)), patch(
            "datus.api.server.configure_logging"
        ), patch("sys.argv", ["server.py", "--action", "stop"]):
            from datus.api.server import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# _build_agent_args additional cases
# ---------------------------------------------------------------------------


class TestBuildAgentArgsExtra:
    def test_load_cp_passed_through(self):
        args = argparse.Namespace(
            namespace="ns",
            config="/etc/agent.yml",
            max_steps=10,
            workflow="fixed",
            load_cp="checkpoint.pkl",
            debug=False,
        )
        result = _build_agent_args(args)
        assert result.load_cp == "checkpoint.pkl"

    def test_workflow_passed_through(self):
        args = argparse.Namespace(
            namespace="ns",
            config=None,
            max_steps=20,
            workflow="reflection",
            load_cp=None,
            debug=True,
        )
        result = _build_agent_args(args)
        assert result.workflow == "reflection"
        assert result.debug is True
