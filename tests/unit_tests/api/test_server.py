# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/api/server.py — server utility functions.

CI-level: zero external dependencies. No network, no process spawning.
"""

import argparse
import os

from datus.api.server import (
    _build_agent_args,
    _ensure_parent_dir,
    _is_process_running,
    _read_pid,
    _remove_pid_file,
    _status,
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
