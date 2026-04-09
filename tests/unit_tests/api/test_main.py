"""Unit tests for datus.api.main — merged FastAPI server entry point.

CI-level: zero external dependencies. No network, no real process spawning.
"""

import argparse
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from datus.api.main import (
    _build_agent_args,
    _build_parser,
    _default_paths,
    _ensure_parent_dir,
    _is_process_running,
    _read_pid,
    _redirect_stdio,
    _remove_pid_file,
    _status,
    _stop,
    _write_pid_file,
)

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


class TestArgumentParser:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATUS_NAMESPACE", None)
            os.environ.pop("DATUS_OUTPUT_DIR", None)
            os.environ.pop("DATUS_LOG_LEVEL", None)
            args = _build_parser().parse_args([])
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.workers == 1
        assert args.reload is False
        assert args.debug is False
        assert args.config is None
        assert args.namespace == "default"
        assert args.output_dir == "./output"
        assert args.log_level == "INFO"
        assert args.max_steps == 20
        assert args.workflow == "fixed"
        assert args.load_cp is None
        assert args.source is None
        assert args.interactive is True
        assert args.action == "start"
        assert args.daemon is False

    def test_custom_host_port_workers(self):
        args = _build_parser().parse_args(["--host", "127.0.0.1", "--port", "9090", "--workers", "4"])
        assert (args.host, args.port, args.workers) == ("127.0.0.1", 9090, 4)

    def test_reload_and_debug_flags(self):
        args = _build_parser().parse_args(["--reload", "--debug"])
        assert args.reload is True
        assert args.debug is True

    def test_source_and_no_interactive(self):
        args = _build_parser().parse_args(["--source", "vscode", "--no-interactive"])
        assert args.source == "vscode"
        assert args.interactive is False

    def test_daemon_action_choices(self):
        args = _build_parser().parse_args(["--daemon", "--action", "restart"])
        assert args.daemon is True
        assert args.action == "restart"

    def test_invalid_log_level_exits(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--log-level", "TRACE"])

    def test_invalid_source_exits(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--source", "emacs"])

    def test_namespace_from_env(self):
        with patch.dict(os.environ, {"DATUS_NAMESPACE": "staging"}):
            args = _build_parser().parse_args([])
        assert args.namespace == "staging"

    def test_output_dir_from_env(self):
        with patch.dict(os.environ, {"DATUS_OUTPUT_DIR": "/custom/out"}):
            args = _build_parser().parse_args([])
        assert args.output_dir == "/custom/out"


# ---------------------------------------------------------------------------
# pid / path helpers
# ---------------------------------------------------------------------------


class TestEnsureParentDir:
    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "file.txt"
        _ensure_parent_dir(nested)
        assert nested.parent.exists()

    def test_idempotent_when_already_exists(self, tmp_path):
        target = tmp_path / "file.txt"
        _ensure_parent_dir(target)
        _ensure_parent_dir(target)


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
        _remove_pid_file(tmp_path / "nonexistent.pid")


class TestIsProcessRunning:
    def test_running_process(self):
        assert _is_process_running(os.getpid()) is True

    def test_non_running_process(self):
        assert _is_process_running(999999999) is False


class TestStatus:
    def test_stopped_when_no_pid_file(self, tmp_path, capsys):
        assert _status(tmp_path / "missing.pid") == 1
        assert "stopped" in capsys.readouterr().out

    def test_stopped_when_process_not_running(self, tmp_path, capsys):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("999999999")
        assert _status(pid_file) == 1
        assert "stopped" in capsys.readouterr().out

    def test_running_when_process_alive(self, tmp_path, capsys):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text(str(os.getpid()))
        assert _status(pid_file) == 0
        assert "running" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_no_pid_file_returns_zero(self, tmp_path):
        assert _stop(tmp_path / "missing.pid", timeout_seconds=0.1) == 0

    def test_stop_stale_pid_removes_file(self, tmp_path):
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("999999999")
        assert _stop(pid_file, timeout_seconds=0.1) == 0
        assert not pid_file.exists()

    def test_stop_running_process_sends_sigterm(self, tmp_path):
        pid_file = tmp_path / "running.pid"
        pid_file.write_text(str(os.getpid()))

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0:
                return
            raise ProcessLookupError("fake stopped")

        with (
            patch("datus.api.main.os.kill", side_effect=fake_kill),
            patch("datus.api.main._is_process_running", side_effect=[True, True, False]),
        ):
            result = _stop(pid_file, timeout_seconds=2.0)

        assert result == 0
        assert any(sig == signal.SIGTERM for _, sig in kill_calls)

    def test_stop_removes_pid_when_process_lookup_error(self, tmp_path):
        pid_file = tmp_path / "proc.pid"
        pid_file.write_text(str(os.getpid()))

        def fake_kill(pid, sig):
            if sig == 0:
                return
            raise ProcessLookupError("not found")

        with patch("datus.api.main.os.kill", side_effect=fake_kill):
            result = _stop(pid_file, timeout_seconds=0.1)

        assert result == 0
        assert not pid_file.exists()


# ---------------------------------------------------------------------------
# _redirect_stdio / _default_paths
# ---------------------------------------------------------------------------


class TestRedirectStdio:
    def test_redirect_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs" / "subdir"
        log_file = log_dir / "test.log"

        with patch("datus.api.main.os.dup2") as mock_dup2:
            _redirect_stdio(log_file)

        assert log_dir.exists()
        assert log_file.exists()
        assert mock_dup2.call_count == 3


class TestDefaultPaths:
    def test_returns_pid_and_log_paths(self):
        pid_file, log_file = _default_paths()
        assert isinstance(pid_file, Path)
        assert isinstance(log_file, Path)


# ---------------------------------------------------------------------------
# _build_agent_args
# ---------------------------------------------------------------------------


class TestBuildAgentArgs:
    def _make(self, **overrides):
        base = dict(
            namespace="ns",
            config="/etc/agent.yml",
            max_steps=20,
            workflow="fixed",
            load_cp=None,
            debug=False,
            source=None,
            interactive=True,
            output_dir="./output",
            log_level="INFO",
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_maps_all_fields(self):
        agent_args = _build_agent_args(
            self._make(
                namespace="myns",
                max_steps=15,
                load_cp="cp.pkl",
                debug=True,
                source="vscode",
                interactive=False,
                output_dir="/tmp/out",
                log_level="DEBUG",
            )
        )
        assert agent_args.namespace == "myns"
        assert agent_args.max_steps == 15
        assert agent_args.load_cp == "cp.pkl"
        assert agent_args.debug is True
        assert agent_args.source == "vscode"
        assert agent_args.interactive is False
        assert agent_args.output_dir == "/tmp/out"
        assert agent_args.log_level == "DEBUG"

    def test_returns_namespace_object(self):
        assert isinstance(_build_agent_args(self._make()), argparse.Namespace)


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_status_action_exits(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        log_file = tmp_path / "test.log"

        with (
            patch("datus.api.main._default_paths", return_value=(pid_file, log_file)),
            patch("datus.api.main.configure_logging"),
            patch.object(sys, "argv", ["datus-api", "--action", "status"]),
        ):
            from datus.api.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code in (0, 1)

    def test_stop_action_exits_zero(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        log_file = tmp_path / "test.log"

        with (
            patch("datus.api.main._default_paths", return_value=(pid_file, log_file)),
            patch("datus.api.main.configure_logging"),
            patch.object(sys, "argv", ["datus-api", "--action", "stop"]),
        ):
            from datus.api.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_config_not_found_exits(self, tmp_path):
        from datus.api.main import main

        with (
            patch("datus.api.main._default_paths", return_value=(tmp_path / "p.pid", tmp_path / "p.log")),
            patch("datus.api.main.configure_logging"),
            patch("datus.api.main.parse_config_path", side_effect=FileNotFoundError("no such file")),
            patch.object(sys, "argv", ["datus-api", "--config", "/nonexistent/agent.yml"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_sets_env_vars_and_runs_server(self, tmp_path):
        from datus.api.main import main

        captured = {}

        def fake_run_server(args, agent_args):
            captured["host"] = args.host
            captured["namespace"] = args.namespace
            captured["source"] = args.source
            captured["interactive"] = args.interactive

        with (
            patch("datus.api.main._default_paths", return_value=(tmp_path / "p.pid", tmp_path / "p.log")),
            patch("datus.api.main.configure_logging"),
            patch("datus.api.main.parse_config_path", return_value="/tmp/agent.yml"),
            patch("datus.api.main._run_server", side_effect=fake_run_server),
            patch.object(
                sys,
                "argv",
                [
                    "datus-api",
                    "--namespace",
                    "test_main_ns",
                    "--source",
                    "web",
                    "--no-interactive",
                ],
            ),
        ):
            main()

        assert os.environ.get("DATUS_NAMESPACE") == "test_main_ns"
        assert os.environ.get("DATUS_CONFIG") == "/tmp/agent.yml"
        assert captured == {
            "host": "0.0.0.0",
            "namespace": "test_main_ns",
            "source": "web",
            "interactive": False,
        }
