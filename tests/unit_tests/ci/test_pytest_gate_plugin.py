"""Failure diagnostics must survive a later interrupted pytest session."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _pytest.reports import TestReport

from ci import pytest_gate_plugin as gate


@pytest.mark.parametrize("showlocals", [False, True])
def test_failure_traceback_is_printed_before_the_next_test(tmp_path, showlocals, monkeypatch):
    # Nightly exports parent-only options, including a basetemp above this cwd.
    parent_options = f"--basetemp={shlex.quote(str(tmp_path.parent))} --no-showlocals --maxfail=1"
    monkeypatch.setenv("PYTEST_ADDOPTS", parent_options)
    source = tmp_path / "test_early_failure.py"
    source.write_text(
        'def test_fails():\n    local_value = 6 * 7\n    assert False, "EARLY_FAILURE"\n'
        'def test_after():\n    print("SECOND_TEST_STARTED", flush=True)\n'
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    report_path = tmp_path / "execution.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-c",
            str(config),
            "-p",
            "no:cacheprovider",
            "-p",
            "ci.pytest_gate_plugin",
            "--gate-report",
            str(report_path),
            "--tb=short",
            "--showlocals" if showlocals else "--no-showlocals",
            "-s",
            "-q",
            str(source),
        ],
        cwd=tmp_path,
        env={
            **{name: value for name, value in os.environ.items() if name != "PYTEST_ADDOPTS"},
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert os.environ["PYTEST_ADDOPTS"] == parent_options
    assert "SECOND_TEST_STARTED" in result.stdout
    before_next_test = result.stdout.split("SECOND_TEST_STARTED", 1)[0]
    assert "AssertionError: EARLY_FAILURE" in before_next_test, result.stdout
    assert ("local_value = 42" in before_next_test) is showlocals
    evidence = json.loads(report_path.read_text())
    assert evidence["outcomes"] == {
        "test_early_failure.py::test_fails": "failed",
        "test_early_failure.py::test_after": "passed",
    }
    assert evidence["exit_code"] == 1


@pytest.mark.parametrize("worker", [False, True])
def test_failure_accounting_survives_missing_reporter_or_worker(worker, monkeypatch):
    with monkeypatch.context() as isolated:
        # This module can itself run under the gate plugin; preserve its live state.
        for name in ("_collected", "_executed"):
            isolated.setattr(gate, name, set())
        for name in ("_outcomes", "_excluded"):
            isolated.setattr(gate, name, {})
        isolated.setattr(gate, "_config", None)
        reporter = Mock()
        config = SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda _name: reporter if worker else None))
        if worker:
            config.workerinput = {}
        gate.pytest_sessionstart(SimpleNamespace(config=config))
        report = TestReport(
            nodeid="test_example.py::test_failure",
            location=("test_example.py", 1, "test_failure"),
            keywords={},
            outcome="failed",
            longrepr="failure detail",
            when="call",
        )
        gate.pytest_runtest_logreport(report)
        assert gate._outcomes == {"test_example.py::test_failure": "failed"}
        assert gate._executed == {"test_example.py::test_failure"}
        assert reporter.mock_calls == []


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("phase", ["call", "setup", "setup_stopped", "pass"])
def test_gate_keeps_first_failure_when_real_pytest_retry_passes(tmp_path, phase, distributed):
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    source = tmp_path / "test_retry.py"
    source.write_text(
        "from pathlib import Path\nimport pytest\nattempt = 0\n"
        "@pytest.fixture\ndef ready():\n"
        "    global attempt\n"
        f'    if {phase!r} in ("setup", "setup_stopped"):\n'
        "        attempt += 1\n"
        f'        assert attempt > 1 and {phase!r} != "setup_stopped", "FIRST_SETUP_FAILURE"\n'
        "def test_retry(ready):\n"
        "    global attempt\n"
        f'    if {phase!r} in ("call", "pass"):\n'
        "        attempt += 1\n"
        f'    assert attempt > 1 or {phase!r} == "pass", "FIRST_CALL_FAILURE"\n'
        '    Path("recovered-attempt").write_text(str(attempt))\n'
    )
    report_path = tmp_path / "execution.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--noconftest",
        "-c",
        str(config),
        "-p",
        "no:cacheprovider",
        "-p",
        "pytest_rerunfailures",
        "-p",
        "ci.pytest_gate_plugin",
        "--gate-report",
        str(report_path),
        "--reruns",
        "1",
        "--reruns-delay",
        "0",
        "--tb=short",
        "--no-showlocals",
        "-q",
        str(source),
    ]
    command += ["-p", "xdist.plugin", "-n", "2"] if distributed else []
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={
            **{name: value for name, value in os.environ.items() if name != "PYTEST_ADDOPTS"},
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    expected_failure = phase != "pass"
    assert result.returncode == (1 if expected_failure else 0), result.stdout + result.stderr
    evidence = json.loads(report_path.read_text())
    assert evidence["outcomes"] == {"test_retry.py::test_retry": "failed" if expected_failure else "passed"}
    assert evidence["executed"] == ([] if phase == "setup_stopped" else ["test_retry.py::test_retry"])
    assert evidence["exit_code"] == (1 if expected_failure else 0)
    marker = tmp_path / "recovered-attempt"
    recovered_attempt = marker.read_text() if marker.exists() else None
    assert recovered_attempt == (None if phase == "setup_stopped" else "1" if phase == "pass" else "2")
    expected_diagnostics = 0 if phase == "pass" else 2 if phase == "setup_stopped" else 1
    assert result.stdout.count("Immediate failure:") == expected_diagnostics, result.stdout
    assert ("FIRST_CALL_FAILURE" in result.stdout) is (phase == "call")
    assert ("FIRST_SETUP_FAILURE" in result.stdout) is (phase in {"setup", "setup_stopped"})


@pytest.mark.parametrize(
    "enabled,initial,expected",
    [(True, pytest.ExitCode.INTERRUPTED, pytest.ExitCode.INTERRUPTED), (False, pytest.ExitCode.OK, pytest.ExitCode.OK)],
)
def test_sticky_failure_preserves_nonzero_or_ungated_exit(monkeypatch, tmp_path, enabled, initial, expected):
    with monkeypatch.context() as isolated:
        isolated.setattr(gate, "_outcomes", {"test_example.py::test_failure": "failed"})
        session = SimpleNamespace(
            config=SimpleNamespace(getoption=lambda _: str(tmp_path / "gate.json") if enabled else None),
            exitstatus=initial,
        )
        gate.pytest_sessionfinish(session, initial)
        assert session.exitstatus == expected
