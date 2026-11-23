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
