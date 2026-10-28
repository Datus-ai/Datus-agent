# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Regression tests for the isolated managed-plugin test runtime."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from tests.integration.conftest import ManagedPluginRuntime


def _runtime(tmp_path: Path) -> ManagedPluginRuntime:
    home = tmp_path / "home"
    home.mkdir()
    return ManagedPluginRuntime(home=home, datus_executable=Path(sys.executable))


def test_managed_plugin_runtime_defaults_to_isolated_home(tmp_path):
    runtime = _runtime(tmp_path)

    completed = runtime.run(
        "-c",
        "import json, os; from pathlib import Path; "
        "print(json.dumps({'cwd': str(Path.cwd()), 'home': os.environ['HOME']}))",
    )

    assert json.loads(completed.stdout) == {
        "cwd": str(runtime.home),
        "home": str(runtime.home),
    }


def test_managed_plugin_runtime_clears_environment_before_raising(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path)
    sentinel = "nightly-secret-must-not-appear-in-showlocals"
    monkeypatch.setenv("DATUS_TEST_SENTINEL_SECRET", sentinel)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        runtime.run("-c", "raise SystemExit(3)")

    run_frame = next(
        frame
        for frame, _ in traceback.walk_tb(raised.value.__traceback__)
        if frame.f_code.co_name == "run" and "env" in frame.f_locals
    )
    assert run_frame.f_locals["env"] == {}
    assert sentinel not in str(raised.value)


def test_managed_plugin_runtime_clears_environment_on_timeout(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path)
    sentinel = "nightly-timeout-secret-must-not-appear-in-showlocals"
    monkeypatch.setenv("DATUS_TEST_SENTINEL_SECRET", sentinel)

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        runtime.run("-c", "import time; time.sleep(10)", timeout=0.01)

    run_frame = next(
        frame
        for frame, _ in traceback.walk_tb(raised.value.__traceback__)
        if frame.f_code.co_name == "run" and "env" in frame.f_locals
    )
    assert run_frame.f_locals["env"] == {}
    assert sentinel not in str(raised.value)
