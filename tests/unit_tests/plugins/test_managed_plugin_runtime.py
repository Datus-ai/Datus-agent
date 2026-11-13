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

from tests.integration.conftest import ManagedPluginRuntime, sql_policy_source

pytestmark = pytest.mark.component


def _runtime(tmp_path: Path) -> ManagedPluginRuntime:
    home = tmp_path / "home"
    home.mkdir()
    return ManagedPluginRuntime(home=home, datus_executable=Path(sys.executable))


def test_policy_release_installs_from_wheel_without_any_source_checkout(monkeypatch, tmp_path):
    wheel = tmp_path / "datus_sql_policies-0.5.3.whl"
    wheel.touch()
    monkeypatch.setenv("DATUS_SQL_POLICY_WHEEL", str(wheel))
    monkeypatch.setenv("EXTERNAL_REPOS_ROOT", str(tmp_path / "missing-sources"))
    artifact = sql_policy_source.__wrapped__()
    monkeypatch.setattr(ManagedPluginRuntime, "run", lambda self, *args: args)
    assert _runtime(tmp_path).install(artifact) == ("plugin", "install", "whl:" + str(wheel.resolve()))


def test_missing_policy_release_does_not_fall_back_to_source(monkeypatch, tmp_path):
    monkeypatch.setenv("DATUS_SQL_POLICY_WHEEL", str(tmp_path / "missing.whl"))
    with pytest.raises(FileNotFoundError):
        sql_policy_source.__wrapped__()


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


@pytest.mark.parametrize("configured", [False, True], ids=["default-build-homes", "custom-build-homes"])
def test_managed_plugin_runtime_preserves_host_build_directories(monkeypatch, tmp_path, configured):
    runtime = _runtime(tmp_path)
    host_home = tmp_path / "host-home"
    monkeypatch.setenv("HOME", str(host_home))
    expected = {}
    for name, suffix in (
        ("UV_CACHE_DIR", ".cache/uv"),
        ("RUSTUP_HOME", ".rustup"),
        ("CARGO_HOME", ".cargo"),
    ):
        directory = tmp_path / name if configured else host_home / suffix
        if configured:
            monkeypatch.setenv(name, str(directory))
        else:
            monkeypatch.delenv(name, raising=False)
        directory.mkdir(parents=True)
        (directory / "build-state").write_text(name, encoding="utf-8")
        expected[name] = {"path": str(directory), "state": name}

    completed = runtime.run(
        "-c",
        "import json, os; from pathlib import Path; "
        "print(json.dumps({name: {'path': os.environ[name], "
        "'state': (Path(os.environ[name]) / 'build-state').read_text()} "
        "for name in ('UV_CACHE_DIR', 'RUSTUP_HOME', 'CARGO_HOME')}))",
    )

    assert json.loads(completed.stdout) == expected


def test_managed_plugin_runtime_reports_bounded_command_output(tmp_path):
    runtime = _runtime(tmp_path)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        runtime.run(
            "-c",
            "import sys; print('discarded build output\\n' * 30, end=''); "
            "print('maturin build failed'); print('Rust toolchain unavailable', file=sys.stderr); sys.exit(1)",
        )

    notes = "\n".join(raised.value.__notes__)
    assert "maturin build failed" in notes
    assert "Rust toolchain unavailable" in notes
    assert notes.count("discarded build output") == 19
    assert raised.value.stdout.endswith("maturin build failed\n")
    assert raised.value.stderr == "Rust toolchain unavailable\n"


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
