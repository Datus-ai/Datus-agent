# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.upgrade_cli.run_upgrade_command``.

CI-level: package enumeration and the upgrade subprocess are
monkey-patched; output is captured via a Rich ``Console`` writing to a
``StringIO`` so we can assert on rendered text.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from datus.cli import upgrade_cli
from datus.cli import upgrade_service as svc


@pytest.fixture
def capture_console(monkeypatch):
    """Force ``run_upgrade_command`` to render into a captured buffer."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    monkeypatch.setattr(upgrade_cli, "Console", lambda *a, **k: console)
    return buffer


def test_all_editable_warns_and_skips_upgrade(capture_console, monkeypatch):
    monkeypatch.setattr(
        svc,
        "enumerate_datus_packages",
        lambda: [svc.DatusPackage("datus-agent", "0.3.1", editable=True)],
    )
    called = []
    monkeypatch.setattr(svc, "upgrade_packages", lambda pkgs: called.append(pkgs))

    rc = upgrade_cli.run_upgrade_command([])
    output = capture_console.getvalue()

    assert rc == 0
    assert called == []  # upgrade_packages must not be invoked
    assert "nothing to upgrade" in output


def test_success_shows_version_diffs_and_clears_cache(capture_console, monkeypatch):
    monkeypatch.setattr(
        svc,
        "enumerate_datus_packages",
        lambda: [svc.DatusPackage("datus-agent", "0.3.1", editable=False)],
    )
    monkeypatch.setattr(
        svc,
        "upgrade_packages",
        lambda pkgs: svc.UpgradeResult(
            ok=True,
            packages=["datus-agent"],
            changes=[("datus-agent", "0.3.1", "0.3.2")],
            label="pip install --upgrade",
        ),
    )
    cache_writes = []
    monkeypatch.setattr(svc, "_cache_path", lambda agent_config=None: "/tmp/fake")
    monkeypatch.setattr(svc, "_write_cache", lambda path, latest: cache_writes.append((path, latest)))

    rc = upgrade_cli.run_upgrade_command([])
    output = capture_console.getvalue()

    assert rc == 0
    assert "datus-agent 0.3.1 -> 0.3.2" in output
    assert "Upgrade complete" in output
    assert "Restart datus" in output
    assert cache_writes == [("/tmp/fake", None)]  # version-check cache cleared


def test_no_changes_reports_already_up_to_date(capture_console, monkeypatch):
    monkeypatch.setattr(
        svc,
        "enumerate_datus_packages",
        lambda: [svc.DatusPackage("datus-agent", "0.3.2", editable=False)],
    )
    monkeypatch.setattr(
        svc,
        "upgrade_packages",
        lambda pkgs: svc.UpgradeResult(ok=True, packages=["datus-agent"], changes=[], label="pip install --upgrade"),
    )
    monkeypatch.setattr(svc, "_cache_path", lambda agent_config=None: "/tmp/fake")
    monkeypatch.setattr(svc, "_write_cache", lambda path, latest: None)

    rc = upgrade_cli.run_upgrade_command([])
    output = capture_console.getvalue()

    assert rc == 0
    assert "already up to date" in output
    assert "->" not in output
    assert "Restart datus" not in output


def test_failure_shows_error_and_stderr_tail(capture_console, monkeypatch):
    monkeypatch.setattr(
        svc,
        "enumerate_datus_packages",
        lambda: [svc.DatusPackage("datus-agent", "0.3.1", editable=False)],
    )
    monkeypatch.setattr(
        svc,
        "upgrade_packages",
        lambda pkgs: svc.UpgradeResult(
            ok=False,
            packages=["datus-agent"],
            label="pip install --upgrade",
            stderr="line-a\nERROR: resolution impossible",
            error="pip install --upgrade exited with code 1",
        ),
    )

    rc = upgrade_cli.run_upgrade_command([])
    output = capture_console.getvalue()

    assert rc == 1
    assert "Upgrade failed" in output
    assert "resolution impossible" in output
    assert "exited with code 1" in output


def test_check_flag_does_not_upgrade(capture_console, monkeypatch):
    monkeypatch.setattr(
        svc,
        "enumerate_datus_packages",
        lambda: [svc.DatusPackage("datus-agent", "0.3.1", editable=False)],
    )
    upgrade_called = []
    monkeypatch.setattr(svc, "upgrade_packages", lambda pkgs: upgrade_called.append(pkgs))
    monkeypatch.setattr(svc, "get_latest_version", lambda **k: "0.4.0")
    monkeypatch.setattr(svc, "newer_version_available", lambda current, **k: "0.4.0")

    rc = upgrade_cli.run_upgrade_command(["--check"])
    output = capture_console.getvalue()

    assert rc == 0
    assert upgrade_called == []
    assert "0.4.0" in output


def test_check_flag_reports_up_to_date(capture_console, monkeypatch):
    monkeypatch.setattr(
        svc,
        "enumerate_datus_packages",
        lambda: [svc.DatusPackage("datus-agent", "0.4.0", editable=False)],
    )
    monkeypatch.setattr(svc, "upgrade_packages", lambda pkgs: pytest.fail("must not upgrade"))
    monkeypatch.setattr(svc, "get_latest_version", lambda **k: "0.4.0")

    rc = upgrade_cli.run_upgrade_command(["--check"])
    output = capture_console.getvalue()

    assert rc == 0
    assert "up to date" in output
