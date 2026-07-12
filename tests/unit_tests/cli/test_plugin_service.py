# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_service`` (install / uninstall / list).

CI-level: subprocess / shutil / entry-point enumeration are all mocked; no
network, no real package operations.
"""

import sys
import types

import pytest

from datus.cli import plugin_service as svc


class _FakeDist:
    def __init__(self, name, version="1.0.0"):
        self.name = name
        self.version = version


class _FakeEntryPoint:
    def __init__(self, name, package="", version="1.0.0", value="mod:Cls"):
        self.name = name
        self.value = value
        self.dist = _FakeDist(package, version) if package else None


def _fake_proc(returncode=0, stdout="ok", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ── _install_command ──────────────────────────────────────────────────────


def test_install_command_prefers_uv(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    cmd, label = svc._install_command("datus-foo")
    assert cmd == ["/usr/bin/uv", "pip", "install", "--python", sys.executable, "datus-foo"]
    assert label == "uv pip install"


def test_install_command_editable_flag(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    cmd, _ = svc._install_command("./local", editable=True)
    assert "-e" in cmd and cmd[-1] == "./local"


def test_install_command_pip_fallback(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    cmd, label = svc._install_command("datus-foo")
    assert cmd == [sys.executable, "-m", "pip", "install", "datus-foo"]
    assert label == "pip install"


# ── install ─────────────────────────────────────────────────────────────


def test_install_reports_new_plugins(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    state = {"installed": False}

    def fake_run(cmd, **kwargs):
        state["installed"] = True
        return _fake_proc()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)

    def fake_iter():
        return [_FakeEntryPoint("statsig")] if state["installed"] else []

    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", fake_iter)
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)

    result = svc.install("datus-statsig-plugin")
    assert result.ok is True
    assert result.new_plugins == ["statsig"]


def test_install_no_new_plugins(monkeypatch):
    """A package that registers no datus.plugins entry point → empty list."""
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", lambda cmd, **k: _fake_proc())
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)
    result = svc.install("some-package")
    assert result.ok is True
    assert result.new_plugins == []


def test_install_failure_returncode(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.subprocess, "run", lambda cmd, **k: _fake_proc(returncode=1, stderr="boom"))
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    result = svc.install("bad-package")
    assert not result.ok
    assert "code 1" in (result.error or "")
    assert result.stderr == "boom"


def test_install_subprocess_raises(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)

    def boom(cmd, **k):
        raise OSError("uv missing")

    monkeypatch.setattr(svc.subprocess, "run", boom)
    monkeypatch.setattr("datus.plugins.registry.iter_plugin_entry_points", lambda: [])
    result = svc.install("x")
    assert not result.ok
    assert "uv missing" in (result.error or "")


def test_install_empty_source():
    result = svc.install("   ")
    assert not result.ok
    assert "no install source" in (result.error or "")


# ── uninstall ─────────────────────────────────────────────────────────────


def test_uninstall_maps_plugin_to_distribution(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "datus.plugins.registry.entry_points_for_group",
        lambda group, name=None: [_FakeEntryPoint("statsig", package="datus-statsig-plugin")],
    )
    monkeypatch.setattr("datus.plugins.registry.invalidate_plugin_cache", lambda: None)
    captured = {}

    def fake_run(cmd, **k):
        captured["cmd"] = cmd
        return _fake_proc()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    result = svc.uninstall("statsig")
    assert result.ok is True
    assert result.package == "datus-statsig-plugin"
    assert "datus-statsig-plugin" in captured["cmd"]


def test_uninstall_unknown_plugin(monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.entry_points_for_group", lambda group, name=None: [])
    result = svc.uninstall("mystery")
    assert not result.ok
    assert "no installed plugin" in (result.error or "")


def test_uninstall_failure_returncode(monkeypatch):
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "datus.plugins.registry.entry_points_for_group",
        lambda group, name=None: [_FakeEntryPoint("statsig", package="datus-statsig-plugin")],
    )
    monkeypatch.setattr(svc.subprocess, "run", lambda cmd, **k: _fake_proc(returncode=1, stderr="nope"))
    result = svc.uninstall("statsig")
    assert not result.ok
    assert "code 1" in (result.error or "")


# ── list_plugins ────────────────────────────────────────────────────────


def test_list_plugins_without_config(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", package="datus-statsig-plugin", version="0.1.0")],
    )
    plugins = svc.list_plugins(agent_config=None)
    assert len(plugins) == 1
    info = plugins[0]
    assert info.name == "statsig"
    assert info.package == "datus-statsig-plugin"
    assert info.version == "0.1.0"
    assert info.active is None  # unknown without a config
    assert info.profiles == []


def test_list_plugins_with_config(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", package="datus-statsig-plugin")],
    )

    class _Cfg:
        plugin_services = {"statsig": {"dev": {}, "prod": {}}}

        def active_plugin_names(self):
            return {"statsig"}

        def active_plugin_profiles(self, name):
            return ["prod"]

    plugins = svc.list_plugins(_Cfg())
    info = plugins[0]
    assert info.profiles == ["dev", "prod"]
    assert info.active is True
    assert info.active_profiles == ["prod"]


def test_list_plugins_inactive_when_not_in_whitelist(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("statsig", package="p")],
    )

    class _Cfg:
        plugin_services = {}

        def active_plugin_names(self):
            return {"other"}  # statsig not listed

        def active_plugin_profiles(self, name):
            return []

    assert svc.list_plugins(_Cfg())[0].active is False


def test_list_plugins_sorted(monkeypatch):
    monkeypatch.setattr(
        "datus.plugins.registry.iter_plugin_entry_points",
        lambda: [_FakeEntryPoint("zeta", package="z"), _FakeEntryPoint("alpha", package="a")],
    )
    assert [p.name for p in svc.list_plugins(None)] == ["alpha", "zeta"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
