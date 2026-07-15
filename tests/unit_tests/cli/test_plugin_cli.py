# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_cli.run_plugin_command`` dispatch.

CI-level: the plugin service, registry lookups, store, and agent-config load are
all mocked; no subprocess, no network.
"""

import pytest

from datus.cli import plugin_cli
from datus.cli.plugin_service import ExportResult, InstallResult, PluginInfo, UninstallResult, UpgradeResult


class _FakeConfig:
    def __init__(self):
        self.calls = []

    def set_plugin_activation(self, name, *, enabled=None, active_profiles=None, clear_profiles=False, persist=True):
        self.calls.append(
            {
                "name": name,
                "enabled": enabled,
                "active_profiles": active_profiles,
                "clear_profiles": clear_profiles,
            }
        )


@pytest.fixture
def fake_config(monkeypatch):
    cfg = _FakeConfig()
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: cfg)
    return cfg


def test_no_subcommand_prints_help(capsys):
    assert plugin_cli.run_plugin_command([]) == 0
    out = capsys.readouterr().out
    assert "install" in out and "export" in out and "upgrade" in out


# ── install ──────────────────────────────────────────────────────────────────


def test_install_success(monkeypatch, capsys):
    monkeypatch.setattr(
        plugin_cli.svc,
        "install",
        lambda source, force=False: InstallResult(ok=True, name="demo", version="1.0", plugin_dir="/p/demo"),
    )
    assert plugin_cli.run_plugin_command(["install", "pip:datus-demo-plugin"]) == 0
    assert "demo" in capsys.readouterr().out


def test_install_forwards_type_spec_and_force(monkeypatch):
    captured = {}

    def fake_install(source, force=False):
        captured["source"] = source
        captured["force"] = force
        return InstallResult(ok=True, name="demo")

    monkeypatch.setattr(plugin_cli.svc, "install", fake_install)
    plugin_cli.run_plugin_command(["install", "git:https://h/x", "--force"])
    assert captured == {"source": "git:https://h/x", "force": True}


def test_install_forwards_bare_source_as_implicit_pip(monkeypatch):
    # A prefix-less source is forwarded verbatim; the service defaults it to pip.
    captured = {}

    def fake_install(source, force=False):
        captured["source"] = source
        captured["force"] = force
        return InstallResult(ok=True, name="demo")

    monkeypatch.setattr(plugin_cli.svc, "install", fake_install)
    plugin_cli.run_plugin_command(["install", "datus-demo-plugin"])
    assert captured == {"source": "datus-demo-plugin", "force": False}


def test_install_no_editable_flag():
    # ``-e`` was removed; argparse must reject it.
    with pytest.raises(SystemExit):
        plugin_cli.run_plugin_command(["install", "src:./x", "-e"])


def test_install_failure_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "install", lambda source, force=False: InstallResult(ok=False, error="boom"))
    assert plugin_cli.run_plugin_command(["install", "pip:bad"]) == 1


# ── pack ─────────────────────────────────────────────────────────────────────


def test_pack_success(monkeypatch, capsys):
    from datus.cli import plugin_pack

    captured = {}

    def fake_pack(source, out_dir=".", with_deps=False):
        captured.update({"source": source, "out_dir": out_dir, "with_deps": with_deps})
        return plugin_pack.PackResult(
            ok=True, bundle_path="dist/datus-demo-plugin-0.1.0.zip", plugin_name="demo", wheel_count=3, with_deps=True
        )

    monkeypatch.setattr(plugin_pack, "pack", fake_pack)
    assert plugin_cli.run_plugin_command(["pack", "./demo", "-o", "dist", "--with-deps"]) == 0
    assert captured == {"source": "./demo", "out_dir": "dist", "with_deps": True}
    assert "datus-demo-plugin-0.1.0.zip" in capsys.readouterr().out


def test_pack_defaults(monkeypatch):
    from datus.cli import plugin_pack

    captured = {}

    def fake_pack(source, out_dir=".", with_deps=False):
        captured.update({"source": source, "out_dir": out_dir, "with_deps": with_deps})
        return plugin_pack.PackResult(ok=True, bundle_path="x.zip", plugin_name="demo", wheel_count=1)

    monkeypatch.setattr(plugin_pack, "pack", fake_pack)
    plugin_cli.run_plugin_command(["pack"])
    assert captured == {"source": ".", "out_dir": ".", "with_deps": False}


def test_pack_failure_returns_1(monkeypatch):
    from datus.cli import plugin_pack

    monkeypatch.setattr(plugin_pack, "pack", lambda *a, **k: plugin_pack.PackResult(ok=False, error="build failed"))
    assert plugin_cli.run_plugin_command(["pack", "./broken"]) == 1


# ── export ───────────────────────────────────────────────────────────────────


def test_export_success(monkeypatch, capsys):
    captured = {}

    def fake_export(name, out_dir="."):
        captured.update({"name": name, "out_dir": out_dir})
        return ExportResult(ok=True, name=name, out_path="out/demo-0.1.0.zip")

    monkeypatch.setattr(plugin_cli.svc, "export", fake_export)
    assert plugin_cli.run_plugin_command(["export", "demo", "-o", "out"]) == 0
    assert captured == {"name": "demo", "out_dir": "out"}
    assert "demo-0.1.0.zip" in capsys.readouterr().out


def test_export_failure_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "export", lambda name, out_dir=".": ExportResult(ok=False, error="nope"))
    assert plugin_cli.run_plugin_command(["export", "mystery"]) == 1


# ── upgrade ──────────────────────────────────────────────────────────────────


def test_upgrade_success(monkeypatch, capsys):
    monkeypatch.setattr(
        plugin_cli.svc,
        "upgrade",
        lambda name: UpgradeResult(ok=True, name=name, old_version="0.1.0", new_version="0.2.0"),
    )
    assert plugin_cli.run_plugin_command(["upgrade", "demo"]) == 0
    assert "0.1.0" in capsys.readouterr().out


def test_upgrade_failure_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "upgrade", lambda name: UpgradeResult(ok=False, error="pinned"))
    assert plugin_cli.run_plugin_command(["upgrade", "demo"]) == 1


# ── uninstall / list / info ──────────────────────────────────────────────────


def test_uninstall_success(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "uninstall", lambda name: UninstallResult(ok=True, plugin=name, package="pkg"))
    assert plugin_cli.run_plugin_command(["uninstall", "demo"]) == 0


def test_uninstall_failure_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "uninstall", lambda name: UninstallResult(ok=False, error="no such plugin"))
    assert plugin_cli.run_plugin_command(["uninstall", "mystery"]) == 1


def test_list(monkeypatch, capsys):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    monkeypatch.setattr(
        plugin_cli.svc,
        "list_plugins",
        lambda cfg: [
            PluginInfo(name="demo", package="datus-demo-plugin", version="0.1.0", install_type="src", source="managed")
        ],
    )
    assert plugin_cli.run_plugin_command(["list"]) == 0
    assert "demo" in capsys.readouterr().out


def test_list_empty(monkeypatch, capsys):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    monkeypatch.setattr(plugin_cli.svc, "list_plugins", lambda cfg: [])
    assert plugin_cli.run_plugin_command(["list"]) == 0
    assert "No plugins installed" in capsys.readouterr().out


def test_info_known(monkeypatch, capsys):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    monkeypatch.setattr(
        plugin_cli.svc,
        "list_plugins",
        lambda cfg: [PluginInfo(name="demo", package="p", version="1.0", install_type="pip", source="managed")],
    )
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: [])
    assert plugin_cli.run_plugin_command(["info", "demo"]) == 0
    out = capsys.readouterr().out
    assert "demo" in out and "pip" in out


def test_info_unknown_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    monkeypatch.setattr(plugin_cli.svc, "list_plugins", lambda cfg: [])
    assert plugin_cli.run_plugin_command(["info", "mystery"]) == 1


# ── enable / disable ─────────────────────────────────────────────────────────


def test_enable_calls_activation(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.store.plugin_dir", lambda name: __import__("pathlib").Path("/nonexistent"))
    assert plugin_cli.run_plugin_command(["enable", "demo"]) == 0
    assert fake_config.calls == [{"name": "demo", "enabled": True, "active_profiles": None, "clear_profiles": True}]


def test_enable_with_profiles(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.store.plugin_dir", lambda name: __import__("pathlib").Path("/nonexistent"))
    plugin_cli.run_plugin_command(["enable", "demo", "--profile", "prod", "--profile", "dev"])
    assert fake_config.calls[0]["active_profiles"] == ["prod", "dev"]
    assert fake_config.calls[0]["clear_profiles"] is False


def test_disable_calls_activation(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    monkeypatch.setattr("datus.plugins.store.plugin_dir", lambda name: __import__("pathlib").Path("/nonexistent"))
    assert plugin_cli.run_plugin_command(["disable", "demo"]) == 0
    assert fake_config.calls[0]["enabled"] is False


def test_enable_unknown_plugin_returns_1(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: False)
    monkeypatch.setattr("datus.plugins.store.plugin_dir", lambda name: __import__("pathlib").Path("/nonexistent"))
    assert plugin_cli.run_plugin_command(["enable", "mystery"]) == 1
    assert fake_config.calls == []


def test_enable_without_config_returns_3(monkeypatch):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    assert plugin_cli.run_plugin_command(["enable", "demo"]) == 3


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
