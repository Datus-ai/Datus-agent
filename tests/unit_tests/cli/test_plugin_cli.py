# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.plugin_cli.run_plugin_command`` dispatch.

CI-level: the plugin service, registry lookups, and agent-config load are all
mocked; no subprocess, no network.
"""

import pytest

from datus.cli import plugin_cli
from datus.cli.plugin_service import InstallResult, PluginInfo, UninstallResult


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
    assert "install" in out and "uninstall" in out


def test_install_success(monkeypatch):
    monkeypatch.setattr(
        plugin_cli.svc,
        "install",
        lambda source, editable=False, force=False: InstallResult(ok=True, new_plugins=["statsig"]),
    )
    assert plugin_cli.run_plugin_command(["install", "datus-statsig-plugin"]) == 0


def test_install_editable_flag_forwarded(monkeypatch):
    captured = {}

    def fake_install(source, editable=False, force=False):
        captured["source"] = source
        captured["editable"] = editable
        captured["force"] = force
        return InstallResult(ok=True, new_plugins=["x"])

    monkeypatch.setattr(plugin_cli.svc, "install", fake_install)
    plugin_cli.run_plugin_command(["install", "./local", "--editable"])
    assert captured == {"source": "./local", "editable": True, "force": False}


def test_install_force_flag_forwarded(monkeypatch):
    captured = {}

    def fake_install(source, editable=False, force=False):
        captured["force"] = force
        return InstallResult(ok=True, new_plugins=["x"])

    monkeypatch.setattr(plugin_cli.svc, "install", fake_install)
    plugin_cli.run_plugin_command(["install", "./bundle.dplug", "--force"])
    assert captured["force"] is True


def test_install_failure_returns_1(monkeypatch):
    monkeypatch.setattr(
        plugin_cli.svc, "install", lambda source, editable=False, force=False: InstallResult(ok=False, error="boom")
    )
    assert plugin_cli.run_plugin_command(["install", "bad"]) == 1


def test_pack_success(monkeypatch, capsys):
    from datus.cli import plugin_pack

    captured = {}

    def fake_pack(source, out_dir=".", python_version=None, platform_tag=None):
        captured.update(
            {"source": source, "out_dir": out_dir, "python_version": python_version, "platform_tag": platform_tag}
        )
        return plugin_pack.PackResult(
            ok=True, bundle_path="dist/datus-plugin-hello-1.0.0-any.dplug", plugin_name="hello", wheel_count=3
        )

    monkeypatch.setattr(plugin_pack, "pack", fake_pack)
    assert (
        plugin_cli.run_plugin_command(
            [
                "pack",
                "./datus-plugin-hello",
                "-o",
                "dist",
                "--python-version",
                "3.12",
                "--platform",
                "manylinux2014_x86_64",
            ]
        )
        == 0
    )
    assert captured == {
        "source": "./datus-plugin-hello",
        "out_dir": "dist",
        "python_version": "3.12",
        "platform_tag": "manylinux2014_x86_64",
    }
    out = capsys.readouterr().out
    assert "datus-plugin-hello-1.0.0-any.dplug" in out


def test_pack_failure_returns_1(monkeypatch):
    from datus.cli import plugin_pack

    monkeypatch.setattr(
        plugin_pack, "pack", lambda *a, **k: plugin_pack.PackResult(ok=False, error="build wheel failed")
    )
    assert plugin_cli.run_plugin_command(["pack", "./broken"]) == 1


def test_uninstall_success(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "uninstall", lambda name: UninstallResult(ok=True, plugin=name, package="pkg"))
    assert plugin_cli.run_plugin_command(["uninstall", "statsig"]) == 0


def test_uninstall_failure_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli.svc, "uninstall", lambda name: UninstallResult(ok=False, error="no such plugin"))
    assert plugin_cli.run_plugin_command(["uninstall", "mystery"]) == 1


def test_list(monkeypatch, capsys):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    monkeypatch.setattr(
        plugin_cli.svc,
        "list_plugins",
        lambda cfg: [PluginInfo(name="statsig", package="datus-statsig-plugin", version="0.1.0", profiles=["dev"])],
    )
    assert plugin_cli.run_plugin_command(["list"]) == 0
    assert "statsig" in capsys.readouterr().out


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
        lambda cfg: [PluginInfo(name="statsig", package="p", version="1.0", profiles=["dev"])],
    )
    monkeypatch.setattr("datus.plugins.registry.plugin_config_schema", lambda name: [])
    assert plugin_cli.run_plugin_command(["info", "statsig"]) == 0
    assert "statsig" in capsys.readouterr().out


def test_info_unknown_returns_1(monkeypatch):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    monkeypatch.setattr(plugin_cli.svc, "list_plugins", lambda cfg: [])
    assert plugin_cli.run_plugin_command(["info", "mystery"]) == 1


def test_enable_calls_activation(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    assert plugin_cli.run_plugin_command(["enable", "statsig"]) == 0
    assert fake_config.calls == [{"name": "statsig", "enabled": True, "active_profiles": None, "clear_profiles": True}]


def test_enable_with_profiles(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    plugin_cli.run_plugin_command(["enable", "statsig", "--profile", "prod", "--profile", "dev"])
    assert fake_config.calls[0]["active_profiles"] == ["prod", "dev"]
    assert fake_config.calls[0]["clear_profiles"] is False


def test_disable_calls_activation(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: True)
    assert plugin_cli.run_plugin_command(["disable", "statsig"]) == 0
    assert fake_config.calls[0]["enabled"] is False


def test_enable_unknown_plugin_returns_1(fake_config, monkeypatch):
    monkeypatch.setattr("datus.plugins.registry.plugin_entry_point_exists", lambda name: False)
    assert plugin_cli.run_plugin_command(["enable", "mystery"]) == 1
    assert fake_config.calls == []


def test_enable_without_config_returns_3(monkeypatch):
    monkeypatch.setattr(plugin_cli, "_load_agent_config", lambda console: None)
    assert plugin_cli.run_plugin_command(["enable", "statsig"]) == 3


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
