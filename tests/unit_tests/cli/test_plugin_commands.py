# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.plugin_commands` (the ``/plugins`` gate).

CI-level: a throwaway ``~/.datus`` home and a stubbed entry-point registry —
no plugin code is imported and no real ``datus.plugins`` entry points leak in
from the dev environment. The regression under test: the gate must consult the
manifest-based ``list_plugins()`` listing, because a disabled managed plugin's
directory is off ``sys.path`` and therefore invisible to entry-point probing.
"""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.plugin_commands import PluginCommands
from datus.plugins import store
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager


@pytest.fixture
def home(tmp_path):
    token = set_current_path_manager(DatusPathManager(datus_home=tmp_path))
    try:
        yield tmp_path
    finally:
        reset_path_manager(token)


def _agent_config(*, plugins_enabled=True):
    config = MagicMock()
    config.plugins_enabled = plugins_enabled
    # Project config lists plugins but none is enabled — the state that keeps
    # every managed plugin directory off sys.path.
    config.active_plugin_names.return_value = set()
    config.active_plugin_profiles.return_value = []
    config.plugin_services = {}
    config.plugin_paths = None
    return config


def _build_commands(agent_config, tui_app=None):
    buf = io.StringIO()
    cli = SimpleNamespace(
        console=Console(file=buf, force_terminal=False, width=120, log_path=False),
        agent_config=agent_config,
        tui_app=tui_app,
    )
    return PluginCommands(cli), buf


def _install_managed_plugin(name="demo"):
    store.write_meta(
        store.plugin_dir(name),
        {
            "name": name,
            "distribution": f"datus-{name}-plugin",
            "version": "0.1.0",
            "entry_point": f"datus_{name}_plugin",
            "install": {"type": "whl"},
        },
    )


def test_master_switch_off_prints_warning_and_skips_manager(home):
    commands, buf = _build_commands(_agent_config(plugins_enabled=False))
    with patch("datus.cli.plugin_commands.PluginApp") as app_cls:
        commands.cmd_plugins("")
    assert "Plugins are disabled" in buf.getvalue()
    app_cls.assert_not_called()


def test_empty_store_prints_install_hint(home):
    commands, buf = _build_commands(_agent_config())
    with (
        patch("datus.plugins.registry.iter_plugin_entry_points", return_value=[]),
        patch("datus.cli.plugin_commands.PluginApp") as app_cls,
    ):
        commands.cmd_plugins("")
    assert "No plugins installed" in buf.getvalue()
    app_cls.assert_not_called()


def test_disabled_managed_plugin_still_opens_manager(home):
    """A managed plugin with project activation off must reach the manager.

    Its directory is off sys.path, so entry-point probing sees nothing — the
    gate must rely on the datus-plugin.json manifest listing instead, else
    /plugins (the tool for re-enabling plugins) becomes unreachable.
    """
    _install_managed_plugin("demo")
    agent_config = _agent_config()
    commands, buf = _build_commands(agent_config)
    with (
        patch("datus.plugins.registry.iter_plugin_entry_points", return_value=[]),
        patch("datus.cli.plugin_commands.PluginApp") as app_cls,
    ):
        commands.cmd_plugins("")
    assert "No plugins installed" not in buf.getvalue()
    app_cls.assert_called_once_with(agent_config, commands.console)
    app_cls.return_value.run.assert_called_once_with()


def test_discovery_failure_degrades_open(home):
    commands, _buf = _build_commands(_agent_config())
    with (
        patch("datus.cli.plugin_service.list_plugins", side_effect=RuntimeError("boom")),
        patch("datus.cli.plugin_commands.PluginApp") as app_cls,
    ):
        commands.cmd_plugins("")
    app_cls.return_value.run.assert_called_once_with()


def test_active_tui_embeds_manager_via_run_wizard(home):
    _install_managed_plugin("demo")
    tui_app = SimpleNamespace(_loop=object(), run_wizard=MagicMock(return_value=None))
    commands, _buf = _build_commands(_agent_config(), tui_app=tui_app)
    with (
        patch("datus.plugins.registry.iter_plugin_entry_points", return_value=[]),
        patch("datus.cli.plugin_commands.PluginApp") as app_cls,
    ):
        commands.cmd_plugins("")
    tui_app.run_wizard.assert_called_once_with(app_cls.return_value.build_embedded_panel)
    app_cls.return_value.run.assert_not_called()
