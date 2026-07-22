# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.sandbox_commands.SandboxCommands``.

CI-level: patches project-override IO and the sandbox mechanism probe so the
dispatcher logic can be exercised without touching the filesystem or spawning
sandbox binaries.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.sandbox_commands import SandboxCommands
from datus.configuration.project_config import ProjectOverride
from datus.tools.func_tool.bash_sandbox import SandboxSettings

_PATCH_LOAD = "datus.cli.sandbox_commands.load_project_override"
_PATCH_SAVE = "datus.cli.sandbox_commands.save_project_override"
_PATCH_MECH = "datus.cli.sandbox_commands.bash_sandbox.detect_mechanism"
_PATCH_AVAILABLE = "datus.cli.sandbox_commands.bash_sandbox.is_available"


def _stub_cli(settings: SandboxSettings | None = None):
    cli = MagicMock()
    cli.console = Console(file=io.StringIO(), no_color=True)
    cli.agent_config = MagicMock()
    cli.agent_config.bash_sandbox = settings if settings is not None else SandboxSettings()
    cli.configuration_manager = MagicMock()
    return cli


@pytest.fixture
def commands():
    cli = _stub_cli()
    return SandboxCommands(cli), cli


class TestSessionToggle:
    def test_on_flips_shared_settings(self, commands):
        cmds, cli = commands
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("on")
        assert cli.agent_config.bash_sandbox.enabled is True
        assert "session only" in cli.console.file.getvalue()
        cli.configuration_manager.update_item.assert_not_called()

    def test_off_flips_shared_settings(self, commands):
        cmds, cli = commands
        cli.agent_config.bash_sandbox.enabled = True
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("off")
        assert cli.agent_config.bash_sandbox.enabled is False

    def test_on_warns_when_mechanism_unavailable(self, commands):
        cmds, cli = commands
        with patch(_PATCH_AVAILABLE, return_value=False), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("on")
        # Still enabled (fail-closed happens at execution time) but loudly warned.
        assert cli.agent_config.bash_sandbox.enabled is True
        assert "REJECTED" in cli.console.file.getvalue()

    def test_invalid_value_errors_without_side_effects(self, commands):
        cmds, cli = commands
        with patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_sandbox("maybe")
        assert cli.agent_config.bash_sandbox.enabled is False
        mock_save.assert_not_called()
        assert "Usage" in cli.console.file.getvalue()


class TestProjectPersistence:
    def test_on_project_saves_override(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_AVAILABLE, return_value=True),
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_sandbox("on --project")
        saved = mock_save.call_args[0][0]
        assert saved.sandbox is True
        assert cli.agent_config.bash_sandbox.enabled is True

    def test_off_project_preserves_other_override_fields(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(target="deepseek", reasoning_effort="high")
        with (
            patch(_PATCH_LOAD, return_value=existing),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_sandbox("off --project")
        saved = mock_save.call_args[0][0]
        assert saved.sandbox is False
        assert saved.target == "deepseek"
        assert saved.reasoning_effort == "high"

    def test_clear_removes_project_override(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(sandbox=True, target="deepseek")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_sandbox("--clear")
        saved = mock_save.call_args[0][0]
        assert saved.sandbox is None
        assert saved.target == "deepseek"

    def test_clear_without_override_is_noop_save(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_sandbox("--clear")
        mock_save.assert_not_called()


class TestModeSwitch:
    def test_strict_enables_and_pins_mode(self, commands):
        cmds, cli = commands
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("strict")
        settings = cli.agent_config.bash_sandbox
        assert settings.enabled is True
        assert settings.mode == "strict"
        assert "session only" in cli.console.file.getvalue()

    def test_normal_resets_mode(self):
        cli = _stub_cli(SandboxSettings(enabled=True, mode="strict"))
        cmds = SandboxCommands(cli)
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("normal")
        assert cli.agent_config.bash_sandbox.mode == "normal"
        assert cli.agent_config.bash_sandbox.enabled is True

    def test_on_off_keep_existing_mode(self):
        cli = _stub_cli(SandboxSettings(enabled=False, mode="strict"))
        cmds = SandboxCommands(cli)
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("on")
        assert cli.agent_config.bash_sandbox.mode == "strict"
        assert cli.agent_config.bash_sandbox.enabled is True
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("off")
        assert cli.agent_config.bash_sandbox.mode == "strict"
        assert cli.agent_config.bash_sandbox.enabled is False

    def test_strict_project_persists_mode_string(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_AVAILABLE, return_value=True),
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_sandbox("strict --project")
        assert mock_save.call_args[0][0].sandbox == "strict"


class TestGlobalPersistence:
    def test_on_global_writes_full_sandbox_section(self):
        settings = SandboxSettings(allow_read=["/data"], allow_write=["/scratch"])
        cli = _stub_cli(settings)
        cmds = SandboxCommands(cli)
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("on --global")
        cli.configuration_manager.update_item.assert_called_once_with(
            "bash",
            {"sandbox": {"enabled": True, "mode": "normal", "allow_read": ["/data"], "allow_write": ["/scratch"]}},
        )
        assert "agent.yml" in cli.console.file.getvalue()

    def test_strict_global_persists_mode(self, commands):
        cmds, cli = commands
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("strict --global")
        cli.configuration_manager.update_item.assert_called_once_with(
            "bash", {"sandbox": {"enabled": True, "mode": "strict"}}
        )

    def test_deny_network_survives_global_save(self):
        cli = _stub_cli(SandboxSettings(deny_network=True))
        cmds = SandboxCommands(cli)
        with patch(_PATCH_AVAILABLE, return_value=True), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("on --global")
        saved = cli.configuration_manager.update_item.call_args[0][1]["sandbox"]
        assert saved["deny_network"] is True

    def test_off_global_omits_empty_lists(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("off --global")
        cli.configuration_manager.update_item.assert_called_once_with(
            "bash", {"sandbox": {"enabled": False, "mode": "normal"}}
        )

    def test_global_notes_conflicting_project_override(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(sandbox=True)
        with patch(_PATCH_LOAD, return_value=existing):
            cmds.cmd_sandbox("off --global")
        assert "project-level override" in cli.console.file.getvalue()


class TestStatus:
    def test_status_shows_state_and_mechanism(self, commands):
        cmds, cli = commands
        with patch(_PATCH_MECH, return_value="seatbelt"), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("status")
        out = cli.console.file.getvalue()
        assert "off" in out
        assert "seatbelt" in out

    def test_empty_args_means_status(self, commands):
        cmds, cli = commands
        with patch(_PATCH_MECH, return_value=None), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("")
        assert "Bash sandbox: off" in cli.console.file.getvalue()

    def test_status_warns_fail_closed_when_enabled_but_unavailable(self):
        cli = _stub_cli(SandboxSettings(enabled=True))
        cmds = SandboxCommands(cli)
        with patch(_PATCH_MECH, return_value=None), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("status")
        out = cli.console.file.getvalue()
        assert "Fail-closed" in out

    def test_status_shows_project_source_and_allowlists(self):
        cli = _stub_cli(SandboxSettings(enabled=True, allow_read=["/data"], allow_write=["/scratch"]))
        cmds = SandboxCommands(cli)
        with patch(_PATCH_MECH, return_value="bwrap"), patch(_PATCH_LOAD, return_value=ProjectOverride(sandbox=True)):
            cmds.cmd_sandbox("status")
        out = cli.console.file.getvalue()
        assert "project" in out
        assert "/data" in out
        assert "/scratch" in out

    def test_status_strict_shows_blocked_datus_home_and_env(self):
        cli = _stub_cli(SandboxSettings(enabled=True, mode="strict", deny_network=True))
        cmds = SandboxCommands(cli)
        with patch(_PATCH_MECH, return_value="bwrap"), patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_sandbox("status")
        out = cli.console.file.getvalue()
        assert "mode: strict" in out
        assert "BLOCKED" in out
        assert "minimal allowlist" in out
        assert "Network: denied" in out
