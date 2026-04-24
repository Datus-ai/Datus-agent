# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.effort_commands.EffortCommands``.

CI-level: patches ``EffortApp.run`` to return scripted :class:`EffortSelection`
values so dispatcher logic can be exercised without a TTY.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.effort_app import EffortSelection
from datus.cli.effort_commands import EffortCommands
from datus.configuration.project_config import ProjectOverride

pytestmark = pytest.mark.ci

_PATCH_LOAD = "datus.cli.effort_commands.load_project_override"
_PATCH_SAVE = "datus.cli.effort_commands.save_project_override"


def _stub_cli(*, project_effort=None, global_effort=None, active_raises=False):
    cli = MagicMock()
    cli.console = Console(file=io.StringIO(), no_color=True)
    cli.agent_config = MagicMock()
    cli.agent_config._target_reasoning_effort = project_effort
    cli.agent_config.set_active_reasoning_effort = MagicMock()
    if active_raises:
        cli.agent_config.active_model.side_effect = RuntimeError("no model")
    else:
        model_cfg = MagicMock()
        model_cfg.reasoning_effort = None
        model_cfg.enable_thinking = False
        cli.agent_config.active_model.return_value = model_cfg
    cli.configuration_manager = MagicMock()
    cli.configuration_manager.get = MagicMock(return_value=global_effort)
    cli.tui_app = None
    return cli


@pytest.fixture
def commands():
    cli = _stub_cli()
    return EffortCommands(cli), cli


class TestDirectValueWithGlobalFlag:
    def test_saves_to_global(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("high --global")
        cli.configuration_manager.update_item.assert_called_once_with("reasoning_effort", "high")
        cli.agent_config.set_active_reasoning_effort.assert_called_once_with("high", persist=False)

    def test_output_mentions_agent_yml(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("medium --global")
        assert "agent.yml" in cli.console.file.getvalue()

    def test_global_does_not_overwrite_project_override(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(reasoning_effort="low")
        with patch(_PATCH_LOAD, return_value=existing):
            cmds.cmd_effort("high --global")
        cli.configuration_manager.update_item.assert_called_once_with("reasoning_effort", "high")
        # Project override still wins in memory — do not sync the global value
        # onto the in-memory target.
        cli.agent_config.set_active_reasoning_effort.assert_not_called()
        assert "project-level override" in cli.console.file.getvalue()


class TestDirectValueWithProjectFlag:
    def test_saves_to_project(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_effort("high --project")
        saved = mock_save.call_args[0][0]
        assert saved.reasoning_effort == "high"
        cli.agent_config.set_active_reasoning_effort.assert_called_once_with("high", persist=False)

    def test_preserves_existing_override_fields(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(target="deepseek", default_datasource="db1")
        with (
            patch(_PATCH_LOAD, return_value=existing),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_effort("low --project")
        saved = mock_save.call_args[0][0]
        assert saved.target == "deepseek"
        assert saved.default_datasource == "db1"
        assert saved.reasoning_effort == "low"


class TestInvalidValue:
    def test_unknown_level_errors_and_no_save(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_effort("nuclear --project")
        mock_save.assert_not_called()
        cli.configuration_manager.update_item.assert_not_called()
        assert "Invalid effort" in cli.console.file.getvalue()


class TestClearFlag:
    def test_clears_project_override(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(target="deepseek", reasoning_effort="high")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_effort("--clear")
        saved = mock_save.call_args[0][0]
        assert saved.reasoning_effort is None
        assert saved.target == "deepseek"

    def test_falls_back_to_global(self):
        cli = _stub_cli(global_effort="medium")
        cmds = EffortCommands(cli)
        existing = ProjectOverride(reasoning_effort="high")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE):
            cmds.cmd_effort("--clear")
        cli.agent_config.set_active_reasoning_effort.assert_called_once_with("medium", persist=False)

    def test_falls_back_to_none_when_no_global(self):
        cli = _stub_cli(global_effort=None)
        cmds = EffortCommands(cli)
        existing = ProjectOverride(reasoning_effort="high")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE):
            cmds.cmd_effort("--clear")
        cli.agent_config.set_active_reasoning_effort.assert_called_once_with(None, persist=False)

    def test_no_save_when_no_project_override(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_effort("--clear")
        mock_save.assert_not_called()


class TestStatus:
    def test_status_reports_project_source(self):
        cli = _stub_cli()
        cmds = EffortCommands(cli)
        existing = ProjectOverride(reasoning_effort="high")
        with patch(_PATCH_LOAD, return_value=existing):
            cmds.cmd_effort("status")
        output = cli.console.file.getvalue()
        assert "high" in output
        assert "project" in output

    def test_status_reports_global_source(self):
        cli = _stub_cli(global_effort="medium")
        cmds = EffortCommands(cli)
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("status")
        output = cli.console.file.getvalue()
        assert "medium" in output
        assert "global" in output

    def test_status_falls_back_to_model_reasoning_effort(self):
        cli = _stub_cli()
        cli.agent_config.active_model.return_value = MagicMock(reasoning_effort="low", enable_thinking=False)
        cmds = EffortCommands(cli)
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("status")
        output = cli.console.file.getvalue()
        assert "low" in output
        assert "model" in output

    def test_status_falls_back_to_enable_thinking_as_medium(self):
        cli = _stub_cli()
        cli.agent_config.active_model.return_value = MagicMock(reasoning_effort=None, enable_thinking=True)
        cmds = EffortCommands(cli)
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("status")
        output = cli.console.file.getvalue()
        assert "medium" in output
        assert "enable_thinking" in output

    def test_status_reports_not_set(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("status")
        assert "not set" in cli.console.file.getvalue()

    def test_status_tolerates_active_model_failure(self):
        cli = _stub_cli(active_raises=True)
        cmds = EffortCommands(cli)
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_effort("status")
        # No exception propagates; output reports "not set".
        assert "not set" in cli.console.file.getvalue()

    def test_status_includes_supports_reasoning_yes(self):
        cli = _stub_cli()
        cli.agent_config.active_model.return_value = MagicMock(
            model="gpt-5.4", type="openai", reasoning_effort=None, enable_thinking=False
        )
        cmds = EffortCommands(cli)
        with patch(_PATCH_LOAD, return_value=None), patch("litellm.supports_reasoning", return_value=True):
            cmds.cmd_effort("status")
        out = cli.console.file.getvalue().lower()
        assert "supports reasoning" in out
        assert "yes" in out

    def test_status_includes_supports_reasoning_no(self):
        cli = _stub_cli()
        cli.agent_config.active_model.return_value = MagicMock(
            model="gpt-4.1", type="openai", reasoning_effort=None, enable_thinking=False
        )
        cmds = EffortCommands(cli)
        with patch(_PATCH_LOAD, return_value=None), patch("litellm.supports_reasoning", return_value=False):
            cmds.cmd_effort("status")
        out = cli.console.file.getvalue().lower()
        assert "supports reasoning" in out
        assert ": no" in out

    def test_status_silent_when_supports_reasoning_raises(self):
        cli = _stub_cli()
        cli.agent_config.active_model.return_value = MagicMock(
            model="some-custom-model", type="openai", reasoning_effort=None, enable_thinking=False
        )
        cmds = EffortCommands(cli)
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch("litellm.supports_reasoning", side_effect=RuntimeError("unknown provider")),
        ):
            cmds.cmd_effort("status")  # must not raise
        out = cli.console.file.getvalue().lower()
        assert "supports reasoning" not in out


class TestOffValue:
    def test_off_saved_as_level(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_effort("off --project")
        saved = mock_save.call_args[0][0]
        assert saved.reasoning_effort == "off"
        cli.agent_config.set_active_reasoning_effort.assert_called_once_with("off", persist=False)


class TestInteractiveFlow:
    def test_interactive_saves_to_project(self, commands):
        cmds, cli = commands
        selection = EffortSelection(code="high", scope="project")
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_effort("")
        saved = mock_save.call_args[0][0]
        assert saved.reasoning_effort == "high"

    def test_interactive_saves_to_global(self, commands):
        cmds, cli = commands
        selection = EffortSelection(code="low", scope="global")
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_effort("")
        cli.configuration_manager.update_item.assert_called_once_with("reasoning_effort", "low")

    def test_interactive_cancel_does_nothing(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_app", return_value=None),
        ):
            cmds.cmd_effort("")
        mock_save.assert_not_called()
        cli.configuration_manager.update_item.assert_not_called()


class TestDirectValueWithScopePicker:
    def test_scope_picker_project(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
            patch.object(cmds, "_run_scope_picker", return_value=EffortSelection(code="minimal", scope="project")),
        ):
            cmds.cmd_effort("minimal")
        saved = mock_save.call_args[0][0]
        assert saved.reasoning_effort == "minimal"

    def test_scope_picker_global(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch.object(cmds, "_run_scope_picker", return_value=EffortSelection(code="medium", scope="global")),
        ):
            cmds.cmd_effort("medium")
        cli.configuration_manager.update_item.assert_called_once_with("reasoning_effort", "medium")

    def test_scope_picker_cancel(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_scope_picker", return_value=None),
        ):
            cmds.cmd_effort("high")
        mock_save.assert_not_called()
        cli.configuration_manager.update_item.assert_not_called()
