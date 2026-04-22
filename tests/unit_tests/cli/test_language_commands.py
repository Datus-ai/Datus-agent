# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.language_commands.LanguageCommands``.

CI-level: patches ``LanguageApp.run`` to return scripted
:class:`LanguageSelection` values so the dispatcher logic can be exercised
without a TTY.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.language_app import LanguageSelection
from datus.cli.language_commands import LanguageCommands
from datus.configuration.project_config import ProjectOverride

pytestmark = pytest.mark.ci

_PATCH_LOAD = "datus.cli.language_commands.load_project_override"
_PATCH_SAVE = "datus.cli.language_commands.save_project_override"


def _stub_cli(language=None, global_language=None):
    cli = MagicMock()
    cli.console = Console(file=io.StringIO(), no_color=True)
    cli.agent_config = MagicMock()
    cli.agent_config.language = language
    cli.configuration_manager = MagicMock()
    cli.configuration_manager.get = MagicMock(return_value=global_language)
    cli.tui_app = None
    return cli


@pytest.fixture
def commands():
    cli = _stub_cli()
    return LanguageCommands(cli), cli


class TestDirectCodeWithGlobalFlag:
    def test_saves_to_global(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_language("zh --global")
        cli.configuration_manager.update_item.assert_called_once_with("language", "zh")
        assert cli.agent_config.language == "zh"

    def test_output_mentions_agent_yml(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_language("en --global")
        output = cli.console.file.getvalue()
        assert "agent.yml" in output


class TestDirectCodeWithProjectFlag:
    def test_saves_to_project(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_language("zh --project")
        assert cli.agent_config.language == "zh"
        saved_override = mock_save.call_args[0][0]
        assert saved_override.language == "zh"

    def test_preserves_existing_override_fields(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(target="deepseek", default_datasource="db1")
        with (
            patch(_PATCH_LOAD, return_value=existing),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
        ):
            cmds.cmd_language("fr --project")
        saved = mock_save.call_args[0][0]
        assert saved.target == "deepseek"
        assert saved.default_datasource == "db1"
        assert saved.language == "fr"


class TestClearFlag:
    def test_clears_project_override(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(target="deepseek", language="zh")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_language("--clear")
        saved = mock_save.call_args[0][0]
        assert saved.language is None
        assert saved.target == "deepseek"

    def test_falls_back_to_global(self):
        cli = _stub_cli(language="zh", global_language="en")
        cmds = LanguageCommands(cli)
        existing = ProjectOverride(language="zh")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE):
            cmds.cmd_language("--clear")
        assert cli.agent_config.language == "en"

    def test_falls_back_to_none_when_no_global(self):
        cli = _stub_cli(language="zh", global_language=None)
        cmds = LanguageCommands(cli)
        existing = ProjectOverride(language="zh")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE):
            cmds.cmd_language("--clear")
        assert cli.agent_config.language is None

    def test_no_save_when_no_project_override(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_language("--clear")
        mock_save.assert_not_called()


class TestAutoCode:
    def test_auto_with_project_flag_clears_project(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(language="zh")
        with patch(_PATCH_LOAD, return_value=existing), patch(_PATCH_SAVE) as mock_save:
            cmds.cmd_language("auto --project")
        saved = mock_save.call_args[0][0]
        assert saved.language is None

    def test_auto_with_global_flag_clears_global(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None):
            cmds.cmd_language("auto --global")
        cli.configuration_manager.update_item.assert_called_once_with("language", None)
        assert cli.agent_config.language is None

    def test_auto_without_flag_opens_scope_picker(self, commands):
        cmds, cli = commands
        existing = ProjectOverride(language="zh")
        with (
            patch(_PATCH_LOAD, return_value=existing),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_scope_picker", return_value=LanguageSelection(code="auto", scope="project")),
        ):
            cmds.cmd_language("auto")
        saved = mock_save.call_args[0][0]
        assert saved.language is None

    def test_auto_scope_picker_cancel(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_scope_picker", return_value=None),
        ):
            cmds.cmd_language("auto")
        mock_save.assert_not_called()
        cli.configuration_manager.update_item.assert_not_called()


class TestUnknownCode:
    def test_warns_but_proceeds(self, commands):
        cmds, cli = commands
        with patch(_PATCH_LOAD, return_value=None), patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml"):
            cmds.cmd_language("ar --project")
        output = cli.console.file.getvalue()
        assert "Warning" in output
        assert cli.agent_config.language == "ar"


class TestInteractiveFlow:
    def test_interactive_saves_to_project(self, commands):
        cmds, cli = commands
        selection = LanguageSelection(code="ja", scope="project")
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_language("")
        assert cli.agent_config.language == "ja"
        mock_save.assert_called_once()

    def test_interactive_saves_to_global(self, commands):
        cmds, cli = commands
        selection = LanguageSelection(code="ko", scope="global")
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_language("")
        cli.configuration_manager.update_item.assert_called_once_with("language", "ko")

    def test_interactive_cancel_does_nothing(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_app", return_value=None),
        ):
            cmds.cmd_language("")
        mock_save.assert_not_called()
        cli.configuration_manager.update_item.assert_not_called()

    def test_interactive_auto_clears_project(self, commands):
        cmds, cli = commands
        selection = LanguageSelection(code="auto", scope="project")
        existing = ProjectOverride(language="zh")
        with (
            patch(_PATCH_LOAD, return_value=existing),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_language("")
        saved = mock_save.call_args[0][0]
        assert saved.language is None

    def test_interactive_auto_clears_global(self, commands):
        cmds, cli = commands
        selection = LanguageSelection(code="auto", scope="global")
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_language("")
        cli.configuration_manager.update_item.assert_called_once_with("language", None)
        assert cli.agent_config.language is None

    def test_interactive_unchanged_skips(self):
        cli = _stub_cli(language="zh")
        cmds = LanguageCommands(cli)
        selection = LanguageSelection(code="zh", scope="project")
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_app", return_value=selection),
        ):
            cmds.cmd_language("")
        mock_save.assert_not_called()
        output = cli.console.file.getvalue()
        assert "unchanged" in output


class TestDirectCodeWithScopePicker:
    def test_scope_picker_project(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE, return_value="/tmp/.datus/config.yml") as mock_save,
            patch.object(cmds, "_run_scope_picker", return_value=LanguageSelection(code="de", scope="project")),
        ):
            cmds.cmd_language("de")
        assert cli.agent_config.language == "de"
        mock_save.assert_called_once()

    def test_scope_picker_global(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch.object(cmds, "_run_scope_picker", return_value=LanguageSelection(code="de", scope="global")),
        ):
            cmds.cmd_language("de")
        cli.configuration_manager.update_item.assert_called_once_with("language", "de")

    def test_scope_picker_cancel(self, commands):
        cmds, cli = commands
        with (
            patch(_PATCH_LOAD, return_value=None),
            patch(_PATCH_SAVE) as mock_save,
            patch.object(cmds, "_run_scope_picker", return_value=None),
        ):
            cmds.cmd_language("de")
        mock_save.assert_not_called()
        cli.configuration_manager.update_item.assert_not_called()
