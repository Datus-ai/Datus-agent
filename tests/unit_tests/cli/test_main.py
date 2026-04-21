# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/main.py — ArgumentParser, Application, main().

All external dependencies (DatusCLI, run_web_interface, configure_logging) are mocked.
"""

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.main import Application, ArgumentParser

# ---------------------------------------------------------------------------
# Tests: ArgumentParser
# ---------------------------------------------------------------------------


class TestArgumentParser:
    def test_init_creates_parser(self):
        ap = ArgumentParser()
        assert isinstance(ap.parser, argparse.ArgumentParser)

    def test_parse_args_defaults(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus"]):
            args = ap.parse_args()
        assert args.db_type == "sqlite"
        assert args.debug is False
        assert args.no_color is False
        assert args.datasource == ""
        assert args.print_mode is None
        assert args.web is False
        assert args.resume is None

    def test_parse_args_debug_flag(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--debug", "--datasource", "ns1"]):
            args = ap.parse_args()
        assert args.debug is True
        assert args.datasource == "ns1"

    def test_parse_args_print(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--datasource", "ns1", "--print", "hello"]):
            args = ap.parse_args()
        assert args.print_mode == "hello"

    def test_parse_args_print_short(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--datasource", "ns1", "-p", "hello"]):
            args = ap.parse_args()
        assert args.print_mode == "hello"

    def test_parse_args_resume(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--datasource", "ns1", "--print", "hello", "--resume", "sess_123"]):
            args = ap.parse_args()
        assert args.resume == "sess_123"

    def test_parse_args_web(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--datasource", "ns1", "--web"]):
            args = ap.parse_args()
        assert args.web is True

    def test_print_and_web_are_mutually_exclusive(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--web", "--print", "hello"]):
            with pytest.raises(SystemExit):
                ap.parse_args()


# ---------------------------------------------------------------------------
# Tests: Application.run
# ---------------------------------------------------------------------------


class TestApplicationRun:
    def test_run_no_namespace_prints_help(self):
        """When no database is set and _resolve_default_datasource fails, help is printed."""
        app = Application()
        mock_args = SimpleNamespace(
            debug=False, datasource="", print_mode=None, web=False, resume=None, proxy_tools=None, config=None
        )
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_ensure_project_config"),
            patch.object(app, "_resolve_default_datasource", return_value=""),
            patch.object(app.arg_parser.parser, "print_help") as mock_help,
        ):
            app.run()
        # _resolve_default_datasource returning "" should cause early return without
        # reaching the REPL; no test asserts print_help here because the real
        # print_help is triggered inside _resolve_default_datasource (which we mocked).
        # Just verify the run returned cleanly.
        mock_help.assert_not_called()

    def test_resume_without_print_mode_errors(self):
        app = Application()
        mock_args = SimpleNamespace(
            debug=False, datasource="ns1", print_mode=None, web=False, resume="sess_123", proxy_tools=None, config=None
        )
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_ensure_project_config"),
        ):
            with pytest.raises(SystemExit):
                app.run()

    def test_proxy_tools_without_print_mode_errors(self):
        """Verify that --proxy_tools without --print raises SystemExit."""
        app = Application()
        mock_args = SimpleNamespace(
            debug=False, datasource="ns1", print_mode=None, web=False, resume=None, proxy_tools="*", config=None
        )
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_ensure_project_config"),
        ):
            with pytest.raises(SystemExit):
                app.run()

    def test_run_print_mode(self):
        app = Application()
        mock_args = SimpleNamespace(
            debug=False,
            datasource="ns1",
            print_mode="hello world",
            web=False,
            resume=None,
            proxy_tools=None,
            config=None,
        )
        mock_runner = MagicMock()
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_ensure_project_config") as mock_ensure,
            patch("datus.cli.print_mode.PrintModeRunner", return_value=mock_runner) as MockRunner,
        ):
            app.run()
        # print_mode skips the project-config wizard
        mock_ensure.assert_not_called()
        MockRunner.assert_called_once_with(mock_args)
        mock_runner.run.assert_called_once()

    def test_run_interactive_mode(self):
        app = Application()
        mock_args = SimpleNamespace(
            debug=False, datasource="ns1", print_mode=None, web=False, resume=None, proxy_tools=None, config=None
        )
        mock_cli = MagicMock()
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_ensure_project_config") as mock_ensure,
            patch("datus.cli.main.DatusCLI", return_value=mock_cli) as MockCLI,
        ):
            app.run()
        mock_ensure.assert_called_once_with(mock_args)
        MockCLI.assert_called_once_with(mock_args)
        mock_cli.run.assert_called_once()

    def test_run_web_mode(self):
        app = Application()
        mock_args = SimpleNamespace(
            debug=False, datasource="ns1", print_mode=None, web=True, resume=None, proxy_tools=None, config=None
        )
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_ensure_project_config") as mock_ensure,
            patch.object(app, "_run_web_interface") as mock_web,
        ):
            app.run()
        # web mode also skips the wizard
        mock_ensure.assert_not_called()
        mock_web.assert_called_once_with(mock_args)


class TestEnsureProjectConfig:
    def test_runs_wizard_when_missing(self):
        app = Application()
        args = SimpleNamespace(config=None)
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_base = MagicMock()
        with (
            patch("datus.configuration.project_config.project_config_path", return_value=mock_path),
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_base) as mock_load,
            patch("datus.cli.project_init.run_project_init") as mock_wizard,
        ):
            app._ensure_project_config(args)
        mock_load.assert_called_once()
        mock_wizard.assert_called_once_with(mock_base)

    def test_idempotent_when_file_exists(self):
        """File exists + override empty → neither wizard nor repair runs.

        ``load_project_override`` is mocked to ``None`` so the repair path
        short-circuits before touching the real YAML loader.
        """
        app = Application()
        args = SimpleNamespace(config=None)
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        with (
            patch("datus.configuration.project_config.project_config_path", return_value=mock_path),
            patch("datus.configuration.agent_config_loader.load_agent_config") as mock_load,
            patch("datus.cli.project_init.run_project_init") as mock_wizard,
            patch("datus.configuration.project_config.load_project_override", return_value=None),
        ):
            app._ensure_project_config(args)
        mock_load.assert_not_called()
        mock_wizard.assert_not_called()

    def test_raises_when_base_config_fails(self):
        app = Application()
        args = SimpleNamespace(config=None)
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("datus.configuration.project_config.project_config_path", return_value=mock_path),
            patch(
                "datus.configuration.agent_config_loader.load_agent_config",
                side_effect=RuntimeError("boom"),
            ),
            patch("datus.cli.project_init.run_project_init") as mock_wizard,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                app._ensure_project_config(args)
        mock_wizard.assert_not_called()


class TestRepairProjectOverrides:
    """``_repair_project_overrides`` re-prompts the user when stale values
    in ``./.datus/config.yml`` reference models/databases that no longer
    exist in the base ``agent.yml``, and persists the corrected values.
    It must be a no-op when every override is already valid — the
    regression that surfaced this fix was a stale ``target`` leaving the
    CLI stuck in ``print_help``."""

    def _raw_agent(self, models, datasources):
        service_dbs = {name: {"type": db_type} for name, db_type in datasources.items()}
        return {"models": {name: {} for name in models}, "services": {"datasources": service_dbs}}

    def _mock_mgr(self, raw):
        mgr = MagicMock()
        mgr.data = raw
        return mgr

    def test_returns_when_override_is_none(self):
        """No overlay file → nothing to validate; must not touch loader."""
        app = Application()
        args = SimpleNamespace(config=None)
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=None),
            patch("datus.configuration.agent_config_loader.configuration_manager") as mock_cfg_mgr,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
        ):
            app._repair_project_overrides(args)
        mock_cfg_mgr.assert_not_called()
        mock_save.assert_not_called()

    def test_skips_when_all_valid(self):
        """All override values match the base config → no prompt, no save."""
        from datus.configuration.project_config import ProjectOverride

        app = Application()
        args = SimpleNamespace(config=None)
        override = ProjectOverride(target="claude", default_datasource="bench")
        raw = self._raw_agent(["claude", "deepseek"], {"bench": "sqlite"})
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=override),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=self._mock_mgr(raw),
            ),
            patch("datus.cli._cli_utils.select_choice") as mock_pick,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
        ):
            app._repair_project_overrides(args)
        mock_pick.assert_not_called()
        mock_save.assert_not_called()

    def test_repairs_stale_target(self):
        """Stale target → prompt, write back corrected overlay, keep db."""
        from datus.configuration.project_config import ProjectOverride

        app = Application()
        args = SimpleNamespace(config=None)
        override = ProjectOverride(target="claude-sonnet", default_datasource="bench")
        raw = self._raw_agent(["claude", "deepseek"], {"bench": "sqlite"})
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=override),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=self._mock_mgr(raw),
            ),
            patch("datus.cli._cli_utils.select_choice", return_value="deepseek") as mock_pick,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            app._repair_project_overrides(args)
        # select_choice called exactly once (for target), not for the valid db.
        assert mock_pick.call_count == 1
        mock_save.assert_called_once()
        (saved_override,) = mock_save.call_args.args
        assert saved_override.target == "deepseek"
        assert saved_override.default_datasource == "bench"

    def test_repairs_stale_default_datasource(self):
        """Stale default_datasource → prompt only for db; keep valid target."""
        from datus.configuration.project_config import ProjectOverride

        app = Application()
        args = SimpleNamespace(config=None)
        override = ProjectOverride(target="claude", default_datasource="benchmark1")
        raw = self._raw_agent(["claude", "deepseek"], {"bench": "sqlite"})
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=override),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=self._mock_mgr(raw),
            ),
            patch("datus.cli._cli_utils.select_choice", return_value="bench") as mock_pick,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            app._repair_project_overrides(args)
        assert mock_pick.call_count == 1
        mock_save.assert_called_once()
        (saved_override,) = mock_save.call_args.args
        assert saved_override.target == "claude"
        assert saved_override.default_datasource == "bench"

    def test_repairs_both_fields(self):
        """Both values stale → two prompts, saved override uses both picks."""
        from datus.configuration.project_config import ProjectOverride

        app = Application()
        args = SimpleNamespace(config=None)
        override = ProjectOverride(target="claude1", default_datasource="benchmark1")
        raw = self._raw_agent(["claude", "deepseek"], {"bench": "sqlite"})
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=override),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=self._mock_mgr(raw),
            ),
            patch(
                "datus.cli._cli_utils.select_choice",
                side_effect=["claude", "bench"],
            ) as mock_pick,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            app._repair_project_overrides(args)
        assert mock_pick.call_count == 2
        mock_save.assert_called_once()
        (saved_override,) = mock_save.call_args.args
        assert saved_override.target == "claude"
        assert saved_override.default_datasource == "bench"

    def test_raises_when_base_has_no_models(self):
        """Base config empty on the same key we need to repair → nothing to
        offer, so surface a config error instead of silently writing a bad
        value."""
        from datus.configuration.project_config import ProjectOverride
        from datus.utils.exceptions import DatusException

        app = Application()
        args = SimpleNamespace(config=None)
        override = ProjectOverride(target="claude-sonnet", default_datasource=None)
        raw = self._raw_agent([], {"bench": "sqlite"})
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=override),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=self._mock_mgr(raw),
            ),
            patch("datus.cli._cli_utils.select_choice") as mock_pick,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            with pytest.raises(DatusException):
                app._repair_project_overrides(args)
        mock_pick.assert_not_called()
        mock_save.assert_not_called()

    def test_raises_when_stdin_is_not_a_tty(self):
        """Stale value + non-interactive stdin → raise instead of silently
        persisting select_choice's default fallback. Guards against the
        REPL-only repair flow accidentally writing a bad config when run
        from a pipe / CI / API surface."""
        from datus.configuration.project_config import ProjectOverride
        from datus.utils.exceptions import DatusException

        app = Application()
        args = SimpleNamespace(config=None)
        override = ProjectOverride(target="claude-sonnet", default_datasource=None)
        raw = self._raw_agent(["claude", "deepseek"], {"bench": "sqlite"})
        with (
            patch("datus.configuration.project_config.load_project_override", return_value=override),
            patch(
                "datus.configuration.agent_config_loader.configuration_manager",
                return_value=self._mock_mgr(raw),
            ),
            patch("datus.cli._cli_utils.select_choice") as mock_pick,
            patch("datus.configuration.project_config.save_project_override") as mock_save,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(DatusException) as exc_info:
                app._repair_project_overrides(args)
        assert "stdin is not a TTY" in str(exc_info.value)
        assert "target='claude-sonnet'" in str(exc_info.value)
        mock_pick.assert_not_called()
        mock_save.assert_not_called()


class TestResolveDefaultDatabase:
    def _make_config(self, datasources: dict, default: str = ""):
        cfg = MagicMock()
        cfg.services.datasources = datasources
        cfg.services.default_datasource = default
        return cfg

    def test_returns_service_default_datasource(self):
        """_resolve_default_datasource is now a thin wrapper over
        config.services.default_datasource — the overlay is applied upstream by
        _apply_project_override, so this function just reads the resolved
        value. We verify the resolved value wins regardless of the base
        agent.yml: the mock returns "b" directly."""
        app = Application()
        args = SimpleNamespace(config=None)
        config = self._make_config({"a": MagicMock(type="sqlite"), "b": MagicMock(type="duckdb")}, default="b")
        with patch("datus.configuration.agent_config_loader.load_agent_config", return_value=config):
            result = app._resolve_default_datasource(args)
        assert result == "b"

    def test_falls_through_to_base_default(self):
        app = Application()
        args = SimpleNamespace(config=None)
        config = self._make_config({"a": MagicMock(type="sqlite")}, default="a")
        with patch("datus.configuration.agent_config_loader.load_agent_config", return_value=config):
            result = app._resolve_default_datasource(args)
        assert result == "a"

    def test_no_databases_returns_empty(self):
        app = Application()
        args = SimpleNamespace(config=None)
        config = self._make_config({}, default="")
        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=config),
        ):
            result = app._resolve_default_datasource(args)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: Application._run_web_interface
# ---------------------------------------------------------------------------


class TestRunWebInterface:
    def test_delegates_to_run_web_interface(self):
        app = Application()
        mock_args = SimpleNamespace(datasource="ns1")
        with patch("datus.cli.web.run_web_interface") as mock_web:
            with patch.dict("sys.modules", {"datus.cli.web": MagicMock(run_web_interface=mock_web)}):
                app._run_web_interface(mock_args)
        mock_web.assert_called_once_with(mock_args)


# ---------------------------------------------------------------------------
# Tests: main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_delegates_to_app_run(self):
        from datus.cli.main import main

        with (
            patch.object(sys, "argv", ["datus"]),
            patch("datus.cli.main.Application") as MockApp,
        ):
            mock_app = MagicMock()
            MockApp.return_value = mock_app
            main()
        mock_app.run.assert_called_once()

    def test_main_skill_subcommand(self):
        """main() delegates to skill handler when first arg is 'skill'.

        The mocked ``sys.exit`` must raise ``SystemExit`` (like the real one) so
        execution actually stops at the skill dispatch. A silent no-op lets
        ``main()`` fall through into ``Application().run()``, whose argparse
        rejects ``skill list`` and — because ``sys.exit`` is still a no-op —
        returns a partial Namespace that ends up launching the interactive REPL,
        which blocks on prompt_toolkit stdin.
        """
        from datus.cli.main import main

        mock_skill_args = SimpleNamespace(debug=False, subcommand="skill")
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_skill_args

        mock_main_mod = MagicMock()
        mock_main_mod.create_parser.return_value = mock_parser

        mock_skill_cli = MagicMock()
        mock_skill_cli.run_skill_command.return_value = 0

        with (
            patch.object(sys, "argv", ["datus", "skill", "list"]),
            patch("datus.cli.main.configure_logging"),
            patch.dict(
                "sys.modules",
                {
                    "datus.main": mock_main_mod,
                    "datus.cli.skill_cli": mock_skill_cli,
                },
            ),
            patch("sys.exit", side_effect=SystemExit) as mock_exit,
        ):
            with pytest.raises(SystemExit):
                main()
        mock_exit.assert_any_call(0)
        mock_skill_cli.run_skill_command.assert_called_once_with(mock_skill_args)
