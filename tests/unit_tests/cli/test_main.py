# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/main.py — ArgumentParser, Application, main().

All external dependencies (DatusCLI, run_web_interface, configure_logging) are mocked.
"""

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
        assert ap.parser is not None

    def test_parse_args_defaults(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus"]):
            args = ap.parse_args()
        assert args.db_type == "sqlite"
        assert args.debug is False
        assert args.no_color is False
        assert args.namespace is None
        assert args.database == ""
        assert args.print_mode is None
        assert args.web is False
        assert args.resume is None

    def test_parse_args_debug_flag(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--debug", "--namespace", "ns1"]):
            args = ap.parse_args()
        assert args.debug is True
        assert args.namespace == "ns1"

    def test_parse_args_print(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--namespace", "ns1", "--print", "hello"]):
            args = ap.parse_args()
        assert args.print_mode == "hello"

    def test_parse_args_print_short(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--namespace", "ns1", "-p", "hello"]):
            args = ap.parse_args()
        assert args.print_mode == "hello"

    def test_parse_args_resume(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--namespace", "ns1", "--print", "hello", "--resume", "sess_123"]):
            args = ap.parse_args()
        assert args.resume == "sess_123"

    def test_parse_args_web(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--namespace", "ns1", "--web"]):
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
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace=None, print_mode=None, web=False, resume=None)
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app.arg_parser.parser, "print_help") as mock_help,
        ):
            app.run()
        mock_help.assert_called_once()

    def test_resume_without_print_mode_errors(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace="ns1", print_mode=None, web=False, resume="sess_123")
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
        ):
            with pytest.raises(SystemExit):
                app.run()

    def test_run_print_mode(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace="ns1", print_mode="hello world", web=False, resume=None)
        mock_runner = MagicMock()
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch("datus.cli.print_mode.PrintModeRunner", return_value=mock_runner) as MockRunner,
        ):
            app.run()
        MockRunner.assert_called_once_with(mock_args)
        mock_runner.run.assert_called_once()

    def test_run_interactive_mode(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace="ns1", print_mode=None, web=False, resume=None)
        mock_cli = MagicMock()
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch("datus.cli.main.DatusCLI", return_value=mock_cli) as MockCLI,
        ):
            app.run()
        MockCLI.assert_called_once_with(mock_args)
        mock_cli.run.assert_called_once()

    def test_run_web_mode(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace="ns1", print_mode=None, web=True, resume=None)
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app, "_run_web_interface") as mock_web,
        ):
            app.run()
        mock_web.assert_called_once_with(mock_args)


# ---------------------------------------------------------------------------
# Tests: Application._run_web_interface
# ---------------------------------------------------------------------------


class TestRunWebInterface:
    def test_delegates_to_run_web_interface(self):
        app = Application()
        mock_args = SimpleNamespace(namespace="ns1")
        with patch("datus.cli.web.run_web_interface") as mock_web:
            with patch.dict("sys.modules", {"datus.cli.web": MagicMock(run_web_interface=mock_web)}):
                app._run_web_interface(mock_args)
        # Just verify no exceptions are raised — the method delegates to lazy import


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
        """main() delegates to skill handler when first arg is 'skill'."""
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
            patch("sys.exit") as mock_exit,
        ):
            main()
        mock_exit.assert_any_call(0)
