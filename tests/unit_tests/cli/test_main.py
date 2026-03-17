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
        assert args.prompt is None
        assert args.web is False

    def test_parse_args_debug_flag(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--debug", "--namespace", "ns1"]):
            args = ap.parse_args()
        assert args.debug is True
        assert args.namespace == "ns1"

    def test_parse_args_prompt(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--namespace", "ns1", "--prompt", "hello"]):
            args = ap.parse_args()
        assert args.prompt == "hello"

    def test_parse_args_web(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--namespace", "ns1", "--web"]):
            args = ap.parse_args()
        assert args.web is True

    def test_prompt_and_web_are_mutually_exclusive(self):
        ap = ArgumentParser()
        with patch.object(sys, "argv", ["datus", "--web", "--prompt", "hello"]):
            with pytest.raises(SystemExit):
                ap.parse_args()


# ---------------------------------------------------------------------------
# Tests: Application.run
# ---------------------------------------------------------------------------


class TestApplicationRun:
    def test_run_no_namespace_prints_help(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace=None, prompt=None, web=False)
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch.object(app.arg_parser.parser, "print_help") as mock_help,
        ):
            app.run()
        mock_help.assert_called_once()

    def test_run_prompt_mode(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace="ns1", prompt="hello world", web=False)
        mock_cli = MagicMock()
        with (
            patch.object(app.arg_parser, "parse_args", return_value=mock_args),
            patch("datus.cli.main.configure_logging"),
            patch("datus.cli.main.DatusCLI", return_value=mock_cli) as MockCLI,
        ):
            app.run()
        MockCLI.assert_called_once_with(mock_args, interactive=False)
        mock_cli.run_prompt.assert_called_once_with("hello world")

    def test_run_interactive_mode(self):
        app = Application()
        mock_args = SimpleNamespace(debug=False, namespace="ns1", prompt=None, web=False)
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
        mock_args = SimpleNamespace(debug=False, namespace="ns1", prompt=None, web=True)
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
