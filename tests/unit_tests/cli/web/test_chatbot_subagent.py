"""Tests for --subagent CLI parameter support in web chatbot."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.main import ArgumentParser


@pytest.mark.ci
class TestSubagentCLIParam:
    """Tests for --subagent argument parsing."""

    def test_argument_parser_has_subagent(self):
        """ArgumentParser should accept --subagent parameter."""
        parser = ArgumentParser()
        args = parser.parser.parse_args(["--subagent", "baisheng"])
        assert args.subagent == "baisheng"

    def test_argument_parser_subagent_default(self):
        """--subagent should default to empty string."""
        parser = ArgumentParser()
        args = parser.parser.parse_args([])
        assert args.subagent == ""

    def test_argument_parser_has_chatbot_dist(self):
        """ArgumentParser should accept --chatbot-dist parameter."""
        parser = ArgumentParser()
        args = parser.parser.parse_args(["--chatbot-dist", "/some/path"])
        assert args.chatbot_dist == "/some/path"

    def test_run_web_interface_creates_app(self):
        """run_web_interface should create a FastAPI app and start uvicorn."""
        from datus.cli.web.chatbot import run_web_interface

        args = Namespace(
            config="conf/agent.yml",
            database="starrocks",
            host="localhost",
            port=8501,
            subagent="baisheng",
            debug=False,
            web=True,
            chatbot_dist=None,
            session_scope=None,
        )

        with (
            patch("datus.cli.web.chatbot.create_web_app") as mock_create,
            patch("datus.cli.web.chatbot.uvicorn") as mock_uvicorn,
            patch("datus.cli.web.chatbot._schedule_browser_open"),
            patch("datus.cli.web.config_manager.get_home_from_config", return_value="~/.datus"),
            patch("datus.utils.path_manager.set_current_path_manager"),
        ):
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            mock_uvicorn.run.side_effect = KeyboardInterrupt

            try:
                run_web_interface(args)
            except KeyboardInterrupt:
                pass

            mock_create.assert_called_once_with(args)
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[1]["host"] == "localhost"
            assert call_kwargs[1]["port"] == 8501


@pytest.mark.ci
class TestSubagentParsing:
    """Tests for subagent arg parsing from argv."""

    def test_parse_subagent_from_argv(self):
        """--subagent should be parsed from ArgumentParser."""
        parser = ArgumentParser()
        args = parser.parser.parse_args(["--subagent", "baisheng"])
        assert args.subagent == "baisheng"

    def test_parse_no_subagent(self):
        """Without --subagent, subagent should be empty string."""
        parser = ArgumentParser()
        args = parser.parser.parse_args([])
        assert args.subagent == ""


@pytest.mark.ci
class TestCreateWebApp:
    """Tests for create_web_app function."""

    def test_creates_fastapi_app(self):
        """create_web_app should return a FastAPI app with chatbot route."""
        from datus.cli.web.chatbot import create_web_app

        args = Namespace(
            database="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist="/nonexistent/path",
            session_scope=None,
        )

        with patch("datus.cli.web.chatbot.create_app") as mock_create_app:
            from fastapi import FastAPI

            mock_app = FastAPI()
            mock_create_app.return_value = mock_app

            app = create_web_app(args)
            assert app is mock_app
            mock_create_app.assert_called_once()

    def test_build_agent_args(self):
        """_build_agent_args should bridge CLI args to API args."""
        from datus.cli.web.chatbot import _build_agent_args

        args = Namespace(
            database="myns",
            config="conf/agent.yml",
            debug=True,
        )

        agent_args = _build_agent_args(args)
        assert agent_args.namespace == "myns"
        assert agent_args.config == "conf/agent.yml"
        assert agent_args.source == "web"
        assert agent_args.interactive is True
        assert agent_args.log_level == "DEBUG"
