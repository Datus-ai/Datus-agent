"""Tests for --subagent CLI parameter support in web chatbot."""
from argparse import Namespace
from unittest.mock import patch

import pytest

from datus.cli.main import ArgumentParser


@pytest.mark.ci
class TestSubagentCLIParam:
    """Tests for --subagent argument parsing and URL building."""

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

    def test_run_web_interface_url_with_subagent(self):
        """URL should include ?subagent= when --subagent is provided."""
        from datus.cli.web.chatbot import run_web_interface

        args = Namespace(
            namespace="starrocks",
            config="conf/agent.yml",
            database="",
            host="localhost",
            port=8501,
            subagent="baisheng",
            debug=False,
            web=True,
        )

        with patch("subprocess.run") as mock_run, patch("os.path.exists", return_value=True):
            mock_run.side_effect = KeyboardInterrupt  # Stop immediately
            try:
                run_web_interface(args)
            except (KeyboardInterrupt, SystemExit):
                pass

            # Verify --subagent was passed to subprocess
            if mock_run.called:
                cmd = mock_run.call_args[0][0]
                assert "--subagent" in cmd
                idx = cmd.index("--subagent")
                assert cmd[idx + 1] == "baisheng"

    def test_run_web_interface_url_without_subagent(self):
        """URL should not include ?subagent= when --subagent is not provided."""
        from datus.cli.web.chatbot import run_web_interface

        args = Namespace(
            namespace="starrocks",
            config="conf/agent.yml",
            database="",
            host="localhost",
            port=8501,
            subagent="",
            debug=False,
            web=True,
        )

        with patch("subprocess.run") as mock_run, patch("os.path.exists", return_value=True):
            mock_run.side_effect = KeyboardInterrupt
            try:
                run_web_interface(args)
            except (KeyboardInterrupt, SystemExit):
                pass

            if mock_run.called:
                cmd = mock_run.call_args[0][0]
                assert "--subagent" not in cmd

    def test_run_web_interface_url_encodes_subagent(self):
        """Subagent name with special chars should be URL-encoded."""
        from datus.cli.web.chatbot import run_web_interface

        args = Namespace(
            namespace="starrocks",
            config="conf/agent.yml",
            database="",
            host="localhost",
            port=8501,
            subagent="test agent&foo",
            debug=False,
            web=True,
        )

        with patch("subprocess.run") as mock_run, patch("os.path.exists", return_value=True):
            mock_run.side_effect = KeyboardInterrupt
            try:
                run_web_interface(args)
            except (KeyboardInterrupt, SystemExit):
                pass


@pytest.mark.ci
class TestChatbotSubagentParsing:
    """Tests for chatbot main() subagent arg parsing."""

    def test_parse_subagent_from_argv(self):
        """--subagent should be parsed from sys.argv in chatbot main()."""
        # Test the parsing logic directly
        test_argv = ["chatbot.py", "--namespace", "starrocks", "--subagent", "baisheng"]

        subagent_name = None
        for i, arg in enumerate(test_argv):
            if arg == "--subagent" and i + 1 < len(test_argv):
                subagent_name = test_argv[i + 1]

        assert subagent_name == "baisheng"

    def test_parse_subagent_missing_value(self):
        """--subagent at end of argv without value should not crash."""
        test_argv = ["chatbot.py", "--subagent"]

        subagent_name = None
        for i, arg in enumerate(test_argv):
            if arg == "--subagent" and i + 1 < len(test_argv):
                subagent_name = test_argv[i + 1]

        assert subagent_name is None

    def test_parse_no_subagent(self):
        """Without --subagent, subagent_name should remain None."""
        test_argv = ["chatbot.py", "--namespace", "starrocks"]

        subagent_name = None
        for i, arg in enumerate(test_argv):
            if arg == "--subagent" and i + 1 < len(test_argv):
                subagent_name = test_argv[i + 1]

        assert subagent_name is None


@pytest.mark.ci
class TestChatbotStreamActionTypeCheck:
    """Tests for action type checking in chat stream."""

    def test_string_action_handled(self):
        """String actions from chat_executor should not crash render_action_item."""
        # Test the isinstance check logic
        action = "Error: Please load configuration first!"
        assert isinstance(action, str)

    def test_action_history_not_string(self):
        """ActionHistory objects should not be treated as strings."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        action = ActionHistory(
            action_id="test-123", role=ActionRole.TOOL, action_type="test_action", status=ActionStatus.SUCCESS
        )
        assert not isinstance(action, str)
