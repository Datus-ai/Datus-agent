# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for web chatbot components.

Tests ChatExecutor and ConfigManager with real DatusCLI + real LLM,
validating the web chatbot's core logic without Streamlit UI.
"""

from unittest.mock import patch

import pytest

from datus.cli.web.chat_executor import ChatExecutor
from datus.cli.web.config_manager import ConfigManager, create_cli_args, get_available_namespaces
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

TESTS_CONF = "tests/conf/agent.yml"


# =============================================================================
# ConfigManager
# =============================================================================


@pytest.mark.nightly
class TestConfigManagerIntegration:
    """Integration tests for ConfigManager functions with real config."""

    def test_get_available_namespaces(self):
        """W1-01: get_available_namespaces returns namespaces from real config."""
        namespaces = get_available_namespaces(TESTS_CONF)

        assert isinstance(namespaces, list)
        assert len(namespaces) > 0, "Should find at least one namespace in agent.yml"
        assert "bird_school" in namespaces, f"bird_school should be in namespaces, got: {namespaces}"

        logger.info(f"Found {len(namespaces)} namespaces: {namespaces}")

    def test_get_available_namespaces_invalid_path(self):
        """W1-02: get_available_namespaces handles invalid config path gracefully."""
        namespaces = get_available_namespaces("/nonexistent/path.yml")
        assert namespaces == []

    def test_create_cli_args(self):
        """W1-03: create_cli_args generates correct argument namespace."""
        args = create_cli_args(config_path=TESTS_CONF, namespace="ssb_sqlite")
        assert args.namespace == "ssb_sqlite"
        assert args.non_interactive is True
        assert hasattr(args, "config")
        assert hasattr(args, "storage_path")

    def test_setup_config_and_models(self):
        """W1-04: ConfigManager.setup_config returns DatusCLI with models available."""
        cm = ConfigManager()
        cli = cm.setup_config(config_path=TESTS_CONF, namespace="ssb_sqlite")

        assert cli is not None, "setup_config should return a DatusCLI instance"
        assert cli.streamlit_mode is True, "streamlit_mode should be True"
        assert cm.cli is cli

        models = cm.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0, "Should have at least one model configured"


# =============================================================================
# ChatExecutor
# =============================================================================


@pytest.mark.nightly
class TestChatExecutorIntegration:
    """Integration tests for ChatExecutor with real DatusCLI."""

    def test_executor_no_cli(self):
        """W2-01: ChatExecutor yields error when CLI is not initialized."""
        executor = ChatExecutor()
        results = list(executor.execute_chat_stream("test query", None))

        assert len(results) == 1
        assert isinstance(results[0], str)
        assert "Error" in results[0]

    def test_execute_chat_stream_produces_actions(self, mock_args):
        """W2-02: ChatExecutor produces actions with real DatusCLI + real LLM."""
        from datus.cli.repl import DatusCLI
        from tests.integration.conftest import wait_for_agent

        with (
            patch("datus.cli.repl.PromptSession.prompt") as mock_prompt,
            patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal,
            patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
        ):
            mock_prompt.side_effect = [EOFError]
            mock_internal.side_effect = ["n"]
            at_data.return_value = [], [], []
            cli = DatusCLI(args=mock_args)
            wait_for_agent(cli)

        executor = ChatExecutor()
        actions = []
        for item in executor.execute_chat_stream("How many schools are in Fresno county?", cli):
            if isinstance(item, ActionHistory):
                actions.append(item)
                logger.info(f"Action: role={item.role}, status={item.status}, type={item.action_type}")
            elif isinstance(item, str):
                logger.info(f"Stream message: {item}")

        assert len(actions) >= 1, f"Should produce at least 1 action, got {len(actions)}"

        # Last stored actions should be accessible
        assert executor.last_actions is not None
        assert len(executor.last_actions) >= 1

    def test_format_action_for_stream(self):
        """W2-04: format_action_for_stream formats different action types."""
        executor = ChatExecutor()

        # Tool action
        tool_action = ActionHistory(
            action_id="t1",
            role=ActionRole.TOOL,
            action_type="describe_table",
            status=ActionStatus.SUCCESS,
            input={"function_name": "describe_table", "table_name": "schools"},
            output={"result": "49 columns"},
        )
        formatted = executor.format_action_for_stream(tool_action)
        assert "✓" in formatted
        assert "describe_table" in formatted

        # Assistant thinking action
        thinking_action = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            action_type="thinking",
            status=ActionStatus.PROCESSING,
            messages="Analyzing the database schema to find school information.",
        )
        formatted = executor.format_action_for_stream(thinking_action)
        assert "💭" in formatted
        assert "Analyzing" in formatted

        # Empty action
        empty_action = ActionHistory(
            action_id="e1",
            role=ActionRole.USER,
            action_type="query",
            status=ActionStatus.PROCESSING,
        )
        formatted = executor.format_action_for_stream(empty_action)
        assert formatted == ""

    def test_extract_sql_and_response_empty_actions(self):
        """W2-05: extract_sql_and_response handles empty actions."""
        executor = ChatExecutor()
        sql, response = executor.extract_sql_and_response([], None)
        assert sql is None
        assert response is None

    def test_extract_sql_and_response_no_success(self):
        """W2-06: extract_sql_and_response handles non-success final action."""
        executor = ChatExecutor()
        actions = [
            ActionHistory(
                action_id="f1",
                role=ActionRole.ASSISTANT,
                action_type="response",
                status=ActionStatus.FAILED,
                output={"error": "Something went wrong"},
            )
        ]
        sql, response = executor.extract_sql_and_response(actions, None)
        assert sql is None
        assert response is None

    def test_extract_sql_and_response_success(self):
        """W2-07: extract_sql_and_response extracts SQL from successful action."""
        executor = ChatExecutor()
        action = ActionHistory(
            action_id="s1",
            role=ActionRole.TOOL,
            action_type="execute_sql",
            status=ActionStatus.SUCCESS,
            input={"function_name": "execute_sql", "query": "SELECT COUNT(*) FROM customer"},
            output={"sql": "SELECT COUNT(*) FROM customer", "response": "There are 30000 customers."},
        )
        sql, response = executor.extract_sql_and_response([action], cli=None)
        assert sql == "SELECT COUNT(*) FROM customer"
        assert response is not None
        assert len(response) > 0

    def test_format_action_tool_processing(self):
        """W2-08: format_action_for_stream formats processing tool with spinner."""
        executor = ChatExecutor()
        action = ActionHistory(
            action_id="p1",
            role=ActionRole.TOOL,
            action_type="describe_table",
            status=ActionStatus.PROCESSING,
            input={"function_name": "describe_table", "table": "customer"},
        )
        formatted = executor.format_action_for_stream(action)
        assert isinstance(formatted, str)
        assert "describe_table" in formatted

    def test_format_action_truncates_long_message(self):
        """W2-09: format_action_for_stream truncates messages longer than 100 chars."""
        executor = ChatExecutor()
        long_message = "A" * 200
        action = ActionHistory(
            action_id="l1",
            role=ActionRole.ASSISTANT,
            action_type="thinking",
            status=ActionStatus.SUCCESS,
            messages=long_message,
        )
        formatted = executor.format_action_for_stream(action)
        assert isinstance(formatted, str)
        assert "..." in formatted
        assert len(formatted) < 200
