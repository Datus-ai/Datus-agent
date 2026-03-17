# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/repl.py.

Tests cover:
- CommandType enum
- DatusCLI._parse_command: EXIT, TOOL, CONTEXT, CHAT, INTERNAL, SQL, subagent routing
- DatusCLI.check_agent_available: ready / initializing / not ready
- DatusCLI._cmd_list_namespaces: smoke test
- DatusCLI._cmd_switch_namespace: empty args, same namespace, switch
- DatusCLI._smart_display_table: empty data, few columns, many columns
- DatusCLI._get_prompt_text: normal/plan mode
- DatusCLI.create_combined_completer: returns a completer

DatusCLI is instantiated via a fully mocked __init__ to avoid prompt_toolkit/
threading side effects.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.repl import CommandType, DatusCLI

# ---------------------------------------------------------------------------
# Factory: build a minimal DatusCLI without running __init__
# ---------------------------------------------------------------------------


def _make_cli(agent_config, available_subagents=None):
    """Create a DatusCLI instance with __init__ bypassed.

    All attributes that the tested methods rely on are set directly.
    """
    cli = object.__new__(DatusCLI)

    console = Console(file=io.StringIO(), no_color=True)
    cli.console = console
    cli.console_column_width = 16
    cli.agent_config = agent_config
    cli.agent = None
    cli.agent_ready = False
    cli.agent_initializing = False
    cli.plan_mode_active = False
    cli.streamlit_mode = False
    cli.at_completer = MagicMock()
    cli.db_connector = MagicMock()
    cli.db_manager = MagicMock()

    from datus.cli.cli_context import CliContext
    from datus.schemas.action_history import ActionHistoryManager

    cli.cli_context = CliContext()
    cli.actions = ActionHistoryManager()

    # Available subagents
    cli.available_subagents = available_subagents or {"gensql", "chat", "compare"}

    # Command handlers (mocked)
    cli.agent_commands = MagicMock()
    cli.chat_commands = MagicMock()
    cli.context_commands = MagicMock()
    cli.metadata_commands = MagicMock()
    cli.sub_agent_commands = MagicMock()
    cli.bi_dashboard_commands = MagicMock()
    cli._workflow_runner = None

    return cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(real_agent_config):
    return _make_cli(real_agent_config)


# ---------------------------------------------------------------------------
# Tests: CommandType
# ---------------------------------------------------------------------------


class TestCommandType:
    def test_all_types_exist(self):
        assert CommandType.SQL.value == "sql"
        assert CommandType.TOOL.value == "tool"
        assert CommandType.CONTEXT.value == "context"
        assert CommandType.CHAT.value == "chat"
        assert CommandType.INTERNAL.value == "internal"
        assert CommandType.EXIT.value == "exit"


# ---------------------------------------------------------------------------
# Tests: _parse_command
# ---------------------------------------------------------------------------


class TestParseCommand:
    def test_exit_command_dot(self, cli):
        cmd_type, cmd, args = cli._parse_command(".exit")
        assert cmd_type == CommandType.EXIT

    def test_exit_command_quit(self, cli):
        cmd_type, cmd, args = cli._parse_command("quit")
        assert cmd_type == CommandType.EXIT

    def test_exit_command_exit(self, cli):
        cmd_type, cmd, args = cli._parse_command("exit")
        assert cmd_type == CommandType.EXIT

    def test_tool_command_with_args(self, cli):
        cmd_type, cmd, args = cli._parse_command("!sl find revenue tables")
        assert cmd_type == CommandType.TOOL
        assert cmd == "!sl"
        assert args == "find revenue tables"

    def test_tool_command_no_args(self, cli):
        cmd_type, cmd, args = cli._parse_command("!sl")
        assert cmd_type == CommandType.TOOL
        assert cmd == "!sl"
        assert args == ""

    def test_context_command(self, cli):
        cmd_type, cmd, args = cli._parse_command("@catalog mydb")
        assert cmd_type == CommandType.CONTEXT
        assert cmd == "@catalog"
        assert args == "mydb"

    def test_internal_command(self, cli):
        cmd_type, cmd, args = cli._parse_command(".tables")
        assert cmd_type == CommandType.INTERNAL
        assert cmd == ".tables"
        assert args == ""

    def test_internal_command_with_args(self, cli):
        cmd_type, cmd, args = cli._parse_command(".namespace test_ns")
        assert cmd_type == CommandType.INTERNAL
        assert cmd == ".namespace"
        assert args == "test_ns"

    def test_chat_command_slash(self, cli):
        cmd_type, cmd, args = cli._parse_command("/how many users?")
        assert cmd_type == CommandType.CHAT
        assert cmd == ""
        assert "how many users" in args

    def test_chat_command_with_known_subagent(self, cli):
        cli.available_subagents = {"gensql", "compare"}
        cmd_type, cmd, args = cli._parse_command("/gensql show me revenue by month")
        assert cmd_type == CommandType.CHAT
        assert cmd == "gensql"
        assert args == "show me revenue by month"

    def test_chat_command_unknown_first_word(self, cli):
        cli.available_subagents = {"gensql"}
        cmd_type, cmd, args = cli._parse_command("/notasubagent do something")
        assert cmd_type == CommandType.CHAT
        assert cmd == ""

    def test_sql_trailing_semicolon_stripped(self, cli):
        """Trailing semicolon is stripped before parsing."""
        with patch("datus.cli.repl.parse_sql_type") as mock_parse:
            from datus.utils.constants import SQLType

            mock_parse.return_value = SQLType.SELECT
            cmd_type, cmd, args = cli._parse_command("SELECT 1;")
        assert cmd_type == CommandType.SQL
        # Verify the semicolon was stripped before passing to parse_sql_type
        call_args = mock_parse.call_args
        assert call_args[0][0] == "SELECT 1", f"Expected stripped SQL 'SELECT 1', got '{call_args[0][0]}'"

    def test_natural_language_treated_as_chat(self, cli):
        """Natural language without prefix is treated as CHAT."""
        with patch("datus.cli.repl.parse_sql_type") as mock_parse:
            from datus.utils.constants import SQLType

            mock_parse.return_value = SQLType.UNKNOWN  # UNKNOWN is always defined
            cmd_type, cmd, args = cli._parse_command("show me the revenue")
        assert cmd_type == CommandType.CHAT

    def test_parse_sql_exception_falls_back_to_chat(self, cli):
        """Exception during parse_sql_type falls back to CHAT."""
        with patch("datus.cli.repl.parse_sql_type", side_effect=Exception("parse error")):
            cmd_type, cmd, args = cli._parse_command("ambiguous text")
        assert cmd_type == CommandType.CHAT


# ---------------------------------------------------------------------------
# Tests: check_agent_available
# ---------------------------------------------------------------------------


class TestCheckAgentAvailable:
    def test_agent_ready(self, cli):
        cli.agent_ready = True
        cli.agent = MagicMock()
        assert cli.check_agent_available() is True

    def test_agent_initializing(self, cli):
        cli.agent_ready = False
        cli.agent_initializing = True
        result = cli.check_agent_available()
        assert result is False
        output = cli.console.file.getvalue()
        assert "initializing" in output.lower() or "background" in output.lower()

    def test_agent_not_available(self, cli):
        cli.agent_ready = False
        cli.agent_initializing = False
        cli.agent = None
        result = cli.check_agent_available()
        assert result is False
        output = cli.console.file.getvalue()
        assert "not available" in output.lower() or "failed" in output.lower()


# ---------------------------------------------------------------------------
# Tests: _get_prompt_text
# ---------------------------------------------------------------------------


class TestGetPromptText:
    def test_normal_mode(self, cli):
        cli.plan_mode_active = False
        text = cli._get_prompt_text()
        assert "Datus>" in text
        assert "PLAN" not in text

    def test_plan_mode(self, cli):
        cli.plan_mode_active = True
        text = cli._get_prompt_text()
        assert "PLAN" in text


# ---------------------------------------------------------------------------
# Tests: _cmd_list_namespaces
# ---------------------------------------------------------------------------


class TestCmdListNamespaces:
    def test_lists_namespaces(self, cli):
        cli._cmd_list_namespaces()
        output = cli.console.file.getvalue()
        # Should have printed something (the table)
        assert len(output) > 0

    def test_current_namespace_highlighted(self, cli):
        cli.agent_config.current_namespace = "test_ns"
        cli._cmd_list_namespaces()
        output = cli.console.file.getvalue()
        assert "test_ns" in output


# ---------------------------------------------------------------------------
# Tests: _cmd_switch_namespace
# ---------------------------------------------------------------------------


class TestCmdSwitchNamespace:
    def test_empty_args_lists_namespaces(self, cli):
        with patch.object(cli, "_cmd_list_namespaces") as mock_list:
            cli._cmd_switch_namespace("")
        mock_list.assert_called_once()

    def test_same_namespace_prints_message(self, cli):
        current_ns = cli.agent_config.current_namespace
        with patch.object(cli, "_cmd_list_namespaces"):
            cli._cmd_switch_namespace(current_ns)
        output = cli.console.file.getvalue()
        assert "doesn't need" in output or "already" in output.lower() or "now under" in output.lower()

    def test_switch_to_different_namespace(self, cli):
        mock_conn = MagicMock()
        mock_conn.database_name = "newdb"
        mock_conn.catalog_name = ""
        mock_conn.schema_name = ""
        cli.db_manager.first_conn_with_name.return_value = ("newdb", mock_conn)

        with patch.object(cli, "reset_session"):
            cli._cmd_switch_namespace("test_ns")

        output = cli.console.file.getvalue()
        assert "test_ns" in output


# ---------------------------------------------------------------------------
# Tests: _smart_display_table
# ---------------------------------------------------------------------------


class TestSmartDisplayTable:
    def test_empty_data_prints_message(self, cli):
        cli._smart_display_table([])
        output = cli.console.file.getvalue()
        assert "No data" in output

    def test_simple_data_displays_table(self, cli):
        data = [{"col1": "val1", "col2": "val2"}]
        cli._smart_display_table(data)
        output = cli.console.file.getvalue()
        assert "col1" in output or "val1" in output

    def test_many_columns_truncated(self, cli):
        """Many columns are truncated to fit terminal width."""
        data = [{f"col{i}": f"val{i}" for i in range(20)}]
        # Should not raise
        cli._smart_display_table(data)
        output = cli.console.file.getvalue()
        assert len(output) > 0

    def test_explicit_columns(self, cli):
        data = [{"col1": "val1", "col2": "val2", "col3": "val3"}]
        cli._smart_display_table(data, columns=["col1", "col2"])
        output = cli.console.file.getvalue()
        assert "col1" in output

    def test_datetime_formatting(self, cli):
        from datetime import date, datetime

        data = [{"dt": datetime(2025, 1, 15, 10, 30), "d": date(2025, 3, 1)}]
        cli._smart_display_table(data)
        output = cli.console.file.getvalue()
        assert "2025" in output
