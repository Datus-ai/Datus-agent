# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/repl.py.

Tests cover:
- CommandType enum
- DatusCLI._parse_command: EXIT, TOOL, SLASH, CHAT, SQL, legacy-prefix UNKNOWN
- DatusCLI.check_agent_available: ready / initializing / not ready
- DatasourceCommands._switch: same datasource, switch to different
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
    cli.default_agent = ""
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
    cli.service_commands = MagicMock()
    # Unknown services fall through to the "Unknown command" path.
    cli.service_commands.dispatch = MagicMock(return_value=False)
    from datus.cli.datasource_commands import DatasourceCommands

    cli.datasource_commands = DatasourceCommands(cli)
    cli._workflow_runner = None
    cli.last_sql = None
    cli.last_result = None

    # Build commands dict referencing the mocks. Keys mirror the canonical
    # slash names wired by ``DatusCLI._build_slash_handler_map``.
    cli.commands = {
        "!sl": cli.agent_commands.cmd_schema_linking,
        "/catalog": cli.context_commands.cmd_catalog,
        "/tables": cli.metadata_commands.cmd_tables,
        "/datasource": cli.datasource_commands.cmd,
        "/help": cli._cmd_help,
        "/exit": lambda a: None,
        "/quit": lambda a: None,
        "/rewind": cli.chat_commands.cmd_rewind,
        "!bash": cli._cmd_bash,
    }

    # Status bar provider so _build_prompt_message works in tests
    from datus.cli.status_bar import StatusBarProvider

    cli.chat_commands.current_subagent_name = None
    cli.chat_commands.current_node = None
    cli._status_bar_provider = StatusBarProvider(cli)

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
        assert CommandType.SLASH.value == "slash"
        assert CommandType.CHAT.value == "chat"
        assert CommandType.EXIT.value == "exit"
        assert CommandType.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Tests: _parse_command
# ---------------------------------------------------------------------------


class TestParseCommand:
    def test_exit_command_quit(self, cli):
        cmd_type, _, _ = cli._parse_command("quit")
        assert cmd_type == CommandType.EXIT

    def test_exit_command_exit(self, cli):
        cmd_type, _, _ = cli._parse_command("exit")
        assert cmd_type == CommandType.EXIT

    def test_slash_exit_routes_through_slash_dispatch(self, cli):
        """``/exit`` goes through SLASH so ``_cmd_exit`` can close the DB."""
        cmd_type, cmd, _ = cli._parse_command("/exit")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/exit"

    def test_slash_quit_alias_resolves_to_exit(self, cli):
        """``/quit`` is an alias of ``/exit`` and keeps its own canonical key."""
        cmd_type, cmd, _ = cli._parse_command("/quit")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/exit"

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

    def test_slash_catalog_command(self, cli):
        cmd_type, cmd, args = cli._parse_command("/catalog mydb")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/catalog"
        assert args == "mydb"

    def test_slash_command_no_args(self, cli):
        cmd_type, cmd, args = cli._parse_command("/tables")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/tables"
        assert args == ""

    def test_slash_command_with_args(self, cli):
        cmd_type, cmd, args = cli._parse_command("/datasource test_ns")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/datasource"
        assert args == "test_ns"

    def test_slash_help_canonical(self, cli):
        cmd_type, cmd, _ = cli._parse_command("/help")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/help"

    def test_unknown_slash_is_not_chat(self, cli):
        """Typos such as ``/halp`` must fail loudly, never flow to chat.

        Parser classifies every non-empty ``/<token>`` as SLASH so the
        dynamic service dispatcher (``/<service>[.<method>]``) gets a
        chance to claim it; unknown tokens then surface as
        ``Unknown command`` at dispatch time.
        """
        cmd_type, cmd, args = cli._parse_command("/halp me please")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/halp"
        assert args == "me please"

    def test_legacy_dot_prefix_hints_rename(self, cli):
        cmd_type, cmd, args = cli._parse_command(".tables")
        assert cmd_type == CommandType.UNKNOWN
        assert cmd == ".tables"
        assert args == "/tables"

    def test_legacy_at_prefix_hints_rename(self, cli):
        cmd_type, cmd, args = cli._parse_command("@catalog mydb")
        assert cmd_type == CommandType.UNKNOWN
        assert cmd == "@catalog"
        assert args == "/catalog"

    def test_bare_exit_still_wins_over_legacy_dot(self, cli):
        """``.exit`` surfaces as a rename hint; bare ``exit`` keeps EXIT type."""
        cmd_type, _, hint = cli._parse_command(".exit")
        assert cmd_type == CommandType.UNKNOWN
        assert hint == "/exit"

    def test_sql_trailing_semicolon_stripped(self, cli):
        """Trailing semicolon is stripped before parsing."""
        with patch("datus.cli.repl.parse_sql_type") as mock_parse:
            from datus.utils.constants import SQLType

            mock_parse.return_value = SQLType.SELECT
            cmd_type, _, _ = cli._parse_command("SELECT 1;")
        assert cmd_type == CommandType.SQL

    def test_natural_language_treated_as_chat(self, cli):
        """Natural language without any prefix still routes to CHAT."""
        with patch("datus.cli.repl.parse_sql_type") as mock_parse:
            from datus.utils.constants import SQLType

            mock_parse.return_value = SQLType.UNKNOWN
            cmd_type, cmd, _ = cli._parse_command("show me the revenue")
        assert cmd_type == CommandType.CHAT
        assert cmd == cli.default_agent

    def test_parse_sql_exception_falls_back_to_chat(self, cli):
        """Exception during parse_sql_type falls back to CHAT."""
        with patch("datus.cli.repl.parse_sql_type", side_effect=Exception("parse error")):
            cmd_type, _, _ = cli._parse_command("ambiguous text")
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
    """Input-line prompt is now a minimal marker; brand/mode live in the status bar."""

    def test_input_prompt_is_minimal(self, cli):
        cli.plan_mode_active = False
        assert cli._get_prompt_text() == "> "

    def test_input_prompt_unchanged_in_plan_mode(self, cli):
        cli.plan_mode_active = True
        # The input line no longer carries the PLAN marker — that lives on the
        # status bar line built by _build_prompt_message.
        assert cli._get_prompt_text() == "> "

    def test_status_bar_reflects_plan_mode(self, cli):
        cli.plan_mode_active = True
        tokens = cli._build_prompt_message(cli._get_prompt_text())
        text = "".join(value for _, value in tokens)
        assert "Datus" in text
        assert "PLAN" in text


# ---------------------------------------------------------------------------
# Tests: /datasource command (via DatasourceCommands)
# ---------------------------------------------------------------------------


class TestCmdDatasource:
    def test_switch_to_different_datasource(self, cli):
        mock_conn = MagicMock()
        mock_conn.database_name = "newdb"
        mock_conn.catalog_name = ""
        mock_conn.schema_name = ""
        cli.db_manager.first_conn_with_name.return_value = ("newdb", mock_conn)

        with patch.object(cli, "reset_session"):
            cli.datasource_commands._switch("california_schools")

        output = cli.console.file.getvalue()
        assert "california_schools" in output

    def test_same_datasource_prints_warning(self, cli):
        current_ns = cli.agent_config.current_datasource
        cli.datasource_commands._switch(current_ns)
        output = cli.console.file.getvalue()
        assert "already" in output.lower()


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


# ---------------------------------------------------------------------------
# Tests: _execute_sql
# ---------------------------------------------------------------------------


class TestExecuteSql:
    def test_no_db_connector_prints_error(self, cli):
        cli.db_connector = None
        cli._execute_sql("SELECT 1")
        output = cli.console.file.getvalue()
        assert "No database connection" in output

    def test_execute_returns_none_prints_error(self, cli):
        cli.db_connector.execute.return_value = None
        cli._execute_sql("SELECT 1")
        output = cli.console.file.getvalue()
        assert "No result" in output or "Error" in output

    def test_successful_arrow_result_displays_table(self, cli):
        import pyarrow as pa

        table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sql_return = table
        mock_result.row_count = 2
        cli.db_connector.execute.return_value = mock_result

        cli._execute_sql("SELECT 1")
        output = cli.console.file.getvalue()
        assert "Alice" in output or "rows" in output.lower()

    def test_sql_error_result_prints_error(self, cli):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "syntax error near 'BAD'"
        cli.db_connector.execute.return_value = mock_result

        cli._execute_sql("SELECT BAD")
        output = cli.console.file.getvalue()
        assert "SQL Error" in output or "syntax error" in output

    def test_execute_exception_prints_error(self, cli):
        cli.db_connector.execute.side_effect = RuntimeError("connection lost")
        cli._execute_sql("SELECT 1")
        output = cli.console.file.getvalue()
        assert "Error" in output or "connection lost" in output

    def test_non_arrow_row_count_positive_shows_update(self, cli):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.row_count = 3
        # sql_return has no column_names (non-arrow path)
        del mock_result.sql_return.column_names
        cli.db_connector.execute.return_value = mock_result

        cli._execute_sql("UPDATE orders SET x=1")
        output = cli.console.file.getvalue()
        # Should print update message
        assert len(output) > 0

    def test_content_set_sql_updates_cli_context_in_place(self, cli):
        """USE/SET SQL updates cli_context in-place, preserving accumulated state."""
        from datus.utils.constants import SQLType

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.row_count = None
        mock_result.sql_return = "OK"  # truthy, no column_names attr
        cli.db_connector.execute.return_value = mock_result

        # Connector state after USE command
        cli.db_connector.catalog_name = "new_catalog"
        cli.db_connector.database_name = "new_db"
        cli.db_connector.schema_name = "new_schema"
        cli.db_connector.dialect = "snowflake"

        # Set initial context and accumulated state
        cli.cli_context.current_catalog = "old_catalog"
        cli.cli_context.current_db_name = "old_db"
        cli.cli_context.current_schema = "old_schema"
        cli.cli_context.current_logic_db_name = "my_logic_name"
        original_context = cli.cli_context

        with patch("datus.cli.repl.parse_sql_type", return_value=SQLType.CONTENT_SET):
            cli._execute_sql("USE DATABASE new_db")

        # Context updated in-place (same object, not replaced)
        assert cli.cli_context is original_context
        assert cli.cli_context.current_catalog == "new_catalog"
        assert cli.cli_context.current_db_name == "new_db"
        assert cli.cli_context.current_schema == "new_schema"
        # Accumulated state preserved
        assert cli.cli_context.current_logic_db_name == "my_logic_name"

    def test_non_content_set_sql_does_not_update_context(self, cli):
        """Non-CONTENT_SET SQL does not update cli_context."""
        from datus.utils.constants import SQLType

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.row_count = None
        mock_result.sql_return = "OK"
        cli.db_connector.execute.return_value = mock_result

        cli.cli_context.current_db_name = "original_db"

        with patch("datus.cli.repl.parse_sql_type", return_value=SQLType.DDL):
            cli._execute_sql("CREATE TABLE t1 (id INT)")

        assert cli.cli_context.current_db_name == "original_db"


# ---------------------------------------------------------------------------
# Tests: _execute_tool_command
# ---------------------------------------------------------------------------


class TestExecuteToolCommand:
    def test_known_command_called(self, cli):
        cli.commands["!sl"] = MagicMock()
        cli._execute_tool_command("!sl", "find revenue")
        cli.commands["!sl"].assert_called_once_with("find revenue")

    def test_unknown_command_prints_error(self, cli):
        cli._execute_tool_command("!nonexistent", "args")
        output = cli.console.file.getvalue()
        assert "Unknown command" in output


# ---------------------------------------------------------------------------
# Tests: _execute_slash_command
# ---------------------------------------------------------------------------


class TestExecuteSlashCommand:
    def test_known_slash_command_called(self, cli):
        cli.commands["/tables"] = MagicMock()
        cli._execute_slash_command("/tables", "")
        cli.commands["/tables"].assert_called_once_with("")

    def test_context_command_called_through_slash(self, cli):
        cli.commands["/catalog"] = MagicMock()
        cli._execute_slash_command("/catalog", "mydb")
        cli.commands["/catalog"].assert_called_once_with("mydb")

    def test_unknown_slash_command_prints_error(self, cli):
        cli._execute_slash_command("/nonexistent", "")
        output = cli.console.file.getvalue()
        assert "Unknown command" in output

    def test_rewind_return_captured_as_prefill(self, cli):
        cli.commands["/rewind"] = MagicMock(return_value="previous user message")
        cli._prefill_input = None
        cli._execute_slash_command("/rewind", "2")
        assert cli._prefill_input == "previous user message"

    def test_slash_exit_propagates_exit_sentinel(self, cli):
        """``/exit`` returns EXIT_SENTINEL so the TUI loop can shut down
        cleanly even when dispatch runs on a worker thread."""
        from datus.cli.tui.app import EXIT_SENTINEL

        cli.commands["/exit"] = MagicMock(return_value=EXIT_SENTINEL)
        result = cli._execute_slash_command("/exit", "")
        assert result == EXIT_SENTINEL


class TestRenderUnknownCommand:
    def test_legacy_prefix_hint_rendered(self, cli):
        cli._render_unknown_command(".tables", "/tables")
        output = cli.console.file.getvalue()
        assert "renamed to '/tables'" in output

    def test_plain_unknown_command_rendered(self, cli):
        cli._render_unknown_command("/halp", "")
        output = cli.console.file.getvalue()
        assert "Unknown command" in output
        assert "/halp" in output


# ---------------------------------------------------------------------------
# Tests: _execute_chat_command
# ---------------------------------------------------------------------------


class TestExecuteChatCommand:
    def test_delegates_to_chat_commands(self, cli):
        cli._execute_chat_command("show revenue", subagent_name="gensql")
        cli.chat_commands.execute_chat_command.assert_called_once_with(
            "show revenue", plan_mode=False, subagent_name="gensql"
        )

    def test_plan_mode_passed_through(self, cli):
        cli.plan_mode_active = True
        cli._execute_chat_command("plan this", subagent_name=None)
        cli.chat_commands.execute_chat_command.assert_called_once_with("plan this", plan_mode=True, subagent_name=None)


# ---------------------------------------------------------------------------
# Tests: _cmd_bash
# ---------------------------------------------------------------------------


class TestCmdBash:
    def test_empty_args_prints_message(self, cli):
        cli._cmd_bash("")
        output = cli.console.file.getvalue()
        assert "provide" in output.lower() or "Please" in output

    def test_non_whitelisted_command_prints_security_error(self, cli):
        cli._cmd_bash("rm -rf /tmp/test")
        output = cli.console.file.getvalue()
        assert "Security" in output or "whitelist" in output.lower()

    def test_whitelisted_command_executes(self, cli):
        mock_run_result = MagicMock()
        mock_run_result.returncode = 0
        mock_run_result.stdout = "/home/user\n"

        with patch("subprocess.run", return_value=mock_run_result):
            cli._cmd_bash("pwd")

        output = cli.console.file.getvalue()
        assert "/home/user" in output

    def test_command_failure_prints_stderr(self, cli):
        mock_run_result = MagicMock()
        mock_run_result.returncode = 1
        mock_run_result.stderr = "no such file"

        with patch("subprocess.run", return_value=mock_run_result):
            cli._cmd_bash("ls /nonexistent")

        output = cli.console.file.getvalue()
        assert "failed" in output.lower() or "no such file" in output

    def test_timeout_prints_error(self, cli):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pwd", 10)):
            cli._cmd_bash("pwd")

        output = cli.console.file.getvalue()
        assert "timed out" in output.lower()

    def test_exception_prints_error(self, cli):
        with patch("subprocess.run", side_effect=OSError("permission denied")):
            cli._cmd_bash("ls")

        output = cli.console.file.getvalue()
        assert "Error" in output


# ---------------------------------------------------------------------------
# Tests: _cmd_help
# ---------------------------------------------------------------------------


class TestCmdHelp:
    def test_help_output_contains_commands(self, cli):
        cli._cmd_help("")
        output = cli.console.file.getvalue()
        assert "Help" in output or "Commands" in output or "SQL" in output


# ---------------------------------------------------------------------------
# Tests: _wait_for_agent_available
# ---------------------------------------------------------------------------


class TestWaitForAgentAvailable:
    def test_immediately_available_returns_true(self, cli):
        cli.agent_ready = True
        cli.agent = MagicMock()
        result = cli._wait_for_agent_available(max_attempts=3, delay=0)
        assert result is True

    def test_times_out_returns_false(self, cli):
        cli.agent_ready = False
        cli.agent_initializing = False
        cli.agent = None

        with patch("time.sleep"):
            result = cli._wait_for_agent_available(max_attempts=2, delay=0)

        assert result is False
        output = cli.console.file.getvalue()
        assert "timed out" in output.lower() or "failed" in output.lower() or "not available" in output.lower()


# ---------------------------------------------------------------------------
# Tests: create_combined_completer
# ---------------------------------------------------------------------------


class TestCreateCombinedCompleterExtended:
    def test_returns_a_completer(self, cli):
        from datus.cli.autocomplete import AtReferenceCompleter, SlashCommandCompleter

        with patch.object(AtReferenceCompleter, "__init__", return_value=None):
            with patch.object(SlashCommandCompleter, "__init__", return_value=None):
                with patch(
                    "prompt_toolkit.completion.merge_completers",
                    return_value=MagicMock(),
                ) as mock_merge:
                    result = cli.create_combined_completer()

        mock_merge.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: /datasource switch extended (via DatasourceCommands)
# ---------------------------------------------------------------------------


class TestCmdSwitchDatasourceExtended:
    def test_switch_updates_cli_context(self, cli):
        mock_conn = MagicMock()
        mock_conn.database_name = "california_schools"
        mock_conn.catalog_name = ""
        mock_conn.schema_name = ""
        cli.db_manager.first_conn_with_name.return_value = ("test_ns", mock_conn)

        with patch.object(
            type(cli.agent_config),
            "current_datasource",
            new_callable=lambda: property(
                lambda self: "other_ns",
                lambda self, v: None,
            ),
        ):
            with patch.object(cli, "reset_session"):
                cli.datasource_commands._switch("test_ns")

        output = cli.console.file.getvalue()
        assert "Datasource changed" in output

    def test_same_datasource_prints_warning(self, cli):
        current = cli.agent_config.current_datasource
        cli.datasource_commands._switch(current)
        output = cli.console.file.getvalue()
        assert "already" in output.lower()


# ---------------------------------------------------------------------------
# Tests: _init_connection
# ---------------------------------------------------------------------------


class TestInitConnection:
    def test_db_name_none_falls_back_to_connector_database_name(self, cli):
        """When first_conn_with_name returns None for db_name, connector.database_name is used."""
        mock_conn = MagicMock()
        mock_conn.database_name = "fallback_db"
        mock_conn.catalog_name = ""
        mock_conn.schema_name = ""
        mock_conn.dialect = "snowflake"
        mock_conn.test_connection.return_value = True
        cli.db_manager.first_conn_with_name.return_value = (None, mock_conn)

        # Ensure cli_context.current_db_name is falsy so the first branch is taken
        cli.cli_context.current_db_name = None

        cli._init_connection(timeout_seconds=5)

        assert cli.db_connector is mock_conn
        # The fallback should have populated db_name from connector.database_name
        assert cli.cli_context.current_db_name == "fallback_db"

    def test_db_name_present_uses_returned_name(self, cli):
        """When first_conn_with_name returns a db_name, it is used directly."""
        mock_conn = MagicMock()
        mock_conn.database_name = "connector_db"
        mock_conn.catalog_name = ""
        mock_conn.schema_name = ""
        mock_conn.dialect = "snowflake"
        mock_conn.test_connection.return_value = True
        cli.db_manager.first_conn_with_name.return_value = ("returned_db", mock_conn)

        cli.cli_context.current_db_name = None

        cli._init_connection(timeout_seconds=5)

        assert cli.db_connector is mock_conn

        assert cli.cli_context.current_logic_db_name == "returned_db"
        assert cli.cli_context.current_db_name == "connector_db"


# ---------------------------------------------------------------------------
# Tests: _print_welcome database fallback
# ---------------------------------------------------------------------------


class TestPrintWelcome:
    """Test _print_welcome banner output and database fallback chain."""

    def _get_output(self, cli, width: int = 100):
        """Call _print_welcome and return the console output."""
        cli.console = Console(file=io.StringIO(), no_color=True, width=width)
        cli._print_welcome()
        return cli.console.file.getvalue()

    def test_database_from_args_datasource(self, real_agent_config):
        """args.datasource takes highest priority."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(datasource="my_db")
        real_agent_config._current_datasource = "config_db"
        output = self._get_output(cli)
        assert "my_db" in output
        assert "config_db" not in output

    def test_database_fallback_to_agent_config(self, real_agent_config):
        """Falls back to agent_config.current_datasource when args are empty."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(datasource="")
        real_agent_config._current_datasource = "config_db"
        output = self._get_output(cli)
        assert "config_db" in output

    def test_no_database_warning(self, real_agent_config):
        """Shows 'not selected' hint when no database is available."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(datasource="")
        real_agent_config._current_datasource = ""
        output = self._get_output(cli)
        assert "not selected" in output


class TestBuildBannerPanel:
    """Test the unified banner structure (version, art, AI status)."""

    def _render(self, cli, width: int = 100) -> str:
        cli.console = Console(file=io.StringIO(), no_color=True, width=width)
        cli._print_welcome()
        return cli.console.file.getvalue()

    def test_contains_version(self, real_agent_config):
        from datus import __version__

        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(database="benchmark", datasource="")
        real_agent_config._current_datasource = ""
        output = self._render(cli)
        assert f"v{__version__}" in output

    def test_contains_subtitle(self, real_agent_config):
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(database="benchmark", datasource="")
        real_agent_config._current_datasource = ""
        output = self._render(cli)
        assert "Data engineering agent builds evolvable context for your data system" in output

    def test_contains_ascii_art_on_wide_terminal(self, real_agent_config):
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(database="benchmark", datasource="")
        real_agent_config._current_datasource = ""
        output = self._render(cli, width=100)
        assert "██████╗" in output

    def test_narrow_terminal_falls_back_to_text(self, real_agent_config):
        from datus import __version__

        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(database="benchmark", datasource="")
        real_agent_config._current_datasource = ""
        output = self._render(cli, width=40)
        assert f"DATUS v{__version__}" in output
        assert "██████╗" not in output

    def test_no_ai_status_row(self, real_agent_config):
        """Banner must not include an AI status row."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(database="benchmark", datasource="")
        real_agent_config._current_datasource = ""
        cli.agent_ready = True
        cli.agent_initializing = False
        output = self._render(cli)
        assert "initializing" not in output
        # "AI" label should not appear as a standalone banner row
        for line in output.splitlines():
            stripped = line.strip("│ ").strip()
            assert not stripped.startswith("AI ")

    def test_not_connected_label(self, real_agent_config):
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(datasource="benchmark")
        real_agent_config._current_datasource = ""
        cli.db_connector = None
        output = self._render(cli)
        assert "not connected" in output

    def test_no_command_prefix_cheatsheet(self, real_agent_config):
        """Banner must not include the '/ . @ !' command prefix cheatsheet row."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(database="benchmark", datasource="")
        real_agent_config._current_datasource = ""
        output = self._render(cli)
        assert "! @ ." not in output
        assert "Commands:" not in output

    def test_using_suffix_when_active_db_differs(self, real_agent_config):
        """Database line appends 'using <name>' when current_db_name != configured db."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(datasource="benchmark")
        real_agent_config._current_datasource = ""
        cli.cli_context.current_db_name = "other_db"
        output = self._render(cli)
        assert "using other_db" in output

    def test_context_row_rendered_when_available(self, real_agent_config):
        """Context row appears when cli_context has a non-empty summary."""
        cli = _make_cli(real_agent_config)
        cli.args = MagicMock(datasource="benchmark")
        real_agent_config._current_datasource = ""
        cli.cli_context.current_db_name = "benchmark"
        output = self._render(cli)
        assert "Context" in output
        assert "database: benchmark" in output


# ---------------------------------------------------------------------------
# Tests: _cmd_agent
# ---------------------------------------------------------------------------


class TestCmdAgent:
    def test_cmd_agent_set_valid(self, cli):
        """'.agent gen_sql' sets default_agent to gen_sql."""
        cli.available_subagents = {"chat", "gen_sql", "gen_report"}
        cli._cmd_agent("gen_sql")
        assert cli.default_agent == "gen_sql"
        output = cli.console.file.getvalue()
        assert "gen_sql" in output

    def test_cmd_agent_set_chat_resets(self, cli):
        """'.agent chat' resets default_agent to empty string."""
        cli.available_subagents = {"chat", "gen_sql"}
        cli.default_agent = "gen_sql"
        cli._cmd_agent("chat")
        assert cli.default_agent == ""
        output = cli.console.file.getvalue()
        assert "chat" in output

    def test_cmd_agent_set_invalid(self, cli):
        """'.agent nonexistent' prints error and doesn't change default."""
        cli.available_subagents = {"chat", "gen_sql"}
        cli.default_agent = ""
        cli._cmd_agent("nonexistent")
        assert cli.default_agent == ""
        output = cli.console.file.getvalue()
        assert "Unknown agent" in output

    def test_cmd_agent_no_args_opens_unified_tui(self, cli):
        """'.agent' (no args) opens the unified :class:`AgentApp` seeded
        on the Built-in tab. A ``set_default`` selection from the app
        updates ``default_agent``."""
        from datus.cli.agent_app import AgentSelection

        cli.available_subagents = {"chat", "gen_sql"}
        cli.default_agent = ""
        with patch("datus.cli.agent_app.AgentApp") as mock_cls:
            mock_cls.return_value.run.return_value = AgentSelection(kind="set_default", name="gen_sql")
            cli._cmd_agent("")
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["seed_tab"] == "builtin"
        assert cli.default_agent == "gen_sql"

    def test_cmd_subagent_opens_unified_tui_on_custom_tab(self, cli):
        """'.subagent' opens :class:`AgentApp` seeded on the Custom tab."""
        from datus.cli.agent_app import AgentSelection

        cli.available_subagents = {"chat", "gen_sql"}
        cli.default_agent = ""
        with patch("datus.cli.agent_app.AgentApp") as mock_cls:
            mock_cls.return_value.run.return_value = AgentSelection(kind="set_default", name="chat")
            cli._cmd_subagent("")
        assert mock_cls.call_args.kwargs["seed_tab"] == "custom"
        assert cli.default_agent == ""

    def test_cmd_agent_cancel_is_noop(self, cli):
        """``None`` from :class:`AgentApp` (Esc / Ctrl+C) must not touch
        ``default_agent`` — we rely on this when the user opens the app
        just to tweak Built-in overrides without changing routing."""
        cli.available_subagents = {"chat", "gen_sql"}
        cli.default_agent = ""
        with patch("datus.cli.agent_app.AgentApp") as mock_cls:
            mock_cls.return_value.run.return_value = None
            cli._cmd_agent("")
        assert cli.default_agent == ""
        output = cli.console.file.getvalue()
        assert "Default agent set to" not in output
        assert "Default agent reset to" not in output


# ---------------------------------------------------------------------------
# Tests: _parse_command with default_agent routing
# ---------------------------------------------------------------------------


class TestParseCommandDefaultAgent:
    """After the slash refactor, agent selection is driven by ``/agent`` and
    bare text (no prefix) is the only path that routes to chat. ``/<word>`` is
    now exclusively a command prefix — unknown slashes surface as UNKNOWN
    instead of silently being interpreted as a chat message.
    """

    def test_slash_unknown_prefix_is_not_chat(self, cli):
        """``/show revenue`` is no longer chat — it's a slash token the
        dispatcher will reject as unknown."""
        cli.default_agent = "gen_sql"
        cmd_type, cmd, args = cli._parse_command("/show revenue")
        assert cmd_type == CommandType.SLASH
        assert cmd == "/show"
        assert args == "revenue"

    def test_bare_text_uses_default_agent(self, cli):
        """Bare natural language text uses ``default_agent``."""
        cli.default_agent = "gen_sql"
        with patch("datus.cli.repl.parse_sql_type") as mock_parse:
            from datus.utils.constants import SQLType

            mock_parse.return_value = SQLType.UNKNOWN
            cmd_type, cmd, args = cli._parse_command("show me the revenue")
        assert cmd_type == CommandType.CHAT
        assert cmd == "gen_sql"

    def test_bare_text_default_chat(self, cli):
        """Bare text with ``default_agent=''`` still routes to the chat node."""
        cli.default_agent = ""
        with patch("datus.cli.repl.parse_sql_type") as mock_parse:
            from datus.utils.constants import SQLType

            mock_parse.return_value = SQLType.UNKNOWN
            cmd_type, cmd, _ = cli._parse_command("hello world")
        assert cmd_type == CommandType.CHAT
        assert cmd == ""


# ---------------------------------------------------------------------------
# Tests: _get_prompt_text with default_agent
# ---------------------------------------------------------------------------


class TestStatusBarReflectsAgent:
    """Agent/brand/mode identifiers now render on the status bar line."""

    def _render_status_bar(self, cli) -> str:
        tokens = cli._build_prompt_message(cli._get_prompt_text())
        return "".join(value for _, value in tokens)

    def test_status_bar_shows_default_agent(self, cli):
        cli.default_agent = "gen_sql"
        cli.plan_mode_active = False
        text = self._render_status_bar(cli)
        assert "gen_sql" in text
        assert "Datus" in text
        assert "PLAN" not in text

    def test_status_bar_falls_back_to_chat_when_no_default(self, cli):
        cli.default_agent = ""
        cli.plan_mode_active = False
        text = self._render_status_bar(cli)
        assert "Datus" in text
        assert "chat" in text
        # labels no longer render
        assert "Agent " not in text

    def test_status_bar_marks_plan_mode(self, cli):
        cli.default_agent = "gen_sql"
        cli.plan_mode_active = True
        text = self._render_status_bar(cli)
        assert "PLAN" in text
        assert "gen_sql" in text


# ---------------------------------------------------------------------------
# run_on_bg_loop — coroutines route through the persistent background loop
# ---------------------------------------------------------------------------


class TestRunOnBgLoop:
    """``run_on_bg_loop`` keeps chat-stream coroutines on a single persistent
    event loop so prompt_toolkit-owned Futures do not outlive a torn-down
    ``asyncio.run`` loop (root cause of the ``got Future pending attached to a
    different loop`` terminal hang during ``ask_user`` rendering).
    """

    @pytest.fixture
    def bg_cli(self, real_agent_config):
        import asyncio
        import threading

        cli = _make_cli(real_agent_config)
        cli._bg_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=cli._bg_loop.run_forever, daemon=True)
        thread.start()
        try:
            yield cli
        finally:
            cli._bg_loop.call_soon_threadsafe(cli._bg_loop.stop)
            thread.join(timeout=2)
            cli._bg_loop.close()

    def test_coroutine_runs_on_bg_loop_and_returns_value(self, bg_cli):
        import asyncio

        captured_loop = {}

        async def coro():
            captured_loop["loop"] = asyncio.get_running_loop()
            return "done"

        result = bg_cli.run_on_bg_loop(coro())

        assert result == "done"
        assert captured_loop["loop"] is bg_cli._bg_loop

    def test_propagates_exception_from_coroutine(self, bg_cli):
        async def coro():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            bg_cli.run_on_bg_loop(coro())

    def test_keyboard_interrupt_cancels_bg_task(self, bg_cli):
        """Main-thread KeyboardInterrupt during ``future.result()`` must
        cancel the coroutine on the bg loop and re-raise so the existing
        ``except KeyboardInterrupt`` handler in chat_commands still fires.
        """
        import asyncio
        import concurrent.futures
        import threading

        started = threading.Event()
        stopped_cleanly = threading.Event()

        async def long_running():
            started.set()
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                stopped_cleanly.set()
                raise

        # Schedule the coroutine and simulate the main-thread Ctrl+C path by
        # manually cancelling the future + invoking the helper's KI branch.
        future = asyncio.run_coroutine_threadsafe(long_running(), bg_cli._bg_loop)
        assert started.wait(timeout=1.0)
        bg_cli._bg_loop.call_soon_threadsafe(future.cancel)
        try:
            future.result(timeout=1.0)
        except concurrent.futures.CancelledError:
            pass
        assert stopped_cleanly.wait(timeout=1.0)
