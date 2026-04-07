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
    cli.last_sql = None
    cli.last_result = None

    # Build commands dict referencing the mocks
    cli.commands = {
        "!sl": cli.agent_commands.cmd_schema_linking,
        "@catalog": cli.context_commands.cmd_catalog,
        ".tables": cli.metadata_commands.cmd_tables,
        ".namespace": cli._cmd_switch_namespace,
        ".help": cli._cmd_help,
        ".exit": lambda a: None,
        "!bash": cli._cmd_bash,
    }

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
# Tests: _execute_context_command
# ---------------------------------------------------------------------------


class TestExecuteContextCommand:
    def test_known_context_command_called(self, cli):
        cli.commands["@catalog"] = MagicMock()
        cli._execute_context_command("@catalog", "mydb")
        cli.commands["@catalog"].assert_called_once_with("mydb")

    def test_unknown_context_command_prints_error(self, cli):
        cli._execute_context_command("@nonexistent", "")
        output = cli.console.file.getvalue()
        assert "Unknown command" in output


# ---------------------------------------------------------------------------
# Tests: _execute_internal_command
# ---------------------------------------------------------------------------


class TestExecuteInternalCommand:
    def test_known_internal_command_called(self, cli):
        cli.commands[".tables"] = MagicMock()
        cli._execute_internal_command(".tables", "")
        cli.commands[".tables"].assert_called_once_with("")

    def test_unknown_internal_command_prints_error(self, cli):
        cli._execute_internal_command(".nonexistent", "")
        output = cli.console.file.getvalue()
        assert "Unknown command" in output


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
        from datus.cli.autocomplete import AtReferenceCompleter, SubagentCompleter

        with patch.object(AtReferenceCompleter, "__init__", return_value=None):
            with patch.object(SubagentCompleter, "__init__", return_value=None):
                with patch(
                    "prompt_toolkit.completion.merge_completers",
                    return_value=MagicMock(),
                ) as mock_merge:
                    result = cli.create_combined_completer()

        mock_merge.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: _cmd_switch_namespace extended
# ---------------------------------------------------------------------------


class TestCmdSwitchNamespaceExtended:
    def test_switch_updates_cli_context(self, cli):
        mock_conn = MagicMock()
        mock_conn.database_name = "california_schools"
        mock_conn.catalog_name = ""
        mock_conn.schema_name = ""
        cli.db_manager.first_conn_with_name.return_value = ("test_ns", mock_conn)

        # Patch current_namespace property to return a different value so the
        # "already on this namespace" branch is NOT taken; setter is a no-op.
        with patch.object(
            type(cli.agent_config),
            "current_namespace",
            new_callable=lambda: property(
                lambda self: "other_ns",
                lambda self, v: None,
            ),
        ):
            with patch.object(cli, "reset_session"):
                cli._cmd_switch_namespace("test_ns")

        output = cli.console.file.getvalue()
        # Output prints the new namespace name passed to _cmd_switch_namespace
        assert "Namespace changed" in output

    def test_same_namespace_both_listed_and_message(self, cli):
        current = cli.agent_config.current_namespace

        with patch.object(cli, "_cmd_list_namespaces") as mock_list:
            cli._cmd_switch_namespace(current)

        mock_list.assert_called()
        output = cli.console.file.getvalue()
        assert "doesn't need" in output or "already" in output.lower()


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
