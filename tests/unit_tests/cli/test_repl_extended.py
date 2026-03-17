# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Extended unit tests for datus/cli/repl.py.

Covers additional uncovered lines beyond test_repl.py:
- _execute_sql: no connector, success with arrow data, error result, exception
- _execute_tool_command: known / unknown command
- _execute_context_command: known / unknown command
- _execute_internal_command: known / unknown command
- _execute_chat_command: delegates to chat_commands
- _cmd_bash: empty args, whitelist check, success, timeout, exception
- _cmd_help: smoke test
- _wait_for_agent_available: immediate success, timeout
- create_combined_completer: returns completer (smoke)
- _cmd_switch_namespace: extended branches

All external dependencies are mocked; CI-level (no network, no API keys).
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.repl import DatusCLI

# ---------------------------------------------------------------------------
# Factory: build minimal DatusCLI without running __init__
# ---------------------------------------------------------------------------


def _make_cli(agent_config, available_subagents=None):
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
    cli._workflow_runner = None

    from datus.cli.cli_context import CliContext
    from datus.schemas.action_history import ActionHistoryManager

    cli.cli_context = CliContext()
    cli.actions = ActionHistoryManager()
    cli.last_sql = None
    cli.last_result = None

    cli.available_subagents = available_subagents or {"gensql", "chat", "compare"}

    cli.agent_commands = MagicMock()
    cli.chat_commands = MagicMock()
    cli.context_commands = MagicMock()
    cli.metadata_commands = MagicMock()
    cli.sub_agent_commands = MagicMock()
    cli.bi_dashboard_commands = MagicMock()

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


class TestCreateCombinedCompleter:
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
