# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/bang_command.py.

Covers the ``!<tool>`` / ``!<plugin>`` dispatcher:
- tool/plugin enumeration (node tools + active plugins)
- tool-before-plugin dispatch priority + unknown-token error
- tool invocation gated by ``run_tool_gate`` (denied never executes)
- plugin invocation via the ``datus <plugin> ...`` subprocess path
- the dim argument-name hint (``param_hint``)
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bang_command import BangCommand
from datus.cli.input_modes import InputMode
from datus.plugins.base import PluginCommand, PluginCommandArg, PluginManifest


def _make_tool(name, *, schema=None, description="", result=None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.params_json_schema = schema if schema is not None else {"properties": {}}
    tool.on_invoke_tool = AsyncMock(return_value=result if result is not None else {"success": 1, "result": "ok"})
    return tool


def _make_cli(tools=None, node_exists=True, plugins_enabled=True):
    node = SimpleNamespace(tools=list(tools or [])) if node_exists else None
    chat = MagicMock()
    chat.current_node = node
    chat.ensure_node_for_bang = MagicMock(return_value=node)
    agent_config = MagicMock()
    agent_config.plugins_enabled = plugins_enabled
    agent_config.plugin_active = MagicMock(return_value=True)
    return SimpleNamespace(
        console=Console(file=io.StringIO(), no_color=True, width=200),
        chat_commands=chat,
        agent_config=agent_config,
        service_commands=MagicMock(),
        input_mode=InputMode.CHAT,
        _send_exec_turn=MagicMock(),
    )


def _hello_manifest():
    return PluginManifest(
        name="hello",
        package_dir=Path("."),
        description="Hello plugin",
        commands=[
            PluginCommand(
                name="sync",
                description="sync tables",
                args=[PluginCommandArg("table", required=True), PluginCommandArg("--limit")],
            ),
            PluginCommand(name="status", description="show status"),
        ],
    )


def _airflow_manifest():
    """A two-level command tree: a ``dags`` group plus a flat ``version`` command."""
    return PluginManifest(
        name="airflow",
        package_dir=Path("."),
        description="Airflow plugin",
        commands=[
            PluginCommand(
                name="dags",
                description="DAG operations",
                subcommands=[
                    PluginCommand(
                        name="trigger",
                        description="Trigger a DAG run",
                        args=[PluginCommandArg("dag_id", required=True), PluginCommandArg("--conf")],
                    ),
                    PluginCommand(name="list", description="List DAGs"),
                ],
            ),
            PluginCommand(name="version", description="Server version"),
        ],
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_tool_map_reads_live_node_tools(self):
        cli = _make_cli(tools=[_make_tool("list_tables"), _make_tool("search_table")])
        bang = BangCommand(cli)
        assert set(bang.tool_map()) == {"list_tables", "search_table"}

    def test_tool_map_excludes_orchestration_tools(self):
        """Agent-orchestration / plan-mode tools are hidden from ``!``."""
        names = ["list_tables", "ask_user", "task", "confirm_plan", "todo_write", "execute_sql"]
        cli = _make_cli(tools=[_make_tool(n) for n in names])
        bang = BangCommand(cli)
        assert set(bang.tool_map()) == {"list_tables", "execute_sql"}

    def test_excluded_tool_name_is_not_dispatched_as_tool(self):
        """A first token naming an excluded tool falls through to the unknown path
        (not invoked as a tool)."""
        tool = _make_tool("task")
        cli = _make_cli(tools=[tool])
        bang = BangCommand(cli)
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[]):
            bang.dispatch("task")
        tool.on_invoke_tool.assert_not_awaited()
        assert "Unknown tool or plugin 'task'" in cli.console.file.getvalue()

    def test_tool_map_create_lazily_builds_node(self):
        cli = _make_cli(node_exists=False)
        # ensure_node_for_bang returns a node with one tool the second time.
        node = SimpleNamespace(tools=[_make_tool("list_tables")])
        cli.chat_commands.ensure_node_for_bang = MagicMock(return_value=node)
        bang = BangCommand(cli)
        assert bang.tool_map(create=False) == {}
        assert set(bang.tool_map(create=True)) == {"list_tables"}
        cli.chat_commands.ensure_node_for_bang.assert_called_once()

    def test_plugin_map_lists_active_plugins(self):
        cli = _make_cli()
        bang = BangCommand(cli)
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]):
            plugins = bang.plugin_map()
        assert set(plugins) == {"hello"}

    def test_plugin_map_filters_inactive(self):
        cli = _make_cli()
        cli.agent_config.plugin_active = MagicMock(return_value=False)
        bang = BangCommand(cli)
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]):
            assert bang.plugin_map() == {}

    def test_plugin_map_empty_when_plugins_disabled(self):
        cli = _make_cli(plugins_enabled=False)
        bang = BangCommand(cli)
        assert bang.plugin_map() == {}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_tool_matches_before_plugin(self):
        """A first token naming both a tool and a plugin resolves to the tool."""
        tool = _make_tool("hello", schema={"properties": {}})
        cli = _make_cli(tools=[tool])
        bang = BangCommand(cli)
        with (
            patch("datus.cli.bash_mode.run_manual_tool_live", return_value=(None, False)) as run_tool,
            patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]),
            patch("datus.cli.bash_mode.run_manual_bash_live") as run_plugin,
        ):
            bang.dispatch("hello")
        run_tool.assert_called_once()
        assert run_tool.call_args.args[1] == "hello"  # tool_name
        run_plugin.assert_not_called()

    def test_plugin_dispatch_feeds_model(self):
        cli = _make_cli(tools=[])
        bang = BangCommand(cli)
        payload = {"kind": "bash", "command": "datus hello sync", "success": True}
        with (
            patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]),
            patch("datus.cli.bash_mode.run_manual_bash_live", return_value=(payload, True)) as run_plugin,
        ):
            bang.dispatch("hello sync mytable --limit=5")
        run_plugin.assert_called_once()
        assert run_plugin.call_args.args[1] == "datus hello sync mytable --limit=5"
        cli._send_exec_turn.assert_called_once_with(payload)

    def test_plugin_denied_not_dispatched(self):
        cli = _make_cli(tools=[])
        bang = BangCommand(cli)
        with (
            patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]),
            patch("datus.cli.bash_mode.run_manual_bash_live", return_value=(None, False)),
        ):
            bang.dispatch("hello status")
        cli._send_exec_turn.assert_not_called()

    def test_unknown_token_prints_error(self):
        cli = _make_cli(tools=[])
        bang = BangCommand(cli)
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[]):
            bang.dispatch("nope")
        assert "Unknown tool or plugin 'nope'" in cli.console.file.getvalue()

    def test_empty_prints_overview(self):
        cli = _make_cli(tools=[_make_tool("list_tables", description="List tables")])
        bang = BangCommand(cli)
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[]):
            bang.dispatch("")
        out = cli.console.file.getvalue()
        assert "list_tables" in out


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------


class TestInvokeTool:
    def test_help_prints_schema_and_does_not_execute(self):
        tool = _make_tool("search_table", schema={"properties": {"query_text": {"type": "string"}}})
        cli = _make_cli(tools=[tool])
        bang = BangCommand(cli)
        bang.dispatch("search_table --help")
        tool.on_invoke_tool.assert_not_awaited()
        cli._send_exec_turn.assert_not_called()
        assert "search_table" in cli.console.file.getvalue()

    def test_parse_error_prints_schema_and_does_not_execute(self):
        tool = _make_tool("search_table", schema={"properties": {"query_text": {"type": "string"}}})
        cli = _make_cli(tools=[tool])
        bang = BangCommand(cli)
        bang.dispatch("search_table --bogus=1")
        tool.on_invoke_tool.assert_not_awaited()
        cli._send_exec_turn.assert_not_called()
        assert "search_table" in cli.console.file.getvalue()

    def test_dispatches_exec_turn_with_parsed_args(self):
        """An approved tool call runs via ``run_manual_tool_live`` and the resulting
        payload is fed to the model via ``_send_exec_turn``; the invoke closure
        passes the parsed args to ``on_invoke_tool``."""
        tool = _make_tool(
            "search_table",
            schema={"properties": {"query_text": {"type": "string"}, "top_n": {"type": "integer"}}},
            result={"success": 1, "result": [{"table": "t1"}]},
        )
        cli = _make_cli(tools=[tool])
        bang = BangCommand(cli)
        payload = {"kind": "tool", "command": "search_table foo --top_n=3", "success": True}
        with patch("datus.cli.bash_mode.run_manual_tool_live", return_value=(payload, True)) as run_tool:
            bang.dispatch("search_table foo --top_n=3")
        cli_arg, tool_name, command, args, invoke_fn = run_tool.call_args.args
        assert cli_arg is cli
        assert tool_name == "search_table"
        assert command == "search_table foo --top_n=3"
        assert args == {"query_text": "foo", "top_n": 3}
        # The closure performs the actual invocation with the parsed args.
        success, output = invoke_fn()
        tool.on_invoke_tool.assert_awaited_once()
        _ctx, args_json = tool.on_invoke_tool.await_args.args
        assert json.loads(args_json) == {"query_text": "foo", "top_n": 3}
        assert success is True
        assert "t1" in output
        cli._send_exec_turn.assert_called_once_with(payload)

    def test_denied_not_dispatched(self):
        tool = _make_tool("execute_sql", schema={"properties": {"sql": {"type": "string"}}})
        cli = _make_cli(tools=[tool])
        bang = BangCommand(cli)
        denial_payload = {"kind": "tool", "command": "execute_sql", "success": False}
        with patch("datus.cli.bash_mode.run_manual_tool_live", return_value=(denial_payload, False)):
            bang.dispatch('execute_sql "DROP TABLE t"')
        cli._send_exec_turn.assert_not_called()


class TestToolResultToOutput:
    def test_func_tool_result_success_unwrapped(self):
        success, output = BangCommand._tool_result_to_output({"success": 1, "result": [{"a": 1}]})
        assert success is True
        assert '"a": 1' in output

    def test_func_tool_result_failure_uses_error(self):
        success, output = BangCommand._tool_result_to_output({"success": 0, "error": "nope"})
        assert success is False
        assert output == "nope"

    def test_plain_string_result(self):
        assert BangCommand._tool_result_to_output("hi") == (True, "hi")

    def test_plain_payload_json_encoded(self):
        success, output = BangCommand._tool_result_to_output({"rows": 3})
        assert success is True
        assert '"rows": 3' in output


# ---------------------------------------------------------------------------
# Argument-name hint
# ---------------------------------------------------------------------------


class TestParamHint:
    def _bang_with_plugin(self, tools=None):
        cli = _make_cli(tools=tools or [])
        bang = BangCommand(cli)
        return cli, bang

    def test_empty_while_typing_name(self):
        _cli, bang = self._bang_with_plugin(tools=[_make_tool("search_table")])
        assert bang.param_hint("!sea") == ""

    def test_non_bang_returns_empty(self):
        _cli, bang = self._bang_with_plugin()
        assert bang.param_hint("select 1") == ""

    def test_tool_hint_shows_required_and_optional(self):
        tool = _make_tool(
            "search_table",
            schema={
                "properties": {"query_text": {"type": "string"}, "top_n": {"type": "integer"}},
                "required": ["query_text"],
            },
        )
        _cli, bang = self._bang_with_plugin(tools=[tool])
        hint = bang.param_hint("!search_table ")
        assert "<query_text>" in hint
        assert "[--top_n]" in hint

    def test_tool_hint_drops_consumed_positional(self):
        tool = _make_tool(
            "search_table",
            schema={
                "properties": {"query_text": {"type": "string"}, "top_n": {"type": "integer"}},
                "required": ["query_text"],
            },
        )
        _cli, bang = self._bang_with_plugin(tools=[tool])
        hint = bang.param_hint("!search_table foo ")
        assert "<query_text>" not in hint
        assert "[--top_n]" in hint

    def test_tool_hint_drops_named_given(self):
        tool = _make_tool(
            "search_table",
            schema={
                "properties": {"query_text": {"type": "string"}, "top_n": {"type": "integer"}},
                "required": ["query_text"],
            },
        )
        _cli, bang = self._bang_with_plugin(tools=[tool])
        hint = bang.param_hint("!search_table --top_n=3 ")
        assert "[--top_n]" not in hint
        assert "<query_text>" in hint

    def test_plugin_hint_lists_commands(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]):
            hint = bang.param_hint("!hello ")
        assert hint == "{sync|status}"

    def test_plugin_hint_shows_command_args(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]):
            hint = bang.param_hint("!hello sync ")
        assert "<table>" in hint
        assert "[--limit]" in hint

    def test_plugin_hint_drops_consumed_arg(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("hello", _hello_manifest())]):
            hint = bang.param_hint("!hello sync mytable ")
        assert "<table>" not in hint
        assert "[--limit]" in hint

    # -- nested command groups (dags -> dags trigger) ----------------------

    def test_plugin_hint_lists_group_subcommands(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("airflow", _airflow_manifest())]):
            hint = bang.param_hint("!airflow dags ")
        assert hint == "{trigger|list}"

    def test_plugin_hint_leaf_subcommand_args(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("airflow", _airflow_manifest())]):
            hint = bang.param_hint("!airflow dags trigger ")
        assert "<dag_id>" in hint
        assert "[--conf]" in hint

    def test_plugin_hint_leaf_drops_consumed_positional(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("airflow", _airflow_manifest())]):
            hint = bang.param_hint("!airflow dags trigger mydag ")
        assert "<dag_id>" not in hint
        assert "[--conf]" in hint

    def test_plugin_hint_top_level_menu_before_group_chosen(self):
        _cli, bang = self._bang_with_plugin()
        with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[("airflow", _airflow_manifest())]):
            hint = bang.param_hint("!airflow ")
        assert hint == "{dags|version}"


@pytest.mark.parametrize("text", ["!", "!  ", "!help"])
def test_dispatch_overview_variants_do_not_raise(text):
    cli = _make_cli(tools=[])
    bang = BangCommand(cli)
    with patch("datus.plugins.registry.iter_plugin_manifests", return_value=[]):
        bang.dispatch(text[1:].strip())
    # Overview always prints the usage line.
    assert "Usage:" in cli.console.file.getvalue()
