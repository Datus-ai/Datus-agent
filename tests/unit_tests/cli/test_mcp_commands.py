# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/mcp_commands.py — MCPCommands.

All external dependencies (MCPTool, MCPServerApp) are mocked.
Tests cover: cmd_mcp routing, list, add, remove, check, call_tool, filter subcommands.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.mcp_commands import MCPCommands

# ---------------------------------------------------------------------------
# Fixture: fake CLI instance
# ---------------------------------------------------------------------------


def _make_cli():
    cli = MagicMock()
    cli.console = MagicMock()
    return cli


def _ok(result=None):
    r = MagicMock()
    r.success = True
    r.result = result or {}
    r.message = ""
    return r


def _err(message="error"):
    r = MagicMock()
    r.success = False
    r.message = message
    return r


@pytest.fixture
def cmd():
    cli = _make_cli()
    with patch("datus.cli.mcp_commands.MCPTool") as MockTool:
        mock_tool = MagicMock()
        MockTool.return_value = mock_tool
        mc = MCPCommands(cli)
        mc.mcp_tool = mock_tool
        yield mc


# ---------------------------------------------------------------------------
# Tests: cmd_mcp routing
# ---------------------------------------------------------------------------


class TestCmdMcpRouting:
    def test_routes_list(self, cmd):
        cmd.cmd_mcp_list = MagicMock()
        cmd.cmd_mcp("list")
        cmd.cmd_mcp_list.assert_called_once()

    def test_routes_add(self, cmd):
        cmd.cmd_mcp_add = MagicMock()
        cmd.cmd_mcp("add foo bar")
        cmd.cmd_mcp_add.assert_called_once_with("foo bar")

    def test_routes_remove(self, cmd):
        cmd.cmd_mcp_remove = MagicMock()
        cmd.cmd_mcp("remove myserver")
        cmd.cmd_mcp_remove.assert_called_once_with("myserver")

    def test_routes_check(self, cmd):
        cmd.cmd_mcp_check = MagicMock()
        cmd.cmd_mcp("check myserver")
        cmd.cmd_mcp_check.assert_called_once_with("myserver")

    def test_routes_call(self, cmd):
        cmd.cmd_call_tool = MagicMock()
        cmd.cmd_mcp("call server.tool {}")
        cmd.cmd_call_tool.assert_called_once_with("server.tool {}")

    def test_routes_filter(self, cmd):
        cmd.cmd_mcp_filter = MagicMock()
        cmd.cmd_mcp("filter set myserver --allowed t1")
        cmd.cmd_mcp_filter.assert_called_once_with("set myserver --allowed t1")

    def test_unknown_command_prints_error(self, cmd):
        cmd.cmd_mcp("unknown_xyz")
        cmd.console.print.assert_called_with("[red]Invalid MCP command[/red]")


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_list
# ---------------------------------------------------------------------------


class TestCmdMcpList:
    def test_list_error_response(self, cmd):
        cmd.mcp_tool.list_servers.return_value = _err("list failed")
        cmd.cmd_mcp_list()
        cmd.console.print.assert_called()

    def test_list_empty(self, cmd):
        cmd.mcp_tool.list_servers.return_value = _ok({"servers": []})
        # result is falsy list
        r = _ok({"servers": []})
        r.result = {"servers": []}
        # make result["servers"] falsy by making result itself falsy
        r2 = MagicMock()
        r2.success = True
        r2.result = None
        cmd.mcp_tool.list_servers.return_value = r2
        cmd.cmd_mcp_list()
        cmd.console.print.assert_called()

    def test_list_fallback_table_on_screen_error(self, cmd):
        servers = [{"name": "srv1", "type": "builtin", "command": "cmd", "args": []}]
        cmd.mcp_tool.list_servers.return_value = _ok({"servers": servers})
        with patch("datus.cli.mcp_commands.MCPServerApp") as MockApp:
            MockApp.return_value.run.side_effect = Exception("screen error")
            cmd.cmd_mcp_list()
        # Should fall back and print table
        cmd.console.print.assert_called()

    def test_display_servers_table(self, cmd):
        servers = [
            {"name": "s1", "type": "builtin", "command": "cmd1", "args": ["a"]},
            {"name": "s2", "type": "user", "command": "cmd2", "args": []},
        ]
        cmd._display_servers_table(servers)
        cmd.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_add
# ---------------------------------------------------------------------------


class TestCmdMcpAdd:
    def test_add_success(self, cmd):
        cmd.mcp_tool.add_server.return_value = _ok()
        with patch("datus.cli.mcp_commands.parse_command_string", return_value=("stdio", "myserver", {})):
            cmd.cmd_mcp_add("stdio myserver")
        cmd.console.print.assert_called()

    def test_add_failure(self, cmd):
        cmd.mcp_tool.add_server.return_value = _err("add failed")
        with patch("datus.cli.mcp_commands.parse_command_string", return_value=("stdio", "myserver", {})):
            cmd.cmd_mcp_add("stdio myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("add failed" in c for c in calls)

    def test_add_exception(self, cmd):
        with patch("datus.cli.mcp_commands.parse_command_string", side_effect=ValueError("bad args")):
            cmd.cmd_mcp_add("bad_input")
        cmd.console.print.assert_called()


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_remove
# ---------------------------------------------------------------------------


class TestCmdMcpRemove:
    def test_remove_no_name(self, cmd):
        cmd.cmd_mcp_remove("")
        cmd.console.print.assert_called_with("[red]Please specify the name of the MCP server to remove[/red]")

    def test_remove_success(self, cmd):
        cmd.mcp_tool.remove_server.return_value = _ok()
        cmd.cmd_mcp_remove("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("myserver" in c for c in calls)

    def test_remove_failure(self, cmd):
        cmd.mcp_tool.remove_server.return_value = _err("not found")
        cmd.cmd_mcp_remove("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("not found" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_check
# ---------------------------------------------------------------------------


class TestCmdMcpCheck:
    def test_check_no_name(self, cmd):
        cmd.cmd_mcp_check("")
        cmd.console.print.assert_called_with("[red]Please specify the name of the MCP server to check[/red]")

    def test_check_success_reachable(self, cmd):
        cmd.mcp_tool.check_connectivity.return_value = _ok(
            {"connectivity": True, "details": {"type": "stdio", "tools_count": 5}}
        )
        cmd.cmd_mcp_check("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("reachable" in c for c in calls)

    def test_check_success_not_reachable(self, cmd):
        cmd.mcp_tool.check_connectivity.return_value = _ok({"connectivity": False, "details": {"error": "timeout"}})
        cmd.cmd_mcp_check("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("not reachable" in c for c in calls)

    def test_check_failure(self, cmd):
        cmd.mcp_tool.check_connectivity.return_value = _err("check failed")
        cmd.cmd_mcp_check("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("check failed" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_call_tool
# ---------------------------------------------------------------------------


class TestCmdCallTool:
    def test_invalid_format(self, cmd):
        cmd.cmd_call_tool("justserver")
        cmd.console.print.assert_called_with("[bold red]Invalid server.tool format[/]")

    def test_call_success_string_result(self, cmd):
        r = _ok({"result": "plain text"})
        cmd.mcp_tool.call_tool.return_value = r
        cmd.cmd_call_tool("server.tool")
        cmd.console.print.assert_called()

    def test_call_success_json_result(self, cmd):
        r = _ok({"result": json.dumps({"key": "val"})})
        cmd.mcp_tool.call_tool.return_value = r
        cmd.cmd_call_tool("server.tool")
        cmd.console.print.assert_called()

    def test_call_success_no_result(self, cmd):
        r = _ok({"result": None})
        cmd.mcp_tool.call_tool.return_value = r
        cmd.cmd_call_tool("server.tool")
        cmd.console.print.assert_called_with("[bold yellow]No result returned[/]")

    def test_call_failure(self, cmd):
        cmd.mcp_tool.call_tool.return_value = _err("tool error")
        cmd.cmd_call_tool("server.tool")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("tool error" in c for c in calls)

    def test_call_with_json_params(self, cmd):
        r = _ok({"result": "ok"})
        cmd.mcp_tool.call_tool.return_value = r
        cmd.cmd_call_tool('server.tool {"key": "value"}')
        cmd.mcp_tool.call_tool.assert_called_with("server", "tool", {"key": "value"})

    def test_call_with_invalid_json_params(self, cmd):
        cmd.cmd_call_tool("server.tool not_json")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("json" in c.lower() for c in calls)

    def test_call_is_error_result(self, cmd):
        r = _ok({"result": {"isError": True, "content": "boom"}})
        cmd.mcp_tool.call_tool.return_value = r
        cmd.cmd_call_tool("server.tool")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("boom" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_filter routing
# ---------------------------------------------------------------------------


class TestCmdMcpFilter:
    def test_filter_routes_set(self, cmd):
        cmd.cmd_mcp_filter_set = MagicMock()
        cmd.cmd_mcp_filter("set myserver --allowed t1")
        cmd.cmd_mcp_filter_set.assert_called_once_with("myserver --allowed t1")

    def test_filter_routes_get(self, cmd):
        cmd.cmd_mcp_filter_get = MagicMock()
        cmd.cmd_mcp_filter("get myserver")
        cmd.cmd_mcp_filter_get.assert_called_once_with("myserver")

    def test_filter_routes_remove(self, cmd):
        cmd.cmd_mcp_filter_remove = MagicMock()
        cmd.cmd_mcp_filter("remove myserver")
        cmd.cmd_mcp_filter_remove.assert_called_once_with("myserver")

    def test_filter_unknown(self, cmd):
        cmd.cmd_mcp_filter("unknown xyz")
        cmd.console.print.assert_called_with("[red]Invalid filter command. Use: set, get, or remove[/red]")


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_filter_set
# ---------------------------------------------------------------------------


class TestCmdMcpFilterSet:
    def test_set_too_few_params(self, cmd):
        cmd.cmd_mcp_filter_set("")
        cmd.console.print.assert_called()

    def test_set_success(self, cmd):
        cmd.mcp_tool.set_tool_filter.return_value = _ok()
        cmd.cmd_mcp_filter_set("myserver --allowed tool1,tool2 --blocked tool3 --enabled true")
        cmd.mcp_tool.set_tool_filter.assert_called_once_with(
            server_name="myserver",
            allowed_tools=["tool1", "tool2"],
            blocked_tools=["tool3"],
            enabled=True,
        )

    def test_set_failure(self, cmd):
        cmd.mcp_tool.set_tool_filter.return_value = _err("set failed")
        cmd.cmd_mcp_filter_set("myserver --allowed tool1")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("set failed" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_filter_get
# ---------------------------------------------------------------------------


class TestCmdMcpFilterGet:
    def test_get_no_name(self, cmd):
        cmd.cmd_mcp_filter_get("")
        cmd.console.print.assert_called_with("[red]Please specify the name of the MCP server[/red]")

    def test_get_with_filter(self, cmd):
        cmd.mcp_tool.get_tool_filter.return_value = _ok(
            {
                "has_filter": True,
                "filter_config": {
                    "enabled": True,
                    "allowed_tool_names": ["t1"],
                    "blocked_tool_names": None,
                },
            }
        )
        cmd.cmd_mcp_filter_get("myserver")
        cmd.console.print.assert_called()

    def test_get_no_filter(self, cmd):
        cmd.mcp_tool.get_tool_filter.return_value = _ok({"has_filter": False})
        cmd.cmd_mcp_filter_get("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("No tool filter" in c for c in calls)

    def test_get_failure(self, cmd):
        cmd.mcp_tool.get_tool_filter.return_value = _err("get failed")
        cmd.cmd_mcp_filter_get("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("get failed" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: cmd_mcp_filter_remove
# ---------------------------------------------------------------------------


class TestCmdMcpFilterRemove:
    def test_remove_no_name(self, cmd):
        cmd.cmd_mcp_filter_remove("")
        cmd.console.print.assert_called_with("[red]Please specify the name of the MCP server[/red]")

    def test_remove_success(self, cmd):
        cmd.mcp_tool.remove_tool_filter.return_value = _ok()
        cmd.cmd_mcp_filter_remove("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("myserver" in c for c in calls)

    def test_remove_failure(self, cmd):
        cmd.mcp_tool.remove_tool_filter.return_value = _err("remove failed")
        cmd.cmd_mcp_filter_remove("myserver")
        calls = [str(c) for c in cmd.console.print.call_args_list]
        assert any("remove failed" in c for c in calls)
