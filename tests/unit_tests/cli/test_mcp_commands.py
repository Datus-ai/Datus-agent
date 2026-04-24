# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.mcp_commands.MCPCommands`.

Covers:

- Argument dispatch (``cmd_mcp`` routes to the correct subcommand).
- Non-interactive paths (``add`` / ``remove`` / ``check`` / ``call`` /
  ``filter``) hit the expected ``MCPTool`` methods.
- The interactive ``_run_menu`` loop converts each :class:`MCPSelection`
  kind into the correct follow-up call.
- Safe-wrappers (``_safe_list_servers`` / ``_safe_list_tools``) swallow
  underlying exceptions.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.mcp_app import MCPSelection
from datus.cli.mcp_commands import MCPCommands


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
    r.result = None
    return r


class _FakeCli:
    def __init__(self):
        self.console = Console(file=io.StringIO(), no_color=True)
        self.tui_app = None


@pytest.fixture
def commands():
    cli = _FakeCli()
    with patch("datus.cli.mcp_commands.MCPTool") as MockTool:
        mock_tool = MagicMock()
        MockTool.return_value = mock_tool
        cmds = MCPCommands(cli)
        cmds.mcp_tool = mock_tool
        yield cmds, cli


def _output(cli) -> str:
    return cli.console.file.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Argument dispatch
# ─────────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_empty_args_opens_menu(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_mcp("")
        menu.assert_called_once_with()

    def test_list_opens_menu(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_mcp("list")
        menu.assert_called_once_with()

    def test_help_prints_usage_table(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("help")
        assert "/mcp commands" in _output(cli)

    def test_add_with_empty_args_opens_add_form(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_mcp("add")
        menu.assert_called_once_with(seed_view="add_form")

    def test_add_with_raw_args_parses_and_adds(self, commands):
        cmds, _ = commands
        with patch("datus.cli.mcp_commands.parse_command_string") as parse:
            parse.return_value = ("stdio", "srv", {"command": "python"})
            cmds.mcp_tool.add_server.return_value = _ok(result={"name": "srv"})
            cmds.cmd_mcp("add --transport stdio srv python")
        cmds.mcp_tool.add_server.assert_called_once_with(name="srv", type="stdio", command="python")

    def test_add_parse_failure_prints_error(self, commands):
        cmds, cli = commands
        with patch("datus.cli.mcp_commands.parse_command_string", side_effect=ValueError("bad")):
            cmds.cmd_mcp("add something invalid")
        assert "Failed to parse" in _output(cli)

    def test_add_missing_name_prints_error(self, commands):
        cmds, cli = commands
        with patch("datus.cli.mcp_commands.parse_command_string") as parse:
            parse.return_value = ("stdio", None, {})
            cmds.cmd_mcp("add raw")
        assert "name is required" in _output(cli).lower()

    def test_remove_without_name_shows_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("remove")
        assert "Usage" in _output(cli)

    def test_remove_with_confirm_runs_remove_server(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.remove_server.return_value = _ok()
        with patch("datus.cli.mcp_commands.confirm_prompt", return_value=True):
            cmds.cmd_mcp("remove srv")
        cmds.mcp_tool.remove_server.assert_called_once_with("srv")

    def test_remove_cancelled_by_confirm(self, commands):
        cmds, cli = commands
        with patch("datus.cli.mcp_commands.confirm_prompt", return_value=False):
            cmds.cmd_mcp("remove srv")
        cmds.mcp_tool.remove_server.assert_not_called()
        assert "Cancelled" in _output(cli)

    def test_check_empty_opens_menu(self, commands):
        cmds, _ = commands
        with patch.object(cmds, "_run_menu") as menu:
            cmds.cmd_mcp("check")
        menu.assert_called_once_with()

    def test_check_with_name_runs_non_interactive(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.check_connectivity.return_value = _ok(
            {"connectivity": True, "details": {"type": "stdio", "tools_count": 5}}
        )
        cmds.cmd_mcp("check srv")
        cmds.mcp_tool.check_connectivity.assert_called_once_with("srv")
        assert "reachable" in _output(cli)

    def test_check_failure_prints_error(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.check_connectivity.return_value = _err("nope")
        cmds.cmd_mcp("check srv")
        assert "nope" in _output(cli)

    def test_check_not_reachable_reports_failure(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.check_connectivity.return_value = _ok(
            {"connectivity": False, "details": {"type": "stdio", "error": "timeout"}}
        )
        cmds.cmd_mcp("check srv")
        assert "not reachable" in _output(cli)

    def test_unknown_command_prints_error_and_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("frobnicate")
        output = _output(cli)
        assert "Unknown mcp command" in output
        assert "/mcp commands" in output

    def test_shlex_parse_error(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp('remove "unterminated')
        assert "Invalid arguments" in _output(cli)


# ─────────────────────────────────────────────────────────────────────
# _cmd_call
# ─────────────────────────────────────────────────────────────────────


class TestCmdCall:
    def test_missing_args_shows_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("call")
        assert "Usage" in _output(cli)

    def test_invalid_identifier_errors(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("call justserver")
        assert "Invalid server.tool format" in _output(cli)

    def test_call_with_string_json_result(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": '{"a": 1}'})
        cmds.cmd_mcp("call server.tool")
        cmds.mcp_tool.call_tool.assert_called_with("server", "tool", None)

    def test_call_with_args_parsed_as_json(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": "ok"})
        # Single-quote the JSON so shlex preserves it verbatim (matches how a
        # shell user would type the command).
        cmds.cmd_mcp('call server.tool \'{"k": "v"}\'')
        cmds.mcp_tool.call_tool.assert_called_with("server", "tool", {"k": "v"})

    def test_call_with_invalid_json_args_errors(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("call server.tool not_json")
        cmds.mcp_tool.call_tool.assert_not_called()
        assert "valid JSON" in _output(cli)

    def test_call_is_error_flag(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": {"isError": True, "content": "boom"}})
        cmds.cmd_mcp("call server.tool")
        assert "boom" in _output(cli)

    def test_call_empty_result(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": None})
        cmds.cmd_mcp("call server.tool")
        assert "No result" in _output(cli)

    def test_call_failure(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.call_tool.return_value = _err("tool error")
        cmds.cmd_mcp("call server.tool")
        assert "tool error" in _output(cli)

    def test_call_dict_result_printed(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": {"k": "v"}})
        cmds.cmd_mcp("call server.tool")
        cmds.mcp_tool.call_tool.assert_called()


# ─────────────────────────────────────────────────────────────────────
# _cmd_filter
# ─────────────────────────────────────────────────────────────────────


class TestCmdFilter:
    def test_empty_shows_usage(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("filter")
        assert "Usage" in _output(cli)

    def test_invalid_sub(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("filter junk")
        assert "Invalid filter command" in _output(cli)

    def test_set_without_server(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("filter set")
        assert "Usage" in _output(cli)

    def test_set_happy_path(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.set_tool_filter.return_value = _ok()
        cmds.cmd_mcp("filter set srv --allowed t1,t2 --blocked t3 --enabled true")
        cmds.mcp_tool.set_tool_filter.assert_called_once_with(
            server_name="srv",
            allowed_tools=["t1", "t2"],
            blocked_tools=["t3"],
            enabled=True,
        )

    def test_set_without_flags_defaults(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.set_tool_filter.return_value = _ok()
        cmds.cmd_mcp("filter set srv")
        cmds.mcp_tool.set_tool_filter.assert_called_once_with(
            server_name="srv",
            allowed_tools=None,
            blocked_tools=None,
            enabled=True,
        )

    def test_get_no_name(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("filter get")
        assert "specify" in _output(cli).lower()

    def test_get_no_filter(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.get_tool_filter.return_value = _ok({"has_filter": False})
        cmds.cmd_mcp("filter get srv")
        assert "No tool filter" in _output(cli)

    def test_get_with_filter_renders_table(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.get_tool_filter.return_value = _ok(
            {
                "has_filter": True,
                "filter_config": {
                    "enabled": True,
                    "allowed_tool_names": ["t1"],
                    "blocked_tool_names": ["t2"],
                },
            }
        )
        cmds.cmd_mcp("filter get srv")
        cmds.mcp_tool.get_tool_filter.assert_called_once_with("srv")

    def test_get_failure_prints_status(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.get_tool_filter.return_value = _err("gone")
        cmds.cmd_mcp("filter get srv")
        assert "gone" in _output(cli)

    def test_remove_no_name(self, commands):
        cmds, cli = commands
        cmds.cmd_mcp("filter remove")
        assert "specify" in _output(cli).lower()

    def test_remove_happy_path(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.remove_tool_filter.return_value = _ok()
        cmds.cmd_mcp("filter remove srv")
        cmds.mcp_tool.remove_tool_filter.assert_called_once_with("srv")


# ─────────────────────────────────────────────────────────────────────
# _do_add / _do_remove business logic
# ─────────────────────────────────────────────────────────────────────


class TestBusinessLogic:
    def test_do_add_success(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.add_server.return_value = _ok(result={"name": "srv"})
        assert cmds._do_add("srv", "stdio", {"command": "python"}) is True
        cmds.mcp_tool.add_server.assert_called_once_with(name="srv", type="stdio", command="python")

    def test_do_add_failure(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.add_server.return_value = _err("dup")
        assert cmds._do_add("srv", "stdio", {"command": "python"}) is False
        assert "dup" in _output(cli)

    def test_do_add_raises(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.add_server.side_effect = RuntimeError("boom")
        assert cmds._do_add("srv", "stdio", {}) is False
        assert "boom" in _output(cli)

    def test_do_remove_requires_name(self, commands):
        cmds, cli = commands
        assert cmds._do_remove("") is False
        assert "required" in _output(cli).lower()

    def test_do_remove_success(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.remove_server.return_value = _ok()
        assert cmds._do_remove("srv") is True

    def test_do_remove_failure(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.remove_server.return_value = _err("not found")
        assert cmds._do_remove("srv") is False
        assert "not found" in _output(cli)

    def test_do_remove_raises(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.remove_server.side_effect = RuntimeError("io")
        assert cmds._do_remove("srv") is False

    def test_do_set_filter_success(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.set_tool_filter.return_value = _ok()
        cmds._do_set_filter("srv", allowed=["t"], blocked=None, enabled=True)
        assert "updated" in _output(cli).lower()

    def test_do_set_filter_failure(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.set_tool_filter.return_value = _err("nope")
        cmds._do_set_filter("srv", allowed=None, blocked=None, enabled=False)
        assert "nope" in _output(cli)

    def test_do_remove_filter_success(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.remove_tool_filter.return_value = _ok()
        cmds._do_remove_filter("srv")
        assert "removed" in _output(cli).lower()

    def test_do_remove_filter_failure(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.remove_tool_filter.return_value = _err("bad")
        cmds._do_remove_filter("srv")
        assert "bad" in _output(cli)


# ─────────────────────────────────────────────────────────────────────
# Additional edge-path coverage (boosts diff-cover beyond 80%)
# ─────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_add_failure_does_not_seed_new_server(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        cmds.mcp_tool.add_server.return_value = _err("duplicate")
        selections = [
            MCPSelection(kind="add", name="srv", server_type="stdio", config={"command": "python"}),
            MCPSelection(kind="cancel"),
        ]
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        # On failure, we re-enter without seeding the new server as focus.
        assert factory.calls[1]["seed_server"] is None

    def test_call_tool_non_dict_non_str_result(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": [1, 2, 3]})
        cmds.cmd_mcp("call server.tool")
        # Just ensure we rendered without crashing.
        cmds.mcp_tool.call_tool.assert_called_once()

    def test_call_tool_non_parseable_string_printed_raw(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.call_tool.return_value = _ok({"result": "plain text"})
        cmds.cmd_mcp("call server.tool")
        cmds.mcp_tool.call_tool.assert_called_once()

    def test_legacy_list_with_no_columnable_rows(self, commands):
        cmds, cli = commands
        # Fake ``build_row_table`` returning ``None`` (happens when the payload
        # doesn't match list-of-dict shape); we expect the ``print_info``
        # fallback to fire.
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "a", "type": "stdio"}]})
        with patch("datus.cli.mcp_commands.build_row_table", return_value=None):
            cmds.cmd_mcp_list()
        assert "No servers to display" in _output(cli)

    def test_prefetch_status_swallows_worker_exception(self, commands):
        cmds, _ = commands

        # Force ``_check_single`` to raise inside the thread-pool worker.
        def _boom(name):
            raise RuntimeError(f"worker-crash-{name}")

        with patch.object(cmds, "_check_single", side_effect=_boom):
            status = cmds._prefetch_status([{"name": "alpha"}])
        assert "alpha" in status
        assert status["alpha"]["connectivity"] is False
        assert "worker-crash-alpha" in status["alpha"]["error"]


# ─────────────────────────────────────────────────────────────────────
# Safe helpers (exception swallowing)
# ─────────────────────────────────────────────────────────────────────


class TestSafeHelpers:
    def test_safe_list_servers_success(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "srv"}]})
        assert cmds._safe_list_servers() == [{"name": "srv"}]

    def test_safe_list_servers_error(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _err("boom")
        assert cmds._safe_list_servers() == []

    def test_safe_list_servers_raises(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.side_effect = RuntimeError("oops")
        assert cmds._safe_list_servers() == []

    def test_safe_list_tools_success(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_tools.return_value = _ok({"tools": [{"name": "t"}]})
        assert cmds._safe_list_tools("srv") == [{"name": "t"}]

    def test_safe_list_tools_no_name(self, commands):
        cmds, _ = commands
        assert cmds._safe_list_tools("") == []

    def test_safe_list_tools_failure(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_tools.return_value = _err("x")
        assert cmds._safe_list_tools("srv") == []

    def test_safe_list_tools_raises(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_tools.side_effect = RuntimeError("io")
        assert cmds._safe_list_tools("srv") == []

    def test_check_single_success(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.check_connectivity.return_value = _ok({"connectivity": True, "details": {"tools_count": 3}})
        status = cmds._check_single("srv")
        assert status["connectivity"] is True
        assert status["tools_count"] == 3

    def test_check_single_failure(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.check_connectivity.return_value = _err("down")
        status = cmds._check_single("srv")
        assert status["connectivity"] is False
        assert status["error"] == "down"

    def test_check_single_raises(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.check_connectivity.side_effect = RuntimeError("net")
        status = cmds._check_single("srv")
        assert status["connectivity"] is False
        assert "net" in status["error"]

    def test_prefetch_status_empty_returns_empty_dict(self, commands):
        cmds, _ = commands
        assert cmds._prefetch_status([]) == {}

    def test_prefetch_status_calls_check_single(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.check_connectivity.return_value = _ok({"connectivity": True, "details": {"tools_count": 1}})
        result = cmds._prefetch_status([{"name": "a"}, {"name": "b"}])
        assert set(result.keys()) == {"a", "b"}
        assert all(v["connectivity"] for v in result.values())

    def test_prefetch_status_skips_nameless_entries(self, commands):
        cmds, _ = commands
        assert cmds._prefetch_status([{"type": "stdio"}]) == {}


# ─────────────────────────────────────────────────────────────────────
# _run_menu selection routing
# ─────────────────────────────────────────────────────────────────────


class _AppFactory:
    """Script :class:`MCPApp` creations with a pre-defined selection sequence."""

    def __init__(self, selections):
        self._selections = list(selections)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        instance = MagicMock()
        sel = self._selections.pop(0) if self._selections else MCPSelection(kind="cancel")
        instance.run.return_value = sel
        return instance


def _patch_app_run(cmds, selections):
    factory = _AppFactory(selections)
    patcher = patch("datus.cli.mcp_commands.MCPApp", side_effect=factory)
    patcher.start()
    return patcher, factory


class TestRunMenuRouting:
    def test_cancel_returns_immediately(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        patcher, factory = _patch_app_run(cmds, [MCPSelection(kind="cancel")])
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        assert len(factory.calls) == 1

    def test_none_selection_treated_as_cancel(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        patcher, factory = _patch_app_run(cmds, [None])
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        assert len(factory.calls) == 1

    def test_add_selection_calls_do_add_and_refetches(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        cmds.mcp_tool.add_server.return_value = _ok(result={"name": "srv"})
        selections = [
            MCPSelection(kind="add", name="srv", server_type="stdio", config={"command": "python"}),
            MCPSelection(kind="cancel"),
        ]
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        cmds.mcp_tool.add_server.assert_called_once_with(name="srv", type="stdio", command="python")
        # Should reopen once with seed_server pointing at the new server.
        assert len(factory.calls) == 2
        assert factory.calls[1]["seed_server"] == "srv"

    def test_remove_selection_invokes_remove_server(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "srv"}]})
        cmds.mcp_tool.remove_server.return_value = _ok()
        selections = [MCPSelection(kind="remove", name="srv"), MCPSelection(kind="cancel")]
        patcher, _ = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        cmds.mcp_tool.remove_server.assert_called_once_with("srv")

    def test_check_selection_updates_status(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "srv"}]})
        cmds.mcp_tool.check_connectivity.return_value = _ok({"connectivity": True, "details": {"tools_count": 2}})
        selections = [MCPSelection(kind="check", name="srv"), MCPSelection(kind="cancel")]
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        assert factory.calls[1]["status_map"]["srv"]["connectivity"] is True

    def test_refresh_refetches_servers(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "a"}]})
        selections = [MCPSelection(kind="refresh"), MCPSelection(kind="cancel")]
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        # Reopened with a fresh tools_cache.
        assert factory.calls[1]["tools_cache"] == {}
        assert cmds.mcp_tool.list_servers.call_count >= 2

    def test_load_tools_populates_cache(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "srv"}]})
        cmds.mcp_tool.list_tools.return_value = _ok({"tools": [{"name": "t1"}]})
        selections = [MCPSelection(kind="load_tools", name="srv"), MCPSelection(kind="cancel")]
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        assert factory.calls[1]["tools_cache"]["srv"] == [{"name": "t1"}]
        assert factory.calls[1]["seed_view"] == "tools"
        assert factory.calls[1]["seed_server"] == "srv"

    def test_set_filter_selection(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "srv"}]})
        cmds.mcp_tool.set_tool_filter.return_value = _ok()
        selections = [
            MCPSelection(
                kind="set_filter",
                name="srv",
                filter_config={"enabled": True, "allowed": ["t"], "blocked": None},
            ),
            MCPSelection(kind="cancel"),
        ]
        patcher, _ = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        cmds.mcp_tool.set_tool_filter.assert_called_once_with(
            server_name="srv",
            allowed_tools=["t"],
            blocked_tools=None,
            enabled=True,
        )

    def test_remove_filter_selection(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": [{"name": "srv"}]})
        cmds.mcp_tool.remove_tool_filter.return_value = _ok()
        selections = [MCPSelection(kind="remove_filter", name="srv"), MCPSelection(kind="cancel")]
        patcher, _ = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        cmds.mcp_tool.remove_tool_filter.assert_called_once_with("srv")

    def test_unknown_kind_terminates(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        selections = [MCPSelection(kind="bogus")]
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        # Only one app opened — unknown kind returns without re-entering.
        assert len(factory.calls) == 1

    def test_max_reentry_guards_infinite_loop(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        # Feed many refreshes; loop should cap at _MAX_REENTRY and stop cleanly.
        selections = [MCPSelection(kind="refresh")] * 20
        patcher, factory = _patch_app_run(cmds, selections)
        try:
            cmds._run_menu()
        finally:
            patcher.stop()
        # _MAX_REENTRY is 8 in the module
        from datus.cli.mcp_commands import _MAX_REENTRY

        assert len(factory.calls) == _MAX_REENTRY


# ─────────────────────────────────────────────────────────────────────
# _run_app stdin suspension
# ─────────────────────────────────────────────────────────────────────


class TestRunApp:
    def test_run_app_no_tui(self, commands):
        cmds, _ = commands
        app = MagicMock()
        app.run.return_value = MCPSelection(kind="cancel")
        result = cmds._run_app(app)
        assert result.kind == "cancel"

    def test_run_app_suspends_tui(self, commands):
        cmds, cli = commands
        tui_app = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        tui_app.suspend_input.return_value = ctx
        cli.tui_app = tui_app
        app = MagicMock()
        app.run.return_value = MCPSelection(kind="cancel")
        cmds._run_app(app)
        tui_app.suspend_input.assert_called_once()
        ctx.__enter__.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# Legacy list path
# ─────────────────────────────────────────────────────────────────────


class TestLegacyList:
    def test_empty_list(self, commands):
        cmds, cli = commands
        cmds.mcp_tool.list_servers.return_value = _ok({"servers": []})
        cmds.cmd_mcp_list()
        assert "No MCP servers" in _output(cli)

    def test_list_renders_table(self, commands):
        cmds, _ = commands
        cmds.mcp_tool.list_servers.return_value = _ok(
            {
                "servers": [
                    {"name": "a", "type": "stdio", "command": "py", "args": ["-m", "x"]},
                    {"name": "b", "type": "http", "url": "https://x"},
                ]
            }
        )
        cmds.cmd_mcp_list()
        cmds.mcp_tool.list_servers.assert_called_once()
