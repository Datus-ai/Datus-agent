# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.mcp_app.MCPApp`.

The Application is never run under a pty — each test builds an ``MCPApp``
and drives its state machine by invoking the action methods directly
(``_on_remove``, ``_enter_add_form``, ``_submit_add_form``, ...).
``Application.exit`` is patched so we can capture the ``MCPSelection``
the app would have returned to its caller.
"""

from __future__ import annotations

import io
from unittest.mock import patch

from rich.console import Console

from datus.cli.mcp_app import _SERVER_TYPES, MCPApp, MCPSelection, _parse_json_map, _truncate, _View


def _console() -> Console:
    return Console(file=io.StringIO(), no_color=True)


def _server(
    name: str,
    *,
    stype: str = "stdio",
    command: str = "python",
    args=None,
    env=None,
    url: str = "",
    headers=None,
    timeout=None,
    tool_filter=None,
):
    data = {"name": name, "type": stype}
    if stype == "stdio":
        data["command"] = command
        data["args"] = list(args or [])
        data["env"] = dict(env or {})
    else:
        data["url"] = url or f"https://{name}.example.com"
        data["headers"] = dict(headers or {})
        data["timeout"] = timeout
    if tool_filter is not None:
        data["tool_filter"] = tool_filter
    return data


def _build(
    *,
    servers=None,
    status_map=None,
    tools_cache=None,
    seed_view: str = "list",
    seed_server=None,
) -> MCPApp:
    return MCPApp(
        _console(),
        servers=servers,
        status_map=status_map,
        tools_cache=tools_cache,
        seed_view=seed_view,
        seed_server=seed_server,
    )


# ─────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_view_is_list(self):
        app = _build()
        assert app._view is _View.LIST

    def test_seed_view_add_form_opens_form(self):
        app = _build(seed_view="add_form")
        assert app._view is _View.ADD_FORM
        assert app._add_type_idx == 0

    def test_seed_server_positions_cursor(self):
        servers = [_server("a"), _server("b"), _server("c")]
        app = _build(servers=servers, seed_server="b")
        assert app._list_cursor == 1
        assert app._focus_server == "b"

    def test_missing_seed_server_does_not_error(self):
        app = _build(servers=[_server("a")], seed_server="ghost")
        assert app._list_cursor == 0
        # focus_server is retained for subsequent drill-down logic
        assert app._focus_server == "ghost"

    def test_status_map_top_level_is_copied(self):
        original = {"a": {"connectivity": True, "tools_count": 3}}
        app = _build(servers=[_server("a")], status_map=original)
        # Inserting at top level in the source must not leak into the app copy.
        original["b"] = {"connectivity": False}
        assert "b" not in app._status_map


# ─────────────────────────────────────────────────────────────────────
# Client-side filtering
# ─────────────────────────────────────────────────────────────────────


class TestFiltering:
    def test_filter_matches_name(self):
        app = _build(servers=[_server("alpha"), _server("beta")])
        app._filter_query = "alp"
        assert [s["name"] for s in app._visible_servers()] == ["alpha"]

    def test_filter_matches_command(self):
        servers = [_server("a", command="sqlite3"), _server("b", command="duckdb")]
        app = _build(servers=servers)
        app._filter_query = "duck"
        assert [s["name"] for s in app._visible_servers()] == ["b"]

    def test_filter_matches_url(self):
        servers = [_server("a", stype="http", url="https://srv-a.example.com")]
        app = _build(servers=servers)
        app._filter_query = "srv-a"
        assert len(app._visible_servers()) == 1

    def test_empty_filter_returns_all(self):
        servers = [_server("a"), _server("b")]
        app = _build(servers=servers)
        assert app._visible_servers() == servers

    def test_apply_search_filter_resets_cursor(self):
        app = _build(servers=[_server("a"), _server("b")])
        app._list_cursor = 5
        app._list_offset = 3
        app._search_input.text = "b"
        app._view = _View.SEARCH_BAR
        app._apply_search_filter()
        assert app._filter_query == "b"
        assert app._view is _View.LIST
        assert app._list_cursor == 0
        assert app._list_offset == 0

    def test_cancel_search_filter_restores_prior_query(self):
        app = _build()
        app._filter_query = "orig"
        app._search_input.text = "typed-but-discarded"
        app._view = _View.SEARCH_BAR
        app._cancel_search_filter()
        assert app._search_input.text == "orig"
        assert app._view is _View.LIST


# ─────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────


class TestRendering:
    def test_truncate_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_truncate_long_text_ellipsised(self):
        out = _truncate("abcdefghij", 5)
        assert out.endswith("\u2026")
        assert len(out) == 5

    def test_truncate_tiny_limit_is_raw_slice(self):
        assert _truncate("abc", 2) == "ab"

    def test_format_status_unknown_when_empty(self):
        assert MCPApp._format_status_cell({}) == "(unknown)"

    def test_format_status_ok(self):
        label = MCPApp._format_status_cell({"connectivity": True})
        assert "ok" in label

    def test_format_status_err_with_message(self):
        label = MCPApp._format_status_cell({"connectivity": False, "error": "boom"})
        assert "err" in label

    def test_format_status_down_without_error(self):
        label = MCPApp._format_status_cell({"connectivity": False})
        assert "down" in label

    def test_format_server_row_stdio(self):
        app = _build(servers=[_server("srv", command="python", args=["-m", "mod"])])
        row = app._format_server_row(app._servers[0])
        assert "srv" in row
        assert "stdio" in row
        assert "python" in row

    def test_render_list_empty(self):
        app = _build()
        parts = app._render_list()
        assert any("no MCP servers" in text for _, text in parts)

    def test_render_list_with_filter_query_shows_hint(self):
        app = _build(servers=[_server("alpha")])
        app._filter_query = "zzz"
        parts = app._render_list()
        assert any("no servers match" in text for _, text in parts)

    def test_render_list_highlights_cursor(self):
        app = _build(servers=[_server("a"), _server("b")])
        parts = app._render_list()
        combined = "".join(text for _, text in parts)
        assert "\u2192" in combined  # SYM_ARROW default

    def test_render_tool_detail_lists_required_parameters(self):
        tools = [
            {
                "name": "query",
                "description": "run a query",
                "inputSchema": {
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "sql text"}},
                    "required": ["sql"],
                },
            }
        ]
        app = _build(servers=[_server("srv")], tools_cache={"srv": tools}, seed_server="srv")
        app._view = _View.TOOL_DETAIL
        app._tools_cursor = 0
        parts = app._render_tool_detail()
        combined = "".join(text for _, text in parts)
        assert "sql" in combined
        assert "required" in combined

    def test_render_tool_detail_handles_no_parameters(self):
        tools = [{"name": "ping", "description": "pong"}]
        app = _build(servers=[_server("srv")], tools_cache={"srv": tools}, seed_server="srv")
        app._view = _View.TOOL_DETAIL
        parts = app._render_tool_detail()
        combined = "".join(text for _, text in parts)
        assert "no parameters" in combined


# ─────────────────────────────────────────────────────────────────────
# Detail field mapping
# ─────────────────────────────────────────────────────────────────────


class TestDetailMapping:
    def test_stdio_fields(self):
        app = _build()
        row = _server("srv", command="uv", args=["run", "mcp"], env={"KEY": "val"})
        fields = dict(app._detail_fields(row))
        assert fields["Command"] == "uv"
        assert "run mcp" == fields["Args"]
        assert "KEY=val" == fields["Env"]

    def test_http_fields(self):
        app = _build()
        row = _server("srv", stype="http", url="https://x", headers={"X-API": "k"}, timeout=30)
        fields = dict(app._detail_fields(row))
        assert fields["URL"] == "https://x"
        assert "X-API=k" == fields["Headers"]
        assert fields["Timeout"] == "30s"

    def test_filter_rendering(self):
        app = _build()
        row = _server(
            "srv",
            tool_filter={
                "enabled": True,
                "allowed_tool_names": ["t1", "t2"],
                "blocked_tool_names": ["bad"],
            },
        )
        fields = dict(app._detail_fields(row))
        assert fields["Filter Enabled"] == "yes"
        assert fields["Allowed"] == "t1, t2"
        assert fields["Blocked"] == "bad"

    def test_no_filter_shown_as_none(self):
        app = _build()
        row = _server("srv")
        fields = dict(app._detail_fields(row))
        assert fields["Filter"] == "(none)"

    def test_status_fields_injected(self):
        status = {"srv": {"connectivity": True, "tools_count": 4}}
        app = _build(servers=[_server("srv")], status_map=status)
        row = app._servers[0]
        fields = dict(app._detail_fields(row))
        assert "ok" in fields["Status"]
        assert fields["Tools Count"] == "4"


# ─────────────────────────────────────────────────────────────────────
# Action handlers → MCPSelection
# ─────────────────────────────────────────────────────────────────────


class TestActions:
    def test_remove_two_press_confirmation(self):
        app = _build(servers=[_server("srv")])
        with patch.object(app._app, "exit") as exit_mock:
            app._on_remove()
            assert exit_mock.call_count == 0
            assert app._pending_remove == "srv"
            assert "Press r again" in (app._error_message or "")
            app._on_remove()
            assert exit_mock.call_count == 1
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "remove"
        assert sel.name == "srv"
        assert app._pending_remove is None

    def test_remove_without_selection_is_noop(self):
        app = _build()
        with patch.object(app._app, "exit") as exit_mock:
            app._on_remove()
        exit_mock.assert_not_called()

    def test_check_emits_selection(self):
        app = _build(servers=[_server("srv")])
        with patch.object(app._app, "exit") as exit_mock:
            app._on_check_current()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "check"
        assert sel.name == "srv"

    def test_refresh_emits_selection(self):
        app = _build()
        with patch.object(app._app, "exit") as exit_mock:
            app._on_refresh()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "refresh"

    def test_enter_tools_with_cache_sets_view(self):
        app = _build(
            servers=[_server("srv")],
            tools_cache={"srv": [{"name": "t"}]},
        )
        app._focus_server = "srv"
        with patch.object(app._app, "exit") as exit_mock:
            app._enter_tools()
        exit_mock.assert_not_called()
        assert app._view is _View.TOOLS

    def test_enter_tools_without_cache_emits_load_tools(self):
        app = _build(servers=[_server("srv")])
        app._focus_server = "srv"
        with patch.object(app._app, "exit") as exit_mock:
            app._enter_tools()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "load_tools"
        assert sel.name == "srv"

    def test_enter_detail_sets_focus_server(self):
        app = _build(servers=[_server("alpha"), _server("beta")])
        app._list_cursor = 1
        app._enter_detail()
        assert app._view is _View.DETAIL
        assert app._focus_server == "beta"

    def test_enter_detail_without_servers_does_nothing(self):
        app = _build()
        app._enter_detail()
        assert app._view is _View.LIST


# ─────────────────────────────────────────────────────────────────────
# Add form submission
# ─────────────────────────────────────────────────────────────────────


class TestAddForm:
    def test_stdio_submission_minimal(self):
        app = _build()
        app._enter_add_form()
        app._add_name.text = "srv"
        app._add_command.text = "python"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "add"
        assert sel.name == "srv"
        assert sel.server_type == "stdio"
        assert sel.config == {"command": "python"}

    def test_stdio_submission_with_args_and_env(self):
        app = _build()
        app._enter_add_form()
        app._add_name.text = "srv"
        app._add_command.text = "uv"
        app._add_args.text = "run, mcp"
        app._add_env.text = '{"KEY": "value"}'
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.config["args"] == ["run", "mcp"]
        assert sel.config["env"] == {"KEY": "value"}

    def test_stdio_requires_command(self):
        app = _build()
        app._enter_add_form()
        app._add_name.text = "srv"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        exit_mock.assert_not_called()
        assert "Command" in (app._error_message or "")

    def test_requires_name(self):
        app = _build()
        app._enter_add_form()
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        exit_mock.assert_not_called()
        assert "Name" in (app._error_message or "")

    def test_http_submission(self):
        app = _build()
        app._enter_add_form()
        app._add_type_idx = _SERVER_TYPES.index("http")
        app._sync_add_focus_order()
        app._add_name.text = "remote"
        app._add_url.text = "https://example.com/mcp"
        app._add_headers.text = '{"Authorization": "Bearer x"}'
        app._add_timeout.text = "15"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.server_type == "http"
        assert sel.config["url"] == "https://example.com/mcp"
        assert sel.config["headers"] == {"Authorization": "Bearer x"}
        assert sel.config["timeout"] == 15.0

    def test_http_requires_url(self):
        app = _build()
        app._enter_add_form()
        app._add_type_idx = _SERVER_TYPES.index("http")
        app._sync_add_focus_order()
        app._add_name.text = "remote"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        exit_mock.assert_not_called()
        assert "URL" in (app._error_message or "")

    def test_invalid_env_json(self):
        app = _build()
        app._enter_add_form()
        app._add_name.text = "srv"
        app._add_command.text = "python"
        app._add_env.text = "not json"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        exit_mock.assert_not_called()
        assert "Env" in (app._error_message or "")

    def test_invalid_headers_json(self):
        app = _build()
        app._enter_add_form()
        app._add_type_idx = _SERVER_TYPES.index("sse")
        app._sync_add_focus_order()
        app._add_name.text = "sse"
        app._add_url.text = "https://example"
        app._add_headers.text = "not json"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        exit_mock.assert_not_called()
        assert "Headers" in (app._error_message or "")

    def test_invalid_timeout(self):
        app = _build()
        app._enter_add_form()
        app._add_type_idx = _SERVER_TYPES.index("http")
        app._sync_add_focus_order()
        app._add_name.text = "x"
        app._add_url.text = "http://x"
        app._add_timeout.text = "not-a-number"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_add_form()
        exit_mock.assert_not_called()
        assert "Timeout" in (app._error_message or "")


# ─────────────────────────────────────────────────────────────────────
# Filter form submission
# ─────────────────────────────────────────────────────────────────────


class TestFilterForm:
    def test_submit_collects_config(self):
        app = _build(servers=[_server("srv")])
        app._focus_server = "srv"
        app._enter_filter_form()
        app._filter_enabled.text = "y"
        app._filter_allowed.text = "t1, t2"
        app._filter_blocked.text = "bad"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_filter_form()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "set_filter"
        assert sel.name == "srv"
        assert sel.filter_config["enabled"] is True
        assert sel.filter_config["allowed"] == ["t1", "t2"]
        assert sel.filter_config["blocked"] == ["bad"]

    def test_submit_without_selection_is_error(self):
        app = _build()
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_filter_form()
        exit_mock.assert_not_called()
        assert "No server" in (app._error_message or "")

    def test_enabled_blank_defaults_to_true(self):
        app = _build(servers=[_server("srv")])
        app._focus_server = "srv"
        app._filter_enabled.text = ""
        app._filter_allowed.text = ""
        app._filter_blocked.text = ""
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_filter_form()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.filter_config["enabled"] is True
        assert sel.filter_config["allowed"] is None

    def test_enabled_no_sets_false(self):
        app = _build(servers=[_server("srv")])
        app._focus_server = "srv"
        app._filter_enabled.text = "no"
        with patch.object(app._app, "exit") as exit_mock:
            app._submit_filter_form()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.filter_config["enabled"] is False

    def test_drop_filter_emits_selection(self):
        app = _build(servers=[_server("srv")])
        app._focus_server = "srv"
        with patch.object(app._app, "exit") as exit_mock:
            app._drop_filter()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "remove_filter"
        assert sel.name == "srv"

    def test_drop_filter_without_selection_errors(self):
        app = _build()
        with patch.object(app._app, "exit") as exit_mock:
            app._drop_filter()
        exit_mock.assert_not_called()
        assert "No server" in (app._error_message or "")

    def test_filter_form_pre_populates_from_existing(self):
        filter_cfg = {
            "enabled": False,
            "allowed_tool_names": ["ra"],
            "blocked_tool_names": ["rb"],
        }
        app = _build(servers=[_server("srv", tool_filter=filter_cfg)], seed_server="srv")
        app._enter_filter_form()
        assert app._filter_enabled.text == "n"
        assert app._filter_allowed.text == "ra"
        assert app._filter_blocked.text == "rb"


# ─────────────────────────────────────────────────────────────────────
# Form focus cycling
# ─────────────────────────────────────────────────────────────────────


class TestFormFocus:
    def test_add_form_dynamic_fields_change_with_type(self):
        app = _build()
        app._enter_add_form()
        stdio_fields = app._dynamic_add_fields()
        app._add_type_idx = _SERVER_TYPES.index("sse")
        sse_fields = app._dynamic_add_fields()
        assert stdio_fields != sse_fields
        assert app._add_command in stdio_fields
        assert app._add_url in sse_fields

    def test_sync_add_focus_order_preserves_idx_on_shrink(self):
        app = _build()
        app._enter_add_form()
        app._form_focus_idx = 10  # far past valid range
        app._sync_add_focus_order()
        assert app._form_focus_idx == len(app._form_focus_order) - 1

    def test_advance_form_focus_wraps(self):
        app = _build()
        app._enter_add_form()
        fields = app._form_focus_order
        assert fields
        app._form_focus_idx = len(fields) - 1
        app._advance_form_focus(+1)
        assert app._form_focus_idx == 0


# ─────────────────────────────────────────────────────────────────────
# Cursor / scroll
# ─────────────────────────────────────────────────────────────────────


class TestCursor:
    def test_clamp_cursor_on_empty_list(self):
        app = _build()
        app._list_cursor = 5
        app._clamp_list_cursor(0)
        assert app._list_cursor == 0
        assert app._list_offset == 0

    def test_clamp_cursor_upper_bound(self):
        app = _build()
        app._list_cursor = 20
        app._clamp_list_cursor(5)
        assert app._list_cursor == 4

    def test_visible_slice_no_scroll_when_small_list(self):
        app = _build()
        start, end = app._visible_slice(0, 0, 3)
        assert (start, end) == (0, 3)

    def test_visible_slice_advances_offset_when_cursor_passes_max(self):
        app = _build()
        app._max_visible = 5
        start, end = app._visible_slice(6, 0, 20)
        assert start == 2
        assert end == 7


# ─────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────


class TestModuleHelpers:
    def test_parse_json_map_returns_dict(self):
        assert _parse_json_map('{"a": 1}') == {"a": 1}

    def test_parse_json_map_non_dict_returns_none(self):
        assert _parse_json_map("[1, 2]") is None

    def test_parse_json_map_invalid_returns_none(self):
        assert _parse_json_map("nope") is None


# ─────────────────────────────────────────────────────────────────────
# MCPSelection dataclass
# ─────────────────────────────────────────────────────────────────────


class TestMCPSelection:
    def test_defaults(self):
        sel = MCPSelection(kind="cancel")
        assert sel.kind == "cancel"
        assert sel.name is None
        assert sel.config is None

    def test_custom_fields(self):
        sel = MCPSelection(
            kind="add",
            name="srv",
            server_type="stdio",
            config={"command": "python"},
        )
        assert sel.server_type == "stdio"
        assert sel.config["command"] == "python"
