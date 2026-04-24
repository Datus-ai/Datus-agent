# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/mcp`` browser rendered as a single prompt_toolkit Application.

Mirrors :mod:`datus.cli.skill_app` / :mod:`datus.cli.model_app`:

- A single Application hosts the server list, detail drill-down, tools
  drill-down, add/filter forms, and two-press remove confirmation — so the
  outer :class:`~datus.cli.tui.app.DatusApp` only releases stdin once via
  :meth:`DatusApp.suspend_input`.
- All slow / blocking I/O (connectivity checks, ``list_tools`` RPCs, config
  writes) runs **after** :meth:`MCPApp.run` returns, driven by
  :class:`~datus.cli.mcp_commands.MCPCommands`.
- Connectivity status and tools lists are pre-fetched once by the caller and
  passed into the Application via ``status_map`` / ``tools_cache``. Pressing
  ``R`` / ``c`` / ``t`` exits with a matching :class:`MCPSelection` so the
  caller can refresh and reopen.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import DynamicContainer, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from datus.cli.cli_styles import CLR_CURSOR, SYM_ARROW, print_error, render_tui_title_bar
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_SERVER_TYPES: Tuple[str, ...] = ("stdio", "sse", "http")


class _View(Enum):
    LIST = "list"
    DETAIL = "detail"
    TOOLS = "tools"
    TOOL_DETAIL = "tool_detail"
    ADD_FORM = "add_form"
    FILTER_FORM = "filter_form"
    SEARCH_BAR = "search_bar"


@dataclass
class MCPSelection:
    """Outcome of an :class:`MCPApp` run.

    ``kind`` discriminates the payload:

    - ``"add"`` — ``name``, ``server_type``, ``config`` for ``add_server``.
    - ``"remove"`` — ``name`` of a server to remove.
    - ``"check"`` — ``name`` of a server to re-ping; ``None`` means "refresh all".
    - ``"load_tools"`` — ``name`` of a server whose tools should be fetched.
    - ``"set_filter"`` — ``name`` + ``filter_config`` to apply via
      ``set_tool_filter``.
    - ``"remove_filter"`` — ``name`` whose filter should be cleared.
    - ``"refresh"`` — caller should re-fetch the full server list & status.
    - ``"cancel"`` — user dismissed the app (also returned for ``None``).
    """

    kind: str
    name: Optional[str] = None
    server_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    filter_config: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None


class MCPApp:
    """Single-Application ``/mcp`` browser.

    The caller (``MCPCommands._run_menu``) is expected to:

    1. Pre-fetch the full server list (``servers``) and an optional status
       map (``status_map``: ``{name: {connectivity, tools_count, error}}``).
    2. Optionally pre-fetch a tools cache keyed by server name.
    3. Wrap :meth:`run` in ``tui_app.suspend_input()`` when the REPL is in
       TUI mode (no-op otherwise).
    4. Apply the returned :class:`MCPSelection` by calling the matching
       :class:`~datus.tools.mcp_tools.mcp_tool.MCPTool` method or by
       re-opening the app after a refresh / tools load.
    """

    def __init__(
        self,
        console: Console,
        *,
        servers: Optional[List[Dict[str, Any]]] = None,
        status_map: Optional[Dict[str, Dict[str, Any]]] = None,
        tools_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        seed_view: str = "list",
        seed_server: Optional[str] = None,
    ) -> None:
        self._console = console
        self._servers: List[Dict[str, Any]] = list(servers or [])
        self._status_map: Dict[str, Dict[str, Any]] = dict(status_map or {})
        self._tools_cache: Dict[str, List[Dict[str, Any]]] = dict(tools_cache or {})

        self._view: _View = _View.LIST
        self._list_cursor: int = 0
        self._list_offset: int = 0
        self._tools_cursor: int = 0
        self._tools_offset: int = 0
        self._filter_query: str = ""

        self._focus_server: Optional[str] = seed_server
        if seed_server is not None:
            for idx, row in enumerate(self._servers):
                if row.get("name") == seed_server:
                    self._list_cursor = idx
                    break

        # Pending two-press remove: armed server name, waiting for second ``r``.
        self._pending_remove: Optional[str] = None

        # Form state for ADD.
        self._add_type_idx: int = 0  # index into _SERVER_TYPES
        self._add_name = TextArea(height=1, multiline=False, prompt="Name:        ", focus_on_click=True)
        self._add_command = TextArea(height=1, multiline=False, prompt="Command:     ", focus_on_click=True)
        self._add_args = TextArea(height=1, multiline=False, prompt="Args (CSV):  ", focus_on_click=True)
        self._add_env = TextArea(height=1, multiline=False, prompt="Env (JSON):  ", focus_on_click=True)
        self._add_url = TextArea(height=1, multiline=False, prompt="URL:         ", focus_on_click=True)
        self._add_headers = TextArea(height=1, multiline=False, prompt="Headers(JSON):", focus_on_click=True)
        self._add_timeout = TextArea(height=1, multiline=False, prompt="Timeout (s): ", focus_on_click=True)

        # Form state for FILTER.
        self._filter_enabled = TextArea(height=1, multiline=False, prompt="Enabled (y/n): ", focus_on_click=True)
        self._filter_allowed = TextArea(height=1, multiline=False, prompt="Allowed (CSV): ", focus_on_click=True)
        self._filter_blocked = TextArea(height=1, multiline=False, prompt="Blocked (CSV): ", focus_on_click=True)

        # Search bar (client-side filter on the LIST view).
        self._search_input = TextArea(height=1, multiline=False, prompt="/", focus_on_click=True)

        self._form_focus_order: List[TextArea] = []
        self._form_focus_idx: int = 0

        self._result: Optional[MCPSelection] = None
        self._error_message: Optional[str] = None

        term_height = shutil.get_terminal_size((120, 40)).lines
        # Title(1) + separator(1) + body + error(1) + separator(1) + hint(1) = 5 overhead.
        self._max_visible: int = max(3, min(18, term_height - 6))

        self._app = self._build_application()

        if seed_view == "add_form":
            self._enter_add_form()
        elif seed_view == "tools" and seed_server:
            self._view = _View.TOOLS
            self._tools_cursor = 0
            self._tools_offset = 0
        elif seed_view == "detail" and seed_server:
            self._view = _View.DETAIL

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Optional[MCPSelection]:
        """Run the Application. Returns ``None`` on KeyboardInterrupt."""
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("MCPApp crashed: %s", exc)
            print_error(self._console, f"/mcp error: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            style="class:mcp-app.list",
            height=Dimension(min=3),
        )
        detail_window = Window(
            content=FormattedTextControl(self._render_detail, focusable=False),
            always_hide_cursor=True,
            style="class:mcp-app.detail",
            height=Dimension(min=3),
        )
        tools_window = Window(
            content=FormattedTextControl(self._render_tools_list, focusable=True),
            always_hide_cursor=True,
            style="class:mcp-app.tools",
            height=Dimension(min=3),
        )
        tool_detail_window = Window(
            content=FormattedTextControl(self._render_tool_detail, focusable=False),
            always_hide_cursor=True,
            style="class:mcp-app.tool-detail",
            height=Dimension(min=3),
        )

        add_form = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_add_header, focusable=False),
                    height=Dimension(min=2, max=4),
                ),
                self._add_name,
                DynamicContainer(self._add_dynamic_body),
            ]
        )

        filter_form = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_filter_header, focusable=False),
                    height=Dimension(min=1, max=3),
                ),
                self._filter_enabled,
                self._filter_allowed,
                self._filter_blocked,
            ]
        )

        search_bar = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_search_header, focusable=False),
                    height=1,
                ),
                self._search_input,
                list_window,
            ]
        )

        def _body_container():
            if self._view == _View.DETAIL:
                return detail_window
            if self._view == _View.TOOLS:
                return tools_window
            if self._view == _View.TOOL_DETAIL:
                return tool_detail_window
            if self._view == _View.ADD_FORM:
                return add_form
            if self._view == _View.FILTER_FORM:
                return filter_form
            if self._view == _View.SEARCH_BAR:
                return search_bar
            return list_window

        body = DynamicContainer(_body_container)

        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
            style="class:mcp-app.hint",
        )
        error_window = Window(
            content=FormattedTextControl(
                lambda: [("class:mcp-app.error", f"  {self._error_message or ''}")] if self._error_message else []
            ),
            height=Dimension(min=0, max=1),
            style="class:mcp-app.error",
        )

        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("MCP Servers")),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                Window(height=1, char="\u2500", style="class:mcp-app.separator"),
                body,
                error_window,
                Window(height=1, char="\u2500", style="class:mcp-app.separator"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=None),
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    # ─────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.LIST:
            hint = (
                "  \u2191\u2193 navigate   Enter details   a add   r remove   c check   "
                "f filter   / find   R refresh   q/Esc quit"
            )
        elif self._view == _View.DETAIL:
            hint = "  t tools   c check   f filter   r remove   Esc back"
        elif self._view == _View.TOOLS:
            hint = "  \u2191\u2193 navigate   Enter details   Esc back"
        elif self._view == _View.TOOL_DETAIL:
            hint = "  Esc back"
        elif self._view == _View.ADD_FORM:
            hint = (
                "  Tab next field   \u2190/\u2192 switch type   Ctrl+S submit   Enter submit (last field)   Esc cancel"
            )
        elif self._view == _View.FILTER_FORM:
            hint = "  Tab next field   Ctrl+S submit   D remove filter   Esc cancel"
        else:
            hint = "  Type to filter   Enter apply   Esc clear"
        return [("class:mcp-app.hint", hint)]

    def _render_list(self) -> List[Tuple[str, str]]:
        items = self._visible_servers()
        if not items:
            msg = "  (no MCP servers configured — press `a` to add one)\n"
            if self._filter_query:
                msg = f"  (no servers match `{self._filter_query}`)\n"
            return [("class:mcp-app.dim", msg)]
        self._clamp_list_cursor(len(items))
        start, end = self._visible_slice(self._list_cursor, self._list_offset_getter(), len(items))
        self._list_offset = start
        lines: List[Tuple[str, str]] = []
        header = f"  {'Name':<24} {'Type':<8} {'Status':<12} {'Tools':>6}  Target\n"
        lines.append(("bold", header))
        if end - start < len(items):
            lines.append(("class:mcp-app.scroll", f"  ({start + 1}-{end} of {len(items)})\n"))
        for i in range(start, end):
            server = items[i]
            label = self._format_server_row(server)
            if i == self._list_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_detail(self) -> List[Tuple[str, str]]:
        row = self._current_server()
        if row is None:
            return [("class:mcp-app.dim", "  (no server selected)\n")]
        lines: List[Tuple[str, str]] = []
        for label, value in self._detail_fields(row):
            lines.append(("bold", f"  {label}: "))
            lines.append(("", f"{value}\n"))
        return lines

    def _render_tools_list(self) -> List[Tuple[str, str]]:
        name = self._focused_server_name()
        if name is None:
            return [("class:mcp-app.dim", "  (no server selected)\n")]
        tools = self._tools_cache.get(name)
        if tools is None:
            return [("class:mcp-app.dim", f"  (loading tools for `{name}`...)\n")]
        if not tools:
            return [("class:mcp-app.dim", f"  (no tools available on `{name}`)\n")]
        self._clamp_tools_cursor(len(tools))
        start, end = self._visible_slice(self._tools_cursor, self._tools_offset, len(tools))
        self._tools_offset = start
        lines: List[Tuple[str, str]] = [("bold", f"  Tools on `{name}` ({len(tools)}):\n")]
        if end - start < len(tools):
            lines.append(("class:mcp-app.scroll", f"  ({start + 1}-{end} of {len(tools)})\n"))
        for i in range(start, end):
            tool = tools[i]
            tool_name = str(tool.get("name", "?"))
            desc = _truncate(str(tool.get("description", "") or ""), 64)
            label = f"{tool_name:<28} {desc}"
            if i == self._tools_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_tool_detail(self) -> List[Tuple[str, str]]:
        name = self._focused_server_name()
        if name is None:
            return [("class:mcp-app.dim", "  (no server selected)\n")]
        tools = self._tools_cache.get(name) or []
        if not tools or self._tools_cursor < 0 or self._tools_cursor >= len(tools):
            return [("class:mcp-app.dim", "  (no tool selected)\n")]
        tool = tools[self._tools_cursor]
        lines: List[Tuple[str, str]] = []
        lines.append(("bold", "  Name: "))
        lines.append(("", f"{tool.get('name', '?')}\n"))
        lines.append(("bold", "  Description: "))
        lines.append(("", f"{tool.get('description', '') or '-'}\n"))
        schema = tool.get("inputSchema") or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        if properties:
            lines.append(("bold", "  Parameters:\n"))
            for param_name, info in properties.items():
                req = "required" if param_name in required else "optional"
                ptype = info.get("type", "unknown") if isinstance(info, dict) else "unknown"
                pdesc = info.get("description", "") if isinstance(info, dict) else ""
                lines.append(("", f"    - {param_name} ({ptype}, {req})"))
                if pdesc:
                    lines.append(("class:mcp-app.dim", f"  {pdesc}\n"))
                else:
                    lines.append(("", "\n"))
        else:
            lines.append(("class:mcp-app.dim", "  (tool takes no parameters)\n"))
        return lines

    def _render_add_header(self) -> List[Tuple[str, str]]:
        active = _SERVER_TYPES[self._add_type_idx]
        parts: List[Tuple[str, str]] = [("bold", "  Add MCP Server\n"), ("", "  Type: ")]
        for t in _SERVER_TYPES:
            style = "reverse bold" if t == active else "class:mcp-app.dim"
            parts.append((style, f" {t} "))
            parts.append(("", " "))
        parts.append(("class:mcp-app.dim", "  (\u2190/\u2192 to switch)\n"))
        return parts

    def _render_filter_header(self) -> List[Tuple[str, str]]:
        name = self._focused_server_name() or "?"
        return [
            ("bold", f"  Tool Filter — `{name}`\n"),
            (
                "class:mcp-app.dim",
                "  Allowed/Blocked are comma-separated names. Ctrl+S to submit, D to drop filter, Esc to cancel.\n",
            ),
        ]

    def _render_search_header(self) -> List[Tuple[str, str]]:
        return [("bold", "  Filter servers\n")]

    def _add_dynamic_body(self):
        active = _SERVER_TYPES[self._add_type_idx]
        if active == "stdio":
            return HSplit([self._add_command, self._add_args, self._add_env])
        # sse / http share the same fields
        return HSplit([self._add_url, self._add_headers, self._add_timeout])

    # ─────────────────────────────────────────────────────────────────
    # Data helpers
    # ─────────────────────────────────────────────────────────────────

    def _visible_servers(self) -> List[Dict[str, Any]]:
        if not self._filter_query:
            return self._servers
        q = self._filter_query.lower()
        return [s for s in self._servers if self._match_server(s, q)]

    @staticmethod
    def _match_server(server: Dict[str, Any], q: str) -> bool:
        haystack = " ".join(
            filter(
                None,
                [
                    str(server.get("name", "")),
                    str(server.get("type", "")),
                    str(server.get("command", "")),
                    str(server.get("url", "")),
                ],
            )
        ).lower()
        return q in haystack

    def _current_server(self) -> Optional[Dict[str, Any]]:
        items = self._visible_servers()
        if not items or self._list_cursor < 0 or self._list_cursor >= len(items):
            return None
        return items[self._list_cursor]

    def _focused_server_name(self) -> Optional[str]:
        """Return the name the user is currently drilling into.

        In DETAIL / TOOLS / TOOL_DETAIL / FILTER_FORM we use the cached
        ``_focus_server``; falling back to the list cursor keeps the
        interactive path working before any drill-down has occurred.
        """
        if self._focus_server is not None:
            return self._focus_server
        row = self._current_server()
        return row.get("name") if row else None

    def _format_server_row(self, server: Dict[str, Any]) -> str:
        name = str(server.get("name", "?"))
        stype = str(server.get("type", "?"))
        target = str(server.get("command") or server.get("url") or "-")
        status = self._status_map.get(name) or {}
        status_label = self._format_status_cell(status)
        tools_count = status.get("tools_count")
        tools_cell = str(tools_count) if isinstance(tools_count, int) else "-"
        return f"{_truncate(name, 23):<24} {stype:<8} {status_label:<12} {tools_cell:>6}  {_truncate(target, 40)}"

    @staticmethod
    def _format_status_cell(status: Dict[str, Any]) -> str:
        if not status:
            return "(unknown)"
        if status.get("connectivity"):
            return "\u2713 ok"
        err = status.get("error")
        if err:
            return "\u2717 err"
        return "\u2717 down"

    def _detail_fields(self, server: Dict[str, Any]) -> List[Tuple[str, str]]:
        name = str(server.get("name", ""))
        stype = str(server.get("type", ""))
        rows: List[Tuple[str, str]] = [
            ("Name", name),
            ("Type", stype),
        ]
        if stype == "stdio":
            rows.append(("Command", str(server.get("command") or "-")))
            args = server.get("args") or []
            rows.append(("Args", " ".join(str(a) for a in args) if args else "-"))
            env = server.get("env") or {}
            rows.append(("Env", ", ".join(f"{k}={v}" for k, v in env.items()) if env else "-"))
        else:
            rows.append(("URL", str(server.get("url") or "-")))
            headers = server.get("headers") or {}
            rows.append(("Headers", ", ".join(f"{k}={v}" for k, v in headers.items()) if headers else "-"))
            timeout = server.get("timeout")
            rows.append(("Timeout", f"{timeout}s" if timeout is not None else "-"))
        tool_filter = server.get("tool_filter") or {}
        if tool_filter:
            allowed = tool_filter.get("allowed_tool_names") or []
            blocked = tool_filter.get("blocked_tool_names") or []
            rows.append(("Filter Enabled", "yes" if tool_filter.get("enabled") else "no"))
            rows.append(("Allowed", ", ".join(allowed) if allowed else "(none)"))
            rows.append(("Blocked", ", ".join(blocked) if blocked else "(none)"))
        else:
            rows.append(("Filter", "(none)"))
        status = self._status_map.get(name) or {}
        rows.append(("Status", self._format_status_cell(status)))
        tools_count = status.get("tools_count")
        rows.append(("Tools Count", str(tools_count) if isinstance(tools_count, int) else "-"))
        if status.get("error"):
            rows.append(("Last Error", str(status.get("error"))))
        return rows

    # ─────────────────────────────────────────────────────────────────
    # Cursor / scroll helpers
    # ─────────────────────────────────────────────────────────────────

    def _clamp_list_cursor(self, total: int) -> None:
        if total <= 0:
            self._list_cursor = 0
            self._list_offset = 0
            return
        if self._list_cursor >= total:
            self._list_cursor = total - 1
        if self._list_cursor < 0:
            self._list_cursor = 0

    def _clamp_tools_cursor(self, total: int) -> None:
        if total <= 0:
            self._tools_cursor = 0
            self._tools_offset = 0
            return
        if self._tools_cursor >= total:
            self._tools_cursor = total - 1
        if self._tools_cursor < 0:
            self._tools_cursor = 0

    def _visible_slice(self, cursor: int, offset: int, total: int) -> Tuple[int, int]:
        max_visible = self._max_visible
        if total <= max_visible:
            return 0, total
        if cursor < offset:
            offset = cursor
        elif cursor >= offset + max_visible:
            offset = cursor - max_visible + 1
        start = max(0, min(offset, total - max_visible))
        return start, start + max_visible

    def _list_offset_getter(self) -> int:
        return self._list_offset

    # ─────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────

    def _enter_detail(self) -> None:
        row = self._current_server()
        if row is None:
            return
        self._focus_server = str(row.get("name")) if row.get("name") else None
        self._view = _View.DETAIL
        self._error_message = None
        self._pending_remove = None

    def _enter_tools(self) -> None:
        name = self._focused_server_name()
        if not name:
            return
        if name not in self._tools_cache:
            # Caller must load tools first; exit with kind="load_tools".
            self._result = MCPSelection(kind="load_tools", name=name)
            self._app.exit(result=self._result)
            return
        self._view = _View.TOOLS
        self._tools_cursor = 0
        self._tools_offset = 0
        self._error_message = None

    def _enter_tool_detail(self) -> None:
        name = self._focused_server_name()
        if not name:
            return
        tools = self._tools_cache.get(name) or []
        if not tools:
            return
        self._view = _View.TOOL_DETAIL
        self._error_message = None

    def _enter_add_form(self) -> None:
        self._view = _View.ADD_FORM
        self._error_message = None
        self._pending_remove = None
        self._form_focus_order = [self._add_name] + self._dynamic_add_fields()
        self._form_focus_idx = 0
        self._app.layout.focus(self._add_name)

    def _enter_filter_form(self) -> None:
        name = self._focused_server_name()
        if not name:
            return
        self._focus_server = name
        self._view = _View.FILTER_FORM
        self._error_message = None
        self._pending_remove = None

        # Pre-populate from existing filter.
        server = next((s for s in self._servers if s.get("name") == name), None)
        tool_filter = (server or {}).get("tool_filter") or {}
        self._filter_enabled.text = "y" if tool_filter.get("enabled", True) else "n"
        allowed = tool_filter.get("allowed_tool_names") or []
        blocked = tool_filter.get("blocked_tool_names") or []
        self._filter_allowed.text = ",".join(allowed)
        self._filter_blocked.text = ",".join(blocked)

        self._form_focus_order = [self._filter_enabled, self._filter_allowed, self._filter_blocked]
        self._form_focus_idx = 0
        self._app.layout.focus(self._filter_enabled)

    def _enter_search_bar(self) -> None:
        self._view = _View.SEARCH_BAR
        self._error_message = None
        self._pending_remove = None
        self._search_input.text = self._filter_query
        self._form_focus_order = [self._search_input]
        self._form_focus_idx = 0
        self._app.layout.focus(self._search_input)

    def _dynamic_add_fields(self) -> List[TextArea]:
        active = _SERVER_TYPES[self._add_type_idx]
        if active == "stdio":
            return [self._add_command, self._add_args, self._add_env]
        return [self._add_url, self._add_headers, self._add_timeout]

    def _sync_add_focus_order(self) -> None:
        self._form_focus_order = [self._add_name] + self._dynamic_add_fields()
        self._form_focus_idx = min(self._form_focus_idx, len(self._form_focus_order) - 1)

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _on_remove(self, *, server_name: Optional[str] = None) -> None:
        name = server_name or self._focused_server_name()
        if not name:
            return
        if self._pending_remove == name:
            self._pending_remove = None
            self._result = MCPSelection(kind="remove", name=name)
            self._app.exit(result=self._result)
            return
        self._pending_remove = name
        self._error_message = f"Delete server `{name}`? Press r again to confirm, any other key to cancel."

    def _on_check_current(self) -> None:
        name = self._focused_server_name()
        if not name:
            return
        self._result = MCPSelection(kind="check", name=name)
        self._app.exit(result=self._result)

    def _on_refresh(self) -> None:
        self._result = MCPSelection(kind="refresh")
        self._app.exit(result=self._result)

    def _submit_add_form(self) -> None:
        name = self._add_name.text.strip()
        if not name:
            self._error_message = "Name is required"
            return
        stype = _SERVER_TYPES[self._add_type_idx]
        config: Dict[str, Any] = {}
        if stype == "stdio":
            command = self._add_command.text.strip()
            if not command:
                self._error_message = "Command is required for stdio"
                return
            config["command"] = command
            args_raw = self._add_args.text.strip()
            if args_raw:
                config["args"] = [a.strip() for a in args_raw.split(",") if a.strip()]
            env_raw = self._add_env.text.strip()
            if env_raw:
                parsed = _parse_json_map(env_raw)
                if parsed is None:
                    self._error_message = "Env must be valid JSON object"
                    return
                config["env"] = parsed
        else:
            url = self._add_url.text.strip()
            if not url:
                self._error_message = "URL is required"
                return
            config["url"] = url
            headers_raw = self._add_headers.text.strip()
            if headers_raw:
                parsed = _parse_json_map(headers_raw)
                if parsed is None:
                    self._error_message = "Headers must be valid JSON object"
                    return
                config["headers"] = parsed
            timeout_raw = self._add_timeout.text.strip()
            if timeout_raw:
                try:
                    config["timeout"] = float(timeout_raw)
                except ValueError:
                    self._error_message = "Timeout must be a number"
                    return
        self._result = MCPSelection(kind="add", name=name, server_type=stype, config=config)
        self._app.exit(result=self._result)

    def _submit_filter_form(self) -> None:
        name = self._focused_server_name()
        if not name:
            self._error_message = "No server selected"
            return
        enabled_raw = self._filter_enabled.text.strip().lower()
        enabled = enabled_raw in ("y", "yes", "true", "1", "on") or enabled_raw == ""
        allowed = [t.strip() for t in self._filter_allowed.text.split(",") if t.strip()]
        blocked = [t.strip() for t in self._filter_blocked.text.split(",") if t.strip()]
        filter_config = {
            "enabled": enabled,
            "allowed": allowed or None,
            "blocked": blocked or None,
        }
        self._result = MCPSelection(kind="set_filter", name=name, filter_config=filter_config)
        self._app.exit(result=self._result)

    def _drop_filter(self) -> None:
        name = self._focused_server_name()
        if not name:
            self._error_message = "No server selected"
            return
        self._result = MCPSelection(kind="remove_filter", name=name)
        self._app.exit(result=self._result)

    def _apply_search_filter(self) -> None:
        self._filter_query = self._search_input.text.strip()
        self._view = _View.LIST
        self._list_cursor = 0
        self._list_offset = 0
        self._error_message = None

    def _cancel_search_filter(self) -> None:
        self._search_input.text = self._filter_query
        self._view = _View.LIST
        self._error_message = None

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view == _View.LIST)
        is_detail = Condition(lambda: self._view == _View.DETAIL)
        is_tools = Condition(lambda: self._view == _View.TOOLS)
        is_tool_detail = Condition(lambda: self._view == _View.TOOL_DETAIL)
        is_add = Condition(lambda: self._view == _View.ADD_FORM)
        is_filter = Condition(lambda: self._view == _View.FILTER_FORM)
        is_search = Condition(lambda: self._view == _View.SEARCH_BAR)

        def _clear_pending() -> None:
            self._pending_remove = None

        # ─── LIST navigation ─────────────────────────────────────────
        @kb.add("up", filter=is_list)
        def _(event):
            items = self._visible_servers()
            if items:
                self._list_cursor = (self._list_cursor - 1) % len(items)
            self._error_message = None
            _clear_pending()

        @kb.add("down", filter=is_list)
        def _(event):
            items = self._visible_servers()
            if items:
                self._list_cursor = (self._list_cursor + 1) % len(items)
            self._error_message = None
            _clear_pending()

        @kb.add("pageup", filter=is_list)
        def _(event):
            self._list_cursor = max(0, self._list_cursor - 10)
            self._error_message = None
            _clear_pending()

        @kb.add("pagedown", filter=is_list)
        def _(event):
            items = self._visible_servers()
            self._list_cursor = min(max(0, len(items) - 1), self._list_cursor + 10)
            self._error_message = None
            _clear_pending()

        @kb.add("enter", filter=is_list)
        def _(event):
            _clear_pending()
            self._enter_detail()

        @kb.add("a", filter=is_list)
        def _(event):
            _clear_pending()
            self._enter_add_form()

        @kb.add("r", filter=is_list)
        def _(event):
            row = self._current_server()
            if row is None:
                return
            self._focus_server = str(row.get("name")) if row.get("name") else None
            self._on_remove(server_name=self._focus_server)

        @kb.add("c", filter=is_list)
        def _(event):
            _clear_pending()
            row = self._current_server()
            if row is not None:
                self._focus_server = str(row.get("name")) if row.get("name") else None
            self._on_check_current()

        @kb.add("f", filter=is_list)
        def _(event):
            _clear_pending()
            row = self._current_server()
            if row is not None:
                self._focus_server = str(row.get("name")) if row.get("name") else None
            self._enter_filter_form()

        @kb.add("/", filter=is_list)
        def _(event):
            _clear_pending()
            self._enter_search_bar()

        @kb.add("R", filter=is_list)
        def _(event):
            _clear_pending()
            self._on_refresh()

        @kb.add("q", filter=is_list)
        def _(event):
            event.app.exit(result=MCPSelection(kind="cancel"))

        @kb.add("escape", filter=is_list)
        def _(event):
            event.app.exit(result=MCPSelection(kind="cancel"))

        # ─── DETAIL view ─────────────────────────────────────────────
        @kb.add("escape", filter=is_detail)
        def _(event):
            self._view = _View.LIST
            self._pending_remove = None

        @kb.add("t", filter=is_detail)
        def _(event):
            self._enter_tools()

        @kb.add("c", filter=is_detail)
        def _(event):
            self._on_check_current()

        @kb.add("f", filter=is_detail)
        def _(event):
            self._enter_filter_form()

        @kb.add("r", filter=is_detail)
        def _(event):
            self._on_remove()

        # ─── TOOLS view ──────────────────────────────────────────────
        @kb.add("up", filter=is_tools)
        def _(event):
            name = self._focused_server_name()
            tools = self._tools_cache.get(name) if name else []
            if tools:
                self._tools_cursor = (self._tools_cursor - 1) % len(tools)

        @kb.add("down", filter=is_tools)
        def _(event):
            name = self._focused_server_name()
            tools = self._tools_cache.get(name) if name else []
            if tools:
                self._tools_cursor = (self._tools_cursor + 1) % len(tools)

        @kb.add("enter", filter=is_tools)
        def _(event):
            self._enter_tool_detail()

        @kb.add("escape", filter=is_tools)
        def _(event):
            self._view = _View.DETAIL

        # ─── TOOL_DETAIL view ────────────────────────────────────────
        @kb.add("escape", filter=is_tool_detail)
        def _(event):
            self._view = _View.TOOLS

        # ─── ADD form ────────────────────────────────────────────────
        @kb.add("tab", filter=is_add)
        def _(event):
            self._sync_add_focus_order()
            self._advance_form_focus(+1)

        @kb.add("s-tab", filter=is_add)
        def _(event):
            self._sync_add_focus_order()
            self._advance_form_focus(-1)

        @kb.add("left", filter=is_add)
        def _(event):
            self._add_type_idx = (self._add_type_idx - 1) % len(_SERVER_TYPES)
            self._sync_add_focus_order()

        @kb.add("right", filter=is_add)
        def _(event):
            self._add_type_idx = (self._add_type_idx + 1) % len(_SERVER_TYPES)
            self._sync_add_focus_order()

        @kb.add("enter", filter=is_add)
        def _(event):
            self._sync_add_focus_order()
            if self._form_focus_idx >= len(self._form_focus_order) - 1:
                self._submit_add_form()
            else:
                self._advance_form_focus(+1)

        @kb.add("c-s", filter=is_add)
        def _(event):
            self._submit_add_form()

        @kb.add("escape", filter=is_add)
        def _(event):
            self._view = _View.LIST
            self._error_message = None

        # ─── FILTER form ─────────────────────────────────────────────
        @kb.add("tab", filter=is_filter)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("s-tab", filter=is_filter)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("enter", filter=is_filter)
        def _(event):
            if self._form_focus_idx >= len(self._form_focus_order) - 1:
                self._submit_filter_form()
            else:
                self._advance_form_focus(+1)

        @kb.add("c-s", filter=is_filter)
        def _(event):
            self._submit_filter_form()

        @kb.add("D", filter=is_filter)
        def _(event):
            self._drop_filter()

        @kb.add("escape", filter=is_filter)
        def _(event):
            self._view = _View.LIST
            self._error_message = None

        # ─── Search bar ──────────────────────────────────────────────
        @kb.add("enter", filter=is_search)
        def _(event):
            self._apply_search_filter()

        @kb.add("escape", filter=is_search)
        def _(event):
            self._cancel_search_filter()

        # ─── Global cancel ───────────────────────────────────────────
        @kb.add("c-c")
        def _(event):
            event.app.exit(result=None)

        return kb

    # ─────────────────────────────────────────────────────────────────
    # Focus helpers
    # ─────────────────────────────────────────────────────────────────

    def _advance_form_focus(self, delta: int) -> None:
        if not self._form_focus_order:
            return
        self._form_focus_idx = (self._form_focus_idx + delta) % len(self._form_focus_order)
        self._app.layout.focus(self._form_focus_order[self._form_focus_idx])


# ─────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 1] + "\u2026"


def _parse_json_map(raw: str) -> Optional[Dict[str, Any]]:
    import json

    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


# Expose select helpers for testing convenience.
_helpers: Dict[str, Callable[..., Any]] = {
    "_truncate": _truncate,
    "_parse_json_map": _parse_json_map,
}


__all__ = ["MCPApp", "MCPSelection"]
