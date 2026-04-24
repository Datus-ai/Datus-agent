# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/mcp`` slash command — unified CLI surface for MCP server management.

Entry point
-----------

- ``/mcp``                                  — open the interactive :class:`MCPApp`.
- ``/mcp list``                             — same as ``/mcp``.
- ``/mcp add``                              — open the app on the Add Server form.
- ``/mcp add <raw cmd>``                    — non-interactive add (scriptable,
  uses :func:`~datus.tools.mcp_tools.parse_command_string`).
- ``/mcp remove <name>``                    — non-interactive remove + confirm.
- ``/mcp check [name]``                     — refresh connectivity; ``name``
  checks a single server, empty arg opens the app with a full refresh.
- ``/mcp call <server.tool> [json]``        — non-interactive tool invocation.
- ``/mcp filter set|get|remove ...``        — non-interactive filter edits
  (retained for scripting; the TUI form covers the interactive path).
- ``/mcp help``                             — command reference.

The interactive path delegates to :class:`datus.cli.mcp_app.MCPApp`, a
single prompt_toolkit Application hosting the server list, detail
drill-down, tools drill-down, add/filter forms, and two-press remove
confirmation in one event loop. All slow I/O (connectivity checks,
``list_tools`` RPCs) runs **after** :meth:`MCPApp.run` returns.
"""

from __future__ import annotations

import json
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich.table import Table

from datus.cli._cli_utils import confirm_prompt
from datus.cli._render_utils import build_kv_table, build_row_table
from datus.cli.cli_styles import (
    TABLE_HEADER_STYLE,
    print_empty_set,
    print_error,
    print_info,
    print_status,
    print_success,
    print_usage,
    print_warning,
)
from datus.cli.mcp_app import MCPApp, MCPSelection
from datus.tools.mcp_tools import MCPTool, parse_command_string
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

# Bounded re-entry when a MCPApp selection requires a refresh / drill-down.
_MAX_REENTRY = 8


class MCPCommands:
    """Handlers for the ``/mcp`` slash command."""

    def __init__(self, cli_instance: "DatusCLI"):
        self.cli = cli_instance
        self.console = cli_instance.console
        self.mcp_tool = MCPTool()

    # ── Entry point ──────────────────────────────────────────────────────

    def cmd_mcp(self, args: str) -> None:
        """Dispatch ``/mcp`` based on its argument shape."""
        token = (args or "").strip()
        if not token:
            self._run_menu()
            return
        try:
            parts = shlex.split(token)
        except ValueError as exc:
            print_error(self.console, f"Invalid arguments: {exc}", prefix=False)
            return
        op, rest = parts[0], parts[1:]
        handlers: Dict[str, Any] = {
            "help": lambda r: self._show_usage(),
            "list": lambda r: self._run_menu(),
            "add": self._cmd_add,
            "remove": self._cmd_remove,
            "check": self._cmd_check,
            "call": self._cmd_call,
            "filter": self._cmd_filter,
        }
        handler = handlers.get(op)
        if handler is None:
            print_error(self.console, f"Unknown mcp command: {op}", prefix=False)
            self._show_usage()
            return
        handler(rest)

    # ── Interactive loop ─────────────────────────────────────────────────

    def _run_menu(
        self,
        *,
        seed_view: str = "list",
        seed_server: Optional[str] = None,
    ) -> None:
        """Drive the :class:`MCPApp` loop, re-entering on refresh / drill-down."""
        servers = self._safe_list_servers()
        status_map = self._prefetch_status(servers)
        tools_cache: Dict[str, List[Dict[str, Any]]] = {}

        current_seed_view = seed_view
        current_seed_server = seed_server

        for _ in range(_MAX_REENTRY):
            app = MCPApp(
                self.console,
                servers=servers,
                status_map=status_map,
                tools_cache=tools_cache,
                seed_view=current_seed_view,
                seed_server=current_seed_server,
            )
            selection = self._run_app(app)
            if selection is None or selection.kind == "cancel":
                return

            if selection.kind == "add":
                created = self._do_add(selection.name or "", selection.server_type or "stdio", selection.config or {})
                if created:
                    servers = self._safe_list_servers()
                    status_map = self._prefetch_status(servers)
                    current_seed_view = "list"
                    current_seed_server = selection.name
                    continue
                current_seed_view = "list"
                current_seed_server = None
                continue

            if selection.kind == "remove":
                name = selection.name or ""
                if self._do_remove(name):
                    servers = self._safe_list_servers()
                    status_map = self._prefetch_status(servers)
                    tools_cache.pop(name, None)
                current_seed_view = "list"
                current_seed_server = None
                continue

            if selection.kind == "check":
                name = selection.name or ""
                if name:
                    status_map[name] = self._check_single(name)
                current_seed_view = "list"
                current_seed_server = name or None
                continue

            if selection.kind == "refresh":
                servers = self._safe_list_servers()
                status_map = self._prefetch_status(servers)
                tools_cache = {}
                current_seed_view = "list"
                current_seed_server = None
                continue

            if selection.kind == "load_tools":
                name = selection.name or ""
                tools = self._safe_list_tools(name)
                tools_cache[name] = tools
                current_seed_view = "tools"
                current_seed_server = name
                continue

            if selection.kind == "set_filter":
                name = selection.name or ""
                filter_config = selection.filter_config or {}
                self._do_set_filter(
                    name,
                    allowed=filter_config.get("allowed"),
                    blocked=filter_config.get("blocked"),
                    enabled=bool(filter_config.get("enabled", True)),
                )
                servers = self._safe_list_servers()
                current_seed_view = "list"
                current_seed_server = name or None
                continue

            if selection.kind == "remove_filter":
                name = selection.name or ""
                self._do_remove_filter(name)
                servers = self._safe_list_servers()
                current_seed_view = "list"
                current_seed_server = name or None
                continue

            logger.debug("MCPApp returned unknown kind=%s", selection.kind)
            return

    def _run_app(self, app: MCPApp) -> Optional[MCPSelection]:
        """Run ``app`` with stdin released by the outer TUI (if any)."""
        tui_app = getattr(self.cli, "tui_app", None)
        ctx = tui_app.suspend_input() if tui_app is not None else nullcontext()
        with ctx:
            return app.run()

    # ── Non-interactive subcommands ──────────────────────────────────────

    def _cmd_add(self, rest: List[str]) -> None:
        if not rest:
            self._run_menu(seed_view="add_form")
            return
        raw = " ".join(rest)
        try:
            transport_type, server_name, config_params = parse_command_string(raw)
        except Exception as exc:
            print_error(self.console, f"Failed to parse add command: {exc}", prefix=False)
            return
        if not server_name:
            print_error(self.console, "Server name is required", prefix=False)
            return
        self._do_add(server_name, transport_type, config_params)

    def _cmd_remove(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/mcp remove <name>")
            return
        name = rest[0]
        if not confirm_prompt(self.console, f"Remove MCP server `{name}`?"):
            print_warning(self.console, "Cancelled.")
            return
        self._do_remove(name)

    def _cmd_check(self, rest: List[str]) -> None:
        if not rest:
            self._run_menu()
            return
        name = rest[0]
        result = self.mcp_tool.check_connectivity(name)
        if not result.success:
            print_status(self.console, f"Error: {result.message}", ok=False)
            return
        details = result.result.get("details", {}) if result.result else {}
        connectivity = bool(result.result.get("connectivity")) if result.result else False
        payload: Dict[str, Any] = {
            "Server": name,
            "Reachable": "yes" if connectivity else "no",
            "Type": details.get("type", "unknown"),
        }
        if "tools_count" in details:
            payload["Tools"] = details["tools_count"]
        if not connectivity and "error" in details:
            payload["Error"] = details["error"]
        table = build_kv_table(payload, title=f"Connectivity: {name}", max_cell_width=80)
        if table is not None:
            self.console.print(table)
        if connectivity:
            print_status(self.console, f"Server '{name}' is reachable", ok=True)
        else:
            print_status(self.console, f"Server '{name}' is not reachable", ok=False)

    def _cmd_call(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/mcp call <server.tool> [json_args]")
            return
        ident = rest[0]
        if "." not in ident:
            print_error(self.console, "Invalid server.tool format", prefix=False)
            return
        server_name, tool_name = ident.split(".", 1)
        tool_params: Optional[Dict[str, Any]] = None
        if len(rest) > 1:
            raw = " ".join(rest[1:]).strip()
            if raw:
                try:
                    tool_params = json.loads(raw)
                except Exception as exc:
                    print_error(self.console, f"Tool arguments must be valid JSON: {exc}", prefix=False)
                    return
        result = self.mcp_tool.call_tool(server_name, tool_name, tool_params)
        if not result.success:
            print_error(self.console, f"Error calling tool: {result.message}", prefix=False)
            return
        payload = result.result or {}
        data = payload.get("result")
        if not data:
            print_empty_set(self.console, "No result returned")
            return
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                self.console.print(data)
                return
        if isinstance(data, dict):
            if data.get("isError"):
                print_error(self.console, f"Call Tool Error: {data.get('content', '')}", prefix=False)
                return
            self.console.print(data)
            return
        self.console.print(data)

    def _cmd_filter(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/mcp filter set|get|remove <server_name> [...]")
            return
        sub = rest[0]
        tail = rest[1:]
        if sub == "set":
            self._cmd_filter_set(tail)
        elif sub == "get":
            self._cmd_filter_get(tail)
        elif sub == "remove":
            self._cmd_filter_remove(tail)
        else:
            print_error(self.console, "Invalid filter command. Use: set, get, or remove", prefix=False)

    def _cmd_filter_set(self, rest: List[str]) -> None:
        if len(rest) < 1:
            print_usage(
                self.console,
                "/mcp filter set <server_name> [--allowed t1,t2] [--blocked t3] [--enabled true|false]",
            )
            return
        server_name = rest[0]
        allowed: Optional[List[str]] = None
        blocked: Optional[List[str]] = None
        enabled = True
        i = 1
        while i < len(rest):
            flag = rest[i]
            value = rest[i + 1] if i + 1 < len(rest) else None
            if flag == "--allowed" and value is not None:
                allowed = [t.strip() for t in value.split(",") if t.strip()]
                i += 2
            elif flag == "--blocked" and value is not None:
                blocked = [t.strip() for t in value.split(",") if t.strip()]
                i += 2
            elif flag == "--enabled" and value is not None:
                enabled = value.lower() in ("true", "1", "yes", "on")
                i += 2
            else:
                i += 1
        self._do_set_filter(server_name, allowed=allowed, blocked=blocked, enabled=enabled)

    def _cmd_filter_get(self, rest: List[str]) -> None:
        if not rest:
            print_error(self.console, "Please specify the name of the MCP server", prefix=False)
            return
        server_name = rest[0]
        result = self.mcp_tool.get_tool_filter(server_name)
        if not result.success:
            print_status(self.console, f"Error getting tool filter: {result.message}", ok=False)
            return
        payload = result.result or {}
        if not payload.get("has_filter"):
            print_warning(self.console, f"No tool filter configured for server '{server_name}'")
            return
        filter_config = payload.get("filter_config") or {}
        view: Dict[str, Any] = {
            "Server": server_name,
            "Enabled": "yes" if filter_config.get("enabled") else "no",
            "Allowed": ", ".join(filter_config.get("allowed_tool_names") or []) or "(none)",
            "Blocked": ", ".join(filter_config.get("blocked_tool_names") or []) or "(none)",
        }
        table = build_kv_table(view, title=f"Tool Filter: {server_name}", max_cell_width=80)
        if table is not None:
            self.console.print(table)

    def _cmd_filter_remove(self, rest: List[str]) -> None:
        if not rest:
            print_error(self.console, "Please specify the name of the MCP server", prefix=False)
            return
        self._do_remove_filter(rest[0])

    # ── Business logic (shared by TUI and CLI paths) ─────────────────────

    def _do_add(self, name: str, server_type: str, config: Dict[str, Any]) -> bool:
        try:
            result = self.mcp_tool.add_server(name=name, type=server_type, **config)
        except Exception as exc:
            print_error(self.console, f"Failed to add server: {exc}", prefix=False)
            return False
        if result.success:
            print_success(self.console, f"Added MCP server `{name}` ({server_type})", symbol=True)
            return True
        print_error(self.console, f"Error adding MCP server: {result.message}", prefix=False)
        return False

    def _do_remove(self, name: str) -> bool:
        if not name:
            print_error(self.console, "Server name is required", prefix=False)
            return False
        try:
            result = self.mcp_tool.remove_server(name)
        except Exception as exc:
            print_error(self.console, f"Failed to remove server: {exc}", prefix=False)
            return False
        if result.success:
            print_success(self.console, f"Removed MCP server `{name}`", symbol=True)
            return True
        print_error(self.console, f"Error removing MCP server: {result.message}", prefix=False)
        return False

    def _do_set_filter(
        self,
        server_name: str,
        *,
        allowed: Optional[List[str]],
        blocked: Optional[List[str]],
        enabled: bool,
    ) -> None:
        result = self.mcp_tool.set_tool_filter(
            server_name=server_name,
            allowed_tools=allowed,
            blocked_tools=blocked,
            enabled=enabled,
        )
        if result.success:
            print_status(self.console, f"Tool filter updated for server '{server_name}'", ok=True)
        else:
            print_status(self.console, f"Error setting tool filter: {result.message}", ok=False)

    def _do_remove_filter(self, server_name: str) -> None:
        result = self.mcp_tool.remove_tool_filter(server_name)
        if result.success:
            print_status(self.console, f"Tool filter removed for server '{server_name}'", ok=True)
        else:
            print_status(self.console, f"Error removing tool filter: {result.message}", ok=False)

    # ── Data helpers (pre-fetch, safe wrappers) ─────────────────────────

    def _safe_list_servers(self) -> List[Dict[str, Any]]:
        try:
            result = self.mcp_tool.list_servers()
        except Exception as exc:
            logger.debug("list_servers raised: %s", exc)
            return []
        if not result.success or not result.result:
            return []
        servers = result.result.get("servers") or []
        return list(servers)

    def _safe_list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        if not server_name:
            return []
        try:
            result = self.mcp_tool.list_tools(server_name)
        except Exception as exc:
            logger.debug("list_tools(%s) raised: %s", server_name, exc)
            return []
        if not result.success or not result.result:
            return []
        return list(result.result.get("tools") or [])

    def _check_single(self, name: str) -> Dict[str, Any]:
        try:
            result = self.mcp_tool.check_connectivity(name)
        except Exception as exc:
            logger.debug("check_connectivity(%s) raised: %s", name, exc)
            return {"connectivity": False, "error": str(exc), "tools_count": None}
        if not result.success or not result.result:
            return {"connectivity": False, "error": result.message, "tools_count": None}
        payload = result.result
        details = payload.get("details") or {}
        return {
            "connectivity": bool(payload.get("connectivity")),
            "tools_count": details.get("tools_count"),
            "error": details.get("error"),
        }

    def _prefetch_status(self, servers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        status: Dict[str, Dict[str, Any]] = {}
        if not servers:
            return status
        names = [str(s.get("name")) for s in servers if s.get("name")]
        if not names:
            return status
        max_workers = min(8, max(1, len(names)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._check_single, name): name for name in names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    status[name] = fut.result()
                except Exception as exc:
                    logger.debug("prefetch connectivity failed for %s: %s", name, exc)
                    status[name] = {"connectivity": False, "error": str(exc), "tools_count": None}
        return status

    # ── Help ─────────────────────────────────────────────────────────────

    def _show_usage(self) -> None:
        table = Table(title="/mcp commands", header_style=TABLE_HEADER_STYLE, show_lines=False)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        rows = [
            ("/mcp", "Open the interactive server browser"),
            ("/mcp list", "Same as `/mcp`"),
            ("/mcp add", "Open the Add Server form"),
            ("/mcp add <raw>", "Non-interactive add (uses `--transport` syntax)"),
            ("/mcp remove <name>", "Remove a server (with confirm)"),
            ("/mcp check [name]", "Check connectivity (interactive refresh if omitted)"),
            ("/mcp call <server.tool> [json]", "Invoke a tool on a server"),
            ("/mcp filter set <name> [...]", "Configure tool filter (whitelist/blacklist)"),
            ("/mcp filter get <name>", "Show tool filter for a server"),
            ("/mcp filter remove <name>", "Disable tool filter"),
            ("/mcp help", "Show this help"),
        ]
        for cmd, desc in rows:
            table.add_row(cmd, desc)
        self.console.print(table)

    # ── Legacy non-interactive list path (kept for scripts / fallback) ──

    def cmd_mcp_list(self) -> None:
        """Print configured servers as a Rich table (no TUI)."""
        servers = self._safe_list_servers()
        if not servers:
            print_empty_set(self.console, "No MCP servers configured.")
            return
        payload = [
            {
                "Name": s.get("name", "?"),
                "Type": s.get("type", "?"),
                "Target": s.get("command") or s.get("url") or "-",
                "Args": " ".join(str(a) for a in (s.get("args") or [])),
            }
            for s in servers
        ]
        table = build_row_table(
            payload,
            title="MCP Servers",
            columns=[
                ("Name", "Name"),
                ("Type", "Type"),
                ("Target", "Target"),
                ("Args", "Args"),
            ],
            max_cell_width=60,
        )
        if table is not None:
            self.console.print(table)
        else:
            print_info(self.console, "No servers to display.")
