# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CLI ``!<tool> [args...]`` / ``!<plugin> <args...>`` command handler.

The ``!`` prefix is a power-user escape hatch that, from the chat REPL, either:

- invokes one of the agent's own function tools directly (``!list_tables``,
  ``!search_table foo --top_n=3``) — gated by the same permission pipeline an
  LLM-driven call gets, then rendered locally (never fed back to the model); or
- runs an installed + activated plugin's CLI in a ``datus <plugin> ...``
  subprocess (``!hello sync --limit=10``).

**Tools are matched first**: if the first token names a live tool it wins;
otherwise a matching plugin name dispatches to the subprocess path; otherwise the
input is rejected with a usage hint. Argument parsing (positional +
``--key=value``) is shared with ``/<service>.<method>`` via
:mod:`datus.cli.tool_arg_parser`.

This handler only runs in CHAT input mode — in SQL/bash mode a leading ``!`` is
part of the statement (see :meth:`DatusCLI._parse_command`).
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import TYPE_CHECKING, Any, Dict, List

from rich.table import Table

from datus.cli import tool_arg_parser
from datus.cli.cli_styles import TABLE_HEADER_STYLE, print_error, print_info, print_warning
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents import FunctionTool

    from datus.cli.repl import DatusCLI
    from datus.plugins.base import PluginManifest

logger = get_logger(__name__)


class BangCommand:
    """Handler + completion source for ``!<tool>`` / ``!<plugin>`` commands."""

    # Agent-orchestration / plan-mode tools that only make sense inside a live
    # LLM turn — invoking them manually via ``!`` (with no run context) would
    # hang or misbehave, so they are hidden from the ``!`` list and dispatch:
    #   - ``ask_user`` needs the interaction broker attached to a running turn;
    #   - ``task`` spawns a sub-agent under the orchestrator;
    #   - ``confirm_plan`` / ``todo_*`` are plan-mode state helpers driven by the
    #     agent, not standalone commands.
    # Kept in sync (conceptually) with ``PermissionHooks._PLAN_MODE_BYPASS_TOOLS``.
    EXCLUDED_TOOLS = frozenset(
        {
            "ask_user",
            "task",
            "confirm_plan",
            "todo_list",
            "todo_read",
            "todo_write",
            "todo_update",
        }
    )

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli

    # ------------------------------------------------------------------ #
    # Enumeration (shared with BangCompleter and the arg-name hint)
    # ------------------------------------------------------------------ #

    def tool_map(self, create: bool = False) -> Dict[str, "FunctionTool"]:
        """Return ``{tool_name: FunctionTool}`` from the live chat node.

        ``create=True`` lazily builds the chat node when absent (execution path,
        where blocking is fine); ``create=False`` only reads an existing node so
        completion / hint rendering never blocks the prompt-toolkit loop.

        Agent-orchestration tools in :attr:`EXCLUDED_TOOLS` are filtered out —
        they are not standalone-invocable (see the set's docstring).
        """
        chat = getattr(self.cli, "chat_commands", None)
        node = None
        if chat is not None:
            if create:
                node = chat.ensure_node_for_bang()
            else:
                node = getattr(chat, "current_node", None)
        tools = getattr(node, "tools", None) or []
        return {name: t for t in tools if (name := getattr(t, "name", "")) and name not in self.EXCLUDED_TOOLS}

    def plugin_map(self) -> Dict[str, "PluginManifest"]:
        """Return ``{plugin_name: PluginManifest}`` for installed, active plugins."""
        agent_config = getattr(self.cli, "agent_config", None)
        if agent_config is None or not getattr(agent_config, "plugins_enabled", True):
            return {}
        try:
            from datus.plugins.registry import iter_plugin_manifests
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Plugin enumeration unavailable: %s", exc)
            return {}
        out: Dict[str, "PluginManifest"] = {}
        for name, manifest in iter_plugin_manifests():
            try:
                if agent_config.plugin_active(name):
                    out[name] = manifest
            except Exception:  # noqa: BLE001 - one bad plugin must not hide the rest
                continue
        return out

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    def dispatch(self, text: str) -> None:
        """Route ``text`` (the input after the leading ``!``) to a tool or plugin."""
        text = (text or "").strip()
        if not text or text in ("help", "?"):
            self._print_overview()
            return

        parts = text.split(maxsplit=1)
        first = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        tools = self.tool_map(create=True)
        if first in tools:
            self._invoke_tool(tools[first], rest)
            return

        plugins = self.plugin_map()
        if first in plugins:
            self._invoke_plugin(first, rest)
            return

        print_error(
            self.cli.console,
            f"Unknown tool or plugin '{first}'. Run `!` to list available tools and plugins.",
            prefix=False,
        )
        print_info(self.cli.console, "Usage: !<tool> [args...]  or  !<plugin> <args...>")

    # ------------------------------------------------------------------ #
    # Tool invocation
    # ------------------------------------------------------------------ #

    def _invoke_tool(self, tool: "FunctionTool", rest: str) -> None:
        console = self.cli.console
        if tool_arg_parser.is_help_request(rest):
            tool_arg_parser.print_schema(console, tool)
            return

        parsed, error = tool_arg_parser.parse_args(rest, tool.params_json_schema or {})
        if parsed is None:
            tool_arg_parser.print_schema(console, tool, hint=error or "Could not parse arguments. Expected schema:")
            return

        from datus.cli.bash_mode import run_manual_tool_live

        rest = rest.strip()
        command = f"{tool.name} {rest}" if rest else tool.name

        def _invoke() -> tuple:
            result = asyncio.run(tool.on_invoke_tool(None, json.dumps(parsed)))
            return self._tool_result_to_output(result)

        # Gate + execute inside a live frame, then feed the result to the model as
        # an execution turn (identical to SQL/bash modes): the call enters the
        # conversation context and triggers a reply. Permission denials render but
        # are not dispatched.
        payload, dispatch = run_manual_tool_live(self.cli, tool.name, command, parsed, _invoke)
        if dispatch and payload is not None:
            self.cli._send_exec_turn(payload)

    @staticmethod
    def _tool_result_to_output(result: Any) -> tuple:
        """Reduce a tool return value to ``(success, output_text)``.

        ``FuncToolResult``-shaped dicts (``{success, error, result}``) are unwrapped
        to their inner result; anything else is treated as a successful payload.
        Structured payloads render as pretty JSON so both the styled block and the
        model see the same readable text.
        """
        if isinstance(result, dict) and "success" in result:
            success = bool(result.get("success"))
            if not success:
                return False, str(result.get("error") or "tool failed")
            payload = result.get("result")
        else:
            success, payload = True, result
        if isinstance(payload, str):
            return success, payload
        return success, json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------ #
    # Plugin invocation (subprocess)
    # ------------------------------------------------------------------ #

    def _invoke_plugin(self, name: str, rest: str) -> None:
        """Run ``datus <plugin> <rest>`` as a subprocess via the bash pipeline.

        Reuses the permission-gated bash execution + live frame; the plugin's own
        CLI permissions apply inside the child process. The output is fed to the
        model as an execution turn (like ``!<tool>``) so the run enters the
        conversation and triggers a reply.
        """
        from datus.cli.bash_mode import run_manual_bash_live

        command = f"datus {name}"
        if rest:
            command = f"{command} {rest}"
        payload, dispatch = run_manual_bash_live(self.cli, command)
        if dispatch and payload is not None:
            self.cli._send_exec_turn(payload)

    # ------------------------------------------------------------------ #
    # Overview listing
    # ------------------------------------------------------------------ #

    def _print_overview(self) -> None:
        console = self.cli.console
        tools = self.tool_map(create=True)
        plugins = self.plugin_map()

        if tools:
            table = Table(title="! tools", show_header=True, header_style=TABLE_HEADER_STYLE)
            table.add_column("Tool")
            table.add_column("Description")
            for name in sorted(tools):
                doc = (getattr(tools[name], "description", "") or "").strip().split("\n", 1)[0]
                table.add_row(name, doc)
            console.print(table)
        else:
            print_warning(console, "No tools available yet (agent still initializing).")

        if plugins:
            table = Table(title="! plugins", show_header=True, header_style=TABLE_HEADER_STYLE)
            table.add_column("Plugin")
            table.add_column("Description")
            for name in sorted(plugins):
                table.add_row(name, plugins[name].description or "")
            console.print(table)

        print_info(
            console,
            "Usage: !<tool> [args...]  or  !<plugin> <args...>  ·  !<tool> --help for parameters",
        )

    # ------------------------------------------------------------------ #
    # Argument-name hint (rendered dim after the input by the TUI)
    # ------------------------------------------------------------------ #

    def param_hint(self, text: str) -> str:
        """Return a dim ``<required> [--optional]`` hint for the current input.

        Consumed by the TUI's ``AfterInput`` processor. Returns ``""`` while the
        user is still typing the tool/plugin name, or on any error (this runs on
        every render, so it must never raise or block — hence non-creating tool
        access only).
        """
        try:
            if not text.startswith("!"):
                return ""
            body = text[1:]
            if " " not in body:
                return ""  # still typing the tool/plugin name
            first, _, rest = body.partition(" ")
            tools = self.tool_map(create=False)
            if first in tools:
                return self._tool_param_hint(tools[first], rest)
            plugins = self.plugin_map()
            if first in plugins:
                return self._plugin_param_hint(plugins[first], rest)
            return ""
        except Exception:  # noqa: BLE001 - a hint must never break rendering
            return ""

    @staticmethod
    def _split_consumed(tokens: List[str]) -> tuple[set, int]:
        """Return ``(named_keys_given, positional_count)`` from arg tokens."""
        named: set = set()
        positional = 0
        for tok in tokens:
            if tok.startswith("--"):
                named.add(tok[2:].split("=", 1)[0])
            else:
                positional += 1
        return named, positional

    def _tool_param_hint(self, tool: "FunctionTool", rest: str) -> str:
        schema = tool.params_json_schema or {}
        props = [k for k in (schema.get("properties") or {}).keys() if k != "self"]
        required = set(schema.get("required", []) or [])
        try:
            tokens = shlex.split(rest) if rest else []
        except ValueError:
            tokens = rest.split()
        named_given, positional = self._split_consumed(tokens)
        parts: List[str] = []
        for idx, name in enumerate(props):
            if name in named_given or idx < positional:
                continue
            parts.append(f"<{name}>" if name in required else f"[--{name}]")
        return " ".join(parts)

    def _plugin_param_hint(self, manifest: "PluginManifest", rest: str) -> str:
        commands = list(manifest.commands)
        if not commands:
            return ""
        # Split into settled tokens + the partial token under the cursor: a
        # trailing space (or empty ``rest``) settles the previous token and opens
        # a new one. While the current token is still being typed we keep showing
        # the level's menu (it may still name a subcommand).
        trailing = rest == "" or rest.endswith(" ")
        words = rest.split()
        settled = words if trailing else words[:-1]
        # Walk settled tokens down the tree. A settled token naming no subcommand
        # at the current level is a positional value — descent stops there and
        # the remaining tokens are ``node``'s positional args.
        node = None
        level = commands
        leftover: List[str] = []
        idx = 0
        while idx < len(settled):
            child = next((c for c in level if c.name == settled[idx]), None)
            if child is None:
                leftover = settled[idx:]
                break
            node, level = child, child.subcommands
            idx += 1
        # Still choosing among subcommands at this level (no positional typed) →
        # show the subcommand menu in declaration order.
        if level and not leftover:
            return "{" + "|".join(c.name for c in level) + "}"
        if node is None:
            return ""
        named_given, positional = self._split_consumed(leftover)
        parts: List[str] = []
        pos_idx = 0
        for arg in node.args:
            if arg.name.startswith("--"):
                if arg.name[2:].split("=", 1)[0] in named_given:
                    continue
                parts.append(f"[{arg.name}]")
            else:
                if pos_idx < positional:
                    pos_idx += 1
                    continue
                pos_idx += 1
                parts.append(f"<{arg.name}>" if arg.required else f"[{arg.name}]")
        return " ".join(parts)
