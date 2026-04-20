# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CLI ``.<service>.<method>`` command handler.

Routes dotted commands to the underlying ``*FuncTool`` instance via
``ServiceClientRegistry``. Read-only by design: any method not listed in
``datus.cli.service_client.READ_METHODS`` is rejected with a clear error
message pointing the user to agent mode.

Argument parsing is intentionally minimal — the allow-listed read methods take
at most three simple arguments (``str`` / ``int`` / ``List[str]``):

- Positional, in schema order: ``.superset.get_dashboard 1``
- Named overrides: ``.superset.get_chart_data 42 --limit=100``
- Lists: ``--subject_path=a,b`` or ``--subject_path=['a','b']``

JSON-blob input is deliberately out of scope — if a method's schema needs it,
that method does not belong in the CLI allow-list.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich.table import Table

from datus.cli.service_client import ServiceClient, ServiceClientRegistry
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents import FunctionTool

    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

_INVOCATION_TIMEOUT_SEC = 60.0


class ServiceCommands:
    """Handler for ``.services`` / ``.<service>`` / ``.<service>.<method>``."""

    def __init__(self, cli_instance: "DatusCLI"):
        self.cli = cli_instance
        self._registry: Optional[ServiceClientRegistry] = None

    # ------------------------------------------------------------------ #
    # Registry access (lazy so ServiceCommands can be created before
    # agent_config fields are populated from background init)
    # ------------------------------------------------------------------ #

    @property
    def registry(self) -> ServiceClientRegistry:
        if self._registry is None:
            self._registry = ServiceClientRegistry(self.cli.agent_config)
        return self._registry

    # ------------------------------------------------------------------ #
    # Entry points wired into DatusCLI.commands / _execute_internal_command
    # ------------------------------------------------------------------ #

    def cmd_services(self, args: str = "") -> None:
        """Handler for the ``.services`` command."""
        rows = self.registry.list_services()
        if not rows:
            self.cli.console.print(
                "[yellow]No services configured. Add entries under "
                "`services.bi_tools`, `services.schedulers`, or "
                "`services.semantic_layer` in agent.yml.[/]"
            )
            return
        table = Table(title="Configured services", show_header=True, header_style="bold green")
        table.add_column("Service")
        table.add_column("Type")
        table.add_column("Status")
        for name, section, status in rows:
            table.add_row(name, section, status)
        self.cli.console.print(table)

    def dispatch(self, cmd: str, args: str) -> bool:
        """Handle a ``.<service>`` or ``.<service>.<method>`` command.

        Returns ``True`` if ``cmd`` was recognised as a service command (and
        therefore handled); ``False`` to let the caller fall through to the
        normal "Unknown command" error path.
        """
        if not cmd.startswith("."):
            return False

        body = cmd[1:]
        head, _, tail = body.partition(".")
        if not head:
            return False

        client = self.registry.get(head)
        if client is None:
            return False

        if not tail:
            self._print_methods(client)
            return True

        self._invoke(client, tail, args)
        return True

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    def _print_methods(self, client: ServiceClient) -> None:
        methods = client.list_methods()
        if not methods:
            self.cli.console.print(
                f"[yellow]Service '{client.service_name}' ({client.service_type}) "
                f"has no read-only methods exposed to the CLI.[/]"
            )
            return
        table = Table(
            title=f"{client.service_name} — read methods",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Method")
        table.add_column("Description")
        for name, doc in methods:
            table.add_row(name, doc or "")
        self.cli.console.print(table)

    def _print_schema(self, tool: "FunctionTool", hint: str = "") -> None:
        schema = tool.params_json_schema or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required", []) or [])
        if hint:
            self.cli.console.print(f"[yellow]{hint}[/]")
        table = Table(
            title=f"{tool.name} — parameters",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Required")
        table.add_column("Description")
        for key, info in props.items():
            if key == "self" or not isinstance(info, dict):
                continue
            table.add_row(
                key,
                str(info.get("type", "")),
                "yes" if key in required else "",
                info.get("description", "") or "",
            )
        self.cli.console.print(table)

    def _render_result(self, result: Any) -> None:
        """Render a ``FuncToolResult``-shaped dict or a bare payload."""
        if isinstance(result, dict) and "success" in result:
            if result.get("success") == 0:
                self.cli.console.print(f"[red]Error:[/] {result.get('error', 'unknown error')}")
                return
            payload = result.get("result")
        else:
            payload = result
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        self.cli.console.print(rendered)

    # ------------------------------------------------------------------ #
    # Invocation
    # ------------------------------------------------------------------ #

    def _invoke(self, client: ServiceClient, method_name: str, args: str) -> None:
        tool = client.get_tool(method_name)
        if tool is None:
            if hasattr(client.tool_instance, method_name):
                # Method exists but is not in the read-only allow-list.
                self.cli.console.print(
                    f"[red]Method '{method_name}' is a write or privileged operation.[/] "
                    f"[dim]The CLI only exposes read-only service methods. "
                    f"Use agent mode to invoke writes.[/]"
                )
            else:
                self.cli.console.print(
                    f"[red]Unknown method '{method_name}' on service '{client.service_name}'.[/] "
                    f"[dim]Run `.{client.service_name}` to list available methods.[/]"
                )
            return

        if self._is_help_request(args):
            self._print_schema(tool)
            return

        parsed = self._parse_args(args, tool.params_json_schema or {})
        if parsed is None:
            self._print_schema(tool, hint="Could not parse arguments. Expected schema:")
            return

        missing = self._missing_required(tool.params_json_schema or {}, parsed)
        if missing:
            self.cli.console.print(f"[red]Missing required argument(s):[/] {', '.join(missing)}")
            self._print_schema(tool)
            return

        try:
            args_json = json.dumps(parsed)
            result = self._run_async(tool.on_invoke_tool(None, args_json))
        except Exception as exc:
            logger.exception(f"Service tool invocation failed for {client.service_name}.{method_name}")
            self.cli.console.print(f"[red]Invocation failed:[/] {exc}")
            return

        self._render_result(result)

    # ------------------------------------------------------------------ #
    # Argument parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_help_request(args: str) -> bool:
        try:
            tokens = shlex.split(args) if args else []
        except ValueError:
            return False
        return "--help" in tokens or "-h" in tokens

    def _parse_args(self, args: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse positional + ``--key=value`` arguments against a JSON schema.

        Returns a ``{key: coerced_value}`` dict, or ``None`` if the input is
        malformed (e.g. more positional args than the schema accepts, or a
        quoting error from ``shlex``).
        """
        try:
            tokens = shlex.split(args) if args else []
        except ValueError:
            return None

        props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        prop_order = [k for k in props.keys() if k != "self"]

        positional: List[str] = []
        named: Dict[str, str] = {}
        for tok in tokens:
            if tok.startswith("--"):
                body = tok[2:]
                if not body:
                    return None
                key, sep, value = body.partition("=")
                if not sep:
                    # Bare ``--flag`` means ``--flag=true`` for boolean fields.
                    named[key] = "true"
                else:
                    named[key] = value
            else:
                positional.append(tok)

        parsed: Dict[str, Any] = {}
        for idx, value in enumerate(positional):
            if idx >= len(prop_order):
                logger.warning(f"Ignoring extra positional argument '{value}' (schema has {len(prop_order)} params)")
                return None
            key = prop_order[idx]
            parsed[key] = self._coerce(value, props.get(key) or {})

        for key, raw in named.items():
            if key not in props:
                logger.warning(f"Unknown parameter '{key}' — ignored")
                continue
            parsed[key] = self._coerce(raw, props.get(key) or {})

        return parsed

    @staticmethod
    def _coerce(raw: str, prop_schema: Dict[str, Any]) -> Any:
        t = (prop_schema.get("type") if isinstance(prop_schema, dict) else None) or ""
        if t == "integer":
            try:
                return int(raw)
            except ValueError:
                return raw
        if t == "number":
            try:
                return float(raw)
            except ValueError:
                return raw
        if t == "boolean":
            return raw.strip().lower() in ("1", "true", "yes", "y")
        if t == "array":
            stripped = raw.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        return raw

    @staticmethod
    def _missing_required(schema: Dict[str, Any], parsed: Dict[str, Any]) -> List[str]:
        """Return the list of required-and-unsupplied parameter names.

        A parameter listed in ``required`` but whose schema property carries a
        ``default`` is treated as effectively optional — Pydantic / the Agents
        SDK tend to include every signature parameter in the OpenAI-style
        ``required`` array even when the Python signature provides a default.
        """
        if not isinstance(schema, dict):
            return []
        required = schema.get("required", []) or []
        props = schema.get("properties", {}) or {}
        missing = []
        for key in required:
            if key in parsed:
                continue
            prop = props.get(key) or {}
            if isinstance(prop, dict) and "default" in prop:
                continue
            missing.append(key)
        return missing

    # ------------------------------------------------------------------ #
    # Async plumbing
    # ------------------------------------------------------------------ #

    def _run_async(self, coro) -> Any:
        """Run the coroutine on the CLI's background loop and block for the result."""
        bg_loop = getattr(self.cli, "_bg_loop", None)
        if bg_loop is None or not bg_loop.is_running():
            # No background loop (e.g. in a unit test); use a fresh loop.
            return asyncio.run(coro)
        future = asyncio.run_coroutine_threadsafe(coro, bg_loop)
        return future.result(timeout=_INVOCATION_TIMEOUT_SEC)
