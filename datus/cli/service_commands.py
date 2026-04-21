# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CLI ``/<service>.<method>`` command handler.

Routes slash-dotted commands to the underlying ``*FuncTool`` instance via
``ServiceClientRegistry``. Read-only by design: any method not listed in
``datus.cli.service_client.READ_METHODS`` is rejected with a clear error
message pointing the user to agent mode.

Argument parsing is intentionally minimal — the allow-listed read methods take
at most three simple arguments (``str`` / ``int`` / ``List[str]``):

- Positional, in schema order: ``/superset.get_dashboard 1``
- Named overrides: ``/superset.get_chart_data 42 --limit=100``
- Lists: ``--subject_path=a,b`` or ``--subject_path=['a','b']``

JSON-blob input is deliberately out of scope — if a method's schema needs it,
that method does not belong in the CLI allow-list.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import shlex
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from rich.table import Table

from datus.cli._render_utils import build_kv_table, build_row_table
from datus.cli.service_client import ServiceClient, ServiceClientRegistry, service_type_label
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents import FunctionTool

    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class ServiceCommands:
    """Handler for ``/services`` / ``/<service>`` / ``/<service>.<method>``."""

    def __init__(self, cli_instance: "DatusCLI"):
        self.cli = cli_instance
        self._registry: Optional[ServiceClientRegistry] = None
        # Populated by ``_parse_args`` when parsing fails in a way that has a
        # specific user-facing hint (e.g. misspelled ``--flag``). ``_invoke``
        # surfaces it alongside the schema so typos fail fast.
        self._last_parse_error: Optional[str] = None

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
        """Handler for the ``/services`` command."""
        rows = self.registry.list_services()
        if not rows:
            self.cli.console.print(
                "[yellow]No services configured. Add entries under "
                "`services.bi_platforms`, `services.schedulers`, or "
                "`services.semantic_layer` in agent.yml.[/]"
            )
            return
        table = Table(title="Configured services", show_header=True, header_style="bold green")
        table.add_column("Service")
        table.add_column("Type")
        table.add_column("Status")
        for name, section, status in rows:
            table.add_row(name, service_type_label(section), status)
        self.cli.console.print(table)

    def dispatch(self, cmd: str, args: str) -> bool:
        """Handle a ``/<service>`` or ``/<service>.<method>`` command.

        Returns ``True`` if ``cmd`` was recognised as a service command (and
        therefore handled); ``False`` to let the caller fall through to the
        normal "Unknown command" error path.
        """
        if not cmd.startswith("/"):
            return False

        body = cmd[1:]
        head, _, tail = body.partition(".")
        if not head:
            return False

        # Only claim the command when the service is actually configured;
        # otherwise fall back to the caller's "Unknown command" path so
        # typoed slash tokens still fail loudly.
        if not self.registry.has(head):
            return False

        # Adapter missing → surface the install hint instead of letting the
        # factory's ImportError drop the route into "Unknown command".
        if not self.registry.adapter_available(head):
            client = self.registry.get(head)
            if client is not None:
                self._print_missing_adapter_hint(client)
            else:
                self.cli.console.print(f"[red]Service '{head}' is configured but its adapter is not installed.[/]")
            return True

        client = self.registry.get(head)
        if client is None:
            self.cli.console.print(f"[red]Service '{head}' could not be loaded.[/]")
            return True

        if not tail:
            self._print_methods(client)
            return True

        self._invoke(client, tail, args)
        return True

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    # Only the platform-specific package needs to be installed — the
    # corresponding ``datus-*-core`` framework is a transitive dependency
    # and pip pulls it in automatically. Listing core here used to confuse
    # users into thinking they had to install two separate packages.
    _ADAPTER_PACKAGE_HINTS = {
        "bi_platforms": "datus-bi-<platform>  (e.g. datus-bi-superset, datus-bi-grafana)",
        "schedulers": "datus-scheduler-<platform>  (e.g. datus-scheduler-airflow)",
        "semantic_layer": "datus-semantic-<type>  (e.g. datus-semantic-metricflow)",
    }

    def _print_missing_adapter_hint(self, client: ServiceClient) -> None:
        """Explain that the service is configured but its adapter isn't installed."""
        pkg_hint = self._ADAPTER_PACKAGE_HINTS.get(client.service_type, "the matching adapter package")
        label = service_type_label(client.service_type)
        self.cli.console.print(
            f"[red]Service '{client.service_name}' ({label}) is configured "
            f"but the adapter is not installed.[/]\n"
            f"[dim]Install {pkg_hint} and restart the CLI, "
            f"then re-run `/services` to confirm.[/]"
        )

    def _print_methods(self, client: ServiceClient) -> None:
        methods = client.list_methods()
        if not methods:
            self.cli.console.print(
                f"[yellow]Service '{client.service_name}' ({service_type_label(client.service_type)}) "
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

    # Cap long cell contents (huge nested ``extra.raw`` blobs, SQL texts,
    # etc.) so a single command doesn't push the terminal through many
    # screenfuls. Wide enough for names and short descriptions at
    # typical terminal widths, truncated in the middle otherwise.
    _MAX_CELL_WIDTH = 120

    def _render_result(self, result: Any, *, service: str = "", method: str = "") -> None:
        """Render a ``FuncToolResult``-shaped dict or a bare payload.

        - ``FuncToolListResult`` envelopes (``{items, total, has_more, extra}``)
          from list_* tools render as a Rich table with a pagination hint
          when more rows exist upstream.
        - Single-dict payloads (``get_dashboard`` / ``get_chart`` / ...)
          render as a two-column Field/Value K/V table.
        - Everything else falls back to indented JSON.
        """
        if isinstance(result, dict) and "success" in result:
            if result.get("success") == 0:
                self.cli.console.print(f"[red]Error:[/] {result.get('error', 'unknown error')}")
                return
            payload = result.get("result")
        else:
            payload = result

        # Fast-path: FuncToolListResult envelope from any list_* tool.
        if self._render_list_envelope(payload, service=service, method=method):
            return
        if self._render_query_envelope(payload):
            return
        if self._render_payload_as_table(payload):
            return
        if self._render_payload_as_kv(payload):
            return
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        self.cli.console.print(rendered)

    # ``DataCompressor.compress`` canonical keys — matched as a set so a
    # near-miss payload (e.g. one missing key) doesn't accidentally hit this
    # branch and hide a real bug downstream.
    _COMPRESSOR_KEYS = frozenset(
        {"original_rows", "original_columns", "is_compressed", "compressed_data", "compression_type"}
    )

    def _render_query_envelope(self, payload: Any) -> bool:
        """Render the ``query_metrics`` result shape.

        Payload: ``{"columns": [...], "data": <compressor envelope>,
        "metadata": {...}}``. ``data.compressed_data`` is a CSV string
        produced by ``DataCompressor``; parse it into rows so the CLI
        shows actual values rather than serializer metadata.
        """
        if not isinstance(payload, dict):
            return False
        data = payload.get("data")
        if not isinstance(data, dict):
            return False
        if not self._COMPRESSOR_KEYS.issubset(data.keys()):
            return False

        compressed = data.get("compressed_data", "")
        rows = self._parse_compressor_csv(compressed) if isinstance(compressed, str) else []

        if rows:
            table = build_row_table(rows, max_cell_width=self._MAX_CELL_WIDTH)
            if table is not None:
                self.cli.console.print(table)
            else:
                self.cli.console.print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
        elif isinstance(compressed, str) and compressed and compressed != "Empty dataset":
            # Compressor produced a non-CSV form (e.g. ``_format_as_table``)
            # or a format we don't parse. Show it verbatim — better than
            # swallowing the payload.
            self.cli.console.print(compressed)
        else:
            self.cli.console.print("[yellow]Empty set.[/]")

        removed = data.get("removed_columns") or []
        total = data.get("original_rows")
        hint_parts: List[str] = []
        if isinstance(total, int) and total > len(rows) and rows:
            hint_parts.append(f"Showing {len(rows)} of {total} rows (compressed).")
        if removed:
            hint_parts.append(f"Omitted columns: {', '.join(removed)}.")
        if hint_parts:
            self.cli.console.print(f"[dim]{' '.join(hint_parts)}[/]")

        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata:
            self.cli.console.print(f"[dim]metadata: {json.dumps(metadata, ensure_ascii=False, default=str)}[/]")
        return True

    @staticmethod
    def _parse_compressor_csv(text: str) -> List[Dict[str, Any]]:
        """Parse ``DataCompressor.compressed_data`` CSV into row dicts.

        Returns ``[]`` for empty / unparseable input so the caller can fall
        back to printing the raw compressed string.
        """
        if not text or text == "Empty dataset":
            return []
        import csv
        import io

        try:
            reader = csv.DictReader(io.StringIO(text))
            return [dict(row) for row in reader]
        except csv.Error:
            return []

    def _render_list_envelope(self, payload: Any, *, service: str, method: str) -> bool:
        """Render ``FuncToolListResult`` envelopes; return True when handled.

        Envelope shape: ``{items, total, has_more, extra}`` — ``items`` is
        always a ``List[Dict]``. After rendering the rows, append a pagination
        hint showing how far into the upstream dataset we are and how to
        fetch the next page.
        """
        if not isinstance(payload, dict):
            return False
        if "items" not in payload or not isinstance(payload["items"], list):
            return False
        items: list = payload["items"]
        total = payload.get("total")
        extra = payload.get("extra") or {}

        if items:
            table = build_row_table(items, max_cell_width=self._MAX_CELL_WIDTH)
            if table is not None:
                self.cli.console.print(table)
            else:
                self.cli.console.print(json.dumps(items, indent=2, ensure_ascii=False, default=str))
        else:
            self.cli.console.print("[yellow]Empty set.[/]")

        # Pagination hint — only when meaningful (another page is reachable).
        next_offset = extra.get("next_offset")
        hint = self._format_pagination_hint(
            shown=len(items),
            total=total,
            next_offset=next_offset,
            service=service,
            method=method,
        )
        if hint:
            self.cli.console.print(hint)
        return True

    @staticmethod
    def _format_pagination_hint(
        *,
        shown: int,
        total: Optional[int],
        next_offset: Optional[int],
        service: str,
        method: str,
    ) -> str:
        """Build the ``Showing X of Y. Next: /<service>.<method> --offset=...``
        hint. Returns an empty string when there's no next page to suggest.
        """
        if next_offset is None:
            # No "another page exists" signal from the adapter. If total is
            # known and we already have it all, stay silent; otherwise silent
            # is still the right call — the tool explicitly didn't hint.
            return ""
        if total is not None and shown >= total:
            return ""
        if total is not None:
            prefix = f"Showing {shown} of {total}."
        else:
            prefix = f"Showing {shown} items."
        cmd_hint = ""
        if service and method:
            cmd_hint = f" Next: /{service}.{method} --offset={next_offset}"
        return f"[dim]{prefix}{cmd_hint}[/]"

    def _render_payload_as_table(self, payload: Any) -> bool:
        """Render a list-of-dict payload as a Rich table.

        Delegates to the shared ``build_row_table`` helper so the visual
        style matches ``/tables`` / ``/databases``. Column set is inferred
        from the union of dict keys; all-empty columns (e.g.
        ``chart_ids`` on BI list responses) are pruned. Returns ``True``
        when a table was printed so the caller skips the JSON fallback.
        """
        table = build_row_table(payload, max_cell_width=self._MAX_CELL_WIDTH)
        if table is None:
            return False
        self.cli.console.print(table)
        return True

    def _render_payload_as_kv(self, payload: Any) -> bool:
        """Render a single-dict payload as a two-column Field/Value table."""
        table = build_kv_table(payload, max_cell_width=self._MAX_CELL_WIDTH)
        if table is None:
            return False
        self.cli.console.print(table)
        return True

    # ------------------------------------------------------------------ #
    # Invocation
    # ------------------------------------------------------------------ #

    def _invoke(self, client: ServiceClient, method_name: str, args: str) -> None:
        # Preflight: the service may be configured in agent.yml but the
        # adapter package (or platform registration) might be missing.
        # Without this check, ``client.get_tool(method_name)`` would still
        # return a wrapper because the allow-list fallback kicks in, and we
        # would only surface "No BI adapter registered" from deep inside
        # ``_build_adapter``. Better to fail fast with an installable hint.
        if not self.registry.adapter_available(client.service_name):
            self._print_missing_adapter_hint(client)
            return

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
                    f"[dim]Run `/{client.service_name}` to list available methods.[/]"
                )
            return

        if self._is_help_request(args):
            self._print_schema(tool)
            return

        parsed = self._parse_args(args, tool.params_json_schema or {})
        if parsed is None:
            hint = self._last_parse_error or "Could not parse arguments. Expected schema:"
            self._print_schema(tool, hint=hint)
            return

        bound_method = getattr(client.tool_instance, method_name, None)
        missing = self._missing_required(bound_method, parsed)
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

        self._render_result(result, service=client.service_name, method=method_name)

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
        malformed (quoting error, extra positional, unknown named flag).
        When parsing fails in a way that has a specific user-facing hint
        (e.g. typoed flag name), the hint is stored on
        ``self._last_parse_error`` so ``_invoke`` can surface it before
        printing the schema.
        """
        self._last_parse_error = None
        try:
            tokens = shlex.split(args) if args else []
        except ValueError:
            self._last_parse_error = "Malformed arguments: unmatched quotes."
            return None

        props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        prop_order = [k for k in props.keys() if k != "self"]
        valid_named = [k for k in prop_order]

        positional: List[str] = []
        named: Dict[str, str] = {}
        for tok in tokens:
            if tok.startswith("--"):
                body = tok[2:]
                if not body:
                    self._last_parse_error = "Empty flag '--'. Expected '--<name>' or '--<name>=<value>'."
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
                self._last_parse_error = (
                    f"Too many positional arguments. Method accepts {len(prop_order)} (got extra: '{value}')."
                )
                return None
            key = prop_order[idx]
            parsed[key] = self._coerce(value, props.get(key) or {})

        for key, raw in named.items():
            if key not in props:
                # Fail fast — a silently dropped ``--limti=1`` or
                # ``--serach=...`` is worse than a parse error because the
                # method executes without the filter the user intended.
                suggestions = ", ".join(valid_named) if valid_named else "(none)"
                self._last_parse_error = f"Unknown parameter '--{key}'. Valid parameters: {suggestions}."
                return None
            parsed[key] = self._coerce(raw, props.get(key) or {})

        return parsed

    @classmethod
    def _coerce(cls, raw: str, prop_schema: Dict[str, Any]) -> Any:
        t = cls._primary_type(prop_schema)
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
            return cls._coerce_collection(raw, expect=list)
        if t == "object":
            return cls._coerce_collection(raw, expect=dict)
        return raw

    @staticmethod
    def _coerce_collection(raw: str, *, expect: type) -> Any:
        """Coerce ``raw`` to ``expect`` (``list`` or ``dict``).

        Attempts, in order:

        1. ``json.loads`` — standard JSON form (``["a"]`` / ``{"k": 1}``).
        2. ``ast.literal_eval`` — Python literal form which tolerates single
           quotes and ``None`` / ``True``. LLMs and humans frequently emit
           ``--metrics=['sales']`` or ``--ctx={'k': 'v'}``; JSON rejects both.
        3. For arrays only: CSV fallback (``a,b,c`` → ``["a", "b", "c"]``).
           For objects, a parse failure returns the raw string so the tool
           can surface a clearer type error than a silently mangled value.
        """
        stripped = raw.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is None:
                try:
                    parsed = ast.literal_eval(stripped)
                except (SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, expect):
                return parsed
        if expect is list:
            return [item.strip() for item in raw.split(",") if item.strip()]
        return raw

    @staticmethod
    def _primary_type(prop_schema: Dict[str, Any]) -> str:
        """Return the primary JSON-schema type, flattening ``anyOf`` / ``oneOf``.

        ``Optional[X]`` is represented by the Agents SDK as
        ``{"anyOf": [{"type": X}, {"type": "null"}]}`` with no top-level
        ``type``. Naively reading ``schema["type"]`` would yield ``""`` and
        cause ``_coerce`` to skip its conversion logic, so e.g. an
        ``Optional[List[str]]`` parameter would receive a raw CSV string
        instead of a list.
        """
        if not isinstance(prop_schema, dict):
            return ""
        t = prop_schema.get("type")
        if isinstance(t, str):
            return t
        if isinstance(t, list):
            for candidate in t:
                if isinstance(candidate, str) and candidate != "null":
                    return candidate
        for key in ("anyOf", "oneOf"):
            variants = prop_schema.get(key)
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                vt = variant.get("type")
                if isinstance(vt, str) and vt != "null":
                    return vt
        return ""

    @staticmethod
    def _missing_required(method: Optional[Callable], parsed: Dict[str, Any]) -> List[str]:
        """Return names of parameters that are truly required but not supplied.

        Uses the Python signature of the bound method as the source of truth —
        Pydantic / the Agents SDK regularly list parameters with
        ``Optional[...] = None`` defaults in the OpenAI-style ``required``
        array, but those are semantically optional and we should not block
        invocation on them.
        """
        if method is None or not callable(method):
            return []
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return []
        missing: List[str] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in parsed:
                continue
            if param.default is inspect.Parameter.empty:
                missing.append(name)
        return missing

    # ------------------------------------------------------------------ #
    # Async plumbing
    # ------------------------------------------------------------------ #

    def _run_async(self, coro) -> Any:
        """Run the tool coroutine on a fresh private loop.

        ``ServiceCommands`` is invoked synchronously from the REPL thread.
        The allow-listed service methods are synchronous Python (typically a
        blocking HTTP call into Superset/Airflow/MetricFlow) and
        ``trans_to_function_tool`` runs them inline inside the coroutine.
        Scheduling this on the shared ``DatusCLI._bg_loop`` would freeze
        every *other* background task (``_async_init_agent``, session
        writes, etc.) for the full duration of the sync call — the 60s
        ``future.result`` timeout only unblocks the REPL; it does not
        interrupt the sync call still running on the loop thread.

        Using ``asyncio.run`` creates a private event loop that lives only
        for this one invocation and is torn down when we return, so a slow
        or hanging backend call cannot leak into the shared loop.
        """
        return asyncio.run(coro)
