# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared CLI argument parsing for JSON-schema-backed ``FunctionTool`` calls.

Both the ``/<service>.<method>`` handler (:mod:`datus.cli.service_commands`) and
the ``!<tool> [args...]`` handler (:mod:`datus.cli.bang_command`) invoke a
``FunctionTool`` by name from the CLI. They share the same minimal argument
grammar — positional args in schema order plus ``--key=value`` named
overrides — coerced against the tool's ``params_json_schema``:

- Positional, in schema order: ``get_dashboard 1``
- Named overrides: ``get_chart_data 42 --limit=100``
- Lists: ``--subject_path=a,b`` or ``--subject_path=['a','b']``
- Bare boolean flags: ``--simple_sample_data`` means ``=true``

JSON-blob input is deliberately out of scope. This module keeps the parsing
free of any service/tool-registry coupling so both callers behave identically.
"""

from __future__ import annotations

import ast
import inspect
import json
import shlex
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from datus.cli.cli_styles import TABLE_HEADER_STYLE, print_warning

if TYPE_CHECKING:
    from agents import FunctionTool


def is_help_request(args: str) -> bool:
    """True when *args* contains a ``--help`` / ``-h`` token."""
    try:
        tokens = shlex.split(args) if args else []
    except ValueError:
        return False
    return "--help" in tokens or "-h" in tokens


def parse_args(args: str, schema: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse positional + ``--key=value`` arguments against a JSON schema.

    Returns ``(parsed, None)`` on success, or ``(None, error)`` when the input
    is malformed (quoting error, extra positional, unknown named flag). The
    error string carries a specific user-facing hint (e.g. a typoed flag name)
    so callers can surface it before printing the schema.
    """
    try:
        tokens = shlex.split(args) if args else []
    except ValueError:
        return None, "Malformed arguments: unmatched quotes."

    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    prop_order = [k for k in props.keys() if k != "self"]
    valid_named = [k for k in prop_order]

    positional: List[str] = []
    named: Dict[str, str] = {}
    for tok in tokens:
        if tok.startswith("--"):
            body = tok[2:]
            if not body:
                return None, "Empty flag '--'. Expected '--<name>' or '--<name>=<value>'."
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
            return None, (f"Too many positional arguments. Method accepts {len(prop_order)} (got extra: '{value}').")
        key = prop_order[idx]
        parsed[key] = coerce(value, props.get(key) or {})

    for key, raw in named.items():
        if key not in props:
            # Fail fast — a silently dropped ``--limti=1`` or ``--serach=...`` is
            # worse than a parse error because the method executes without the
            # filter the user intended.
            suggestions = ", ".join(valid_named) if valid_named else "(none)"
            return None, f"Unknown parameter '--{key}'. Valid parameters: {suggestions}."
        parsed[key] = coerce(raw, props.get(key) or {})

    return parsed, None


def coerce(raw: str, prop_schema: Dict[str, Any]) -> Any:
    """Coerce a raw string token to the JSON-schema type of *prop_schema*."""
    t = primary_type(prop_schema)
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
        return coerce_collection(raw, expect=list)
    if t == "object":
        return coerce_collection(raw, expect=dict)
    return raw


def coerce_collection(raw: str, *, expect: type) -> Any:
    """Coerce *raw* to *expect* (``list`` or ``dict``).

    Attempts, in order:

    1. ``json.loads`` — standard JSON form (``["a"]`` / ``{"k": 1}``).
    2. ``ast.literal_eval`` — Python literal form which tolerates single
       quotes and ``None`` / ``True``. LLMs and humans frequently emit
       ``--metrics=['sales']`` or ``--ctx={'k': 'v'}``; JSON rejects both.
    3. For arrays only: CSV fallback (``a,b,c`` → ``["a", "b", "c"]``).
       For objects, a parse failure returns the raw string so the tool can
       surface a clearer type error than a silently mangled value.
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


def primary_type(prop_schema: Dict[str, Any]) -> str:
    """Return the primary JSON-schema type, flattening ``anyOf`` / ``oneOf``.

    ``Optional[X]`` is represented by the Agents SDK as
    ``{"anyOf": [{"type": X}, {"type": "null"}]}`` with no top-level ``type``.
    Naively reading ``schema["type"]`` would yield ``""`` and cause
    :func:`coerce` to skip its conversion logic, so e.g. an
    ``Optional[List[str]]`` parameter would receive a raw CSV string instead of
    a list.
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


def missing_required(method: Optional[Callable], parsed: Dict[str, Any]) -> List[str]:
    """Return names of parameters that are truly required but not supplied.

    Uses the Python signature of the bound method as the source of truth —
    Pydantic / the Agents SDK regularly list parameters with ``Optional[...] =
    None`` defaults in the OpenAI-style ``required`` array, but those are
    semantically optional and we should not block invocation on them.
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


def print_schema(console: Console, tool: "FunctionTool", hint: str = "") -> None:
    """Render a ``FunctionTool``'s parameter schema as a Rich table."""
    schema = tool.params_json_schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required", []) or [])
    if hint:
        print_warning(console, hint)
    table = Table(
        title=f"{tool.name} — parameters",
        show_header=True,
        header_style=TABLE_HEADER_STYLE,
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
    console.print(table)
