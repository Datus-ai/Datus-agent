# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Manual SQL/bash execution records — the "execution turn" message format.

When the user runs SQL or bash from the TUI input modes, the command and its
result are packed into a single chat message and sent to the model, triggering
a normal chat turn. The message carries a machine-parsable sentinel + JSON
payload so it survives session persistence and can be decoded on ``/resume``
and re-rendered in exactly the same styled block — the single source of truth
for both the live turn and the restored transcript.

The renderer (``render_exec_block``) and the decoder live here so that
``action_display.renderers`` (live + resume), the ``/resume`` picker preview,
and the web SSE builder can all share one implementation without importing
each other.

Layering: this module may import ``cli_styles`` and Rich, but must NOT import
``action_display`` (which imports this module).
"""

import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError
from rich.box import HORIZONTALS
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from datus.cli.cli_styles import CODE_THEME, TABLE_HEADER_STYLE
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Sentinel prefix marking a manual-execution chat message. Uses mathematical
# white square brackets (U+27E6/U+27E7) — printable, JSON/DB-safe, and
# effectively never present in real user input. Deliberately avoids ``@`` /
# ``[`` / ``/`` so ``parse_at_context`` and Rich markup never mis-handle it.
EXEC_SENTINEL = "⟦datus-exec⟧"

# Caps so a queued execution record never bloats the model context or the
# re-rendered transcript. Mirrors the row/char budgets used elsewhere.
MAX_ROWS = 20
MAX_CELL_CHARS = 200
MAX_OUTPUT_CHARS = 3000

# Per-kind prompt chrome (mirrors the input-mode prompt colours).
_KIND_CHROME = {
    "sql": ("sql> ", "bold red", "red"),
    "bash": ("bash> ", "bold yellow", "yellow"),
    # ``!<tool>`` manual tool executions — distinct cyan chrome.
    "tool": ("! ", "bold cyan", "cyan"),
}


def is_exec_message(text: Any) -> bool:
    """True when *text* is a manual-execution record message."""
    return isinstance(text, str) and text.startswith(EXEC_SENTINEL)


# Action type used for the live streamed execution frame (PROCESSING → terminal).
MANUAL_EXEC_ACTION_TYPE = "manual_exec"


class SqlExecPayload(BaseModel):
    """Validated shape of a persisted SQL execution record."""

    kind: Literal["sql"]
    command: str
    success: bool
    columns: List[str] = []
    rows: List[List[str]] = []
    row_count: Optional[int] = None
    truncated: bool = False
    meta: str = ""
    error: Optional[str] = None


class BashExecPayload(BaseModel):
    """Validated shape of a persisted bash execution record."""

    kind: Literal["bash"]
    command: str
    success: bool
    output: str = ""
    meta: str = ""
    error: Optional[str] = None


class ToolExecPayload(BaseModel):
    """Validated shape of a persisted ``!<tool>`` execution record.

    Mirrors the bash payload: ``command`` is the ``<tool> <args>`` invocation and
    ``output`` is the tool result rendered as text (JSON for structured returns).
    """

    kind: Literal["tool"]
    command: str
    success: bool
    output: str = ""
    meta: str = ""
    error: Optional[str] = None


_PAYLOAD_MODELS: Dict[str, type[BaseModel]] = {
    "sql": SqlExecPayload,
    "bash": BashExecPayload,
    "tool": ToolExecPayload,
}


def mode_prompt(kind: str) -> Tuple[str, str]:
    """Return the ``(prompt, prompt_style)`` for a kind (``sql`` / ``bash``)."""
    prompt, prompt_style, _ = _KIND_CHROME.get(kind, ("> ", "bold", ""))
    return prompt, prompt_style


def encode_exec_message(payload: Dict[str, Any]) -> str:
    """Serialize an execution *payload* into a sentinel-prefixed chat message."""
    return EXEC_SENTINEL + json.dumps(payload, ensure_ascii=False)


def decode_exec_message(text: Any) -> Optional[Dict[str, Any]]:
    """Parse a manual-execution message back into its payload dict.

    The decoded JSON is validated against the typed payload model for its
    ``kind``; ordinary messages and malformed/incompatible records (e.g. a
    corrupted persisted row where ``rows`` is not a list of string lists)
    return ``None`` so callers fall back to plain user-message rendering
    instead of crashing the downstream renderer / SSE builder.
    """
    if not is_exec_message(text):
        return None
    try:
        raw = json.loads(text[len(EXEC_SENTINEL) :])
    except (ValueError, TypeError) as exc:
        logger.debug("Failed to decode exec message: %s", exc)
        return None
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    model = _PAYLOAD_MODELS.get(kind) if isinstance(kind, str) else None
    if model is None:
        return None
    try:
        return model.model_validate(raw).model_dump()
    except ValidationError as exc:
        logger.debug("Invalid exec payload dropped: %s", exc)
        return None


def exec_preview(text: Any) -> str:
    """One-line preview (e.g. for the ``/resume`` picker) of an exec message.

    Falls back to the raw text when it is not an exec record.
    """
    payload = decode_exec_message(text)
    if payload is None:
        return text if isinstance(text, str) else str(text)
    prompt = _KIND_CHROME[payload["kind"]][0]
    command = " ".join(str(payload.get("command", "")).split())
    return f"{prompt}{command}"


def _md_fence(content: str) -> str:
    """Pick a backtick fence longer than any backtick run in *content*.

    A command or output that itself contains ```` ``` ```` would otherwise
    close the code block early and let the remainder render as Markdown; a
    fence of ``max_run + 1`` backticks (at least 3) keeps the block intact.
    """
    longest = 0
    run = 0
    for ch in content:
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def _md_escape(value: Any) -> str:
    """Escape Markdown-structural characters in inline / table-cell text.

    Neutralises pipes (table columns), newlines (row/line breaks), emphasis,
    links, inline code and raw HTML so arbitrary command output can never alter
    the surrounding Markdown structure.
    """
    out: List[str] = []
    for ch in str(value):
        if ch in "\r\n":
            out.append(" ")
        elif ch in "\\`*_{}[]()#+!|<>":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def exec_to_markdown(text: Any) -> str:
    """Render a manual-execution message as readable Markdown (for web/SSE).

    CLI clients decode the payload into a styled block; non-CLI consumers
    (web history / SSE) get an equivalent fenced Markdown rendering so the raw
    sentinel never leaks. Command / output text goes in collision-safe code
    fences and all inline / table content is escaped, so arbitrary SQL, shell
    commands and results cannot break out of the Markdown structure. Returns
    *text* unchanged when it is not an exec record.
    """
    payload = decode_exec_message(text)
    if payload is None:
        return text if isinstance(text, str) else str(text)
    prompt = _KIND_CHROME[payload["kind"]][0]
    lang = "sql" if payload["kind"] == "sql" else "bash"
    command = str(payload.get("command", ""))
    cmd_fence = _md_fence(command)
    lines = [f"`{prompt.strip()}`", f"{cmd_fence}{lang}", command, cmd_fence]
    if payload.get("meta"):
        lines.append(f"_{_md_escape(payload['meta'])}_")
    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if columns:
        lines.append("")
        lines.append("| " + " | ".join(_md_escape(c) for c in columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_md_escape(cell) for cell in row) + " |")
        if payload.get("truncated"):
            lines.append(f"_(showing first {len(rows)} of {payload.get('row_count', len(rows))} rows)_")
    output = payload.get("output")
    if output:
        out_fence = _md_fence(str(output))
        lines.append("")
        lines.append(out_fence)
        lines.append(str(output))
        lines.append(out_fence)
    if not payload.get("success") and payload.get("error"):
        lines.append("")
        lines.append(f"**Error:** {_md_escape(payload['error'])}")
    return "\n".join(lines)


def _coerce_cell(value: Any) -> str:
    """Stringify a result cell, capping oversized values."""
    if value is None:
        text = ""
    else:
        text = str(value)
    if len(text) > MAX_CELL_CHARS:
        text = text[:MAX_CELL_CHARS] + "…"
    return text


def build_sql_payload(
    command: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
    row_count: int,
    exec_time: float,
) -> Dict[str, Any]:
    """Build a successful SQL result payload (arrow/tabular path)."""
    shown = rows[:MAX_ROWS]
    coerced = [[_coerce_cell(row.get(col)) for col in columns] for row in shown]
    return {
        "kind": "sql",
        "command": command,
        "success": True,
        "columns": list(columns),
        "rows": coerced,
        "row_count": row_count,
        "truncated": row_count > len(shown),
        "meta": f"{row_count} rows in {exec_time:.2f}s",
    }


def build_sql_message_payload(command: str, success: bool, meta: str) -> Dict[str, Any]:
    """Build a non-tabular SQL payload (DML row count / DDL / USE-SET)."""
    return {"kind": "sql", "command": command, "success": success, "meta": meta}


def build_sql_error_payload(command: str, error: str) -> Dict[str, Any]:
    """Build a failed SQL payload."""
    return {"kind": "sql", "command": command, "success": False, "error": error, "meta": "failed"}


def build_bash_payload(
    command: str, success: bool, output: str, error: Optional[str], exec_time: float
) -> Dict[str, Any]:
    """Build a bash execution payload (output capped)."""
    output = output or ""
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n… [truncated]"
    meta = f"exit 0 in {exec_time:.2f}s" if success else f"failed in {exec_time:.2f}s"
    payload: Dict[str, Any] = {
        "kind": "bash",
        "command": command,
        "success": success,
        "output": output,
        "meta": meta,
    }
    if error and not success:
        payload["error"] = error
    return payload


def build_tool_payload(
    command: str, success: bool, output: str, error: Optional[str], exec_time: float
) -> Dict[str, Any]:
    """Build a ``!<tool>`` execution payload (output capped like bash)."""
    output = output or ""
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n… [truncated]"
    meta = f"ok in {exec_time:.2f}s" if success else f"failed in {exec_time:.2f}s"
    payload: Dict[str, Any] = {
        "kind": "tool",
        "command": command,
        "success": success,
        "output": output,
        "meta": meta,
    }
    if error and not success:
        payload["error"] = error
    return payload


def _split_stderr(output: str) -> Tuple[str, str]:
    """Split BashTool output into (stdout, stderr) on the ``[stderr]`` marker."""
    stdout, marker, stderr = output.partition("\n[stderr]\n")
    return (stdout, stderr if marker else "")


def _command_renderable(payload: Dict[str, Any]) -> Text:
    """Prompt-prefixed command line: SQL is syntax-highlighted, bash is plain."""
    prompt, prompt_style, _ = _KIND_CHROME[payload["kind"]]
    command = str(payload.get("command", ""))
    line = Text(prompt, style=prompt_style)
    if payload["kind"] == "sql":
        highlighted = Syntax("", "sql", theme=CODE_THEME, background_color="default").highlight(command)
        highlighted.rstrip()
        line.append_text(highlighted)
    else:
        line.append(command)
    return line


def _sql_result_table(payload: Dict[str, Any]) -> Optional[Table]:
    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not columns:
        return None
    table = Table(show_header=True, header_style=TABLE_HEADER_STYLE)
    for col in columns:
        table.add_column(str(col), overflow="fold", no_wrap=False)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    return table


def render_exec_block(payload: Dict[str, Any]) -> RenderableType:
    """Render a manual-execution payload as a bordered, mode-coloured block.

    Used identically by the live chat turn and the ``/resume`` transcript, so
    a restored execution looks exactly like it did when first run.

    Framed like a normal user message — top/bottom rules only (``HORIZONTALS``,
    no side edges so drag-copy stays clean) — but in the mode colour (SQL red /
    bash yellow) instead of the user green. Inside: the ``sql>``/``bash>``
    command line, the result (table / output), and a ✓/✗ status footer with
    timing. The content keeps its normal background so the SQL table and the
    syntax-highlighted command stay legible.
    """
    kind = payload["kind"]
    _, _, border_style = _KIND_CHROME[kind]
    body: List[RenderableType] = [_command_renderable(payload)]

    if kind == "sql":
        table = _sql_result_table(payload)
        if table is not None:
            body.append(table)
        if payload.get("truncated"):
            shown = len(payload.get("rows") or [])
            body.append(Text(f"(showing first {shown} of {payload.get('row_count', shown)} rows)", style="dim"))
    else:
        stdout, stderr = _split_stderr(payload.get("output", ""))
        if stdout.strip():
            body.append(Text(stdout.rstrip("\n")))
        if stderr.strip():
            body.append(Text(stderr.rstrip("\n"), style="red"))

    if not payload.get("success") and payload.get("error"):
        body.append(Text(f"✗ {payload['error']}", style="red"))

    # Tool-call-like status footer: ✓/✗ glyph + meta, dim.
    glyph = "✓" if payload.get("success") else "✗"
    meta = str(payload.get("meta") or "")
    footer_style = "dim" if payload.get("success") else "red"
    body.append(Text(f"{glyph} {meta}".rstrip(), style=footer_style))

    return Panel(Group(*body), border_style=border_style, box=HORIZONTALS, padding=(0, 1), expand=True)
