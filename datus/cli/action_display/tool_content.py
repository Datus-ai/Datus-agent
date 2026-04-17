# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified tool call content rendering — style-agnostic content building with registry support.

Architecture:
- ToolCallContent: style-agnostic data container
- ToolCallContentBuilder: unified build entry + registry
- Per-tool builder functions: each tool's custom visualization is a standalone function
  registered via builder.register(), fully decoupled from the generic logic.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from datus.schemas.action_history import ActionHistory, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallContent:
    """Style-agnostic content for a tool call display."""

    label: str  # tool identifier, e.g. "search_table"
    status_mark: str  # "✓" or "✗"
    duration_str: str  # " (1.2s)" or ""
    output_preview: str = ""  # legacy compact-mode summary; kept for back-compat
    args_lines: List[str] = field(default_factory=list)  # verbose: Rich markup formatted
    output_lines: List[str] = field(default_factory=list)  # verbose: Rich markup formatted
    # Compact-mode fields used by the new header + └─ line layout:
    args_summary: str = ""  # concise args for header, e.g. '"orders"' or 'pattern: "*.py"'
    compact_result: str = ""  # concise result for └─ line, e.g. '3 tables: a, b, c'


# Type alias for custom content builder functions
ToolCallContentFn = Callable[[ActionHistory, bool], ToolCallContent]


# ── Shared helpers ─────────────────────────────────────────────────


def calc_duration(action: ActionHistory) -> str:
    """Calculate duration string from action timestamps.

    Sub-tenth-of-a-second durations are rendered as ``(<0.1s)`` so they stay
    visible without misleading rounding.
    """
    if action.end_time and action.start_time:
        duration_sec = (action.end_time - action.start_time).total_seconds()
        if duration_sec < 0.1:
            return " (<0.1s)"
        return f" ({duration_sec:.1f}s)"
    return ""


# ── Compact-layout helpers (header args + └─ result line) ──────────


def _parse_args_dict(action: ActionHistory) -> dict:
    """Return the tool's arguments as a dict, or an empty dict."""
    if not action.input:
        return {}
    args = action.input.get("arguments")
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}


def _collapse_whitespace(text: str) -> str:
    """Collapse any whitespace run (newlines, tabs, repeated spaces) into one space."""
    return " ".join(str(text).split())


def _truncate_middle(text: str, max_len: int = 60) -> str:
    """Flatten whitespace, then truncate the middle if the result exceeds max_len."""
    flat = _collapse_whitespace(text)
    if len(flat) <= max_len:
        return flat
    keep = max(1, (max_len - 5) // 2)
    return flat[:keep] + " ... " + flat[-keep:]


def _format_positional(args: dict, *keys: str, max_len: int = 60) -> str:
    """Pick the first non-empty key and format as a quoted positional arg."""
    for key in keys:
        val = args.get(key)
        if val not in (None, "", [], {}):
            return f'"{_truncate_middle(val, max_len)}"'
    return ""


def _format_kw(args: dict, *keys: str, max_len: int = 60) -> str:
    """Format selected args as ``key: "value"`` pairs (only non-empty keys)."""
    parts: List[str] = []
    for key in keys:
        val = args.get(key)
        if val in (None, "", [], {}):
            continue
        parts.append(f'{key}: "{_truncate_middle(val, max_len)}"')
    return ", ".join(parts)


def _fallback_args_summary(args: dict, max_kv: int = 2, max_len: int = 40) -> str:
    """Generic fallback: show the first few non-empty args as ``k: "v"``."""
    parts: List[str] = []
    for key, val in args.items():
        if val in (None, "", [], {}):
            continue
        parts.append(f'{key}: "{_truncate_middle(val, max_len)}"')
        if len(parts) >= max_kv:
            break
    return ", ".join(parts)


def _extract_error_message(output_data) -> str:
    """Return a concise error message from the tool output, or an empty string."""
    data = parse_output_data(output_data)
    if not data:
        return ""
    err = data.get("error")
    if err and str(err) not in ("None", "null", ""):
        return _truncate_middle(str(err), 80)
    return ""


def set_default_args_summary(tc: ToolCallContent, action: ActionHistory) -> None:
    """Populate ``tc.args_summary`` from ``action.input`` using the fallback."""
    if tc.args_summary:
        return
    args = _parse_args_dict(action)
    if args:
        tc.args_summary = _fallback_args_summary(args)


def set_error_as_result(tc: ToolCallContent, action: ActionHistory) -> None:
    """If the action failed, set ``compact_result`` to the error message."""
    if action.status == ActionStatus.FAILED or (
        isinstance(action.output, dict) and action.output.get("success") is False
    ):
        err = _extract_error_message(action.output)
        if err:
            tc.compact_result = err


def _item_name(item) -> str:
    """Best-effort extraction of a short name from a list item (str or dict)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("name", "table_name", "database", "schema", "identifier", "title", "id"):
            val = item.get(key)
            if val:
                return str(val)
    return ""


def _fmt_count_with_preview(
    count: int,
    noun_singular: str,
    noun_plural: str,
    items: Optional[list] = None,
    max_show: int = 3,
) -> str:
    """Format ``N nouns: item1, item2, item3`` when item names are available."""
    noun = noun_singular if count == 1 else noun_plural
    header = f"{count} {noun}"
    if not items:
        return header
    names: List[str] = []
    for item in items[:max_show]:
        name = _item_name(item)
        if name:
            names.append(name)
        if len(names) >= max_show:
            break
    if not names:
        return header
    preview = ", ".join(names)
    if count > len(names):
        preview += ", ..."
    return f"{header}: {preview}"


def _strip_legacy_preview(preview: str) -> str:
    """Convert the legacy ``\u2713 xxx`` / ``\u2717 yyy`` preview into a bare result.

    The renderer now draws the status mark on the ``\u2514\u2500`` line itself,
    so per-builder previews must not prefix their own mark. When an older
    builder still emits one, strip it to keep the output clean.
    """
    if not preview:
        return ""
    text = preview.strip()
    for prefix in ("\u2713 ", "\u2717 ", "\u00d7 "):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


# ── Per-tool argument summary formatters ───────────────────────────
#
# Maps a function_name to a callable(args_dict) -> str that produces the
# concise arg string rendered inside the ``tool(...)`` header. Only tools
# listed here get a hand-tuned summary; others fall back to the generic
# key:"value" pairs via ``set_default_args_summary``.
_TOOL_ARGS_FORMATTERS: Dict[str, Callable[[dict], str]] = {
    # DB tools
    "list_tables": lambda a: _format_kw(a, "catalog", "schema_name"),
    "describe_table": lambda a: _format_positional(a, "table_name", "name"),
    "search_table": lambda a: _format_positional(a, "query_text", "query"),
    "read_query": lambda a: _format_positional(a, "query", "sql"),
    "query": lambda a: _format_positional(a, "query", "sql"),
    "get_table_ddl": lambda a: _format_positional(a, "table_name", "name"),
    "list_databases": lambda _a: "",
    "list_schemas": lambda a: _format_positional(a, "database", "catalog"),
    # Filesystem tools
    "read_file": lambda a: _format_positional(a, "file_path", "path"),
    "write_file": lambda a: _format_positional(a, "file_path", "path"),
    "edit_file": lambda a: _format_positional(a, "file_path", "path"),
    "glob": lambda a: _format_kw(a, "pattern", "path"),
    "grep": lambda a: _format_kw(a, "pattern", "path"),
    # Context search tools
    "list_subject_tree": lambda _a: "",
    "search_metrics": lambda a: _format_positional(a, "query", "query_text"),
    "get_metrics": lambda a: _format_positional(a, "metric_name", "metric_id", "name"),
    "search_reference_sql": lambda a: _format_positional(a, "query", "query_text"),
    "get_reference_sql": lambda a: _format_positional(a, "ref_id", "sql_id", "name"),
    "search_semantic_objects": lambda a: _format_positional(a, "query", "query_text"),
    "search_knowledge": lambda a: _format_positional(a, "query", "query_text"),
    "get_knowledge": lambda a: _format_positional(a, "doc_id", "id", "name"),
    # Date parsing tools
    "parse_temporal_expressions": lambda a: _format_positional(a, "expression", "text", "query"),
    # Reference template tools
    "search_reference_template": lambda a: _format_positional(a, "query", "query_text"),
    "get_reference_template": lambda a: _format_positional(a, "template_id", "name"),
    "render_reference_template": lambda a: _format_positional(a, "template_id", "name"),
    "execute_reference_template": lambda a: _format_positional(a, "template_id", "name"),
    # Platform document tools
    "list_document_nav": lambda _a: "",
    "get_document": lambda a: _format_positional(a, "doc_id", "doc_name", "name"),
    "search_document": lambda a: _format_positional(a, "query", "query_text"),
    "web_search_document": lambda a: _format_positional(a, "query", "query_text"),
    # Skill tools
    "load_skill": lambda a: _format_positional(a, "skill_name", "name"),
    "skill_execute_command": lambda a: _format_kw(a, "skill_name", "command"),
    "execute_command": lambda a: _format_kw(a, "skill_name", "command"),
}


def set_tool_specific_args_summary(tc: ToolCallContent, action: ActionHistory) -> None:
    """Populate ``tc.args_summary`` using a per-tool formatter when available."""
    if tc.args_summary:
        return
    function_name = action.input.get("function_name", "") if action.input else ""
    formatter = _TOOL_ARGS_FORMATTERS.get(function_name)
    if formatter is None:
        return
    summary = formatter(_parse_args_dict(action))
    if summary:
        tc.args_summary = summary


def extract_args(action: ActionHistory) -> List[str]:
    """Extract argument lines from action input as YAML-like key: value pairs."""
    lines: List[str] = []
    if not action.input or not action.input.get("arguments"):
        return lines
    args = action.input["arguments"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            lines.append(f"args: {args}")
            return lines
    if isinstance(args, dict):
        for k, v in args.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append(f"args: {args}")
    return lines


def make_base_content(action: ActionHistory) -> ToolCallContent:
    """Create a ToolCallContent with label / status_mark / duration_str filled in."""
    function_name = action.input.get("function_name", "") if action.input else ""
    return ToolCallContent(
        label=function_name or action.messages or "",
        status_mark="\u2713" if action.status == ActionStatus.SUCCESS else "\u2717",
        duration_str=calc_duration(action),
    )


def parse_output_data(output_data) -> Optional[dict]:
    """Normalize output_data to a parsed dict. Returns None on failure."""
    if not output_data:
        return None
    if isinstance(output_data, str):
        try:
            output_data = json.loads(output_data)
        except Exception:
            return None
    if not isinstance(output_data, dict):
        return None
    data = output_data.get("raw_output", output_data)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return data


def format_output_verbose(output_data, indent: str = "") -> List[str]:
    """Format tool output fully for verbose mode (no truncation)."""
    lines: List[str] = []
    if not output_data:
        return lines

    if isinstance(output_data, str):
        try:
            output_data = json.loads(output_data)
        except Exception:
            lines.append(f"{indent}output: {output_data}")
            return lines

    if not isinstance(output_data, dict):
        lines.append(f"{indent}output: {output_data}")
        return lines

    data = output_data.get("raw_output", output_data)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            lines.append(f"{indent}output: {data}")
            return lines

    if isinstance(data, dict):
        for k, v in data.items():
            v_str = str(v)
            if "\n" in v_str:
                lines.append(f"{indent}{k}:")
                for sub_line in v_str.split("\n"):
                    lines.append(f"{indent}  {sub_line}")
            else:
                lines.append(f"{indent}{k}: {v_str}")
    else:
        lines.append(f"{indent}output: {data}")

    return lines


def format_generic_preview(output_data, enable_truncation: bool = True) -> str:
    """Generic output preview: error detection -> list counting -> success/completed.

    This is the fallback for tools without a registered custom builder.
    """
    if not output_data:
        return ""

    data = parse_output_data(output_data)
    if data is None:
        return "\u2713 Completed (preview unavailable)"

    # Error detection
    if "success" in data and not data["success"]:
        if "error" in data:
            error = data["error"] if len(data["error"]) <= 50 else data["error"][:50] + "..."
            return f"\u2717 Failed:({error})"
        return "\u2717 Failed"

    # Try to extract a countable list from data.text or data.result
    items = _extract_items(data, enable_truncation)
    if isinstance(items, str):
        # _extract_items returned a text preview string
        return items
    if isinstance(items, list):
        return f"\u2713 {len(items)} items"

    # Generic fallback
    raw = output_data if isinstance(output_data, dict) else {}
    if "success" in raw:
        return "\u2713 Success" if raw["success"] else "\u2717 Failed"

    return "\u2713 Completed"


def _extract_items(data: dict, enable_truncation: bool):
    """Try to extract items list from data. Returns list, text preview str, or None."""
    if "text" in data and isinstance(data["text"], str):
        text_content = data["text"]
        try:
            cleaned_text = text_content.replace("'", '"').replace("None", "null")
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            if enable_truncation and len(text_content) > 50:
                return f"{text_content[:50]}..."
            return text_content
    if "result" in data and isinstance(data["result"], list):
        return data["result"]
    return None


# ── Rich markup format helpers ─────────────────────────────────────

# Keys whose values should be highlighted as SQL
_SQL_KEYS = {"sql", "sql_query", "query", "definition", "sql_return"}


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in user-provided text."""
    return text.replace("[", "\\[")


def extract_args_markup(action: ActionHistory) -> List[str]:
    """Extract argument lines with Rich markup formatting."""
    lines: List[str] = []
    if not action.input or not action.input.get("arguments"):
        return lines
    args = action.input["arguments"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            lines.append(f"args: {_escape_markup(args)}")
            return lines
    if isinstance(args, dict):
        for k, v in args.items():
            v_str = str(v)
            v_esc = _escape_markup(v_str)
            if k.lower() in _SQL_KEYS:
                lines.append(f"[bold]{k}[/bold]:")
                for sql_line in v_str.split("\n"):
                    lines.append(f"  [bright_cyan]{_escape_markup(sql_line)}[/bright_cyan]")
            elif "\n" in v_str:
                lines.append(f"[bold]{k}[/bold]:")
                for sub_line in v_str.split("\n"):
                    lines.append(f"  [dim]{_escape_markup(sub_line)}[/dim]")
            else:
                lines.append(f"[bold]{k}[/bold]: {v_esc}")
    else:
        lines.append(f"args: {_escape_markup(str(args))}")
    return lines


def format_output_verbose_markup(output_data, indent: str = "") -> List[str]:
    """Format tool output with Rich markup for verbose mode."""
    lines: List[str] = []
    if not output_data:
        return lines

    if isinstance(output_data, str):
        try:
            output_data = json.loads(output_data)
        except Exception:
            lines.append(f"{indent}output: {_escape_markup(output_data)}")
            return lines

    if not isinstance(output_data, dict):
        lines.append(f"{indent}output: {_escape_markup(str(output_data))}")
        return lines

    data = output_data.get("raw_output", output_data)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            lines.append(f"{indent}output: {_escape_markup(data)}")
            return lines

    if isinstance(data, dict):
        for k, v in data.items():
            v_str = str(v)
            v_esc = _escape_markup(v_str)
            if k.lower() == "error" and v_str:
                lines.append(f"{indent}[bold red]{k}: {v_esc}[/bold red]")
            elif k.lower() in _SQL_KEYS:
                lines.append(f"{indent}[bold]{k}[/bold]:")
                for sql_line in v_str.split("\n"):
                    lines.append(f"{indent}  [bright_cyan]{_escape_markup(sql_line)}[/bright_cyan]")
            elif "\n" in v_str:
                lines.append(f"{indent}[bold]{k}[/bold]:")
                for sub_line in v_str.split("\n"):
                    lines.append(f"{indent}  {_escape_markup(sub_line)}")
            else:
                lines.append(f"{indent}[bold]{k}[/bold]: {v_esc}")
    else:
        lines.append(f"{indent}output: {_escape_markup(str(data))}")

    return lines


_METADATA_KEYS = {"success", "error"}


def _format_result_only_markup(output_data, indent: str = "") -> List[str]:
    """Format tool output for verbose mode, skipping success/error metadata on success.

    When the tool succeeded (success is truthy or absent), only show the result data.
    When the tool failed, show the error.
    Falls back to format_output_verbose_markup for unparseable data.
    """
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose_markup(output_data, indent)

    # If there's an error, show it
    error = data.get("error")
    if error and str(error) not in ("None", "null", ""):
        return [f"{indent}[bold red]error: {_escape_markup(str(error))}[/bold red]"]

    # Extract result and format it
    result = data.get("result")
    if result is None:
        # No result key — show all non-metadata keys
        lines: List[str] = []
        for k, v in data.items():
            if k.lower() in _METADATA_KEYS:
                continue
            v_str = str(v)
            v_esc = _escape_markup(v_str)
            if k.lower() in _SQL_KEYS:
                lines.append(f"{indent}[bold]{k}[/bold]:")
                for sql_line in v_str.split("\n"):
                    lines.append(f"{indent}  [bright_cyan]{_escape_markup(sql_line)}[/bright_cyan]")
            elif "\n" in v_str:
                lines.append(f"{indent}[bold]{k}[/bold]:")
                for sub_line in v_str.split("\n"):
                    lines.append(f"{indent}  {_escape_markup(sub_line)}")
            else:
                lines.append(f"{indent}[bold]{k}[/bold]: {v_esc}")
        return lines if lines else format_output_verbose_markup(output_data, indent)

    return _format_value_markup(result, indent)


def _format_value_markup(value, indent: str = "") -> List[str]:
    """Format a single value (result field) with Rich markup."""
    if isinstance(value, list):
        lines: List[str] = []
        for idx, item in enumerate(value):
            if isinstance(item, dict):
                # Estimate total content length to decide layout
                total_len = sum(len(str(k)) + len(str(v)) + 4 for k, v in item.items())
                if total_len <= 120 and not any("\n" in str(v) for v in item.values()):
                    # Compact: single line
                    parts = []
                    for k, v in item.items():
                        parts.append(f"[bold]{k}[/bold]: {_escape_markup(str(v))}")
                    lines.append(f"{indent}{', '.join(parts)}")
                else:
                    # Multi-line: each key on its own line
                    if idx > 0:
                        lines.append(f"{indent}[dim]---[/dim]")
                    lines.extend(_format_dict_markup(item, indent))
            else:
                lines.append(f"{indent}{_escape_markup(str(item))}")
        return lines
    elif isinstance(value, dict):
        return _format_dict_markup(value, indent)
    elif isinstance(value, str):
        if "\n" in value:
            return [f"{indent}{_escape_markup(line)}" for line in value.split("\n")]
        return [f"{indent}{_escape_markup(value)}"]
    else:
        return [f"{indent}{_escape_markup(str(value))}"]


def _format_dict_markup(d: dict, indent: str = "") -> List[str]:
    """Format a single dict with Rich markup, one key per line."""
    lines: List[str] = []
    for k, v in d.items():
        v_str = str(v)
        v_esc = _escape_markup(v_str)
        if k.lower() in _SQL_KEYS:
            lines.append(f"{indent}[bold]{k}[/bold]:")
            for sql_line in v_str.split("\n"):
                lines.append(f"{indent}  [bright_cyan]{_escape_markup(sql_line)}[/bright_cyan]")
        elif "\n" in v_str:
            lines.append(f"{indent}[bold]{k}[/bold]:")
            for sub_line in v_str.split("\n"):
                lines.append(f"{indent}  {_escape_markup(sub_line)}")
        elif len(v_str) > 100:
            lines.append(f"{indent}[bold]{k}[/bold]:")
            lines.append(f"{indent}  {v_esc}")
        else:
            lines.append(f"{indent}[bold]{k}[/bold]: {v_esc}")
    return lines


def _format_describe_table_output_markup(output_data) -> List[str]:
    """Format describe_table output with Rich markup and aligned columns."""
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose_markup(output_data)

    lines: List[str] = []

    if "success" in data:
        lines.append(f"[bold]success[/bold]: {data['success']}")
    if "error" in data and data["error"]:
        lines.append(f"[bold red]error: {_escape_markup(str(data['error']))}[/bold red]")

    result = data.get("result")
    if not isinstance(result, dict):
        return lines if lines else format_output_verbose_markup(output_data)

    columns = result.get("columns")
    if not isinstance(columns, list):
        for k, v in result.items():
            lines.append(f"[bold]{k}[/bold]: {_escape_markup(str(v))}")
        return lines

    lines.append(f"[bold]columns[/bold] ({len(columns)}):")

    max_name_len = max((len(col.get("name", "")) for col in columns if isinstance(col, dict)), default=0)
    max_type_len = max((len(col.get("type", "")) for col in columns if isinstance(col, dict)), default=0)
    max_name_len = min(max_name_len, 30)
    max_type_len = min(max_type_len, 20)

    for col in columns:
        if isinstance(col, dict):
            name = col.get("name", "?")
            col_type = col.get("type", "?")
            comment = col.get("comment", "")
            name_padded = _escape_markup(name.ljust(max_name_len))
            type_padded = _escape_markup(col_type.ljust(max_type_len))
            if comment:
                lines.append(f"  [cyan]{name_padded}[/cyan]  [dim]{type_padded}[/dim]  {_escape_markup(comment)}")
            else:
                lines.append(f"  [cyan]{name_padded}[/cyan]  [dim]{type_padded}[/dim]")
        else:
            lines.append(f"  {_escape_markup(str(col))}")

    return lines


def _format_read_query_output_markup(output_data) -> List[str]:
    """Format read_query output with Rich markup — show data preview."""
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose_markup(output_data)

    if "error" in data and data["error"]:
        return [f"[bold red]error: {_escape_markup(str(data['error']))}[/bold red]"]

    result = data.get("result")
    if not isinstance(result, dict):
        if result is not None:
            return [f"[bold]result[/bold]: {_escape_markup(str(result))}"]
        return format_output_verbose_markup(output_data)

    compressed_data = result.get("compressed_data")
    if isinstance(compressed_data, str) and compressed_data:
        return _format_csv_preview_markup(compressed_data)

    return format_output_verbose_markup(output_data)


def _format_csv_preview_markup(csv_text: str, max_rows: int = 10, max_col_width: int = 35) -> List[str]:
    """Format CSV data as a readable table preview with Rich markup."""
    import csv
    import io

    from tabulate import tabulate

    stripped = csv_text.strip()
    if not stripped:
        return []

    reader = csv.reader(io.StringIO(stripped))
    all_rows = list(reader)
    if not all_rows:
        return []

    max_cols = 8
    header = all_rows[0][:max_cols]
    total_cols = len(all_rows[0])
    total_data_rows = len(all_rows) - 1

    data_rows: List[List[str]] = []
    for row in all_rows[1:]:
        if all(c.strip() == "..." for c in row):
            if data_rows:
                data_rows.append(["..."] * len(header))
            continue
        truncated = []
        for val in row[:max_cols]:
            val = val.strip()
            if len(val) > max_col_width:
                val = val[: max_col_width - 3] + "..."
            truncated.append(val)
        data_rows.append(truncated)
        if len([r for r in data_rows if r[0] != "..."]) >= max_rows:
            break

    table_str = tabulate(data_rows, headers=header, tablefmt="simple")
    lines = ["[bold]data[/bold]:"]
    for table_line in table_str.splitlines():
        lines.append(f"  {_escape_markup(table_line)}")

    if total_data_rows > max_rows:
        lines.append(f"  [dim]... ({total_data_rows} rows total)[/dim]")
    if total_cols > max_cols:
        lines.append(f"  [dim]({total_cols - max_cols} more columns not shown)[/dim]")

    return lines


def _format_get_table_ddl_output_markup(output_data) -> List[str]:
    """Format get_table_ddl output with Rich markup and highlighted DDL."""
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose_markup(output_data)

    lines: List[str] = []

    if "success" in data:
        lines.append(f"[bold]success[/bold]: {data['success']}")
    if "error" in data and data["error"]:
        lines.append(f"[bold red]error: {_escape_markup(str(data['error']))}[/bold red]")

    result = data.get("result")
    if not isinstance(result, dict):
        if result is not None:
            lines.append(f"[bold]result[/bold]: {_escape_markup(str(result))}")
        return lines if lines else format_output_verbose_markup(output_data)

    identifier = result.get("identifier")
    if identifier:
        lines.append(f"[bold]table[/bold]: [cyan]{_escape_markup(identifier)}[/cyan]")

    table_type = result.get("table_type")
    if table_type:
        lines.append(f"[bold]type[/bold]: {_escape_markup(table_type)}")

    definition = result.get("definition")
    if isinstance(definition, str) and definition:
        lines.append("[bold]definition[/bold]:")
        for ddl_line in definition.split("\n"):
            lines.append(f"  [bright_cyan]{_escape_markup(ddl_line)}[/bright_cyan]")

    return lines if lines else format_output_verbose_markup(output_data)


# ── Verbose format helpers for specific tools ─────────────────────


def _format_csv_preview(csv_text: str, max_rows: int = 10, max_col_width: int = 35) -> List[str]:
    """Format compressed CSV data as a readable table preview using csv + tabulate."""
    import csv
    import io

    from tabulate import tabulate

    stripped = csv_text.strip()
    if not stripped:
        return []

    reader = csv.reader(io.StringIO(stripped))
    all_rows = list(reader)
    if not all_rows:
        return []

    max_cols = 8
    header = all_rows[0][:max_cols]
    total_cols = len(all_rows[0])
    total_data_rows = len(all_rows) - 1

    # Collect data rows, skip "..." placeholder rows
    data_rows: List[List[str]] = []
    for row in all_rows[1:]:
        if all(c.strip() == "..." for c in row):
            if data_rows:
                data_rows.append(["..."] * len(header))
            continue
        # Truncate long cell values
        truncated = []
        for val in row[:max_cols]:
            val = val.strip()
            if len(val) > max_col_width:
                val = val[: max_col_width - 3] + "..."
            truncated.append(val)
        data_rows.append(truncated)
        if len([r for r in data_rows if r[0] != "..."]) >= max_rows:
            break

    table_str = tabulate(data_rows, headers=header, tablefmt="simple")
    lines = ["data:"]
    for table_line in table_str.splitlines():
        lines.append(f"  {table_line}")

    if total_data_rows > max_rows:
        lines.append(f"  ... ({total_data_rows} rows total)")
    if total_cols > max_cols:
        lines.append(f"  ({total_cols - max_cols} more columns not shown)")

    return lines


def _format_read_query_output_verbose(output_data) -> List[str]:
    """Format read_query output for verbose mode — only show data preview."""
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose(output_data)

    # Show error if present
    if "error" in data and data["error"]:
        return [f"error: {data['error']}"]

    result = data.get("result")
    if not isinstance(result, dict):
        if result is not None:
            return [f"result: {result}"]
        return format_output_verbose(output_data)

    compressed_data = result.get("compressed_data")
    if isinstance(compressed_data, str) and compressed_data:
        return _format_csv_preview(compressed_data)

    return format_output_verbose(output_data)


def _format_describe_table_output_verbose(output_data) -> List[str]:
    """Format describe_table output for verbose mode with structured column display."""
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose(output_data)

    lines: List[str] = []

    if "success" in data:
        lines.append(f"success: {data['success']}")
    if "error" in data and data["error"]:
        lines.append(f"error: {data['error']}")

    result = data.get("result")
    if not isinstance(result, dict):
        return lines if lines else format_output_verbose(output_data)

    columns = result.get("columns")
    if not isinstance(columns, list):
        for k, v in result.items():
            lines.append(f"{k}: {v}")
        return lines

    lines.append(f"columns ({len(columns)}):")

    max_name_len = max((len(col.get("name", "")) for col in columns if isinstance(col, dict)), default=0)
    max_type_len = max((len(col.get("type", "")) for col in columns if isinstance(col, dict)), default=0)
    max_name_len = min(max_name_len, 30)
    max_type_len = min(max_type_len, 20)

    for col in columns:
        if isinstance(col, dict):
            name = col.get("name", "?")
            col_type = col.get("type", "?")
            comment = col.get("comment", "")
            name_padded = name.ljust(max_name_len)
            type_padded = col_type.ljust(max_type_len)
            if comment:
                lines.append(f"  {name_padded}  {type_padded}  {comment}")
            else:
                lines.append(f"  {name_padded}  {type_padded}")
        else:
            lines.append(f"  {col}")

    return lines


def _format_get_table_ddl_output_verbose(output_data) -> List[str]:
    """Format get_table_ddl output for verbose mode with formatted DDL."""
    data = parse_output_data(output_data)
    if data is None:
        return format_output_verbose(output_data)

    lines: List[str] = []

    if "success" in data:
        lines.append(f"success: {data['success']}")
    if "error" in data and data["error"]:
        lines.append(f"error: {data['error']}")

    result = data.get("result")
    if not isinstance(result, dict):
        if result is not None:
            lines.append(f"result: {result}")
        return lines if lines else format_output_verbose(output_data)

    identifier = result.get("identifier")
    if identifier:
        lines.append(f"table: {identifier}")

    table_type = result.get("table_type")
    if table_type:
        lines.append(f"type: {table_type}")

    definition = result.get("definition")
    if isinstance(definition, str) and definition:
        lines.append("definition:")
        for ddl_line in definition.split("\n"):
            lines.append(f"  {ddl_line}")

    return lines if lines else format_output_verbose(output_data)


# ── Per-tool builder functions ─────────────────────────────────────


def _build_list_tables(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_tables / table_overview: show table count and first few names."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        items = _get_items_from_output(action.output)
        if isinstance(items, list):
            tc.compact_result = _fmt_count_with_preview(len(items), "table", "tables", items)
    return tc


def _build_describe_table(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """describe_table: show column count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_describe_table_output_markup(action.output)
    else:
        data = parse_output_data(action.output)
        columns = None
        if isinstance(data, dict):
            result = data.get("result")
            if isinstance(result, dict) and isinstance(result.get("columns"), list):
                columns = result["columns"]
            elif isinstance(result, list):
                columns = result
        if columns is None:
            items = _get_items_from_output(action.output)
            if isinstance(items, list):
                columns = items
        if isinstance(columns, list):
            tc.compact_result = f"{len(columns)} columns"
    return tc


def _build_read_query(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """read_query / query: show row × column count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_read_query_output_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            # Check top-level and nested result for rows / columns
            rows = data.get("original_rows")
            cols = None
            if rows is None:
                result = data.get("result")
                if isinstance(result, dict):
                    rows = result.get("original_rows")
                    cols = result.get("column_count")
                    if cols is None:
                        compressed = result.get("compressed_data")
                        if isinstance(compressed, str) and compressed:
                            first_line = compressed.split("\n", 1)[0]
                            if first_line:
                                cols = len(first_line.split(","))
            if rows is not None:
                if cols:
                    tc.compact_result = f"{rows} \u00d7 {cols} result"
                else:
                    tc.compact_result = f"{rows} rows"
            else:
                items = _get_items_from_output(action.output)
                if isinstance(items, list):
                    tc.compact_result = f"{len(items)} items"
    return tc


def _build_search_table(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_table: show metadata + sample_data counts."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            metadata_count = len(data.get("metadata") or [])
            sample_data = data.get("sample_data")
            if isinstance(sample_data, dict):
                sample_count = sample_data.get("original_rows", 0) or 0
            elif isinstance(sample_data, list):
                sample_count = len(sample_data)
            else:
                sample_count = 0
            tc.compact_result = (
                f"{metadata_count} {_plural_unit(metadata_count, 'table', 'tables')} "
                f"and {sample_count} {_plural_unit(sample_count, 'sample row', 'sample rows')}"
            )
    return tc


def _build_search_metrics(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_metrics: show metric count."""
    return _build_search_generic(action, verbose, "metric", "metrics")


def _build_search_reference_sql(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_reference_sql: show reference SQL count."""
    return _build_search_generic(action, verbose, "reference SQL", "reference SQLs")


def _build_search_external_knowledge(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_external_knowledge / search_knowledge: show knowledge count."""
    return _build_search_generic(action, verbose, "knowledge entry", "knowledge entries")


def _build_search_documents(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_documents: show document count."""
    return _build_search_generic(action, verbose, "document", "documents")


def _plural_unit(count: int, singular: str, plural: str) -> str:
    """Pick ``singular`` for ``count == 1``, ``plural`` otherwise."""
    return singular if count == 1 else plural


def _build_search_generic(
    action: ActionHistory,
    verbose: bool,
    unit_singular: str,
    unit_plural: str,
) -> ToolCallContent:
    """Shared builder for search_* tools that count items from parsed output."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        items = _get_items_from_output(action.output)
        count = len(items) if isinstance(items, list) else 0
        tc.compact_result = f"{count} {_plural_unit(count, unit_singular, unit_plural)} matched"
    return tc


def _build_get_table_ddl(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_table_ddl: show the table identifier and DDL size."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_get_table_ddl_output_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                identifier = result.get("identifier", "")
                definition = result.get("definition") or ""
                chars = len(definition) if isinstance(definition, str) else 0
                if identifier and chars:
                    tc.compact_result = f"{identifier} \u00b7 {chars:,} chars"
                elif chars:
                    tc.compact_result = f"{chars:,} chars"
                elif identifier:
                    tc.compact_result = str(identifier)
                else:
                    tc.compact_result = "DDL retrieved"
    return tc


def _build_edit_file(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """edit_file: show GitHub-style diff for edits."""
    tc = make_base_content(action)
    if verbose:
        args_lines: List[str] = []
        edits = []
        if action.input and action.input.get("arguments"):
            args = action.input["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                for k, v in args.items():
                    if k == "edits" and isinstance(v, list):
                        edits = v
                    else:
                        args_lines.append(f"[bold]{k}[/bold]: {_escape_markup(str(v))}")

        # Render each edit as a diff block with colors
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                continue
            old_text = edit.get("oldText", "")
            new_text = edit.get("newText", "")
            if len(edits) > 1:
                args_lines.append(f"[bold]edit {i + 1}[/bold]:")
                prefix = "  "
            else:
                prefix = ""
            for line in old_text.split("\n"):
                args_lines.append(f"{prefix}[red]- {_escape_markup(line)}[/red]")
            for line in new_text.split("\n"):
                args_lines.append(f"{prefix}[green]+ {_escape_markup(line)}[/green]")

        tc.args_lines = args_lines
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        args = _parse_args_dict(action)
        edits_count = len(args.get("edits", [])) if isinstance(args.get("edits"), list) else 0
        if data:
            result = data.get("result")
            if edits_count:
                noun = "edit" if edits_count == 1 else "edits"
                tc.compact_result = f"{edits_count} {noun} applied"
            elif isinstance(result, str) and "edit" in result.lower():
                tc.compact_result = result.split("(")[-1].rstrip(")") if "(" in result else "Edited"
            elif data.get("success"):
                tc.compact_result = "Edited"
    return tc


def _build_write_file(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """write_file: show file path and write result."""
    tc = make_base_content(action)
    if verbose:
        # Custom args: show path/file_type, truncate long content
        args_lines: List[str] = []
        if action.input and action.input.get("arguments"):
            args = action.input["arguments"]
            if isinstance(args, dict):
                for k, v in args.items():
                    if k == "content" and isinstance(v, str):
                        content_lines = v.split("\n")
                        if len(content_lines) > 5:
                            args_lines.append(f"[bold]content[/bold]: [dim]({len(content_lines)} lines)[/dim]")
                            for line in content_lines[:5]:
                                args_lines.append(f"  [dim]{_escape_markup(line)}[/dim]")
                            args_lines.append(f"  [dim]... ({len(content_lines) - 5} more lines)[/dim]")
                        else:
                            args_lines.append("[bold]content[/bold]:")
                            for line in content_lines:
                                args_lines.append(f"  [dim]{_escape_markup(line)}[/dim]")
                    else:
                        args_lines.append(f"[bold]{k}[/bold]: {_escape_markup(str(v))}")
        tc.args_lines = args_lines
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        args = _parse_args_dict(action)
        content = args.get("content")
        if isinstance(content, str) and content:
            line_count = len(content.splitlines()) or 1
            tc.compact_result = f"wrote {line_count} lines"
        elif data:
            result = data.get("result")
            if isinstance(result, str) and "success" in result.lower():
                tc.compact_result = "File written"
            elif data.get("success"):
                tc.compact_result = "File written"
    return tc


# ── Shared builder patterns ────────────────────────────────────────


def _build_simple_list(action: ActionHistory, verbose: bool, unit: str) -> ToolCallContent:
    """Shared builder for tools that return a list of items in result."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        items = None
        if data:
            result = data.get("result")
            if isinstance(result, list):
                items = result
        if items is None:
            items = _get_items_from_output(action.output)
        if isinstance(items, list):
            singular = unit[:-1] if unit.endswith("s") else unit
            tc.compact_result = _fmt_count_with_preview(len(items), singular, unit, items)
    return tc


def _build_get_detail(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """Shared builder for get_metrics / get_reference_sql — extract result['name']."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                name = result.get("name", "")
                tc.compact_result = f"{name}" if name else "Retrieved"
            elif isinstance(result, str):
                tc.compact_result = _truncate_middle(result, 50) if result else "Retrieved"
    return tc


def _build_simple_action(action: ActionHistory, verbose: bool, success_label: str) -> ToolCallContent:
    """Shared builder for tools with simple success/fail output."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            if action.status == ActionStatus.FAILED:
                tc.compact_result = f"{action.output if isinstance(action.output, str) else 'Failed'}"
            else:
                tc.compact_result = f"{success_label}"
    return tc


def _build_doc_search_result(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """Shared builder for search_document / web_search_document — use result['doc_count']."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                doc_count = result.get("doc_count", 0)
                tc.compact_result = f"{doc_count} docs found"
            else:
                items = _get_items_from_output(action.output)
                count = len(items) if isinstance(items, list) else 0
                tc.compact_result = f"{count} docs found"
    return tc


# ── Custom builder functions for specific tools ────────────────────


def _build_list_databases(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_databases: show database count."""
    return _build_simple_list(action, verbose, "databases")


def _build_list_schemas(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_schemas: show schema count."""
    return _build_simple_list(action, verbose, "schemas")


def _build_list_subject_tree(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_subject_tree: show domain count, verbose renders tree structure."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                tc.output_lines = _format_subject_tree_markup(result)
            else:
                tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                tc.compact_result = f"{len(result)} domains"
    return tc


_LEAF_KEYS = {"metrics", "reference_sql", "knowledge"}


def _format_subject_tree_markup(tree: dict, indent: str = "") -> List[str]:
    """Render a subject tree dict as a tree with Rich markup.

    Uses box-drawing characters (├──, └──, │) for structure.
    Leaf keys (metrics, reference_sql, knowledge) are rendered as counts.
    """
    lines: List[str] = []
    items = list(tree.items())
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "\u2514\u2500\u2500" if is_last else "\u251c\u2500\u2500"
        child_indent = indent + ("    " if is_last else "\u2502   ")

        if key in _LEAF_KEYS and isinstance(value, list):
            count = len(value)
            lines.append(f"{indent}{connector} [dim]{_escape_markup(key)}[/dim]: [cyan]{count}[/cyan]")
        elif isinstance(value, dict):
            lines.append(f"{indent}{connector} [bold]{_escape_markup(key)}[/bold]")
            lines.extend(_format_subject_tree_markup(value, child_indent))
        else:
            lines.append(f"{indent}{connector} [bold]{_escape_markup(key)}[/bold]: {_escape_markup(str(value))}")
    return lines


def _build_get_metrics(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_metrics: show metric name."""
    return _build_get_detail(action, verbose)


def _build_get_reference_sql(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_reference_sql: show reference SQL name."""
    return _build_get_detail(action, verbose)


def _build_search_reference_template(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_reference_template: show template count."""
    return _build_search_generic(action, verbose, "template", "templates")


def _build_get_reference_template(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_reference_template: show template name."""
    return _build_get_detail(action, verbose)


def _build_render_reference_template(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """render_reference_template: show rendered SQL preview."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                name = result.get("template_name", "")
                tc.compact_result = f"{name}" if name else "Rendered"
    return tc


def _build_execute_reference_template(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """execute_reference_template: show rendered SQL and query results."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                query_result = result.get("query_result")
                if isinstance(query_result, dict):
                    rows = query_result.get("original_rows")
                    if rows is not None:
                        tc.compact_result = f"{rows} rows"
                    else:
                        tc.compact_result = "Executed"
                else:
                    tc.compact_result = "Executed"
    return tc


def _build_search_semantic_objects(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_semantic_objects: show semantic object count."""
    return _build_search_generic(action, verbose, "semantic object", "semantic objects")


def _build_search_knowledge(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_knowledge: show knowledge entry count."""
    return _build_search_generic(action, verbose, "knowledge entry", "knowledge entries")


def _build_get_knowledge(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_knowledge: show the fetched knowledge entry name."""
    return _build_get_detail(action, verbose)


def _build_list_metrics_semantic(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_metrics (SemanticTools): show metric count."""
    return _build_search_generic(action, verbose, "metric", "metrics")


def _build_get_dimensions(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_dimensions: show dimension count."""
    return _build_simple_list(action, verbose, "dimensions")


def _build_query_metrics(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """query_metrics: show query result as CSV table."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, dict):
                    compressed_data = result.get("data") or result.get("compressed_data")
                    if isinstance(compressed_data, str) and compressed_data:
                        tc.output_lines = _format_csv_preview_markup(compressed_data)
                    else:
                        tc.output_lines = _format_result_only_markup(action.output)
                else:
                    tc.output_lines = _format_result_only_markup(action.output)
            else:
                tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
                row_count = metadata.get("row_count") if isinstance(metadata, dict) else None
                if row_count is not None:
                    tc.compact_result = f"{row_count} rows"
                else:
                    tc.compact_result = "Query completed"
            else:
                tc.compact_result = "Query completed"
    return tc


def _build_validate_semantic(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """validate_semantic: show validation result."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                if result.get("valid"):
                    tc.compact_result = "Valid"
                else:
                    issues = result.get("issues", [])
                    count = len(issues) if isinstance(issues, list) else 0
                    tc.compact_result = f"{count} validation errors"
    return tc


def _build_attribution_analyze(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """attribution_analyze: show analyzed dimensions count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                dims = result.get("selected_dimensions") or result.get("dimension_ranking", [])
                count = len(dims) if isinstance(dims, list) else 0
                tc.compact_result = f"{count} dimensions analyzed"
    return tc


def _build_read_file(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """read_file: show line count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, str):
                    content_lines = result.split("\n")
                    total = len(content_lines)
                    max_show = 20
                    lines: List[str] = [f"[bold]content[/bold]: [dim]({total} lines)[/dim]"]
                    for line in content_lines[:max_show]:
                        lines.append(f"  [dim]{_escape_markup(line)}[/dim]")
                    if total > max_show:
                        lines.append(f"  [dim]... ({total - max_show} more lines)[/dim]")
                    tc.output_lines = lines
                else:
                    tc.output_lines = _format_result_only_markup(action.output)
            else:
                tc.output_lines = _format_result_only_markup(action.output)
    else:
        content = _extract_file_content(action.output)
        if content is not None:
            tc.compact_result = f"{len(content.splitlines()) or 1} lines"
        else:
            # Content pulled from another channel (e.g. summary) — just mark
            # success so the user does not see an empty result line.
            tc.compact_result = "read"
    return tc


# Keys that typically hold the real file body inside a wrapper (FuncToolResult,
# parsed JSON envelope, etc.). Ordered most-to-least specific.
_FILE_BODY_KEYS = ("result", "content", "text", "body", "data", "output")

# When falling back to top-level probing of the action.output dict, we narrow
# the key set to avoid picking up JSON-encoded wrapper payloads that some SDKs
# expose under generic names like ``text`` / ``data`` / ``output``.
_FILE_BODY_STRICT_KEYS = ("result", "content", "body")


def _extract_file_content(output_data) -> Optional[str]:
    """Pull the raw file body out of a read_file action output.

    Looks in the following order so we never accidentally return a JSON
    envelope as the file body:

    1. ``output_data.raw_output`` when it is a dict — the FuncToolResult shape.
    2. ``output_data.raw_output`` when it is a string — parse JSON first, then
       fall back to treating the string itself as the body.
    3. ``output_data`` is a string — treat it as the body.
    4. ``output_data`` top-level strict keys (``result`` / ``content`` /
       ``body``) as a last resort. ``text`` / ``data`` / ``output`` are
       intentionally excluded at the top level because MCP-style tools often
       put a serialized wrapper under those names.
    """
    if output_data is None:
        return None
    if isinstance(output_data, str):
        return output_data
    if not isinstance(output_data, dict):
        return None

    raw = output_data.get("raw_output")
    if isinstance(raw, dict):
        for key in _FILE_BODY_KEYS:
            val = raw.get(key)
            if isinstance(val, str):
                return val
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in _FILE_BODY_KEYS:
                    val = parsed.get(key)
                    if isinstance(val, str):
                        return val
        except Exception:
            pass
        # Raw string that is not JSON — assume it is the file body itself.
        return raw

    # Last-resort top-level probing using strict keys only.
    for key in _FILE_BODY_STRICT_KEYS:
        val = output_data.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _build_glob(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """glob: show file count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                files = result.get("files", [])
                tc.compact_result = f"{len(files)} files found"
            elif isinstance(result, list):
                tc.compact_result = f"{len(result)} files found"
    return tc


def _build_grep(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """grep: show match count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, dict):
                    matches = result.get("matches", [])
                    lines: List[str] = []
                    for m in matches[:30]:
                        f = _escape_markup(str(m.get("file", "?")))
                        ln = m.get("line", "?")
                        content = _escape_markup(str(m.get("content", "")))
                        lines.append(f"  [bold]{f}:{ln}[/bold]: [dim]{content}[/dim]")
                    if len(matches) > 30:
                        lines.append(f"  [dim]... ({len(matches) - 30} more matches)[/dim]")
                    tc.output_lines = lines
                else:
                    tc.output_lines = _format_result_only_markup(action.output)
            else:
                tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                matches = result.get("matches", [])
                file_count = len({m.get("file") for m in matches if isinstance(m, dict) and m.get("file")})
                if file_count:
                    tc.compact_result = f"{len(matches)} matches in {file_count} files"
                else:
                    tc.compact_result = f"{len(matches)} matches"
    return tc


def _build_list_document_nav(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_document_nav: show platform and doc count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                platform = result.get("platform", "")
                total_docs = result.get("total_docs", 0)
                tc.compact_result = f"{platform} \u2014 {total_docs} docs"
    return tc


def _build_get_document(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_document: show title and chunk count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                title = result.get("title", "")
                chunk_count = result.get("chunk_count", 0)
                tc.compact_result = f"{title} ({chunk_count} chunks)"
    return tc


def _build_todo_read(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """todo_read: show todo list count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                total = result.get("total_lists", 0)
                tc.compact_result = f"{total} todo lists"
    return tc


def _build_todo_write(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """todo_write: show success."""
    return _build_simple_action(action, verbose, "Todo list updated")


def _build_todo_update(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """todo_update: show updated status."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                item = result.get("updated_item", {})
                status = item.get("status", "") if isinstance(item, dict) else ""
                tc.compact_result = f"{status}" if status else "Updated"
    return tc


def _build_check_exists(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """check_semantic_object_exists / check_semantic_model_exists: show exists/not found."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                tc.compact_result = "Exists" if result.get("exists") else "Not found"
    return tc


def _build_end_generation(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """end_semantic_model_generation: show file count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                files = result.get("semantic_model_files", [])
                count = len(files) if isinstance(files, list) else 0
                tc.compact_result = f"{count} semantic models generated"
    return tc


def _build_end_metric_generation(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """end_metric_generation: show success."""
    return _build_simple_action(action, verbose, "Metric generated")


def _build_generate_sql_summary_id(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """generate_sql_summary_id: show success."""
    return _build_simple_action(action, verbose, "ID generated")


def _build_parse_dates(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """parse_temporal_expressions: show parsed date range when single, count otherwise."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                dates = result.get("extracted_dates", [])
                if isinstance(dates, list) and len(dates) == 1 and isinstance(dates[0], dict):
                    first = dates[0]
                    start = first.get("start_date") or first.get("start") or first.get("date")
                    end = first.get("end_date") or first.get("end")
                    if start and end and start != end:
                        tc.compact_result = f"{start} \u2192 {end}"
                    elif start:
                        tc.compact_result = str(start)
                    else:
                        tc.compact_result = "1 date expression"
                else:
                    count = len(dates) if isinstance(dates, list) else 0
                    tc.compact_result = f"{count} date expressions"
    return tc


def _build_analyze_relationships(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """analyze_table_relationships: show relationship count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                rels = result.get("relationships", [])
                count = len(rels) if isinstance(rels, list) else 0
                tc.compact_result = f"{count} relationships found"
    return tc


def _build_get_multiple_ddl(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_multiple_tables_ddl: show DDL count, verbose highlights DDL."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, list):
                    lines: List[str] = []
                    for item in result:
                        if isinstance(item, dict):
                            table_name = item.get("table_name", "?")
                            if "error" in item:
                                lines.append(
                                    f"[bold red]{_escape_markup(table_name)}: "
                                    f"{_escape_markup(str(item['error']))}[/bold red]"
                                )
                            else:
                                defn = item.get("definition", "")
                                lines.append(f"[bold]{_escape_markup(table_name)}[/bold]:")
                                if isinstance(defn, str):
                                    for ddl_line in defn.split("\n")[:15]:
                                        lines.append(f"  [bright_cyan]{_escape_markup(ddl_line)}[/bright_cyan]")
                                    total = len(defn.split("\n"))
                                    if total > 15:
                                        lines.append(f"  [dim]... ({total - 15} more lines)[/dim]")
                    tc.output_lines = lines
                else:
                    tc.output_lines = _format_result_only_markup(action.output)
            else:
                tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, list):
                tc.compact_result = f"{len(result)} DDLs retrieved"
    return tc


def _build_analyze_columns(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """analyze_column_usage_patterns: show column count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, dict):
                patterns = result.get("column_patterns", {})
                count = len(patterns) if isinstance(patterns, dict) else 0
                tc.compact_result = f"{count} columns analyzed"
    return tc


def _build_execute_command(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """execute_command / skill_execute_command: show success."""
    return _build_simple_action(action, verbose, "Command executed")


def _build_load_skill(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """load_skill: show success."""
    return _build_simple_action(action, verbose, "Skill loaded")


def _build_ask_user(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """ask_user: show answer count or error in compact, formatted Q&A in verbose."""
    tc = make_base_content(action)
    if verbose:
        # Args: show each question with options
        args_lines: List[str] = []
        try:
            args = action.input.get("arguments", {}) if action.input else {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, dict):
                raw_qs = args.get("questions", [])
                if isinstance(raw_qs, str):
                    raw_qs = json.loads(raw_qs)
                if isinstance(raw_qs, list):
                    for i, q in enumerate(raw_qs):
                        if isinstance(q, dict):
                            q_text = q.get("question", "")
                            options = q.get("options")
                            args_lines.append(f"[bold]Q{i + 1}[/bold]: {_escape_markup(q_text)}")
                            if options and isinstance(options, list):
                                args_lines.append(
                                    f"  [dim]options: {', '.join(_escape_markup(str(o)) for o in options)}[/dim]"
                                )
        except Exception:
            args_lines = []
        tc.args_lines = args_lines if args_lines else extract_args_markup(action)

        # Output: show Q→A pairs or error
        output_lines: List[str] = []
        try:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, str):
                    result = json.loads(result)
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict):
                            q = item.get("question", "")
                            a = item.get("answer", "")
                            short_q = q[:40] + "..." if len(q) > 40 else q
                            output_lines.append(
                                f"{_escape_markup(short_q)} \u2192 [bold]{_escape_markup(str(a))}[/bold]"
                            )
        except Exception:
            output_lines = []
        if action.output:
            tc.output_lines = output_lines if output_lines else _format_result_only_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            if data.get("success") and data.get("result"):
                result = data["result"]
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except Exception:
                        pass
                if isinstance(result, list):
                    tc.compact_result = f"{len(result)} answer(s)"
                else:
                    tc.compact_result = "Answered"
            elif not data.get("success") and data.get("error"):
                error = str(data["error"])
                error = error[:50] + "..." if len(error) > 50 else error
                tc.compact_result = f"{error}"
    return tc


# ── Output item extraction helper ──────────────────────────────────


def _get_items_from_output(output_data) -> Optional[list]:
    """Extract a list of items from output_data (text JSON array or result key)."""
    data = parse_output_data(output_data)
    if data is None:
        return None
    if "text" in data and isinstance(data["text"], str):
        try:
            cleaned = data["text"].replace("'", '"').replace("None", "null")
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    if "result" in data and isinstance(data["result"], list):
        return data["result"]
    return None


# ── Builder class ──────────────────────────────────────────────────


class ToolCallContentBuilder:
    """Unified tool call content builder with registry for custom tool visualizations.

    Built-in tools are auto-registered on construction. External code can
    register additional tools via ``builder.register(name, fn)``.
    """

    def __init__(self, enable_truncation: bool = True):
        self.enable_truncation = enable_truncation
        self._registry: Dict[str, ToolCallContentFn] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in per-tool builders."""
        # Database tools
        self._registry["list_tables"] = _build_list_tables
        self._registry["table_overview"] = _build_list_tables
        self._registry["describe_table"] = _build_describe_table
        self._registry["read_query"] = _build_read_query
        self._registry["query"] = _build_read_query
        self._registry["search_table"] = _build_search_table
        self._registry["get_table_ddl"] = _build_get_table_ddl
        self._registry["list_databases"] = _build_list_databases
        self._registry["list_schemas"] = _build_list_schemas

        # Context search tools
        self._registry["search_metrics"] = _build_search_metrics
        self._registry["search_reference_sql"] = _build_search_reference_sql
        self._registry["search_external_knowledge"] = _build_search_external_knowledge
        self._registry["search_knowledge"] = _build_search_knowledge
        self._registry["get_knowledge"] = _build_get_knowledge
        self._registry["search_documents"] = _build_search_documents
        self._registry["search_document"] = _build_doc_search_result
        self._registry["list_subject_tree"] = _build_list_subject_tree
        self._registry["get_metrics"] = _build_get_metrics
        self._registry["get_reference_sql"] = _build_get_reference_sql
        self._registry["search_semantic_objects"] = _build_search_semantic_objects

        # Reference template tools
        self._registry["search_reference_template"] = _build_search_reference_template
        self._registry["get_reference_template"] = _build_get_reference_template
        self._registry["render_reference_template"] = _build_render_reference_template
        self._registry["execute_reference_template"] = _build_execute_reference_template

        # Semantic tools
        self._registry["list_metrics"] = _build_list_metrics_semantic
        self._registry["get_dimensions"] = _build_get_dimensions
        self._registry["query_metrics"] = _build_query_metrics
        self._registry["validate_semantic"] = _build_validate_semantic
        self._registry["attribution_analyze"] = _build_attribution_analyze

        # Filesystem tools
        self._registry["edit_file"] = _build_edit_file
        self._registry["write_file"] = _build_write_file
        self._registry["read_file"] = _build_read_file
        self._registry["glob"] = _build_glob
        self._registry["grep"] = _build_grep

        # Platform document tools
        self._registry["list_document_nav"] = _build_list_document_nav
        self._registry["get_document"] = _build_get_document
        self._registry["web_search_document"] = _build_doc_search_result

        # Plan tools
        self._registry["todo_read"] = _build_todo_read
        self._registry["todo_write"] = _build_todo_write
        self._registry["todo_update"] = _build_todo_update

        # Generation tools
        self._registry["check_semantic_object_exists"] = _build_check_exists
        self._registry["check_semantic_model_exists"] = _build_check_exists
        self._registry["end_semantic_model_generation"] = _build_end_generation
        self._registry["end_metric_generation"] = _build_end_metric_generation
        self._registry["generate_sql_summary_id"] = _build_generate_sql_summary_id

        # Date parsing tools
        self._registry["parse_temporal_expressions"] = _build_parse_dates

        # Semantic model generation tools
        self._registry["analyze_table_relationships"] = _build_analyze_relationships
        self._registry["get_multiple_tables_ddl"] = _build_get_multiple_ddl
        self._registry["analyze_column_usage_patterns"] = _build_analyze_columns

        # Skill tools
        self._registry["execute_command"] = _build_execute_command
        self._registry["skill_execute_command"] = _build_execute_command
        self._registry["load_skill"] = _build_load_skill

        # Interaction tools
        self._registry["ask_user"] = _build_ask_user

    def register(self, function_name: str, fn: ToolCallContentFn) -> None:
        """Register a custom content builder for a specific tool function."""
        self._registry[function_name] = fn

    def build(self, action: ActionHistory, verbose: bool) -> ToolCallContent:
        """Build tool call content. Checks registry first, falls back to default."""
        function_name = action.input.get("function_name", "") if action.input else ""

        if function_name in self._registry:
            tc = self._registry[function_name](action, verbose)
        else:
            tc = self._build_default(action, verbose)

        if not verbose:
            # Populate compact-layout fields if the per-tool builder skipped them.
            set_tool_specific_args_summary(tc, action)
            set_default_args_summary(tc, action)
            if not tc.compact_result:
                tc.compact_result = _strip_legacy_preview(tc.output_preview)
            set_error_as_result(tc, action)

        return tc

    def _build_default(self, action: ActionHistory, verbose: bool) -> ToolCallContent:
        """Default content building logic — no tool-specific knowledge."""
        tc = make_base_content(action)

        if verbose:
            tc.args_lines = extract_args_markup(action)
            if action.output:
                tc.output_lines = _format_result_only_markup(action.output)
        else:
            if action.status == ActionStatus.SUCCESS and action.output:
                tc.output_preview = format_generic_preview(action.output, self.enable_truncation)

        return tc

    def _format_output_preview(self, output_data, function_name: str = "") -> str:
        """Legacy delegation: build a temporary action-like preview.

        Used by ActionContentGenerator._get_tool_output_preview for backward compat.
        """
        return format_generic_preview(output_data, self.enable_truncation)
