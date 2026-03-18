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
    output_preview: str = ""  # compact mode output summary
    args_lines: List[str] = field(default_factory=list)  # verbose: Rich markup formatted
    output_lines: List[str] = field(default_factory=list)  # verbose: Rich markup formatted


# Type alias for custom content builder functions
ToolCallContentFn = Callable[[ActionHistory, bool], ToolCallContent]


# ── Shared helpers ─────────────────────────────────────────────────


def calc_duration(action: ActionHistory) -> str:
    """Calculate duration string from action timestamps."""
    if action.end_time and action.start_time:
        duration_sec = (action.end_time - action.start_time).total_seconds()
        return f" ({duration_sec:.1f}s)"
    return ""


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
    """list_tables / table_overview: show table count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        items = _get_items_from_output(action.output)
        if isinstance(items, list):
            tc.output_preview = f"\u2713 {len(items)} tables"
    return tc


def _build_describe_table(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """describe_table: show column count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_describe_table_output_markup(action.output)
    else:
        items = _get_items_from_output(action.output)
        if isinstance(items, list):
            tc.output_preview = f"\u2713 {len(items)} columns"
    return tc


def _build_read_query(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """read_query / query: show row count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_read_query_output_markup(action.output)
    else:
        data = parse_output_data(action.output)
        if data:
            # Check top-level and nested result for original_rows
            rows = data.get("original_rows")
            if rows is None:
                result = data.get("result")
                if isinstance(result, dict):
                    rows = result.get("original_rows")
            if rows is not None:
                tc.output_preview = f"\u2713 {rows} rows"
            else:
                items = _get_items_from_output(action.output)
                if isinstance(items, list):
                    tc.output_preview = f"\u2713 {len(items)} items"
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
            sample_count = len(data.get("sample_data") or [])
            tc.output_preview = f"\u2713 {metadata_count} tables and {sample_count} sample rows"
    return tc


def _build_search_metrics(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_metrics: show metric count."""
    return _build_search_generic(action, verbose, "metrics")


def _build_search_reference_sql(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_reference_sql: show reference SQL count."""
    return _build_search_generic(action, verbose, "reference SQLs")


def _build_search_external_knowledge(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_external_knowledge / search_knowledge: show knowledge count."""
    return _build_search_generic(action, verbose, "knowledge entries")


def _build_search_documents(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_documents: show document count."""
    return _build_search_generic(action, verbose, "documents")


def _build_search_generic(action: ActionHistory, verbose: bool, unit: str) -> ToolCallContent:
    """Shared builder for search_* tools that count items from parsed output."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            tc.output_lines = _format_result_only_markup(action.output)
    else:
        items = _get_items_from_output(action.output)
        count = len(items) if isinstance(items, list) else 0
        tc.output_preview = f"\u2713 {count} {unit}"
    return tc


def _build_get_table_ddl(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_table_ddl: show DDL definition."""
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
                tc.output_preview = f"\u2713 {identifier}" if identifier else "\u2713 DDL retrieved"
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
        if data:
            result = data.get("result")
            if isinstance(result, str) and "edit" in result.lower():
                tc.output_preview = f"\u2713 {result.split('(')[-1].rstrip(')')}" if "(" in result else "\u2713 Edited"
            elif data.get("success"):
                tc.output_preview = "\u2713 Edited"
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
        if data:
            result = data.get("result")
            if isinstance(result, str) and "success" in result.lower():
                tc.output_preview = "\u2713 File written"
            elif data.get("success"):
                tc.output_preview = "\u2713 File written"
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
        if data:
            result = data.get("result")
            if isinstance(result, list):
                tc.output_preview = f"\u2713 {len(result)} {unit}"
            else:
                items = _get_items_from_output(action.output)
                if isinstance(items, list):
                    tc.output_preview = f"\u2713 {len(items)} {unit}"
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
                tc.output_preview = f"\u2713 {name}" if name else "\u2713 Retrieved"
            elif isinstance(result, str):
                tc.output_preview = f"\u2713 {result[:50]}" if result else "\u2713 Retrieved"
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
                tc.output_preview = f"\u2717 {action.output if isinstance(action.output, str) else 'Failed'}"
            else:
                tc.output_preview = f"\u2713 {success_label}"
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
                tc.output_preview = f"\u2713 {doc_count} docs found"
            else:
                items = _get_items_from_output(action.output)
                count = len(items) if isinstance(items, list) else 0
                tc.output_preview = f"\u2713 {count} docs found"
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
                tc.output_preview = f"\u2713 {len(result)} domains"
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
            lines.append(f"{indent}{connector} [dim]{_escape_markup(key)}[/dim]: " f"[cyan]{count}[/cyan]")
        elif isinstance(value, dict):
            lines.append(f"{indent}{connector} [bold]{_escape_markup(key)}[/bold]")
            lines.extend(_format_subject_tree_markup(value, child_indent))
        else:
            lines.append(f"{indent}{connector} [bold]{_escape_markup(key)}[/bold]: " f"{_escape_markup(str(value))}")
    return lines


def _build_get_metrics(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_metrics: show metric name."""
    return _build_get_detail(action, verbose)


def _build_get_reference_sql(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_reference_sql: show reference SQL name."""
    return _build_get_detail(action, verbose)


def _build_search_semantic_objects(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_semantic_objects: show semantic object count."""
    return _build_search_generic(action, verbose, "semantic objects")


def _build_search_knowledge(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_knowledge: show knowledge entry count."""
    return _build_search_generic(action, verbose, "knowledge entries")


def _build_get_knowledge(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """get_knowledge: show knowledge entry count."""
    return _build_search_generic(action, verbose, "knowledge entries")


def _build_list_metrics_semantic(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_metrics (SemanticTools): show metric count."""
    return _build_search_generic(action, verbose, "metrics")


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
                    tc.output_preview = f"\u2713 {row_count} rows"
                else:
                    tc.output_preview = "\u2713 Query completed"
            else:
                tc.output_preview = "\u2713 Query completed"
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
                    tc.output_preview = "\u2713 Valid"
                else:
                    issues = result.get("issues", [])
                    count = len(issues) if isinstance(issues, list) else 0
                    tc.output_preview = f"\u2717 {count} validation errors"
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
                tc.output_preview = f"\u2713 {count} dimensions analyzed"
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
        data = parse_output_data(action.output)
        if data:
            result = data.get("result")
            if isinstance(result, str):
                line_count = len(result.split("\n"))
                tc.output_preview = f"\u2713 {line_count} lines"
    return tc


def _build_read_multiple_files(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """read_multiple_files: show file count."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, dict):
                    lines: List[str] = []
                    for path, content in result.items():
                        line_count = len(content.split("\n")) if isinstance(content, str) else "?"
                        lines.append(f"[bold]{_escape_markup(path)}[/bold]: [dim]{line_count} lines[/dim]")
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
                tc.output_preview = f"\u2713 {len(result)} files read"
    return tc


def _build_create_directory(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """create_directory: show success."""
    return _build_simple_action(action, verbose, "Directory created")


def _build_list_directory(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """list_directory: show item count."""
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
                            name = item.get("name", "?")
                            item_type = item.get("type", "")
                            suffix = "/" if item_type == "directory" else ""
                            lines.append(f"  {_escape_markup(name)}{suffix}")
                        else:
                            lines.append(f"  {_escape_markup(str(item))}")
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
                tc.output_preview = f"\u2713 {len(result)} items"
    return tc


def _build_directory_tree(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """directory_tree: show tree."""
    tc = make_base_content(action)
    if verbose:
        tc.args_lines = extract_args_markup(action)
        if action.output:
            data = parse_output_data(action.output)
            if data:
                result = data.get("result")
                if isinstance(result, str):
                    lines: List[str] = ["[bold]tree[/bold]:"]
                    for tree_line in result.split("\n")[:30]:
                        lines.append(f"  [dim]{_escape_markup(tree_line)}[/dim]")
                    total = len(result.split("\n"))
                    if total > 30:
                        lines.append(f"  [dim]... ({total - 30} more lines)[/dim]")
                    tc.output_lines = lines
                else:
                    tc.output_lines = _format_result_only_markup(action.output)
            else:
                tc.output_lines = _format_result_only_markup(action.output)
    else:
        tc.output_preview = "\u2713 Tree generated"
    return tc


def _build_move_file(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """move_file: show success."""
    return _build_simple_action(action, verbose, "Moved")


def _build_search_files(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """search_files: show file count."""
    return _build_simple_list(action, verbose, "files found")


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
                tc.output_preview = f"\u2713 {platform} \u2014 {total_docs} docs"
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
                tc.output_preview = f"\u2713 {title} ({chunk_count} chunks)"
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
                tc.output_preview = f"\u2713 {total} todo lists"
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
                tc.output_preview = f"\u2713 {status}" if status else "\u2713 Updated"
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
                tc.output_preview = "\u2713 Exists" if result.get("exists") else "\u2713 Not found"
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
                tc.output_preview = f"\u2713 {count} semantic models generated"
    return tc


def _build_end_metric_generation(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """end_metric_generation: show success."""
    return _build_simple_action(action, verbose, "Metric generated")


def _build_generate_sql_summary_id(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """generate_sql_summary_id: show success."""
    return _build_simple_action(action, verbose, "ID generated")


def _build_parse_dates(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """parse_temporal_expressions: show date expression count."""
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
                count = len(dates) if isinstance(dates, list) else 0
                tc.output_preview = f"\u2713 {count} date expressions"
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
                tc.output_preview = f"\u2713 {count} relationships found"
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
                tc.output_preview = f"\u2713 {len(result)} DDLs retrieved"
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
                tc.output_preview = f"\u2713 {count} columns analyzed"
    return tc


def _build_execute_command(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """execute_command / skill_execute_command: show success."""
    return _build_simple_action(action, verbose, "Command executed")


def _build_load_skill(action: ActionHistory, verbose: bool) -> ToolCallContent:
    """load_skill: show success."""
    return _build_simple_action(action, verbose, "Skill loaded")


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
        self._registry["read_multiple_files"] = _build_read_multiple_files
        self._registry["create_directory"] = _build_create_directory
        self._registry["list_directory"] = _build_list_directory
        self._registry["directory_tree"] = _build_directory_tree
        self._registry["move_file"] = _build_move_file
        self._registry["search_files"] = _build_search_files

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

    def register(self, function_name: str, fn: ToolCallContentFn) -> None:
        """Register a custom content builder for a specific tool function."""
        self._registry[function_name] = fn

    def build(self, action: ActionHistory, verbose: bool) -> ToolCallContent:
        """Build tool call content. Checks registry first, falls back to default."""
        function_name = action.input.get("function_name", "") if action.input else ""

        if function_name in self._registry:
            return self._registry[function_name](action, verbose)

        return self._build_default(action, verbose)

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
