# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import sys
import threading
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

logger = get_logger(__name__)

# Blinking dot animation frames for PROCESSING status
_BLINK_FRAMES = ["○", "●"]

# In compact mode, only show the last N subagent actions in the Live overlay
_SUBAGENT_ROLLING_WINDOW_SIZE = 2


def _truncate_middle(text: str, max_len: int = 120) -> str:
    """Truncate text in the middle if too long, keeping head and tail."""
    if len(text) <= max_len:
        return text
    keep = (max_len - 5) // 2  # 5 chars for " ... "
    return text[:keep] + " ... " + text[-keep:]


def _get_assistant_content(action: ActionHistory) -> str:
    """Extract display content from an ASSISTANT action, preferring output.raw_output."""
    if action.output and isinstance(action.output, dict):
        raw = action.output.get("raw_output", "")
        if raw:
            return raw
    return action.messages or ""


class BaseActionContentGenerator:
    def __init__(self) -> None:
        self.role_colors = {
            ActionRole.SYSTEM: "bright_magenta",
            ActionRole.ASSISTANT: "bright_blue",
            ActionRole.USER: "bright_green",
            ActionRole.TOOL: "bright_cyan",
            ActionRole.WORKFLOW: "bright_yellow",
            ActionRole.INTERACTION: "bright_yellow",
        }
        self.status_icons = {
            ActionStatus.PROCESSING: "⏳",
            ActionStatus.SUCCESS: "✅",
            ActionStatus.FAILED: "❌",
        }
        self.status_dots = {
            ActionStatus.SUCCESS: "🟢",  # Green for success
            ActionStatus.FAILED: "🔴",  # Red for failed
            ActionStatus.PROCESSING: "🟡",  # Yellow for warning/pending
        }

        self.role_dots = {
            ActionRole.TOOL: "🔧",  # Cyan for tools
            ActionRole.ASSISTANT: "💬",  # Grey for thinking/messages
            ActionRole.SYSTEM: "🟣",  # Purple for system
            ActionRole.USER: "🟢",  # Green for user
            ActionRole.WORKFLOW: "🟡",  # Yellow for workflow
            ActionRole.INTERACTION: "❓",  # Question mark for interaction requests
        }

    def _get_action_dot(self, action: ActionHistory) -> str:
        """Get the appropriate colored dot for an action based on role and status"""
        # For tools, use cyan dot
        if action.role == ActionRole.TOOL:
            return self.role_dots[ActionRole.TOOL]
        # For assistant messages, use grey dot
        elif action.role == ActionRole.ASSISTANT:
            return self.role_dots[ActionRole.ASSISTANT]
        # For others, use status-based dots
        else:
            return self.status_dots.get(action.status, "⚫")

    def get_status_icon(self, action: ActionHistory) -> str:
        """Get the appropriate colored dot for an action based on status"""
        return self.status_icons.get(action.status, "⚡")


class ActionContentGenerator(BaseActionContentGenerator):
    """Generates rich content for action history display - separated from display logic"""

    def __init__(self, enable_truncation: bool = True):
        super().__init__()
        self.enable_truncation = enable_truncation

    def format_streaming_action(self, action: ActionHistory) -> str:
        """Format a single action for streaming display"""
        dot = self._get_action_dot(action)
        # Base action text with dot
        # For tools, messages already contains full info (function name + args) from models layer
        text = f"{dot} {action.messages}"

        # Add status info for tools
        if action.role == ActionRole.TOOL:
            # Don't add arguments - messages already contains everything
            if action.status == ActionStatus.PROCESSING:
                pass
            else:
                # Show completion status with output preview
                status_text = "✓" if action.status == ActionStatus.SUCCESS else "✗"
                duration = ""
                if action.end_time and action.start_time:
                    duration_sec = (action.end_time - action.start_time).total_seconds()
                    duration = f" ({duration_sec:.1f}s)"

                # Add output preview for successful tool calls on next line
                output_preview = ""
                if action.status == ActionStatus.SUCCESS and action.output:
                    function_name = action.input.get("function_name", "") if action.input else ""
                    preview = self._get_tool_output_preview(action.output, function_name)
                    if preview:
                        output_preview = f"\n    {preview}"

                text += f" - {status_text}{output_preview}{duration}"

        return text

    def format_inline_completed(self, action: ActionHistory) -> List[str]:
        """Format a completed action for inline display. Returns list of lines."""
        if action.role == ActionRole.ASSISTANT:
            return [f"⏺ 💬 {_get_assistant_content(action)}"]
        elif action.role == ActionRole.TOOL:
            summary = self.format_streaming_action(action)
            status_dot = "[green]⏺[/green]" if action.status == ActionStatus.SUCCESS else "[red]⏺[/red]"
            # Replace the role dot prefix with status dot
            # summary starts with "🔧 ..."
            return [f"{status_dot} {summary}"]
        elif action.role == ActionRole.WORKFLOW:
            return [f"⏺ 🟡 {action.messages}"]
        elif action.role == ActionRole.SYSTEM:
            return [f"⏺ 🟣 {action.messages}"]
        # INTERACTION: skip (handled by chat_commands)
        return []

    def format_inline_expanded(self, action: ActionHistory) -> List[str]:
        """Format a completed action in expanded (verbose) mode. Returns list of lines.

        Verbose mode shows full content without truncation: complete arguments,
        full output data, and untruncated thinking/messages.
        """
        lines = []
        if action.role == ActionRole.TOOL:
            function_name = action.input.get("function_name", "") if action.input else ""
            status_text = "✓" if action.status == ActionStatus.SUCCESS else "✗"
            duration = ""
            if action.end_time and action.start_time:
                duration_sec = (action.end_time - action.start_time).total_seconds()
                duration = f" ({duration_sec:.1f}s)"
            lines.append(f"⏺ 🔧 {function_name or action.messages} - {status_text}{duration}")
            # Show full arguments (no truncation), with tool-specific formatting
            if action.input and action.input.get("arguments"):
                args = action.input["arguments"]
                # Parse JSON string args into dict for display
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                arg_lines = self._format_tool_args_verbose(function_name, args)
                if arg_lines:
                    lines.extend(arg_lines)
                elif isinstance(args, dict):
                    for k, v in args.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    args: {args}")
            # Blank line between args and output
            if action.input and action.input.get("arguments") and action.output:
                lines.append("")
            # Show full output with tool-specific formatting
            if action.output:
                output_lines = self._format_tool_output_verbose_by_type(function_name, action.output, indent="    ")
                lines.extend(output_lines)
            # Blank line after action for visual separation
            lines.append("")
        elif action.role == ActionRole.ASSISTANT:
            lines.append(f"⏺ 💬 {_get_assistant_content(action)}")
        elif action.role == ActionRole.WORKFLOW:
            lines.append(f"⏺ 🟡 {action.messages}")
        elif action.role == ActionRole.SYSTEM:
            lines.append(f"⏺ 🟣 {action.messages}")
        return lines

    def _format_tool_output_verbose(self, output_data, indent: str = "    ") -> List[str]:
        """Format tool output fully for verbose mode (no truncation)."""
        import json

        lines = []
        if not output_data:
            return lines

        # Normalize to dict
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except Exception:
                lines.append(f"{indent}output: {output_data}")
                return lines

        if not isinstance(output_data, dict):
            lines.append(f"{indent}output: {output_data}")
            return lines

        # Use raw_output if available
        data = output_data.get("raw_output", output_data)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                lines.append(f"{indent}output: {data}")
                return lines

        if isinstance(data, dict):
            # Auto-unwrap FuncToolResult structure: skip success/error, show result only
            if "success" in data and "result" in data:
                data = {"result": data["result"]}
            for k, v in data.items():
                if v is None:
                    continue
                v_str = str(v)
                # Multi-line values: indent continuation lines
                if "\n" in v_str:
                    lines.append(f"{indent}{k}:")
                    for sub_line in v_str.split("\n"):
                        lines.append(f"{indent}  {sub_line}")
                else:
                    lines.append(f"{indent}{k}: {v_str}")
        else:
            lines.append(f"{indent}output: {data}")

        return lines

    # -- tool-specific verbose formatters ------------------------------------

    @staticmethod
    def _extract_result_data(output_data):
        """Extract the inner result from a FuncToolResult-wrapped output dict.

        Returns the ``result`` value when the output follows the standard
        ``{"success": ..., "result": ...}`` shape, or *None* when parsing fails.
        """
        if not output_data:
            return None
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except Exception:
                return None
        if not isinstance(output_data, dict):
            return None
        raw = output_data.get("raw_output", output_data)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return None
        if isinstance(raw, dict):
            if "result" in raw:
                return raw["result"]
            return raw
        return None

    def _render_rich_to_lines(self, renderable, indent: str) -> List[str]:
        """Render a Rich renderable to indented string lines."""
        buf = StringIO()
        temp_console = Console(file=buf, width=100, no_color=True, highlight=False)
        temp_console.print(renderable)
        output = buf.getvalue()
        return [f"{indent}{line}" for line in output.rstrip("\n").split("\n")]

    def _format_tool_output_verbose_by_type(self, function_name: str, output_data, indent: str = "    ") -> List[str]:
        """Format tool output for verbose mode using tool-specific formatters."""
        data = self._extract_result_data(output_data)
        if data is None:
            return self._format_tool_output_verbose(output_data, indent)

        formatter = self._TOOL_FORMATTERS.get(function_name)
        if formatter:
            try:
                result = formatter(self, data, indent)
                if result:
                    return result
            except Exception:
                pass
        return self._format_tool_output_verbose(output_data, indent)

    # -- individual formatters ------------------------------------------------

    def _fmt_read_query(self, data, indent: str) -> List[str]:
        """Format SQL query results, handling both compressed and raw formats."""
        if not isinstance(data, dict):
            return []

        # Handle compressed format from DataCompressor.compress()
        compressed_data = data.get("compressed_data")
        if compressed_data is not None:
            lines = []
            original_rows = data.get("original_rows", 0)
            original_columns = data.get("original_columns", [])
            col_count = len(original_columns) if isinstance(original_columns, list) else 0
            is_compressed = data.get("is_compressed", False)

            # Header with row/column counts
            status = "compressed" if is_compressed else "complete"
            lines.append(f"{indent}Query returned {original_rows} rows, {col_count} columns ({status})")

            # Show removed columns if any
            removed = data.get("removed_columns", [])
            if removed:
                lines.append(f"{indent}[dim]Omitted columns: {', '.join(str(c) for c in removed)}[/dim]")

            # Render compressed_data as a table if it looks like CSV
            if isinstance(compressed_data, str) and compressed_data.strip():
                csv_lines = compressed_data.strip().split("\n")
                if len(csv_lines) >= 2 and "," in csv_lines[0]:
                    # Parse CSV into a Rich Table
                    headers = csv_lines[0].split(",")
                    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
                    for h in headers:
                        table.add_column(h.strip())
                    for row_line in csv_lines[1:]:
                        cells = row_line.split(",")
                        table.add_row(*[c.strip() for c in cells])
                    lines.extend(self._render_rich_to_lines(table, indent))
                else:
                    # Plain text format - just indent each line
                    for text_line in csv_lines:
                        lines.append(f"{indent}{text_line}")
            return lines

        # Handle raw format: {"columns": [...], "data": [[...], ...]}
        columns = data.get("columns")
        rows = data.get("data")
        if not columns or not isinstance(columns, list):
            return []

        table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
        for col in columns:
            table.add_column(str(col))

        if rows and isinstance(rows, list):
            max_rows = 20
            for row in rows[:max_rows]:
                if isinstance(row, (list, tuple)):
                    table.add_row(*[str(v) for v in row])
            if len(rows) > max_rows:
                table.add_row(*([f"... +{len(rows) - max_rows} more"] + [""] * (len(columns) - 1)), style="dim")

        lines = self._render_rich_to_lines(table, indent)
        row_count = data.get("row_count") or (len(rows) if rows else 0)
        lines.append(f"{indent}[dim]{row_count} rows returned[/dim]")
        return lines

    def _fmt_list_tables(self, data, indent: str) -> List[str]:
        """Format table listings as a bullet list."""
        if not isinstance(data, list):
            return []
        lines = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("name", "?")
                ttype = item.get("type", "table")
                lines.append(f"{indent}• {name}  [dim]({ttype})[/dim]")
            else:
                lines.append(f"{indent}• {item}")
        return lines

    def _fmt_list_simple(self, data, indent: str) -> List[str]:
        """Format simple list results (databases, schemas) as bullet list."""
        if isinstance(data, list):
            if len(data) <= 10:
                return [f"{indent}• {item}" for item in data]
            lines = [f"{indent}• {item}" for item in data[:10]]
            lines.append(f"{indent}[dim]... and {len(data) - 10} more[/dim]")
            return lines
        return []

    def _fmt_describe_table(self, data, indent: str) -> List[str]:
        """Format table column metadata as a Rich table."""
        if not isinstance(data, dict):
            return []
        lines = []
        # Table description
        table_info = data.get("table")
        if isinstance(table_info, dict):
            desc = table_info.get("description", "")
            name = table_info.get("name", "")
            if name or desc:
                lines.append(f"{indent}[bold]{name}[/bold]{': ' + desc if desc else ''}")

        columns = data.get("columns")
        if not columns or not isinstance(columns, list):
            return lines

        table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
        table.add_column("Column")
        table.add_column("Type")
        table.add_column("Comment")
        for col in columns:
            if isinstance(col, dict):
                table.add_row(
                    col.get("name", ""),
                    col.get("type", ""),
                    col.get("comment", "") or "",
                )
        lines.extend(self._render_rich_to_lines(table, indent))
        return lines

    def _fmt_get_table_ddl(self, data, indent: str) -> List[str]:
        """Format DDL statement with SQL syntax highlighting."""
        if not isinstance(data, dict):
            return []
        lines = []
        identifier = data.get("identifier", "")
        if identifier:
            lines.append(f"{indent}[bold]table:[/bold] {identifier}")
        definition = data.get("definition", "")
        if definition:
            syntax = Syntax(definition, "sql", theme="monokai", line_numbers=False, word_wrap=True)
            lines.extend(self._render_rich_to_lines(syntax, indent))
        return lines

    def _fmt_search_table(self, data, indent: str) -> List[str]:
        """Format table search results as a numbered list."""
        if not isinstance(data, dict):
            return []
        lines = []
        metadata = data.get("metadata", [])
        if isinstance(metadata, list):
            for i, item in enumerate(metadata, 1):
                if isinstance(item, dict):
                    name = item.get("table_name") or item.get("identifier", "?")
                    ttype = item.get("table_type", "")
                    desc = item.get("description", "")
                    schema = item.get("schema_name", "")
                    prefix = f"{schema}." if schema else ""
                    line = f"{indent}{i}. [bold]{rich_escape(prefix)}{rich_escape(name)}[/bold]"
                    if ttype:
                        line += f" [dim]({rich_escape(ttype)})[/dim]"
                    lines.append(line)
                    if desc:
                        lines.append(f"{indent}   {rich_escape(desc)}")
        sample_data = data.get("sample_data", [])
        if isinstance(sample_data, list) and sample_data:
            lines.append(f"{indent}[dim]{len(sample_data)} tables with sample data[/dim]")
        return lines

    def _fmt_search_metrics(self, data, indent: str) -> List[str]:
        """Format metric search results."""
        if not isinstance(data, list):
            return []
        lines = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            name = item.get("name", "?")
            desc = item.get("description", "")
            lines.append(f"{indent}{i}. [bold]{rich_escape(name)}[/bold]")
            if desc:
                lines.append(f"{indent}   {rich_escape(desc)}")
            sql = item.get("sql_query", "")
            if sql:
                syntax = Syntax(sql.strip(), "sql", theme="monokai", line_numbers=False, word_wrap=True)
                lines.extend(self._render_rich_to_lines(syntax, indent + "   "))
            dims = item.get("dimensions")
            if isinstance(dims, list) and dims:
                lines.append(f"{indent}   [dim]dimensions: {rich_escape(', '.join(str(d) for d in dims))}[/dim]")
        return lines

    def _fmt_search_reference_sql(self, data, indent: str) -> List[str]:
        """Format reference SQL search results."""
        if not isinstance(data, list):
            return []
        lines = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            name = item.get("name", "?")
            summary = item.get("summary", "")
            lines.append(f"{indent}{i}. [bold]{rich_escape(name)}[/bold]")
            if summary:
                lines.append(f"{indent}   {rich_escape(summary)}")
            sql = item.get("sql", "")
            if sql:
                syntax = Syntax(sql.strip(), "sql", theme="monokai", line_numbers=False, word_wrap=True)
                lines.extend(self._render_rich_to_lines(syntax, indent + "   "))
            tags = item.get("tags")
            if isinstance(tags, list) and tags:
                lines.append(f"{indent}   [dim]tags: {rich_escape(', '.join(str(t) for t in tags))}[/dim]")
        return lines

    def _fmt_search_knowledge(self, data, indent: str) -> List[str]:
        """Format knowledge search results."""
        if not isinstance(data, list):
            return []
        lines = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            text = item.get("search_text", "")
            explanation = item.get("explanation", "")
            if text:
                lines.append(f"{indent}{i}. [bold]{rich_escape(text)}[/bold]")
            if explanation:
                lines.append(f"{indent}   {rich_escape(explanation)}")
        return lines

    def _fmt_search_documents(self, data, indent: str) -> List[str]:
        """Format document search results."""
        if not isinstance(data, list):
            return []
        lines = []
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                title = item.get("title", item.get("name", "?"))
                lines.append(f"{indent}{i}. {rich_escape(str(title))}")
            else:
                lines.append(f"{indent}{i}. {rich_escape(str(item))}")
        return lines

    def _fmt_search_semantic(self, data, indent: str) -> List[str]:
        """Format semantic object search results."""
        if not isinstance(data, list):
            return []
        lines = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            kind = item.get("kind", "")
            name = item.get("name", "?")
            desc = item.get("description", "")
            tag = f" [dim]({rich_escape(kind)})[/dim]" if kind else ""
            lines.append(f"{indent}{i}. [bold]{rich_escape(name)}[/bold]{tag}")
            if desc:
                lines.append(f"{indent}   {rich_escape(desc)}")
        return lines

    def _fmt_todo(self, data, indent: str) -> List[str]:
        """Format todo/plan results."""
        if isinstance(data, dict):
            msg = data.get("message", "")
            lines = []
            if msg:
                lines.append(f"{indent}{msg}")
            lists = data.get("lists", [])
            if isinstance(lists, list):
                for lst in lists:
                    items = lst.get("items", []) if isinstance(lst, dict) else []
                    for item in items:
                        if isinstance(item, dict):
                            status = item.get("status", "pending")
                            content = item.get("content", "")
                            icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(status, "•")
                            lines.append(f"{indent}  {icon} {content}")
            # Handle todo_write / todo_update that return updated_item or todo_list
            updated = data.get("updated_item")
            if isinstance(updated, dict):
                status = updated.get("status", "")
                content = updated.get("content", "")
                icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(status, "•")
                lines.append(f"{indent}  {icon} {content}")
            return lines
        return []

    def _fmt_date_parse(self, data, indent: str) -> List[str]:
        """Format temporal expression parsing results."""
        if not isinstance(data, dict):
            return []
        lines = []
        dates = data.get("extracted_dates", [])
        if isinstance(dates, list):
            for d in dates:
                if isinstance(d, dict):
                    text = d.get("text", "")
                    start = d.get("start_date", "")
                    end = d.get("end_date", "")
                    dtype = d.get("type", "")
                    line = f'{indent}📅 "{text}"'
                    if dtype:
                        line += f" [dim]({dtype})[/dim]"
                    lines.append(line)
                    if start or end:
                        lines.append(f"{indent}   {start} → {end}")
        context = data.get("date_context", "")
        if context:
            lines.append(f"{indent}[dim]{context}[/dim]")
        return lines

    def _fmt_current_date(self, data, indent: str) -> List[str]:
        """Format current date result."""
        if not isinstance(data, dict):
            return []
        date = data.get("current_date", "")
        is_ref = data.get("is_reference_date", False)
        label = "reference date" if is_ref else "current date"
        return [f"{indent}📅 {label}: {date}"]

    def _fmt_get_knowledge(self, data, indent: str) -> List[str]:
        """Format get_knowledge result: show knowledge entries with name and explanation."""
        if not isinstance(data, list):
            return []
        lines = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            explanation = item.get("explanation", "")
            subject_path = item.get("subject_path", [])
            path_str = " > ".join(subject_path) if isinstance(subject_path, list) and subject_path else ""
            header = f"{indent}{i}. [bold]{name}[/bold]"
            if path_str:
                header += f" [dim]({path_str})[/dim]"
            lines.append(header)
            if explanation:
                lines.append(f"{indent}   {explanation}")
        return lines

    def _fmt_list_subject_tree(self, data, indent: str) -> List[str]:
        """Format list_subject_tree result: render as a tree structure."""
        if not isinstance(data, dict):
            return []
        tree = Tree("[bold]Knowledge Tree[/bold]")
        self._build_tree(tree, data)
        return self._render_rich_to_lines(tree, indent)

    def _build_tree(self, parent, node: dict) -> None:
        """Recursively build a Rich Tree from a nested dict."""
        for key, value in node.items():
            if isinstance(value, dict):
                branch = parent.add(f"[bold]{key}[/bold]")
                self._build_tree(branch, value)
            elif isinstance(value, list):
                branch = parent.add(f"[bold]{key}[/bold]")
                for item in value:
                    branch.add(str(item))
            else:
                parent.add(f"{key}: {value}")

    def _format_tool_args_verbose(self, function_name: str, args) -> List[str]:
        """Format tool arguments with tool-specific formatting. Returns [] to use default."""
        # Parse JSON string args into dict if needed
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        formatter = self._TOOL_ARG_FORMATTERS.get(function_name)
        if formatter and isinstance(args, dict):
            try:
                result = formatter(self, args, "    ")
                if result:
                    return result
            except Exception:
                pass
        return []

    def _fmt_args_write_file(self, args: dict, indent: str) -> List[str]:
        """Format write_file args: show path and diff-style content (green + lines)."""
        lines = []
        path = args.get("path", "")
        if path:
            lines.append(f"{indent}path: {path}")
        content = args.get("content", "")
        if isinstance(content, str) and content:
            content_lines = content.split("\n")
            line_count = len(content_lines)
            if not content.endswith("\n"):
                pass
            else:
                line_count = max(line_count - 1, 1)
            lines.append(f"{indent}content: ({line_count} lines)")
            for cl in content_lines:
                lines.append(f"{indent}  [green]+ {cl}[/green]")
        file_type = args.get("file_type", "")
        if file_type:
            lines.append(f"{indent}file_type: {file_type}")
        return lines

    def _fmt_args_edit_file(self, args: dict, indent: str) -> List[str]:
        """Format edit_file args: show path and diff-style edits."""
        lines = []
        path = args.get("path", "")
        if path:
            lines.append(f"{indent}path: {path}")
        edits = args.get("edits", [])
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except Exception:
                lines.append(f"{indent}edits: {edits}")
                return lines
        if isinstance(edits, list):
            for i, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    continue
                old_text = edit.get("oldText", "")
                new_text = edit.get("newText", "")
                if len(edits) > 1:
                    lines.append(f"{indent}edit {i + 1}:")
                for ol in old_text.split("\n"):
                    lines.append(f"{indent}  [red]- {ol}[/red]")
                for nl in new_text.split("\n"):
                    lines.append(f"{indent}  [green]+ {nl}[/green]")
        return lines

    def _fmt_args_read_query(self, args: dict, indent: str) -> List[str]:
        """Format read_query args: syntax-highlight the SQL statement."""
        lines = []
        sql = args.get("sql", "")
        database = args.get("database", "")
        if database:
            lines.append(f"{indent}database: {database}")
        if sql:
            syntax = Syntax(sql.strip(), "sql", theme="monokai", line_numbers=False, word_wrap=True)
            lines.extend(self._render_rich_to_lines(syntax, indent))
        # Show any other args besides sql/database
        for k, v in args.items():
            if k not in ("sql", "database"):
                lines.append(f"{indent}{k}: {v}")
        return lines

    def _fmt_args_describe_table(self, args: dict, indent: str) -> List[str]:
        """Format describe_table/get_table_ddl args: show table name prominently, hide empty fields."""
        lines = []
        table_name = args.get("table_name", "")
        database = args.get("database", "")
        schema_name = args.get("schema_name", "")
        catalog = args.get("catalog", "")
        # Build qualified name: catalog.database.table
        parts = [p for p in [catalog, schema_name or database, table_name] if p]
        if parts:
            lines.append(f"{indent}[bold]{'.'.join(parts)}[/bold]")
        # Show any other non-empty, non-standard args
        skip_keys = {"table_name", "database", "schema_name", "catalog"}
        for k, v in args.items():
            if k not in skip_keys and v not in (None, "", 0, False):
                lines.append(f"{indent}{k}: {v}")
        return lines

    def _fmt_args_search_query(self, args: dict, indent: str) -> List[str]:
        """Format search_knowledge/search_table args: highlight the query text."""
        lines = []
        query = args.get("query_text", "")
        if query:
            lines.append(f"{indent}[italic]\"{query}\"[/italic]")
        # Show non-empty, non-default args
        skip_keys = {"query_text"}
        for k, v in args.items():
            if k not in skip_keys and v not in (None, "", 0, False):
                lines.append(f"{indent}{k}: {v}")
        return lines

    def _fmt_args_get_knowledge(self, args: dict, indent: str) -> List[str]:
        """Format get_knowledge args: show paths as readable tree paths."""
        lines = []
        paths = args.get("paths", [])
        if isinstance(paths, list):
            for path in paths:
                if isinstance(path, list):
                    lines.append(f"{indent}{' > '.join(path)}")
                else:
                    lines.append(f"{indent}{path}")
        for k, v in args.items():
            if k != "paths" and v not in (None, "", 0, False):
                lines.append(f"{indent}{k}: {v}")
        return lines

    def _fmt_args_list_tables(self, args: dict, indent: str) -> List[str]:
        """Format list_tables args: show database/schema, hide empty fields."""
        lines = []
        database = args.get("database", "")
        schema_name = args.get("schema_name", "")
        catalog = args.get("catalog", "")
        target = ".".join(p for p in [catalog, schema_name or database] if p) or "(default)"
        lines.append(f"{indent}scope: {target}")
        skip_keys = {"database", "schema_name", "catalog"}
        for k, v in args.items():
            if k not in skip_keys and v not in (None, "", 0, False):
                lines.append(f"{indent}{k}: {v}")
        return lines

    def _fmt_args_file_path(self, args: dict, indent: str) -> List[str]:
        """Format read_file/move_file args: show path(s) prominently."""
        lines = []
        for key in ("path", "source", "destination", "file_path"):
            val = args.get(key, "")
            if val:
                lines.append(f"{indent}{key}: {val}")
        skip_keys = {"path", "source", "destination", "file_path"}
        for k, v in args.items():
            if k not in skip_keys and v not in (None, "", 0, False):
                lines.append(f"{indent}{k}: {v}")
        return lines

    _TOOL_ARG_FORMATTERS = {
        "write_file": _fmt_args_write_file,
        "edit_file": _fmt_args_edit_file,
        "read_query": _fmt_args_read_query,
        "query": _fmt_args_read_query,
        "describe_table": _fmt_args_describe_table,
        "get_table_ddl": _fmt_args_describe_table,
        "search_knowledge": _fmt_args_search_query,
        "search_external_knowledge": _fmt_args_search_query,
        "search_table": _fmt_args_search_query,
        "search_metrics": _fmt_args_search_query,
        "search_reference_sql": _fmt_args_search_query,
        "search_documents": _fmt_args_search_query,
        "search_semantic_objects": _fmt_args_search_query,
        "get_knowledge": _fmt_args_get_knowledge,
        "list_tables": _fmt_args_list_tables,
        "list_databases": _fmt_args_list_tables,
        "list_schemas": _fmt_args_list_tables,
        "read_file": _fmt_args_file_path,
        "move_file": _fmt_args_file_path,
        "search_files": _fmt_args_file_path,
    }

    def _fmt_read_file(self, data, indent: str) -> List[str]:
        """Format read_file result: show line count only, not file content."""
        if isinstance(data, str):
            line_count = data.count("\n") + (1 if data and not data.endswith("\n") else 0)
            return [f"{indent}{line_count} lines read"]
        if isinstance(data, dict):
            # read_multiple_files returns a dict of {path: content}
            lines = []
            for path, content in data.items():
                if isinstance(content, str):
                    lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
                    lines.append(f"{indent}{path}: {lc} lines read")
                else:
                    lines.append(f"{indent}{path}: read")
            return lines
        return []

    def _fmt_search_files(self, data, indent: str) -> List[str]:
        """Format search_files result: show file list."""
        if isinstance(data, list):
            lines = [f"{indent}{len(data)} files found:"]
            for f in data:
                lines.append(f"{indent}  {rich_escape(str(f))}")
            return lines
        return []

    # Mapping from function_name to formatter method
    _TOOL_FORMATTERS = {
        "read_query": _fmt_read_query,
        "query": _fmt_read_query,
        "list_tables": _fmt_list_tables,
        "table_overview": _fmt_list_tables,
        "list_databases": _fmt_list_simple,
        "list_schemas": _fmt_list_simple,
        "describe_table": _fmt_describe_table,
        "get_table_ddl": _fmt_get_table_ddl,
        "search_table": _fmt_search_table,
        "search_metrics": _fmt_search_metrics,
        "search_reference_sql": _fmt_search_reference_sql,
        "search_knowledge": _fmt_search_knowledge,
        "search_external_knowledge": _fmt_search_knowledge,
        "search_documents": _fmt_search_documents,
        "search_semantic_objects": _fmt_search_semantic,
        "todo_read": _fmt_todo,
        "todo_write": _fmt_todo,
        "todo_update": _fmt_todo,
        "parse_temporal_expressions": _fmt_date_parse,
        "get_current_date": _fmt_current_date,
        "read_file": _fmt_read_file,
        "read_multiple_files": _fmt_read_file,
        "get_knowledge": _fmt_get_knowledge,
        "list_subject_tree": _fmt_list_subject_tree,
        "search_files": _fmt_search_files,
    }

    def format_inline_processing(self, action: ActionHistory, frame: str) -> str:
        """Format a PROCESSING tool for blinking display."""
        function_name = action.input.get("function_name", "") if action.input else ""
        return f"{frame} 🔧 {function_name or action.messages}..."

    def get_role_color(self, role: ActionRole) -> str:
        """Get the appropriate color for an action role"""
        return self.role_colors.get(role, "white")

    def generate_streaming_content(self, actions: List[ActionHistory]):
        """Generate content for streaming display with optional truncation and rich formatting"""

        if not actions:
            return "[dim]Waiting for actions...[/dim]"

        # If truncation is enabled, use simple text format
        if self.enable_truncation:
            return self._generate_simple_text_content(actions)
        else:
            # If truncation is disabled, use rich Panel + Table format
            return self._generate_rich_panel_content(actions)

    def _generate_simple_text_content(
        self,
        actions: List[ActionHistory],
    ) -> str:
        """Generate simple text content with truncation (current logic)"""
        content_lines = []

        for action in actions:
            # Skip TOOL actions that are still PROCESSING (not yet completed)
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                continue
            formatted_action = self.format_streaming_action(action)
            content_lines.append(formatted_action)

        return "\n".join(content_lines)

    def _generate_rich_panel_content(self, actions: List[ActionHistory]):
        """Generate rich Panel + Table content for non-truncated display"""
        from rich.console import Group

        content_elements = []

        for action in actions:
            if action.role == ActionRole.TOOL:
                tool_call_line = self.format_streaming_action(action)
                content_elements.append(tool_call_line)

                # 2. Add result table if there's meaningful output
                if action.output and action.status == ActionStatus.SUCCESS:
                    result_table = self._create_result_table(action)
                    if result_table:
                        content_elements.append(result_table)
            else:
                formatted_action = self.format_streaming_action(action)
                content_elements.append(Panel(formatted_action))

        # Return Group to combine all elements
        if content_elements:
            return Group(*content_elements)
        else:
            return "[dim]No actions to display[/dim]"

    def _create_result_table(self, action: ActionHistory):
        """Create a result table for tool output (simplified format)"""
        import json

        from rich.table import Table

        if not action.output:
            return None

        # Normalize output_data to dict format
        output_data = action.output
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except Exception:
                return None

        if not isinstance(output_data, dict):
            return None

        # Use raw_output if available, otherwise use the data directly
        data = output_data.get("raw_output", output_data)

        # If data is a string, parse it as JSON first
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return None

        # Extract result items for table display
        items = None
        if "text" in data and isinstance(data["text"], str):
            text_content = data["text"]
            # Try to parse as JSON array
            try:
                cleaned_text = text_content.replace("'", '"').replace("None", "null")
                items = json.loads(cleaned_text)
            except Exception:
                return None
        elif "result" in data and isinstance(data["result"], list):
            items = data["result"]

        # Create table only if we have list items
        if not items or not isinstance(items, list) or len(items) == 0:
            return None

        # Create table with dynamic columns based on first item
        first_item = items[0]
        if not isinstance(first_item, dict):
            return None

        table = Table(show_header=True, header_style="bold cyan", box=None)

        # Add columns based on first item keys
        for key in first_item.keys():
            table.add_column(str(key).title(), style="white")

        # Add rows (limit to first 10 rows to avoid overwhelming display)
        max_rows = min(len(items), 10)
        for item in items[:max_rows]:
            if isinstance(item, dict):
                row_values = []
                for key in first_item.keys():
                    value = item.get(key, "")
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    row_values.append(str(value))
                table.add_row(*row_values)

        # Add summary row if there are more items
        if len(items) > max_rows:
            summary_row = [f"... and {len(items) - max_rows} more rows"] + [""] * (len(first_item.keys()) - 1)
            table.add_row(*summary_row, style="dim")

        return table

    def format_data(self, data) -> str:
        """Format input/output data for display with truncation control"""
        if isinstance(data, dict):
            # Pretty print JSON-like data
            formatted = []
            for key, value in data.items():
                # Don't truncate SQL queries if truncation is disabled
                if key.lower() in ["sql_query", "sql", "query", "sql_return"] and isinstance(value, str):
                    formatted.append(f"  {key}: {value}")
                elif isinstance(value, str) and self.enable_truncation and len(value) > 50:
                    value = value[:50] + "..."
                    formatted.append(f"  {key}: {value}")
                else:
                    formatted.append(f"  {key}: {value}")
            return "\n".join(formatted)
        elif isinstance(data, str):
            if self.enable_truncation and len(data) > 100:
                return data[:100] + "..."
            return data
        else:
            return str(data)

    def get_data_summary(self, data) -> str:
        """Get a brief summary of data with truncation control"""
        if isinstance(data, dict):
            if "success" in data:
                status = "✅" if data["success"] else "❌"
                # Show SQL query with truncation control
                if "sql_query" in data and data["sql_query"]:
                    sql_preview = data["sql_query"]
                    if self.enable_truncation and len(sql_preview) > 200:
                        sql_preview = sql_preview[:200] + "..."
                    return f"{status} SQL: {sql_preview}"
                return f"{status} {len(data)} fields"
            else:
                return f"{len(data)} fields"
        elif isinstance(data, str):
            if self.enable_truncation and len(data) > 30:
                return data[:30] + "..."
            return data
        else:
            data_str = str(data)
            if self.enable_truncation and len(data_str) > 30:
                return data_str[:30]
            return data_str

    def _get_tool_args_preview(self, input_data: dict) -> str:
        """Get a brief preview of tool arguments with truncation control"""
        if "arguments" in input_data and input_data["arguments"]:
            args = input_data["arguments"]
            if isinstance(args, dict):
                # Show first key-value pair or query if present
                if "query" in args:
                    query = str(args["query"])
                    if self.enable_truncation and len(query) > 200:
                        return f"query='{query[:200]}...'"
                    return f"query='{query}'"
                elif args:
                    key, value = next(iter(args.items()))
                    value_str = str(value)
                    if self.enable_truncation and len(value_str) > 50:
                        return f"{key}='{value_str[:50]}...'"
                    return f"{key}='{value_str}'"
            else:
                args_str = str(args)
                if self.enable_truncation and len(args_str) > 50:
                    return f"'{args_str[:50]}...'"
                return f"'{args_str}'"
        return ""

    def _get_tool_output_preview(self, output_data: dict, function_name: str = "") -> str:
        """Get a brief preview of tool output results with truncation control"""
        import json

        if not output_data:
            return ""

        # Normalize output_data to dict format
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except Exception:
                return "✓ Completed (preview unavailable)"

        if not isinstance(output_data, dict):
            return "✓ Completed (preview unavailable)"

        # Use raw_output if available, otherwise use the data directly
        data = output_data.get("raw_output", output_data)
        logger.debug(f"raw_output for extracting text: {data}")

        # If data is a string, parse it as JSON first
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return "✓ Completed (preview unavailable)"
        items = None
        if "success" in data and not data["success"]:
            if "error" in data:
                error = data["error"] if len(data["error"]) <= 50 else data["error"][:50] + "..."
                return f"✗ Failed:({error})"
            return "✗ Failed"

        # Parse data.text for counting items or showing text preview
        if "text" in data and isinstance(data["text"], str):
            text_content = data["text"]
            # First try to parse as JSON array for counting
            try:
                cleaned_text = text_content.replace("'", '"').replace("None", "null")
                items = json.loads(cleaned_text)
            except Exception:
                # If JSON parsing fails, treat as plain text and show preview
                if self.enable_truncation and len(text_content) > 50:
                    return f"{text_content[:50]}..."
                return text_content
        elif "result" in data:
            items = data["result"]
        if items and isinstance(items, list):
            count = len(items)
            # Return appropriate label based on function name
            if function_name in ["list_tables", "table_overview"]:
                return f"✓ {count} tables"
            elif function_name in ["describe_table"]:
                return f"✓ {count} columns"
            else:
                return f"✓ {count} items"
        if function_name in ["read_query", "query"] and "original_rows" in items:
            return f"✓ {items['original_rows']} rows"
        if function_name == "search_table":
            metadata_count = len(items.get("metadata") or [])
            sample_count = len(items.get("sample_data") or [])
            return f"✓ {metadata_count} tables and {sample_count} sample rows"
        if function_name == "search_metrics":
            return f"✓ {len(items) if items else 0} metrics"
        if function_name == "search_reference_sql":
            return f"✓ {len(items) if items else 0} reference SQLs"
        if function_name == "search_external_knowledge":
            return f"✓ {len(items) if items else 0} extensions of knowledge"
        if function_name == "search_documents":
            return f"✓ {len(items) if items else 0} documents"
        # Generic fallback
        if "success" in output_data:
            return "✓ Success" if output_data["success"] else "✗ Failed"

        return "✓ Completed"


class ActionHistoryDisplay:
    """Display ActionHistory in a flat inline format (Claude Code style)"""

    def __init__(self, console: Optional[Console] = None, enable_truncation: bool = True):
        self.console = console or Console()
        self.enable_truncation = enable_truncation

        # Create content generator with truncation setting
        self.content_generator = ActionContentGenerator(enable_truncation=enable_truncation)

        # Reference to current streaming context for live control
        self._current_context: Optional["InlineStreamingContext"] = None

    def format_action_summary(self, action: ActionHistory) -> str:
        """Format a single action as a summary line"""
        status_icon = self.content_generator.get_status_icon(action)
        role_color = self.content_generator.get_role_color(action.role)

        return f"[{role_color}]{status_icon} {action.messages}[/{role_color}]"

    def format_action_detail(self, action: ActionHistory) -> Panel:
        """Format a single action as a detailed panel"""
        status_icon = self.content_generator.get_status_icon(action)
        role_color = self.content_generator.get_role_color(action.role)

        # Create header
        header = Text()
        header.append(f"{status_icon} ", style="bold")
        header.append(action.messages, style=f"bold {role_color}")
        header.append(f" ({action.action_type})", style="dim")

        # Create content
        content = []

        # Add messages
        if action.messages:
            content.append(Text(f"💬 {action.messages}", style="italic"))

        # Add status and duration
        duration = ""
        if action.end_time and action.start_time:
            duration_seconds = (action.end_time - action.start_time).total_seconds()
            duration = f" ({duration_seconds:.2f}s)"
        content.append(Text(f"📊 Status: {action.status.upper()}{duration}", style="bold yellow"))

        # Add input if present
        if action.input:
            content.append(Text("📥 Input:", style="bold cyan"))
            input_text = self.content_generator.format_data(action.input)
            content.append(Text(input_text, style="cyan"))

        # Add output if present
        if action.output:
            content.append(Text("📤 Output:", style="bold green"))
            output_text = self.content_generator.format_data(action.output)
            content.append(Text(output_text, style="green"))

        # Add timing
        if action.start_time:
            content.append(Text(f"🕐 Started: {action.start_time.strftime('%H:%M:%S')}", style="dim"))
        if action.end_time:
            content.append(Text(f"🏁 Ended: {action.end_time.strftime('%H:%M:%S')}", style="dim"))

        # Combine all content
        panel_content = Text("\n").join(content)

        return Panel(
            panel_content,
            title=f"[{role_color}]{action.role.upper()}[/{role_color}]",
            border_style=role_color,
            padding=(1, 2),
        )

    # -- unified render helpers for action history --------------------------

    def _render_subagent_header(self, action: ActionHistory, verbose: bool) -> None:
        """Print sub-agent group header (type + prompt/description)."""
        subagent_type = action.action_type or "subagent"
        prompt = action.messages or ""
        if prompt.startswith("User: "):
            prompt = prompt[6:]
        description = ""
        if action.input and isinstance(action.input, dict):
            description = action.input.get("_task_description", "")

        goal = description or (("" if verbose else _truncate_middle(prompt, max_len=200)) if prompt else "")
        goal_esc = rich_escape(goal) if goal else ""
        header = (
            f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]({goal_esc})"
            if goal
            else f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]"
        )
        if verbose and prompt:
            header += f"\n  ⎿  [yellow]prompt:[/yellow] [dim]{rich_escape(prompt)}[/dim]"
        self.console.print(header)

    def _render_subagent_action(self, action: ActionHistory, verbose: bool) -> None:
        """Print a single sub-agent action line."""
        if action.role == ActionRole.USER:
            # Prompt already shown in header, skip (consistent with streaming path)
            return
        if action.role == ActionRole.TOOL:
            function_name = action.input.get("function_name", "") if action.input else ""
            if verbose:
                label = rich_escape(function_name or action.messages)
            else:
                label = rich_escape(_truncate_middle(action.messages or function_name, max_len=200))
            status_text = "✓" if action.status == ActionStatus.SUCCESS else "✗"
            duration = ""
            if action.end_time and action.start_time:
                dur = (action.end_time - action.start_time).total_seconds()
                duration = f" ({dur:.1f}s)"
            line = f"  ⎿  🔧 {label} - {status_text}{duration}"
            self.console.print(f"[dim]{line}[/dim]")
            if verbose:
                if action.input and action.input.get("arguments"):
                    args = action.input["arguments"]
                    # Try tool-specific formatter first
                    arg_lines = self.content_generator._format_tool_args_verbose(function_name, args)
                    if arg_lines:
                        for al in arg_lines:
                            self.console.print(f"[dim]  ⎿  {al}[/dim]")
                    else:
                        # Fallback: parse JSON string and show key-value pairs
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if isinstance(args, dict):
                            for k, v in args.items():
                                self.console.print(
                                    f"[dim]  ⎿      {rich_escape(str(k))}: {rich_escape(str(v))}[/dim]"
                                )
                        else:
                            self.console.print(f"[dim]  ⎿      args: {rich_escape(str(args))}[/dim]")
                # Blank line between args and output
                if action.input and action.input.get("arguments") and action.output:
                    self.console.print("")
                if action.output:
                    output_lines = self.content_generator._format_tool_output_verbose_by_type(
                        function_name, action.output, indent="  ⎿      "
                    )
                    for ol in output_lines:
                        self.console.print(f"[dim]{ol}[/dim]")
                # Blank line after action for visual separation
                self.console.print("")
            return
        if action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                self.console.print(f"[dim]  ⎿  💬 {rich_escape(content)}[/dim]")
            return
        # Other roles
        label = action.messages or action.action_type
        if not verbose:
            label = _truncate_middle(label, max_len=200)
        self.console.print(f"[dim]  ⎿  {rich_escape(label)}[/dim]")

    def _render_subagent_done(
        self, tool_count: int, start_time: Optional[datetime], next_action: ActionHistory
    ) -> None:
        """Print the Done summary line for a sub-agent group."""
        end_time = next_action.end_time or datetime.now()
        dur_str = ""
        if start_time:
            dur_sec = (end_time - start_time).total_seconds()
            dur_str = f" \u00b7 {dur_sec:.1f}s"
        summary = f"  \u23bf  Done ({tool_count} tool uses{dur_str})"
        self.console.print(f"[dim]{summary}[/dim]")

    def _render_subagent_collapsed(
        self,
        first_action: ActionHistory,
        tool_count: int,
        start_time: Optional[datetime],
        end_action: ActionHistory,
    ) -> None:
        """Render a completed subagent group as collapsed: header line + Done summary line."""
        subagent_type = first_action.action_type or "subagent"
        prompt = first_action.messages or ""
        if prompt.startswith("User: "):
            prompt = prompt[6:]
        description = ""
        if first_action.input and isinstance(first_action.input, dict):
            description = first_action.input.get("_task_description", "")
        goal = description or (_truncate_middle(prompt, max_len=200) if prompt else "")
        goal_esc = rich_escape(goal) if goal else ""
        header = (
            f"[bold bright_cyan]\u23f4 {subagent_type}[/bold bright_cyan]({goal_esc})"
            if goal
            else f"[bold bright_cyan]\u23f4 {subagent_type}[/bold bright_cyan]"
        )
        self.console.print(header)
        status_mark = "\u2713" if end_action.status != ActionStatus.FAILED else "\u2717"
        end_time = end_action.end_time or datetime.now()
        dur_str = ""
        if start_time:
            dur_sec = (end_time - start_time).total_seconds()
            dur_str = f" \u00b7 {dur_sec:.1f}s"
        summary = f"  \u23bf  Done {status_mark} ({tool_count} tool uses{dur_str})"
        self.console.print(f"[dim]{summary}[/dim]")

    @staticmethod
    def _extract_subagent_response(output: dict) -> str:
        """Extract the response string from a task tool action output.

        The action output has ``raw_output`` which is the FuncToolResult
        serialization — either a JSON string or a dict with structure
        ``{"success": 1, "result": {"response": "..."}}``.
        """
        import json as _json

        raw = output.get("raw_output", output)
        if isinstance(raw, str):
            try:
                raw = _json.loads(raw)
            except (ValueError, TypeError):
                return ""
        if isinstance(raw, dict):
            # FuncToolResult: {"success":1, "result": {"response": "..."}}
            result = raw.get("result")
            if isinstance(result, dict):
                return result.get("response", "")
            # Flat dict: {"response": "..."}
            return raw.get("response", "")
        return ""

    def _render_subagent_response(self, action: ActionHistory) -> None:
        """Show the subagent response value after the Done line (verbose only)."""
        output = action.output
        if not output or not isinstance(output, dict):
            return
        response = self._extract_subagent_response(output)
        if response:
            lines = response.splitlines()
            for i, line in enumerate(lines):
                if i == 0:
                    self.console.print(f"  ⎿  [yellow]response:[/yellow] [dim]{rich_escape(line)}[/dim]")
                else:
                    self.console.print(f"[dim]  ⎿  {rich_escape(line)}[/dim]")

    def _render_deferred_group(
        self,
        group: dict,
        end_action: ActionHistory,
        task_tool_action: ActionHistory,
        verbose: bool,
    ) -> None:
        """Render a completed subagent group together with its task tool response.

        This keeps each group's header, actions, Done line, and response together
        so that parallel subagents don't interleave their responses.
        """
        for buffered in group.get("actions", []):
            if buffered is group["first_action"]:
                self._render_subagent_header(buffered, verbose)
            self._render_subagent_action(buffered, verbose)
        self._render_subagent_done(group["tool_count"], group["start_time"], end_action)
        if verbose:
            self._render_subagent_response(task_tool_action)

    def _flush_deferred_groups(
        self,
        deferred_groups: List[Tuple[dict, ActionHistory]],
        verbose: bool,
    ) -> None:
        """Render any deferred groups that never got a task tool response."""
        while deferred_groups:
            grp, end_act = deferred_groups.pop(0)
            for buffered in grp.get("actions", []):
                if buffered is grp["first_action"]:
                    self._render_subagent_header(buffered, verbose)
                self._render_subagent_action(buffered, verbose)
            self._render_subagent_done(grp["tool_count"], grp["start_time"], end_act)

    def _render_task_tool_as_subagent(self, action: ActionHistory, verbose: bool) -> None:
        """Render a standalone 'task' tool action as a subagent summary.

        Used for resume sessions where depth>0 actions are not persisted
        but the task tool call/result is available.
        """
        input_data = action.input or {}
        subagent_type = input_data.get("type", "subagent")
        prompt = input_data.get("prompt", "")
        description = input_data.get("description", "")

        # Header
        goal = description or (("" if verbose else _truncate_middle(prompt, max_len=200)) if prompt else "")
        goal_esc = rich_escape(goal) if goal else ""
        header = (
            f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]({goal_esc})"
            if goal
            else f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]"
        )
        if verbose and prompt:
            header += f"\n  ⎿  [yellow]prompt:[/yellow] [dim]{rich_escape(prompt)}[/dim]"
        self.console.print(header)

        # Output summary
        output = action.output
        if output and isinstance(output, dict):
            status_text = "✓" if action.status == ActionStatus.SUCCESS else "✗"
            duration = ""
            if action.end_time and action.start_time:
                dur = (action.end_time - action.start_time).total_seconds()
                duration = f" ({dur:.1f}s)"
            if verbose:
                response = self._extract_subagent_response(output) if isinstance(output, dict) else ""
                self.console.print(f"[dim]  ⎿  result - {status_text}{duration}[/dim]")
                if response:
                    lines = response.splitlines()
                    for i, line in enumerate(lines):
                        if i == 0:
                            self.console.print(f"  ⎿  [yellow]response:[/yellow] [dim]{rich_escape(line)}[/dim]")
                        else:
                            self.console.print(f"[dim]  ⎿  {rich_escape(line)}[/dim]")
            else:
                # Compact: show one-line summary
                preview = self.content_generator._get_tool_output_preview(output, "task")
                line = f"  ⎿  result - {status_text}{duration}"
                if preview:
                    line += f"  {rich_escape(preview)}"
                self.console.print(f"[dim]{line}[/dim]")

    def _render_main_action(self, action: ActionHistory, verbose: bool) -> None:
        """Print a depth=0 completed action."""
        if action.role == ActionRole.TOOL:
            fn = action.input.get("function_name", "") if action.input else ""
            if fn == "task":
                self._render_task_tool_as_subagent(action, verbose)
                return
        if action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                self.console.print(Markdown(f"⏺ 💬 {content}"))
            return
        if action.role == ActionRole.USER:
            # Extract user message (strip "User: " prefix if present)
            msg = action.messages
            if msg.startswith("User: "):
                msg = msg[6:]
            self.console.print(f"[green bold]Datus> [/green bold]{msg}")
            return
        if verbose:
            lines = self.content_generator.format_inline_expanded(action)
        else:
            lines = self.content_generator.format_inline_completed(action)
        for line in lines:
            self.console.print(line)

    def render_action_history(
        self,
        actions: List[ActionHistory],
        verbose: bool = False,
        show_partial_done: bool = True,
        collapse_completed: bool = False,
    ) -> None:
        """Render a list of completed actions using the unified inline format.

        This is the single source of truth for rendering an action history list.
        Used by _reprint_history (Ctrl+O toggle), display_action_list (.resume),
        and display_inline_trace_details (post-run Ctrl+O).

        Args:
            actions: List of ActionHistory to render.
            verbose: If True, show full arguments/output (no truncation).
            show_partial_done: If True, print a Done summary even when the
                sub-agent group is still open (useful for completed histories).
                Set to False when reprinting mid-execution (the streaming
                context will handle the running group).
            collapse_completed: If True and not verbose, completed subagent
                groups are rendered as a single collapsed summary line instead
                of showing all individual actions.
        """
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        # Whether to actually collapse completed groups
        do_collapse = collapse_completed and not verbose

        # Multi-group state: key = parent_action_id (None for legacy groups)
        subagent_groups: Dict[Optional[str], dict] = {}
        pending_task_tool_skips = 0  # number of depth=0 TOOL(task) actions to skip
        # Completed groups waiting for their depth=0 TOOL(task) response
        # before being rendered (verbose only). Each entry is (group, end_action).
        deferred_groups: List[Tuple[dict, ActionHistory]] = []

        for action in actions:
            # Skip INTERACTION actions (handled elsewhere)
            if action.role == ActionRole.INTERACTION:
                continue
            # Skip PROCESSING TOOL entries (only render SUCCESS/FAILED)
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                continue
            # Skip node final actions (e.g. chat_response) — rendered separately
            if action.role == ActionRole.ASSISTANT and action.action_type and action.action_type.endswith("_response"):
                continue

            # -- subagent_complete action closes a group --
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                group = subagent_groups.pop(group_key, None)
                if group:
                    if do_collapse:
                        self._render_subagent_collapsed(
                            group["first_action"], group["tool_count"], group["start_time"], action
                        )
                    else:
                        # Defer rendering until we pair with the task tool response,
                        # so that each group's response stays under its own header.
                        deferred_groups.append((group, action))
                    pending_task_tool_skips += 1
                continue

            # -- sub-agent group handling --
            if action.depth > 0:
                group_key = action.parent_action_id
                if group_key not in subagent_groups:
                    subagent_groups[group_key] = {
                        "start_time": action.start_time,
                        "tool_count": 0,
                        "subagent_type": action.action_type or "subagent",
                        "first_action": action,
                        "actions": [],
                    }
                if action.role == ActionRole.TOOL:
                    subagent_groups[group_key]["tool_count"] += 1
                # Always buffer — render as a complete group on close
                subagent_groups[group_key]["actions"].append(action)
                continue

            # -- leaving legacy sub-agent group (no parent_action_id) via depth transition --
            if None in subagent_groups:
                group = subagent_groups.pop(None)
                if do_collapse:
                    self._render_subagent_collapsed(
                        group["first_action"], group["tool_count"], group["start_time"], action
                    )
                else:
                    for buffered in group.get("actions", []):
                        if buffered is group["first_action"]:
                            self._render_subagent_header(buffered, verbose)
                        self._render_subagent_action(buffered, verbose)
                    self._render_subagent_done(group["tool_count"], group["start_time"], action)
                pending_task_tool_skips += 1
                continue

            # Skip depth=0 TOOL(task) after subagent groups (Done line already covers them).
            # Use a counter so that multiple completed groups + interleaved non-task
            # actions don't cause subsequent TOOL(task) actions to leak through.
            if pending_task_tool_skips > 0:
                if action.role == ActionRole.TOOL:
                    fn = action.input.get("function_name", "") if action.input else ""
                    if fn == "task":
                        # Pair with the oldest deferred group and render together
                        if deferred_groups:
                            grp, end_act = deferred_groups.pop(0)
                            self._render_deferred_group(grp, end_act, action, verbose)
                        pending_task_tool_skips -= 1
                        continue
                    # Non-task tool: flush any remaining deferred groups first
                    self._flush_deferred_groups(deferred_groups, verbose)

            # -- normal main-agent action --
            self._render_main_action(action, verbose)

        # Flush deferred groups that never got a task tool result
        self._flush_deferred_groups(deferred_groups, verbose)

        # If any sub-agent groups never closed (still active).
        # When show_partial_done=False, skip rendering unclosed groups entirely —
        # the streaming context (Live display) handles them to avoid duplication.
        if subagent_groups and show_partial_done:
            for group in subagent_groups.values():
                if group.get("first_action"):
                    self._render_subagent_header(group["first_action"], verbose)
                for buffered in group.get("actions", []):
                    self._render_subagent_action(buffered, verbose)
                dur_str = ""
                if group["start_time"]:
                    dur_sec = (datetime.now() - group["start_time"]).total_seconds()
                    dur_str = f" \u00b7 {dur_sec:.1f}s"
                summary = f"  \u23bf  Done ({group['tool_count']} tool uses{dur_str})"
                self.console.print(f"[dim]{summary}[/dim]")

    def render_multi_turn_history(
        self,
        turns: List[Tuple[str, List[ActionHistory]]],
        verbose: bool = False,
        show_partial_done: bool = True,
        collapse_completed: bool = False,
    ) -> None:
        """Render all historical turns, each preceded by a user-message header."""
        for user_message, actions in turns:
            self.console.print(f"[green bold]Datus> [/green bold]{user_message}")
            self.console.print("[dim]" + "\u2500" * 40 + "[/dim]")
            # Filter out top-level (depth==0) USER actions to avoid duplicate user message display
            # (user message is already rendered as the turn header above).
            # Keep depth>0 USER actions — they are needed for subagent group creation/headers.
            non_user_actions = [a for a in actions if not (a.role == ActionRole.USER and a.depth == 0)]
            self.render_action_history(
                non_user_actions,
                verbose=verbose,
                show_partial_done=show_partial_done,
                collapse_completed=collapse_completed,
            )
            self.console.print()

    def display_action_list(self, actions: List[ActionHistory]) -> None:
        """Display a list of actions in a tree-like format (delegates to unified renderer)."""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return
        self.render_action_history(actions, verbose=False)

    def display_streaming_actions(
        self,
        actions: List[ActionHistory],
        history_turns: Optional[List[Tuple[str, List[ActionHistory]]]] = None,
        current_user_message: str = "",
    ) -> "InlineStreamingContext":
        """Create an inline streaming display context for actions (Claude Code style)"""
        return InlineStreamingContext(
            actions,
            self,
            history_turns=history_turns or [],
            current_user_message=current_user_message,
        )

    def stop_live(self) -> None:
        """Stop the live display temporarily for user interaction."""
        if self._current_context:
            try:
                self._current_context.stop_display()
            except Exception as e:
                logger.debug(f"Error stopping live display: {e}")

    def restart_live(self) -> None:
        """Restart the live display after user interaction."""
        if self._current_context:
            try:
                self._current_context.restart_display()
            except Exception as e:
                logger.debug(f"Error restarting live display: {e}")

    def display_final_action_history(self, actions: List[ActionHistory]) -> None:
        """Display the final action history with complete SQL queries and reasoning results"""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        tree = Tree("[bold]Action History[/bold]")

        for action in actions:
            status_icon = self.content_generator.get_status_icon(action)
            role_color = self.content_generator.get_role_color(action.role)

            # Create main node
            duration = ""
            if action.end_time and action.start_time:
                duration_seconds = (action.end_time - action.start_time).total_seconds()
                duration = f" [dim]({duration_seconds:.2f}s)[/dim]"

            main_text = f"[{role_color}]{status_icon} {action.messages}[/{role_color}]{duration}"
            action_node = tree.add(main_text)

            # Add details as child nodes using rich_util formatting
            if action.input:
                if isinstance(action.input, dict):
                    input_tree = dict_to_tree(action.input, console=self.console)
                    input_node = action_node.add("[cyan]📥 Input:[/cyan]")
                    for child in input_tree.children:
                        input_node.add(child.label)
                else:
                    action_node.add(f"[cyan]📥 Input:[/cyan] {str(action.input)}")

            if action.output:
                if isinstance(action.output, dict):
                    output_tree = dict_to_tree(action.output, console=self.console)
                    output_node = action_node.add("[green]📤 Output:[/green]")
                    for child in output_tree.children:
                        output_node.add(child.label)
                else:
                    action_node.add(f"[green]📤 Output:[/green] {str(action.output)}")

        self.console.print(tree)

    def _get_data_summary_with_full_sql(self, data) -> str:
        """Get a data summary with full SQL queries for final display"""
        if isinstance(data, dict):
            if "success" in data:
                status = "✅" if data["success"] else "❌"
                # Show full SQL query if present
                if "sql_query" in data and data["sql_query"]:
                    return f"{status} SQL: {data['sql_query']}"
                return f"{status} {len(data)} fields"
            else:
                return f"{len(data)} fields"
        elif isinstance(data, str):
            return data
        else:
            return str(data)


class InlineStreamingContext:
    """Context manager for flat inline streaming display (Claude Code style).

    Actions are printed permanently to the console as they arrive/complete.
    PROCESSING tools are shown with a blinking dot animation via Rich Live.
    Completed actions are printed permanently and never refreshed in-place.
    """

    def __init__(
        self,
        actions_list: List[ActionHistory],
        display_instance: ActionHistoryDisplay,
        history_turns: Optional[List[Tuple[str, List[ActionHistory]]]] = None,
        current_user_message: str = "",
    ):
        self.actions = actions_list
        self.display = display_instance
        self._history_turns: List[Tuple[str, List[ActionHistory]]] = history_turns or []
        self._current_user_message = current_user_message
        self._processed_index = 0
        self._tick = 0
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._print_lock = threading.Lock()
        self._live: Optional[Live] = None
        self._paused = False
        self._subagent_groups: Dict[Optional[str], Dict] = {}
        self._completed_group_ids: set = set()
        self._subagent_live: Optional[Live] = None
        self._verbose = False
        self._verbose_frozen = False  # True = in frozen verbose mode, no real-time processing
        self._verbose_toggle_event = threading.Event()

    @property
    def live(self) -> Optional[Live]:
        """Compatibility property: expose the current mini-Live (if any)."""
        return self._live

    def stop_display(self) -> None:
        """Stop display for INTERACTION pause."""
        self._paused = True
        self._stop_processing_live()

    def restart_display(self) -> None:
        """Resume display after INTERACTION pause."""
        self._paused = False

    def toggle_verbose(self) -> None:
        """Toggle verbose mode (called from Ctrl+O key callback)."""
        self._verbose_toggle_event.set()

    def recreate_live_display(self):
        """Compatibility shim: restart display after interaction."""
        self.restart_display()

    # -- context manager --------------------------------------------------

    def __enter__(self):
        self._processed_index = 0
        self._stop_event.clear()
        self._paused = False

        # Register with display instance
        self.display._current_context = self

        # Start background refresh thread
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        self._verbose_frozen = False  # Ensure unfreeze before flush
        self._stop_event.set()
        self._stop_processing_live()
        self._stop_subagent_live()

        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)

        # Flush remaining actions after thread has stopped (no more concurrent access)
        self._flush_remaining_actions()

        # Unregister
        if self.display._current_context is self:
            self.display._current_context = None

    # -- refresh loop (daemon thread) --------------------------------------

    def _refresh_loop(self) -> None:
        """Background thread: poll actions ~4x/sec and dispatch print/Live."""
        while not self._stop_event.is_set():
            # Check for verbose toggle request (Ctrl+O)
            if self._verbose_toggle_event.is_set():
                self._verbose_toggle_event.clear()
                self._verbose = not self._verbose

                if self._verbose:
                    # Entering verbose mode: freeze — show snapshot, stop all Live displays
                    self._verbose_frozen = True
                    self._stop_processing_live()
                    self._stop_subagent_live()
                    with self._print_lock:
                        self.display.console.clear()
                        sys.stdout.write("\033[3J")
                        sys.stdout.flush()
                        self.display.console.print("[bold bright_black]  ⎯ switched to verbose mode (frozen) ⎯[/]")
                    self._reprint_history_verbose_snapshot()
                    # Do NOT restart any Live displays — screen is frozen
                else:
                    # Returning to compact mode: unfreeze — resume real-time processing
                    self._verbose_frozen = False
                    self._stop_processing_live()
                    self._stop_subagent_live()
                    with self._print_lock:
                        self.display.console.clear()
                        sys.stdout.write("\033[3J")
                        sys.stdout.flush()
                        self.display.console.print("[bold bright_black]  ⎯ switched to compact mode ⎯[/]")
                    self._reprint_history()
                    # Restart Live for any remaining active subagent groups
                    if self._subagent_groups:
                        with self._print_lock:
                            self._update_subagent_groups_live()

            if not self._paused and not self._verbose_frozen:
                self._process_actions()
                self._tick += 1
            self._stop_event.wait(timeout=0.25)

    # -- reprint history on mode toggle ------------------------------------

    def _reprint_history(self) -> None:
        """Reprint all already-processed actions in the current verbose mode.

        Called when the user toggles Ctrl+O mid-execution so that the
        history retroactively reflects the new display style.
        """
        with self._print_lock:
            # Render previous turns first
            if self._history_turns:
                self.display.render_multi_turn_history(
                    self._history_turns,
                    verbose=self._verbose,
                    show_partial_done=True,
                    collapse_completed=not self._verbose,
                )
            # Render current turn header + already-processed actions
            if self._current_user_message:
                self.display.console.print(f"[green bold]Datus> [/green bold]{self._current_user_message}")
                self.display.console.print("[dim]" + "\u2500" * 40 + "[/dim]")
            # Filter out top-level (depth==0) USER actions to avoid duplicate user message display
            # (user message is already rendered as the turn header above).
            # Keep depth>0 USER actions — they are needed for subagent group creation/headers.
            current_actions = [
                a for a in self.actions[: self._processed_index] if not (a.role == ActionRole.USER and a.depth == 0)
            ]
            self.display.render_action_history(
                current_actions,
                verbose=self._verbose,
                collapse_completed=not self._verbose,
                show_partial_done=False,
            )

    def _reprint_history_verbose_snapshot(self) -> None:
        """Reprint all processed actions + active subagent groups as a static verbose snapshot.

        Called when user enters verbose/frozen mode (Ctrl+O). Renders everything
        with verbose=True, collapse_completed=False, and appends active subagent
        groups as static output (no Live display).
        """
        with self._print_lock:
            # Render previous turns
            if self._history_turns:
                self.display.render_multi_turn_history(
                    self._history_turns,
                    verbose=True,
                    show_partial_done=True,
                    collapse_completed=False,
                )
            # Render current turn header + already-processed actions
            if self._current_user_message:
                self.display.console.print(f"[green bold]Datus> [/green bold]{self._current_user_message}")
                self.display.console.print("[dim]" + "\u2500" * 40 + "[/dim]")
            current_actions = [
                a for a in self.actions[: self._processed_index] if not (a.role == ActionRole.USER and a.depth == 0)
            ]
            self.display.render_action_history(
                current_actions,
                verbose=True,
                collapse_completed=False,
                show_partial_done=False,
            )
            # Render active subagent groups as static verbose output
            for _group_key, group in self._subagent_groups.items():
                first_action = group.get("first_action")
                if first_action:
                    self.display._render_subagent_header(first_action, verbose=True)
                for buffered in group.get("actions", []):
                    self.display._render_subagent_action(buffered, verbose=True)
                # Show "in progress" indicator
                dur_str = ""
                if group["start_time"]:
                    dur_sec = (datetime.now() - group["start_time"]).total_seconds()
                    dur_str = f" \u00b7 {dur_sec:.1f}s"
                self.display.console.print(
                    f"[dim]  \u23bf  \u23f3 in progress ({group['tool_count']} tool uses{dur_str})...[/dim]"
                )

    # -- core processing ---------------------------------------------------

    def _process_actions(self) -> None:
        """Walk from _processed_index forward and handle each action."""
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            # Skip INTERACTION actions (handled by chat_commands)
            if action.role == ActionRole.INTERACTION:
                self._processed_index += 1
                continue

            # -- subagent_complete action closes a group --
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                # Advance index BEFORE ending the group so that _reprint_with_collapse
                # includes this subagent_complete action in its slice.
                self._processed_index += 1
                self._end_subagent_group_by_key(group_key, action)
                continue

            # -- Sub-agent action (depth > 0) --
            if action.depth > 0:
                # Skip PROCESSING tools first — ensures the subagent group header
                # is created from the same action as render_action_history uses.
                if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                    self._processed_index += 1
                    continue
                group_key = action.parent_action_id
                if group_key not in self._subagent_groups:
                    # New sub-agent group: stop current Live, print header
                    self._stop_processing_live()
                    self._start_subagent_group(action, group_key)
                # Update current action display
                self._update_subagent_display(action, group_key)
                self._processed_index += 1
                continue

            # -- Main agent action (depth == 0), but preceded by legacy sub-agent group --
            if None in self._subagent_groups:
                # Advance index BEFORE ending the group so that _reprint_with_collapse
                # includes this action in its slice.
                self._processed_index += 1
                self._end_subagent_group_by_key(None, action)
                continue

            # TOOL with PROCESSING -> show blinking Live
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                self._update_processing_live(action)
                # Don't advance index yet; wait for status change
                return

            # Completed action (non-PROCESSING) -> print permanently
            self._stop_processing_live()
            self._print_completed_action(action)
            self._processed_index += 1

    def _flush_remaining_actions(self) -> None:
        """Flush all remaining actions at exit time without waiting for status changes."""
        # First drain all remaining actions (keeping groups open so tool_count stays accurate)
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            if action.role == ActionRole.INTERACTION:
                self._processed_index += 1
                continue

            # Skip PROCESSING tool entries — their SUCCESS version follows in the list
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                self._processed_index += 1
                continue

            # subagent_complete closes the group
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                # Advance index BEFORE ending the group so that _reprint_with_collapse
                # includes this subagent_complete action in its slice.
                self._processed_index += 1
                self._end_subagent_group_by_key(group_key, action)
                continue

            # depth>0: render inside the sub-agent group
            if action.depth > 0:
                group_key = action.parent_action_id
                if group_key not in self._subagent_groups:
                    self._start_subagent_group(action, group_key)
                self._update_subagent_display(action, group_key)
                self._processed_index += 1
                continue

            self._stop_processing_live()
            self._print_completed_action(action)
            self._processed_index += 1

        # Now close any sub-agent groups that are still open (no subagent_complete arrived)
        if self._subagent_groups:
            self._stop_processing_live()
            self._stop_subagent_live()
            for group_key in list(self._subagent_groups.keys()):
                group = self._subagent_groups.pop(group_key)
                first_action = group.get("first_action")
                if first_action:
                    self.display._render_subagent_header(first_action, self._verbose)
                for buffered in group.get("actions", []):
                    self.display._render_subagent_action(buffered, self._verbose)
                duration_sec = (datetime.now() - group["start_time"]).total_seconds()
                summary = f"  \u23bf  Done ({group['tool_count']} tool uses \u00b7 {duration_sec:.1f}s)"
                self.display.console.print(f"[dim]{summary}[/dim]")

    # -- sub-agent group display --------------------------------------------

    @staticmethod
    def _truncate_middle(text: str, max_len: int = 120) -> str:
        """Truncate text in the middle if too long, keeping head and tail."""
        return _truncate_middle(text, max_len)

    def _start_subagent_group(self, first_action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Create sub-agent group and update the Live display."""
        subagent_type = first_action.action_type or "subagent"

        with self._print_lock:
            self._subagent_groups[group_key] = {
                "start_time": first_action.start_time,
                "tool_count": 0,
                "subagent_type": subagent_type,
                "first_action": first_action,
                "actions": [],
            }
            self._update_subagent_groups_live()

    def _update_subagent_display(self, action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Buffer sub-agent action and update the grouped Live display."""
        group = self._subagent_groups.get(group_key)
        if group is None:
            return
        if action.role == ActionRole.TOOL:
            group["tool_count"] += 1
        if action.role == ActionRole.USER:
            # Prompt already shown in header, skip
            return
        group["actions"].append(action)
        with self._print_lock:
            self._update_subagent_groups_live()

    def _end_subagent_group_by_key(self, group_key: Optional[str], end_action: ActionHistory) -> None:
        """End sub-agent group: stop Live, permanently print completed group, restart Live for remaining.

        In compact mode, triggers a full reprint with completed groups collapsed.
        In verbose mode, permanently prints the completed group (header + actions + Done).
        """
        self._stop_processing_live()
        self._stop_subagent_live()

        group = self._subagent_groups.pop(group_key, None)
        if group is None:
            return

        if group_key is not None:
            self._completed_group_ids.add(group_key)

        if not self._verbose:
            # Compact mode: clear screen and reprint with completed groups collapsed
            self._reprint_with_collapse()
        else:
            # Verbose mode: permanently print the completed group
            duration = ""
            end_time = end_action.end_time or datetime.now()
            if group["start_time"]:
                duration_sec = (end_time - group["start_time"]).total_seconds()
                duration = f" \u00b7 {duration_sec:.1f}s"

            tool_count = group["tool_count"]
            summary = f"  \u23bf  Done ({tool_count} tool uses{duration})"

            with self._print_lock:
                first_action = group.get("first_action")
                if first_action:
                    self.display._render_subagent_header(first_action, self._verbose)
                for buffered in group.get("actions", []):
                    self.display._render_subagent_action(buffered, self._verbose)
                self.display.console.print(f"[dim]{summary}[/dim]")

        # Restart Live for remaining active groups
        if self._subagent_groups:
            with self._print_lock:
                self._update_subagent_groups_live()

    def _reprint_with_collapse(self) -> None:
        """Clear screen and reprint all history with completed groups collapsed."""
        with self._print_lock:
            self.display.console.clear()
            sys.stdout.write("\033[3J")
            sys.stdout.flush()
            # Reprint historical turns
            if self._history_turns:
                self.display.render_multi_turn_history(
                    self._history_turns,
                    verbose=self._verbose,
                    show_partial_done=True,
                    collapse_completed=True,
                )
            # Reprint current turn header
            if self._current_user_message:
                self.display.console.print(f"[green bold]Datus> [/green bold]{self._current_user_message}")
                self.display.console.print("[dim]" + "\u2500" * 40 + "[/dim]")
            # Reprint current actions with completed groups collapsed
            current_actions = [
                a for a in self.actions[: self._processed_index] if not (a.role == ActionRole.USER and a.depth == 0)
            ]
            self.display.render_action_history(
                current_actions,
                verbose=self._verbose,
                collapse_completed=True,
                show_partial_done=False,
            )

    # -- subagent Live display ------------------------------------------------

    def _update_subagent_groups_live(self) -> None:
        """Start or update the Live display showing all active subagent groups."""
        renderable = self._build_subagent_groups_renderable()
        if self._subagent_live is None:
            self._subagent_live = Live(renderable, console=self.display.console, refresh_per_second=4, transient=True)
            self._subagent_live.start()
        else:
            self._subagent_live.update(renderable)

    def _stop_subagent_live(self) -> None:
        """Stop the subagent Live display if running."""
        with self._print_lock:
            if self._subagent_live is not None:
                try:
                    self._subagent_live.stop()
                except Exception as e:
                    logger.debug(f"Error stopping subagent live display: {e}")
                self._subagent_live = None

    def _build_subagent_groups_renderable(self) -> Group:
        """Build a Group renderable showing all active subagent groups with actions grouped.

        Each line is a separate Text.from_markup element so that markup in
        one line cannot break formatting in another (e.g. dynamic labels
        containing ``[`` / ``]`` characters).
        """
        items: List[Text] = []
        for _group_key, group in self._subagent_groups.items():
            first_action = group.get("first_action")
            if first_action:
                subagent_type = first_action.action_type or "subagent"
                prompt = first_action.messages or ""
                if prompt.startswith("User: "):
                    prompt = prompt[6:]
                description = ""
                if first_action.input and isinstance(first_action.input, dict):
                    description = first_action.input.get("_task_description", "")
                goal = description or (
                    ("" if self._verbose else _truncate_middle(prompt, max_len=200)) if prompt else ""
                )
                goal_esc = rich_escape(goal)
                header = (
                    f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]({goal_esc})"
                    if goal
                    else f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]"
                )
                items.append(Text.from_markup(header))
                if self._verbose and prompt:
                    prompt_esc = rich_escape(prompt)
                    items.append(Text.from_markup(f"  \u23bf  [yellow]prompt:[/yellow] [dim]{prompt_esc}[/dim]"))
            actions_list = group.get("actions", [])
            if self._verbose:
                display_actions = actions_list
            else:
                display_actions = actions_list[-_SUBAGENT_ROLLING_WINDOW_SIZE:]
                hidden = len(actions_list) - len(display_actions)
                if hidden > 0:
                    items.append(Text.from_markup(f"[dim]  \u23bf  ... {hidden} earlier action(s) ...[/dim]"))
            for action in display_actions:
                action_items = self._format_subagent_action_items(action)
                items.extend(action_items)
        return Group(*items) if items else Group(Text(""))

    def _format_subagent_action_items(self, action: ActionHistory) -> List[Text]:
        """Format a single subagent action as a list of Text renderables."""
        if action.role == ActionRole.TOOL:
            function_name = action.input.get("function_name", "") if action.input else ""
            label = rich_escape(action.messages or function_name)
            if not self._verbose:
                label = rich_escape(self._truncate_middle(action.messages or function_name, max_len=200))
            status_text = "\u2713" if action.status == ActionStatus.SUCCESS else "\u2717"
            duration = ""
            if action.end_time and action.start_time:
                dur = (action.end_time - action.start_time).total_seconds()
                duration = f" ({dur:.1f}s)"
            result = [Text.from_markup(f"[dim]  \u23bf  \U0001f527 {label} - {status_text}{duration}[/dim]")]
            if self._verbose:
                if action.input and action.input.get("arguments"):
                    args = action.input["arguments"]
                    if isinstance(args, dict):
                        for k, v in args.items():
                            result.append(
                                Text.from_markup(
                                    f"[dim]  \u23bf      {rich_escape(str(k))}: {rich_escape(str(v))}[/dim]"
                                )
                            )
                    else:
                        result.append(Text.from_markup(f"[dim]  \u23bf      args: {rich_escape(str(args))}[/dim]"))
                if action.output:
                    output_lines = self.display.content_generator._format_tool_output_verbose_by_type(
                        function_name, action.output, indent="  ⎿      "
                    )
                    result.extend(Text.from_markup(f"[dim]{rich_escape(ol)}[/dim]") for ol in output_lines)
            return result
        elif action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                return [Text.from_markup(f"[dim]  \u23bf  \U0001f4ac {rich_escape(content)}[/dim]")]
            return []
        else:
            label = action.messages or action.action_type
            if not self._verbose:
                label = self._truncate_middle(label, max_len=200)
            return [Text.from_markup(f"[dim]  \u23bf  {rich_escape(label)}[/dim]")]

    def _format_subagent_action_line(self, action: ActionHistory) -> str:
        """Format a single subagent action as a plain string (for tests)."""
        items = self._format_subagent_action_items(action)
        return "\n".join(item.plain for item in items)

    # -- completed action printing -------------------------------------------

    def _print_completed_action(self, action: ActionHistory) -> None:
        """Print a completed action permanently to the console."""
        # "task" tool calls — subagent group already displayed header/actions/Done,
        # but response is only available now. Show it in verbose mode.
        if action.role == ActionRole.TOOL:
            fn = action.input.get("function_name", "") if action.input else ""
            if fn == "task":
                if self._verbose and action.status == ActionStatus.SUCCESS and action.output:
                    with self._print_lock:
                        self.display._render_subagent_response(action)
                return
        if action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                with self._print_lock:
                    self.display.console.print(Markdown(f"⏺ 💬 {content}"))
            return
        if self._verbose:
            lines = self.display.content_generator.format_inline_expanded(action)
        else:
            lines = self.display.content_generator.format_inline_completed(action)
        if not lines:
            return
        with self._print_lock:
            for line in lines:
                self.display.console.print(line)

    # -- blinking Live for PROCESSING tools --------------------------------

    def _update_processing_live(self, action: ActionHistory) -> None:
        """Create or update the mini-Live for a PROCESSING tool."""
        frame = _BLINK_FRAMES[self._tick % len(_BLINK_FRAMES)]
        text = self.display.content_generator.format_inline_processing(action, frame)
        renderable = Text.from_markup(f"[white]{text}[/white]")

        with self._print_lock:
            if self._live is None:
                self._live = Live(renderable, console=self.display.console, refresh_per_second=4, transient=True)
                self._live.start()
            else:
                self._live.update(renderable)

    def _stop_processing_live(self) -> None:
        """Stop the current mini-Live if running."""
        with self._print_lock:
            if self._live is not None:
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None


def create_action_display(console: Optional[Console] = None, enable_truncation: bool = True) -> ActionHistoryDisplay:
    """Factory function to create ActionHistoryDisplay with truncation control"""
    return ActionHistoryDisplay(console, enable_truncation)
