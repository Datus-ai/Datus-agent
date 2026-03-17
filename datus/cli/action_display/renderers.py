# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified rendering primitives for action history display.

All methods return List[Text/Markdown] renderables without printing,
providing a single source of truth for both sync and streaming paths.
"""

import json as _json
import re
from datetime import datetime
from typing import List, Optional, Sequence, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.syntax import Syntax
from rich.text import Text

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ── Module-level utilities ──────────────────────────────────────────────────


_MARKUP_TAG_RE = re.compile(r"\[/?[^\]]*\]")


def _strip_markup(text: str) -> str:
    """Remove Rich markup tags from text, restoring escaped brackets."""
    stripped = _MARKUP_TAG_RE.sub("", text)
    return stripped.replace("\\[", "[")


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


# ── Icon/color mappings ─────────────────────────────────────────────────────


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
            ActionStatus.PROCESSING: "\u23f3",
            ActionStatus.SUCCESS: "\u2705",
            ActionStatus.FAILED: "\u274c",
        }
        self.status_dots = {
            ActionStatus.SUCCESS: "\U0001f7e2",  # Green for success
            ActionStatus.FAILED: "\U0001f534",  # Red for failed
            ActionStatus.PROCESSING: "\U0001f7e1",  # Yellow for warning/pending
        }

        self.role_dots = {
            ActionRole.TOOL: "\U0001f527",  # Cyan for tools
            ActionRole.ASSISTANT: "\U0001f4ac",  # Grey for thinking/messages
            ActionRole.SYSTEM: "\U0001f7e3",  # Purple for system
            ActionRole.USER: "\U0001f7e2",  # Green for user
            ActionRole.WORKFLOW: "\U0001f7e1",  # Yellow for workflow
            ActionRole.INTERACTION: "\u2753",  # Question mark for interaction requests
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
            return self.status_dots.get(action.status, "\u26ab")

    def get_status_icon(self, action: ActionHistory) -> str:
        """Get the appropriate colored dot for an action based on status"""
        return self.status_icons.get(action.status, "\u26a1")


# ── Content helpers (str output for Web UI / legacy) ────────────────────────


class ActionContentGenerator(BaseActionContentGenerator):
    """Generates rich content for action history display - separated from display logic"""

    def __init__(self, enable_truncation: bool = True):
        super().__init__()
        self.enable_truncation = enable_truncation
        # Lazy import to avoid circular dependency
        from datus.cli.action_display.tool_content import ToolCallContentBuilder

        self.tool_content_builder = ToolCallContentBuilder(enable_truncation=enable_truncation)

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
                tc = self.tool_content_builder.build(action, verbose=False)
                output_preview = ""
                if tc.output_preview:
                    output_preview = f"\n    {tc.output_preview}"
                text += f" - {tc.status_mark}{output_preview}{tc.duration_str}"

        return text

    def format_inline_completed(self, action: ActionHistory) -> List[str]:
        """Format a completed action for inline display. Returns list of lines."""
        if action.role == ActionRole.ASSISTANT:
            return [f"\u23fa \U0001f4ac {_get_assistant_content(action)}"]
        elif action.role == ActionRole.TOOL:
            summary = self.format_streaming_action(action)
            status_dot = "[green]\u23fa[/green]" if action.status == ActionStatus.SUCCESS else "[red]\u23fa[/red]"
            # Replace the role dot prefix with status dot
            # summary starts with "\U0001f527 ..."
            return [f"{status_dot} {summary}"]
        elif action.role == ActionRole.WORKFLOW:
            return [f"\u23fa \U0001f7e1 {action.messages}"]
        elif action.role == ActionRole.SYSTEM:
            return [f"\u23fa \U0001f7e3 {action.messages}"]
        elif action.role == ActionRole.INTERACTION:
            return []
        return []

    def format_inline_expanded(self, action: ActionHistory) -> List[str]:
        """Format a completed action in expanded (verbose) mode. Returns list of lines.

        Verbose mode shows full content without truncation: complete arguments,
        full output data, and untruncated thinking/messages.
        """
        lines = []
        if action.role == ActionRole.TOOL:
            tc = self.tool_content_builder.build(action, verbose=True)
            lines.append(f"\u23fa \U0001f527 {tc.label} - {tc.status_mark}{tc.duration_str}")
            for arg_line in tc.args_lines:
                lines.append(f"    {_strip_markup(arg_line)}")
            if tc.args_lines and tc.output_lines:
                lines.append("")
            for out_line in tc.output_lines:
                lines.append(f"    {_strip_markup(out_line)}")
            lines.append("")
        elif action.role == ActionRole.ASSISTANT:
            lines.append(f"\u23fa \U0001f4ac {_get_assistant_content(action)}")
        elif action.role == ActionRole.WORKFLOW:
            lines.append(f"\u23fa \U0001f7e1 {action.messages}")
        elif action.role == ActionRole.SYSTEM:
            lines.append(f"\u23fa \U0001f7e3 {action.messages}")
        return lines

    def format_inline_processing(self, action: ActionHistory, frame: str) -> str:
        """Format a PROCESSING tool for blinking display."""
        function_name = action.input.get("function_name", "") if action.input else ""
        return f"{frame} \U0001f527 {function_name or action.messages}..."

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
        from rich.panel import Panel

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
                status = "\u2705" if data["success"] else "\u274c"
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

    def _get_tool_output_preview(self, output_data, function_name: str = "") -> str:
        """Delegate to tool_content_builder for output preview."""
        return self.tool_content_builder._format_output_preview(output_data, function_name)


# ── Unified Renderer ────────────────────────────────────────────────────────


class ActionRenderer:
    """Unified rendering primitives for action history.

    All render_* methods return List[Union[Text, Markdown]] without side effects.
    Use ``print_renderables`` to output them to a console.
    """

    def __init__(self, content_generator: Optional[ActionContentGenerator] = None, enable_truncation: bool = True):
        if content_generator is not None:
            self.content_generator = content_generator
        else:
            self.content_generator = ActionContentGenerator(enable_truncation=enable_truncation)
        # Shorthand alias used internally
        self.cg = self.content_generator

    # -- subagent group primitives ------------------------------------------

    def render_subagent_header(self, action: ActionHistory, verbose: bool) -> List[Text]:
        """Render sub-agent group header (type + prompt/description)."""
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
        result: List[Text] = [Text.from_markup(header)]
        if verbose and prompt:
            result.append(Text.from_markup(f"      [yellow]prompt:[/yellow] [dim]{rich_escape(prompt)}[/dim]"))
        return result

    def render_subagent_action(self, action: ActionHistory, verbose: bool) -> List[Text]:
        """Render a single sub-agent action line."""
        if action.role == ActionRole.USER:
            return []
        if action.role == ActionRole.TOOL:
            tc = self.cg.tool_content_builder.build(action, verbose)
            label = rich_escape(tc.label)
            result: List[Text] = [
                Text.from_markup(f"[dim]  \u23bf  \U0001f527 {label} - {tc.status_mark}{tc.duration_str}[/dim]")
            ]
            if verbose:
                for line in tc.args_lines:
                    result.append(Text.from_markup(f"[dim]          {line}[/dim]"))
                if tc.args_lines and tc.output_lines:
                    result.append(Text(""))
                for line in tc.output_lines:
                    result.append(Text.from_markup(f"[dim]          {line}[/dim]"))
                result.append(Text(""))
            else:
                if tc.output_preview:
                    result.append(Text.from_markup(f"[dim]          {rich_escape(tc.output_preview)}[/dim]"))
            return result
        if action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                return [Text.from_markup(f"[dim]  \u23bf  \U0001f4ac {rich_escape(content)}[/dim]")]
            return []
        # Other roles
        label = action.messages or action.action_type
        if not verbose:
            label = _truncate_middle(label, max_len=200)
        return [Text.from_markup(f"[dim]  \u23bf  {rich_escape(label)}[/dim]")]

    def render_subagent_done(self, tool_count: int, start_time: Optional[datetime], end_action: ActionHistory) -> Text:
        """Render the Done summary line for a sub-agent group."""
        end_time = end_action.end_time or datetime.now()
        dur_str = ""
        if start_time:
            dur_sec = (end_time - start_time).total_seconds()
            dur_str = f" \u00b7 {dur_sec:.1f}s"
        summary = f"  \u23bf  Done ({tool_count} tool uses{dur_str})"
        return Text.from_markup(f"[dim]{summary}[/dim]")

    def render_subagent_collapsed(
        self,
        first_action: ActionHistory,
        tool_count: int,
        start_time: Optional[datetime],
        end_action: ActionHistory,
    ) -> List[Text]:
        """Render a completed subagent group as collapsed: header + Done summary."""
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
        status_mark = "\u2713" if end_action.status != ActionStatus.FAILED else "\u2717"
        end_time = end_action.end_time or datetime.now()
        dur_str = ""
        if start_time:
            dur_sec = (end_time - start_time).total_seconds()
            dur_str = f" \u00b7 {dur_sec:.1f}s"
        summary = f"  \u23bf  Done {status_mark} ({tool_count} tool uses{dur_str})"
        return [Text.from_markup(header), Text.from_markup(f"[dim]{summary}[/dim]")]

    def render_subagent_response(self, action: ActionHistory) -> List[Text]:
        """Render the subagent response value after the Done line (verbose only)."""
        output = action.output
        if not output or not isinstance(output, dict):
            return []
        response = self._extract_subagent_response(output)
        if not response:
            return []
        result: List[Text] = []
        lines = response.splitlines()
        for i, line in enumerate(lines):
            if i == 0:
                result.append(Text.from_markup(f"      [yellow]response:[/yellow] [dim]{rich_escape(line)}[/dim]"))
            else:
                result.append(Text.from_markup(f"[dim]      {rich_escape(line)}[/dim]"))
        return result

    # -- interaction primitives ---------------------------------------------

    def render_interaction_request(self, action: ActionHistory, verbose: bool) -> List[Union[Text, Markdown, Syntax]]:
        """Render INTERACTION PROCESSING -- request content.

        Choices are NOT rendered here since the content field typically already
        describes available options, and the actual selection UI (select_choice)
        is handled by the input_collector callback.
        """
        input_data = action.input or {}
        content = input_data.get("content", "")
        content_type = input_data.get("content_type", "text")

        result: List[Union[Text, Markdown, Syntax]] = []
        result.append(Text.from_markup("[bold bright_yellow]\u2753 Interaction Request[/bold bright_yellow]"))

        if content:
            if content_type == "yaml":
                result.append(Syntax(content, "yaml", theme="monokai", line_numbers=True))
            elif content_type == "sql":
                result.append(Syntax(content, "sql", theme="monokai", line_numbers=True))
            elif content_type == "markdown":
                result.append(Markdown(content))
            else:
                result.append(Text(content))

        return result

    def render_interaction_success(self, action: ActionHistory, verbose: bool) -> List[Union[Text, Markdown, Syntax]]:
        """Render INTERACTION SUCCESS -- user choice + result content."""
        output_data = action.output or {}
        content = output_data.get("content", "") or action.messages or ""
        content_type = output_data.get("content_type", "markdown")
        user_choice = output_data.get("user_choice", "")

        result: List[Union[Text, Markdown, Syntax]] = []

        if user_choice:
            result.append(Text.from_markup(f"\u2705 [dim]Selected: {rich_escape(str(user_choice))}[/dim]"))

        if content:
            if content_type == "yaml":
                result.append(Syntax(content, "yaml", theme="monokai", line_numbers=True))
            elif content_type == "sql":
                result.append(Syntax(content, "sql", theme="monokai", line_numbers=True))
            elif content_type == "markdown":
                result.append(Markdown(content))
            else:
                result.append(Text(content))

        if not result:
            result.append(Text.from_markup("\u2705 [dim]Interaction completed[/dim]"))

        return result

    # -- main agent primitives ----------------------------------------------

    def render_main_action(self, action: ActionHistory, verbose: bool) -> List[Union[Text, Markdown, Syntax]]:
        """Render a depth=0 completed action directly as Rich objects."""
        # Task tool -> render as subagent
        if action.role == ActionRole.TOOL:
            fn = action.input.get("function_name", "") if action.input else ""
            if fn == "task":
                return self.render_task_tool_as_subagent(action, verbose)

        # INTERACTION -> skip in history (only shown during live interaction)
        if action.role == ActionRole.INTERACTION:
            return []

        # ASSISTANT -> Markdown
        if action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                return [Markdown(f"\u23fa \U0001f4ac {content}")]
            return []

        # USER -> styled Text
        if action.role == ActionRole.USER:
            msg = action.messages
            if msg.startswith("User: "):
                msg = msg[6:]
            return [Text.from_markup(f"[green bold]Datus> [/green bold]{msg}")]

        # TOOL / WORKFLOW / SYSTEM -> generate Rich Text directly
        if action.role == ActionRole.TOOL:
            return self._render_main_tool(action, verbose)

        if action.role == ActionRole.WORKFLOW:
            return [Text.from_markup(f"\u23fa \U0001f7e1 {rich_escape(action.messages or '')}")]

        if action.role == ActionRole.SYSTEM:
            return [Text.from_markup(f"\u23fa \U0001f7e3 {rich_escape(action.messages or '')}")]

        # Fallback: delegate to content generator str methods (backward compat)
        if verbose:
            lines = self.cg.format_inline_expanded(action)
        else:
            lines = self.cg.format_inline_completed(action)
        return [Text(line) if isinstance(line, str) else line for line in lines]

    def _render_main_tool(self, action: ActionHistory, verbose: bool) -> List[Union[Text, Markdown, Syntax]]:
        """Render a main-agent TOOL action directly as Rich Text objects."""
        tc = self.cg.tool_content_builder.build(action, verbose)
        if verbose:
            result: List[Union[Text, Markdown, Syntax]] = [
                Text.from_markup(f"\u23fa \U0001f527 {rich_escape(tc.label)} - {tc.status_mark}{tc.duration_str}")
            ]
            for line in tc.args_lines:
                result.append(Text.from_markup(f"    {line}"))
            if tc.args_lines and tc.output_lines:
                result.append(Text(""))
            for line in tc.output_lines:
                result.append(Text.from_markup(f"        {line}"))
            result.append(Text(""))
            return result
        else:
            status_dot = "[green]\u23fa[/green]" if action.status == ActionStatus.SUCCESS else "[red]\u23fa[/red]"
            header = f"\U0001f527 {rich_escape(tc.label)} - {tc.status_mark}{tc.duration_str}"
            if tc.output_preview:
                header += f"\n    {rich_escape(tc.output_preview)}"
            return [Text.from_markup(f"{status_dot} {header}")]

    @staticmethod
    def _parse_task_tool_input(input_data: dict) -> tuple:
        """Extract (subagent_type, prompt, description) from a task tool action's input.

        The task tool input has two possible layouts:
        1. Direct: {"type": "gen_sql", "prompt": "...", "description": "..."}
        2. Wrapped: {"function_name": "task", "arguments": '{"type": "gen_sql", ...}'}
        """
        subagent_type = input_data.get("type", "")
        prompt = input_data.get("prompt", "")
        description = input_data.get("description", "")

        if not subagent_type:
            args = input_data.get("arguments", "")
            if isinstance(args, str) and args:
                try:
                    args = _json.loads(args)
                except (ValueError, TypeError):
                    args = {}
            if isinstance(args, dict):
                subagent_type = args.get("type", "")
                prompt = prompt or args.get("prompt", "")
                description = description or args.get("description", "")

        return subagent_type or "subagent", prompt, description

    def render_task_tool_as_subagent(self, action: ActionHistory, verbose: bool) -> List[Union[Text, Markdown, Syntax]]:
        """Render a standalone 'task' tool action as a subagent summary."""
        input_data = action.input or {}
        subagent_type, prompt, description = self._parse_task_tool_input(input_data)

        # Header
        goal = description or (("" if verbose else _truncate_middle(prompt, max_len=200)) if prompt else "")
        goal_esc = rich_escape(goal) if goal else ""
        header = (
            f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]({goal_esc})"
            if goal
            else f"[bold bright_cyan]\u23fa {subagent_type}[/bold bright_cyan]"
        )
        if verbose and prompt:
            header += f"\n      [yellow]prompt:[/yellow] [dim]{rich_escape(prompt)}[/dim]"
        result: List[Union[Text, Markdown]] = [Text.from_markup(header)]

        # Output summary
        output = action.output
        if output and isinstance(output, dict):
            status_text = "\u2713" if action.status == ActionStatus.SUCCESS else "\u2717"
            duration = ""
            if action.end_time and action.start_time:
                dur = (action.end_time - action.start_time).total_seconds()
                duration = f" ({dur:.1f}s)"
            if verbose:
                response = self._extract_subagent_response(output) if isinstance(output, dict) else ""
                result.append(Text.from_markup(f"[dim]  \u23bf  result - {status_text}{duration}[/dim]"))
                if response:
                    lines = response.splitlines()
                    for i, line in enumerate(lines):
                        if i == 0:
                            result.append(
                                Text.from_markup(f"      [yellow]response:[/yellow] [dim]{rich_escape(line)}[/dim]")
                            )
                        else:
                            result.append(Text.from_markup(f"[dim]      {rich_escape(line)}[/dim]"))
            else:
                preview = self.cg._get_tool_output_preview(output, "task")
                line = f"  \u23bf  result - {status_text}{duration}"
                if preview:
                    line += f"  {rich_escape(preview)}"
                result.append(Text.from_markup(f"[dim]{line}[/dim]"))
        return result

    # -- processing animation -----------------------------------------------

    def render_processing(self, action: ActionHistory, frame: str) -> Text:
        """Render blinking animation frame for a PROCESSING tool."""
        function_name = action.input.get("function_name", "") if action.input else ""
        text = f"{frame} \U0001f527 {function_name or action.messages}..."
        return Text.from_markup(f"[white]{text}[/white]")

    # -- utility renderables ------------------------------------------------

    def render_user_header(self, message: str) -> Text:
        """Render 'Datus> ...' user message header."""
        return Text.from_markup(f"[green bold]Datus> [/green bold]{message}")

    def render_separator(self) -> Text:
        """Render horizontal separator."""
        return Text.from_markup("[dim]" + "\u2500" * 40 + "[/dim]")

    # -- convenience --------------------------------------------------------

    @staticmethod
    def print_renderables(console: Console, renderables: Sequence[Union[Text, Markdown, Syntax]]) -> None:
        """Print a list of renderables to a console."""
        for r in renderables:
            console.print(r)

    @staticmethod
    def _extract_subagent_response(output: dict) -> str:
        """Extract the response string from a task tool action output."""
        raw = output.get("raw_output", output)
        if isinstance(raw, str):
            try:
                raw = _json.loads(raw)
            except (ValueError, TypeError):
                return ""
        if isinstance(raw, dict):
            result = raw.get("result")
            if isinstance(result, dict):
                return result.get("response", "")
            return raw.get("response", "")
        return ""
