# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import sys
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

logger = get_logger(__name__)

# Blinking dot animation frames for PROCESSING status
_BLINK_FRAMES = ["○", "●"]


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
            # Show full arguments (no truncation)
            if action.input and action.input.get("arguments"):
                args = action.input["arguments"]
                if isinstance(args, dict):
                    for k, v in args.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    args: {args}")
            # Show full output
            if action.output:
                output_lines = self._format_tool_output_verbose(action.output, indent="    ")
                lines.extend(output_lines)
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
            for k, v in data.items():
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
        header = f"\u23fa {subagent_type}({goal})" if goal else f"\u23fa {subagent_type}"
        if verbose and prompt:
            header += f"\n  ⎿  [yellow]prompt:[/yellow] [dim]{prompt}[/dim]"
        self.console.print(header)

    def _render_subagent_action(self, action: ActionHistory, verbose: bool) -> None:
        """Print a single sub-agent action line."""
        if action.role == ActionRole.USER:
            # Prompt already shown in header, skip (consistent with streaming path)
            return
        if action.role == ActionRole.TOOL:
            function_name = action.input.get("function_name", "") if action.input else ""
            label = action.messages or function_name
            if not verbose:
                label = _truncate_middle(label, max_len=200)
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
                    if isinstance(args, dict):
                        for k, v in args.items():
                            self.console.print(f"[dim]  ⎿      {k}: {v}[/dim]")
                    else:
                        self.console.print(f"[dim]  ⎿      args: {args}[/dim]")
                if action.output:
                    output_lines = self.content_generator._format_tool_output_verbose(action.output, indent="  ⎿      ")
                    for ol in output_lines:
                        self.console.print(f"[dim]{ol}[/dim]")
            return
        if action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                self.console.print(f"[dim]  ⎿  💬 {content}[/dim]")
            return
        # Other roles
        label = action.messages or action.action_type
        if not verbose:
            label = _truncate_middle(label, max_len=200)
        self.console.print(f"[dim]  ⎿  {label}[/dim]")

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
                    self.console.print(f"  ⎿  [yellow]response:[/yellow] [dim]{line}[/dim]")
                else:
                    self.console.print(f"[dim]  ⎿  {line}[/dim]")

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
        header = f"\u23fa {subagent_type}({goal})" if goal else f"\u23fa {subagent_type}"
        if verbose and prompt:
            header += f"\n  ⎿  [yellow]prompt:[/yellow] [dim]{prompt}[/dim]"
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
                            self.console.print(f"  ⎿  [yellow]response:[/yellow] [dim]{line}[/dim]")
                        else:
                            self.console.print(f"[dim]  ⎿  {line}[/dim]")
            else:
                # Compact: show one-line summary
                preview = self.content_generator._get_tool_output_preview(output, "task")
                line = f"  ⎿  result - {status_text}{duration}"
                if preview:
                    line += f"  {preview}"
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
        """
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        # Multi-group state: key = parent_action_id (None for legacy groups)
        subagent_groups: Dict[Optional[str], dict] = {}
        skip_task_tools = False  # skip depth=0 TOOL(task) after subagent group

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
                    self._render_subagent_done(group["tool_count"], group["start_time"], action)
                # Skip the following depth=0 TOOL(task) result(s)
                skip_task_tools = True
                continue

            # -- sub-agent group handling --
            if action.depth > 0:
                group_key = action.parent_action_id
                if group_key not in subagent_groups:
                    subagent_groups[group_key] = {
                        "start_time": action.start_time,
                        "tool_count": 0,
                        "subagent_type": action.action_type or "subagent",
                    }
                    self._render_subagent_header(action, verbose)
                if action.role == ActionRole.TOOL:
                    subagent_groups[group_key]["tool_count"] += 1
                self._render_subagent_action(action, verbose)
                continue

            # -- leaving legacy sub-agent group (no parent_action_id) via depth transition --
            if None in subagent_groups:
                group = subagent_groups.pop(None)
                self._render_subagent_done(group["tool_count"], group["start_time"], action)
                skip_task_tools = True
                continue

            # Skip depth=0 TOOL(task) after a subagent group (Done line already covers it).
            # Only reset the flag when a non-task TOOL action appears; ASSISTANT/USER
            # actions between subagent_complete and TOOL(task) should not reset it.
            if skip_task_tools:
                if action.role == ActionRole.TOOL:
                    fn = action.input.get("function_name", "") if action.input else ""
                    if fn == "task":
                        if verbose:
                            self._render_subagent_response(action)
                        continue
                    skip_task_tools = False

            # -- normal main-agent action --
            self._render_main_action(action, verbose)

        # If any sub-agent groups never closed
        if subagent_groups and show_partial_done:
            for group in subagent_groups.values():
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
    ) -> None:
        """Render all historical turns, each preceded by a user-message header."""
        for user_message, actions in turns:
            self.console.print(f"[green bold]Datus> [/green bold]{user_message}")
            self.console.print("[dim]" + "\u2500" * 40 + "[/dim]")
            # Filter out top-level (depth==0) USER actions to avoid duplicate user message display
            # (user message is already rendered as the turn header above).
            # Keep depth>0 USER actions — they are needed for subagent group creation/headers.
            non_user_actions = [a for a in actions if not (a.role == ActionRole.USER and a.depth == 0)]
            self.render_action_history(non_user_actions, verbose=verbose, show_partial_done=show_partial_done)
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
        self._verbose = False
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
        self._stop_event.set()
        self._stop_processing_live()

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
                mode_label = "verbose" if self._verbose else "compact"
                self._stop_processing_live()
                with self._print_lock:
                    self.display.console.clear()
                    sys.stdout.write("\033[3J")
                    sys.stdout.flush()
                    self.display.console.print(f"[bold bright_black]  ⎯ switched to {mode_label} mode ⎯[/]")
                self._reprint_history()
            if not self._paused:
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
                show_partial_done=False,
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
                self._end_subagent_group_by_key(group_key, action)
                self._processed_index += 1
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
                self._end_subagent_group_by_key(None, action)
                # Skip the task SUCCESS normal rendering (Done line already covers it)
                self._processed_index += 1
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
                self._end_subagent_group_by_key(group_key, action)
                self._processed_index += 1
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
            for group_key in list(self._subagent_groups.keys()):
                group = self._subagent_groups.pop(group_key)
                duration_sec = (datetime.now() - group["start_time"]).total_seconds()
                summary = f"  \u23bf  Done ({group['tool_count']} tool uses \u00b7 {duration_sec:.1f}s)"
                self.display.console.print(f"[dim]{summary}[/dim]")

    # -- sub-agent group display --------------------------------------------

    @staticmethod
    def _truncate_middle(text: str, max_len: int = 120) -> str:
        """Truncate text in the middle if too long, keeping head and tail."""
        return _truncate_middle(text, max_len)

    def _start_subagent_group(self, first_action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Print sub-agent group header with prompt/description and start tracking."""
        subagent_type = first_action.action_type or "subagent"

        # Extract prompt from first action messages
        prompt = first_action.messages or ""
        # Strip "User: " prefix if present
        if prompt.startswith("User: "):
            prompt = prompt[6:]
        # Extract description from action input if available
        description = ""
        if first_action.input and isinstance(first_action.input, dict):
            description = first_action.input.get("_task_description", "")

        goal = description or (("" if self._verbose else _truncate_middle(prompt, max_len=200)) if prompt else "")
        header = f"\u23fa {subagent_type}({goal})" if goal else f"\u23fa {subagent_type}"
        if self._verbose and prompt:
            header += f"\n  ⎿  [yellow]prompt:[/yellow] [dim]{prompt}[/dim]"

        with self._print_lock:
            self.display.console.print(header)
            self._subagent_groups[group_key] = {
                "start_time": first_action.start_time,
                "tool_count": 0,
                "subagent_type": subagent_type,
            }

    def _update_subagent_display(self, action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Print sub-agent action permanently (non-collapsing, appended)."""
        group = self._subagent_groups.get(group_key)
        if action.role == ActionRole.TOOL and group is not None:
            group["tool_count"] += 1

        # Build display label
        if action.role == ActionRole.USER:
            # Prompt already shown in header, skip
            return
        elif action.role == ActionRole.TOOL:
            function_name = action.input.get("function_name", "") if action.input else ""
            label = action.messages or function_name
            if not self._verbose:
                label = self._truncate_middle(label, max_len=200)
            # Show completion status
            status_text = "✓" if action.status == ActionStatus.SUCCESS else "✗"
            duration = ""
            if action.end_time and action.start_time:
                duration_sec = (action.end_time - action.start_time).total_seconds()
                duration = f" ({duration_sec:.1f}s)"
            line = f"  ⎿  🔧 {label} - {status_text}{duration}"

            # In verbose mode, show full arguments and full output
            extra_lines = []
            if self._verbose:
                if action.input and action.input.get("arguments"):
                    args = action.input["arguments"]
                    if isinstance(args, dict):
                        for k, v in args.items():
                            extra_lines.append(f"  ⎿      {k}: {v}")
                    else:
                        extra_lines.append(f"  ⎿      args: {args}")
                if action.output:
                    output_lines = self.display.content_generator._format_tool_output_verbose(
                        action.output, indent="  ⎿      "
                    )
                    extra_lines.extend(output_lines)

            with self._print_lock:
                self.display.console.print(f"[dim]{line}[/dim]")
                for el in extra_lines:
                    self.display.console.print(f"[dim]{el}[/dim]")
            return

        elif action.role == ActionRole.ASSISTANT:
            content = _get_assistant_content(action)
            if content:
                with self._print_lock:
                    self.display.console.print(f"[dim]  ⎿  💬 {content}[/dim]")
            return
        else:
            label = action.messages or action.action_type
            if not self._verbose:
                label = self._truncate_middle(label, max_len=200)
            line = f"  ⎿  {label}"

        with self._print_lock:
            self.display.console.print(f"[dim]{line}[/dim]")

    def _end_subagent_group_by_key(self, group_key: Optional[str], end_action: ActionHistory) -> None:
        """Print Done summary line and end sub-agent group identified by *group_key*."""
        self._stop_processing_live()

        group = self._subagent_groups.pop(group_key, None)
        if group is None:
            return

        duration = ""
        end_time = end_action.end_time or datetime.now()
        if group["start_time"]:
            duration_sec = (end_time - group["start_time"]).total_seconds()
            duration = f" \u00b7 {duration_sec:.1f}s"

        tool_count = group["tool_count"]
        summary = f"  \u23bf  Done ({tool_count} tool uses{duration})"

        with self._print_lock:
            self.display.console.print(f"[dim]{summary}[/dim]")

        if group_key is not None:
            self._completed_group_ids.add(group_key)

    # -- completed action printing -------------------------------------------

    def _print_completed_action(self, action: ActionHistory) -> None:
        """Print a completed action permanently to the console."""
        # Skip "task" tool calls — already represented by the subagent group display
        if action.role == ActionRole.TOOL:
            fn = action.input.get("function_name", "") if action.input else ""
            if fn == "task":
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
