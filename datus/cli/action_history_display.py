import os
import sys
from collections import deque
from contextlib import contextmanager
from io import StringIO
from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

logger = get_logger(__name__)


class ActionHistoryDisplay:
    """Display ActionHistory in a rich format similar to Claude Code"""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.role_colors = {
            ActionRole.SYSTEM: "bright_magenta",
            ActionRole.ASSISTANT: "bright_blue",
            ActionRole.USER: "bright_green",
            ActionRole.TOOL: "bright_cyan",
            ActionRole.WORKFLOW: "bright_yellow",
        }
        self.status_icons = {
            ActionStatus.PROCESSING: "⏳",
            ActionStatus.SUCCESS: "✅",
            ActionStatus.FAILED: "❌",
        }
        # New simplified dot system for status and role
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
        }

        # Sliding window for managing content overflow
        self._action_window = None
        self._max_actions = None

    def _get_terminal_height(self) -> int:
        """Get terminal height, fallback to reasonable default"""
        try:
            return os.get_terminal_size().lines
        except (OSError, ValueError):
            return 24  # Fallback to standard terminal height

    def _calculate_max_actions(self) -> int:
        """Calculate maximum number of actions that can fit in terminal"""
        terminal_height = self._get_terminal_height()
        # Reserve space for: panel borders (4 lines), title (1 line), some padding
        # Each action typically takes 1-3 lines depending on content
        available_height = max(terminal_height - 8, 5)  # Minimum of 5 actions
        # Assume average of 2 lines per action for conservative estimate
        return max(available_height // 2, 5)

    @contextmanager
    def _capture_external_output(self):
        """Context manager to capture stdout/stderr during Live display to prevent interference"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create string buffers to capture output
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        try:
            # Redirect stdout/stderr to buffers
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield stdout_buffer, stderr_buffer
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Optionally display captured output after Live session
            captured_stdout = stdout_buffer.getvalue()
            captured_stderr = stderr_buffer.getvalue()

            if captured_stdout.strip():
                logger.debug(f"Captured stdout during Live display: {captured_stdout}")
            if captured_stderr.strip():
                logger.debug(f"Captured stderr during Live display: {captured_stderr}")

    def format_action_summary(self, action: ActionHistory) -> str:
        """Format a single action as a summary line"""
        status_icon = self.status_icons.get(action.status, "⚡")
        role_color = self.role_colors.get(action.role, "white")

        return f"[{role_color}]{status_icon} {action.messages}[/{role_color}]"

    def format_action_detail(self, action: ActionHistory) -> Panel:
        """Format a single action as a detailed panel"""
        status_icon = self.status_icons.get(action.status, "⚡")
        role_color = self.role_colors.get(action.role, "white")

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
            input_text = self._format_data(action.input)
            content.append(Text(input_text, style="cyan"))

        # Add output if present
        if action.output:
            content.append(Text("📤 Output:", style="bold green"))
            output_text = self._format_data(action.output)
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

    def _format_data(self, data) -> str:
        """Format input/output data for display"""
        if isinstance(data, dict):
            # Pretty print JSON-like data
            formatted = []
            for key, value in data.items():
                # Don't truncate SQL queries - they're important to see in full
                if key.lower() in ["sql_query", "sql", "query", "sql_return"] and isinstance(value, str):
                    formatted.append(f"  {key}: {value}")
                elif isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                    formatted.append(f"  {key}: {value}")
                else:
                    formatted.append(f"  {key}: {value}")
            return "\n".join(formatted)
        elif isinstance(data, str):
            return data if len(data) <= 100 else data[:100] + "..."
        else:
            return str(data)

    def display_action_list(self, actions: List[ActionHistory]) -> None:
        """Display a list of actions in a tree-like format"""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        tree = Tree("[bold]Action History[/bold]")

        for i, action in enumerate(actions, 1):
            status_icon = self.status_icons.get(action.status, "⚡")
            role_color = self.role_colors.get(action.role, "white")

            # Create main node with duration
            duration = ""
            if action.end_time and action.start_time:
                duration_seconds = (action.end_time - action.start_time).total_seconds()
                duration = f" [dim]({duration_seconds:.2f}s)[/dim]"

            main_text = f"[{role_color}]{status_icon} {action.messages}[/{role_color}]{duration}"
            action_node = tree.add(main_text)

            # Add details as child nodes
            if action.input:
                input_summary = self._get_data_summary(action.input)
                action_node.add(f"[cyan]📥 Input: {input_summary}[/cyan]")

            if action.output:
                output_summary = self._get_data_summary(action.output)
                action_node.add(f"[green]📤 Output: {output_summary}[/green]")

        self.console.print(tree)

    def _get_data_summary(self, data) -> str:
        """Get a brief summary of data for tree display"""
        if isinstance(data, dict):
            if "success" in data:
                status = "✅" if data["success"] else "❌"
                # Show SQL query if present - increase limit for SQL queries
                if "sql_query" in data and data["sql_query"]:
                    sql_preview = data["sql_query"][:200] + "..." if len(data["sql_query"]) > 200 else data["sql_query"]
                    return f"{status} SQL: {sql_preview}"
                return f"{status} {len(data)} fields"
            else:
                return f"{len(data)} fields"
        elif isinstance(data, str):
            return data[:30] + "..." if len(data) > 30 else data
        else:
            return str(data)[:30]

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

    def _format_streaming_action(self, action: ActionHistory, dot: str) -> str:
        """Format a single action for streaming display"""
        # Base action text with dot
        text = f"{dot} {action.messages}"

        # Add status info for tools
        if action.role == ActionRole.TOOL:
            if action.input and isinstance(action.input, dict):
                function_name = action.input.get("function_name", "unknown")
                args_preview = self._get_tool_args_preview(action.input)
                text += f" - {function_name}({args_preview})"
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

    def _get_tool_args_preview(self, input_data: dict) -> str:
        """Get a brief preview of tool arguments"""
        if "arguments" in input_data and input_data["arguments"]:
            args = input_data["arguments"]
            if isinstance(args, dict):
                # Show first key-value pair or query if present
                if "query" in args:
                    query = str(args["query"])
                    return f"query='{query[:200]}...'" if len(query) > 200 else f"query='{query}'"
                elif args:
                    key, value = next(iter(args.items()))
                    value_str = str(value)
                    return f"{key}='{value_str[:50]}...'" if len(value_str) > 50 else f"{key}='{value_str}'"
            else:
                args_str = str(args)
                return f"'{args_str[:50]}...'" if len(args_str) > 50 else f"'{args_str}'"
        return ""

    def _get_tool_output_preview(self, output_data: dict, function_name: str = "") -> str:
        """Get a brief preview of tool output results"""
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

        # Parse data.text for counting items or showing text preview
        if "text" in data and isinstance(data["text"], str):
            text_content = data["text"]

            # First try to parse as JSON array for counting
            try:
                cleaned_text = text_content.replace("'", '"').replace("None", "null")
                items = json.loads(cleaned_text)

                if isinstance(items, list):
                    count = len(items)
                    # Return appropriate label based on function name
                    if function_name in ["list_tables", "table_overview"]:
                        return f"✓ {count} tables"
                    elif function_name in ["describe_table"]:
                        return f"✓ {count} columns"
                    elif function_name in ["read_query", "query"]:
                        return f"✓ {count} rows"
                    else:
                        return f"✓ {count} items"
            except Exception:
                # If JSON parsing fails, treat as plain text and show preview
                return f"{text_content[:50]}..." if len(text_content) > 50 else text_content

        # Generic fallback
        if "success" in output_data:
            return "✓ Success" if output_data["success"] else "✗ Failed"

        return "✓ Completed"

    def display_streaming_actions(self, actions: List[ActionHistory]) -> "StreamingActionContext":
        """Create a live display for streaming actions with sliding window and output capture"""

        # Initialize sliding window if needed
        if self._max_actions is None:
            self._max_actions = self._calculate_max_actions()

        if self._action_window is None:
            self._action_window = deque(maxlen=self._max_actions)
        else:
            # Update maxlen if terminal size changed
            current_max = self._calculate_max_actions()
            if current_max != self._max_actions:
                # Create new deque with updated size, preserving recent actions
                new_window = deque(self._action_window, maxlen=current_max)
                self._action_window = new_window
                self._max_actions = current_max

        return StreamingActionContext(actions, self)

    def display_final_action_history(self, actions: List[ActionHistory]) -> None:
        """Display the final action history with complete SQL queries and reasoning results"""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        tree = Tree("[bold]Action History[/bold]")

        for action in actions:
            status_icon = self.status_icons.get(action.status, "⚡")
            role_color = self.role_colors.get(action.role, "white")

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
                    input_node = action_node.add(f"[cyan]📥 Input:[/cyan] {str(action.input)}")

            if action.output:
                if isinstance(action.output, dict):
                    output_tree = dict_to_tree(action.output, console=self.console)
                    output_node = action_node.add("[green]📤 Output:[/green]")
                    for child in output_tree.children:
                        output_node.add(child.label)
                else:
                    output_node = action_node.add(f"[green]📤 Output:[/green] {str(action.output)}")

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


class StreamingActionContext:
    """Context manager for streaming actions display with output capture"""

    def __init__(self, actions_list: List[ActionHistory], display_instance: ActionHistoryDisplay):
        self.actions = actions_list
        self.display = display_instance
        self.live = None

    def __enter__(self):
        # Create the content renderer
        class StreamingContent:
            def __init__(self, actions_list, display_instance):
                self.actions = actions_list
                self.display = display_instance

            def __rich_console__(self, console, options):  # pylint: disable=unused-argument
                # Update sliding window with current actions
                self.display._action_window.clear()
                for action in self.actions:
                    self.display._action_window.append(action)

                # Always create the same panel structure to avoid duplicate headers
                if not self.display._action_window:
                    content = "[dim]Waiting for actions...[/dim]"
                else:
                    # Create simple list of actions with colored dots from sliding window
                    lines = []
                    for action in self.display._action_window:
                        # Get appropriate dot based on role and status
                        dot = self.display._get_action_dot(action)
                        # Format the action line
                        action_line = self.display._format_streaming_action(action, dot)
                        lines.append(action_line)

                    content = "\n".join(lines)

                    # Add indicator if we're showing partial results
                    if len(self.actions) > len(self.display._action_window):
                        truncated_count = len(self.actions) - len(self.display._action_window)
                        content = f"[dim]({truncated_count} older actions hidden)[/dim]\n{content}"

                # Always yield the same panel structure
                yield Panel(content, title="[bold cyan]Action Stream[/bold cyan]", border_style="cyan")

        # Create the content object that will update dynamically
        content = StreamingContent(self.actions, self.display)

        # Create Live display
        self.live = Live(content, refresh_per_second=4)

        # Start the live display
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        if self.live:
            self.live.stop()


def create_action_display(console: Optional[Console] = None) -> ActionHistoryDisplay:
    """Factory function to create ActionHistoryDisplay"""
    return ActionHistoryDisplay(console)
