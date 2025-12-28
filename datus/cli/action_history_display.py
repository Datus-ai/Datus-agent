# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import sys
from collections import deque
from contextlib import contextmanager
from io import StringIO
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

logger = get_logger(__name__)


class BaseActionContentGenerator:
    def __init__(self) -> None:
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
    """Display ActionHistory in a rich format with separated content generation logic"""

    def __init__(self, console: Optional[Console] = None, enable_truncation: bool = True):
        self.console = console or Console()
        self.enable_truncation = enable_truncation

        # Create content generator with truncation setting
        self.content_generator = ActionContentGenerator(enable_truncation=enable_truncation)

        # Sliding window for managing content overflow
        self._action_window: Optional[deque] = None
        self._max_actions: Optional[int] = None

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

    def display_action_list(self, actions: List[ActionHistory]) -> None:
        """Display a list of actions in a tree-like format"""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        tree = Tree("[bold]Action History[/bold]")

        for _, action in enumerate(actions, 1):
            status_icon = self.content_generator.get_status_icon(action)
            role_color = self.content_generator.get_role_color(action.role)

            # Create main node with duration
            duration = ""
            if action.end_time and action.start_time:
                duration_seconds = (action.end_time - action.start_time).total_seconds()
                duration = f" [dim]({duration_seconds:.2f}s)[/dim]"

            main_text = f"[{role_color}]{status_icon} {action.messages}[/{role_color}]{duration}"
            action_node = tree.add(main_text)

            # Add details as child nodes
            if action.input:
                input_summary = self.content_generator.get_data_summary(action.input)
                action_node.add(f"[cyan]📥 Input: {input_summary}[/cyan]")

            if action.output:
                output_summary = self.content_generator.get_data_summary(action.output)
                action_node.add(f"[green]📤 Output: {output_summary}[/green]")

        self.console.print(tree)

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
        self._content_renderer = None  # Keep reference to content for recreation
        self._display_checkpoint = 0  # Track how many actions were shown before recreation

    def recreate_live_display(self):
        """
        Recreate a brand new Live display from current cursor position.

        This is used in plan mode to create a fresh display after showing
        static content (menus, plans), avoiding overlap with previous content.
        """
        from datus.cli.execution_state import execution_controller

        # Stop and discard the old Live display
        if self.live:
            try:
                self.live.stop()
            except Exception:
                # Ignore any errors when stopping the old display
                pass

        # Set checkpoint to current number of actions
        # This ensures only new actions (after recreation) will be displayed
        self._display_checkpoint = len(self.actions)

        # Create a new Live display from current cursor position
        # Reuse the same content renderer (it will respect the checkpoint)
        if self._content_renderer:
            self.live = Live(self._content_renderer, refresh_per_second=4)
            self.live.start()

            # Register with execution controller
            try:
                execution_controller.register_live_display(self.live)
            except Exception as e:
                logger.warning(f"Failed to register recreated live display: {e}")

        return self.live

    def __enter__(self):
        # Create the content renderer
        class StreamingContent:
            def __init__(self, actions_list, display_instance: ActionHistoryDisplay, context):
                self.actions = actions_list
                self.display = display_instance
                self.context = context  # Reference to StreamingActionContext for checkpoint

            def __rich_console__(self, console, options):  # pylint: disable=unused-argument
                # Filter actions based on checkpoint
                # Only show actions that came after the display was recreated
                filtered_actions = self.actions[self.context._display_checkpoint :]

                # Update sliding window with filtered actions
                if self.display._action_window is not None:
                    self.display._action_window.clear()
                    for action in filtered_actions:
                        self.display._action_window.append(action)

                # Generate content using content generator
                window_actions = list(self.display._action_window) if self.display._action_window else []
                content: str | Group = self.display.content_generator.generate_streaming_content(window_actions)

                # Always yield the same panel structure
                yield Panel(content, title="[bold cyan]Action Stream[/bold cyan]", border_style="cyan")

        # Create the content object that will update dynamically
        content = StreamingContent(self.actions, self.display, self)
        self._content_renderer = content  # Save for potential recreation

        # Create Live display
        self.live = Live(content, refresh_per_second=4)

        # Start the live display
        self.live.start()

        # Register with execution controller (including context for recreation)
        try:
            from datus.cli.execution_state import execution_controller

            execution_controller.register_live_display(self.live, context=self)
        except Exception as e:
            logger.warning(f"Failed to register live display: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        # Unregister from execution controller
        try:
            from datus.cli.execution_state import execution_controller

            execution_controller.unregister_live_display()
        except Exception as e:
            logger.warning(f"Failed to unregister live display: {e}")

        if self.live:
            self.live.stop()


def create_action_display(console: Optional[Console] = None, enable_truncation: bool = True) -> ActionHistoryDisplay:
    """Factory function to create ActionHistoryDisplay with truncation control"""
    return ActionHistoryDisplay(console, enable_truncation)
