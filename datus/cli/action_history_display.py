from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.rich_util import dict_to_tree


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
            ActionRole.ASSISTANT: "⚫",  # Grey for thinking/messages
            ActionRole.SYSTEM: "🟣",  # Purple for system
            ActionRole.USER: "🟢",  # Green for user
            ActionRole.WORKFLOW: "🟡",  # Yellow for workflow
        }

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
            if action.status == ActionStatus.PROCESSING:
                # Show input for processing tools
                if action.input and isinstance(action.input, dict):
                    function_name = action.input.get("function_name", "unknown")
                    args_preview = self._get_tool_args_preview(action.input)
                    text += f" - {function_name}({args_preview})"
            else:
                # Show completion status
                status_text = "✓" if action.status == ActionStatus.SUCCESS else "✗"
                duration = ""
                if action.end_time and action.start_time:
                    duration_sec = (action.end_time - action.start_time).total_seconds()
                    duration = f" ({duration_sec:.1f}s)"
                text += f" - {status_text}{duration}"

        # Add duration for completed actions
        elif action.status != ActionStatus.PROCESSING and action.end_time and action.start_time:
            duration_sec = (action.end_time - action.start_time).total_seconds()
            text += f" [dim]({duration_sec:.1f}s)[/dim]"

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
                    return f"{key}='{value_str[:20]}...'" if len(value_str) > 20 else f"{key}='{value_str}'"
            else:
                args_str = str(args)
                return f"'{args_str[:30]}...'" if len(args_str) > 30 else f"'{args_str}'"
        return ""

    def display_streaming_actions(self, actions: List[ActionHistory]) -> Live:
        """Create a live display for streaming actions with simplified list format"""

        # Create a class to hold the updatable content
        class StreamingContent:
            def __init__(self, actions_list, display_instance):
                self.actions = actions_list
                self.display = display_instance

            def __rich_console__(self, console, options):
                if not self.actions:
                    yield Panel("[dim]Waiting for actions...[/dim]", title="Real-time Action History")
                    return

                # Create simple list of actions with colored dots
                lines = []
                lines.append("[bold]Real-time Action History[/bold]\n")

                for i, action in enumerate(self.actions, 1):
                    # Get appropriate dot based on role and status
                    dot = self.display._get_action_dot(action)

                    # Format the action line
                    action_line = self.display._format_streaming_action(action, dot)
                    lines.append(action_line)

                # Join all lines and display in a panel
                content = "\n".join(lines)
                yield Panel(content, title="[bold cyan]Action Stream[/bold cyan]", border_style="cyan")

        # Create the content object that will update dynamically
        content = StreamingContent(actions, self)

        # Create Live display with the updatable content
        return Live(content, refresh_per_second=4)

    def display_final_action_history_with_full_sql(self, actions: List[ActionHistory]) -> None:
        """Display the final action history with complete SQL queries and reasoning results"""
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        tree = Tree("[bold]Action History[/bold]")

        for i, action in enumerate(actions, 1):
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
                input_tree = dict_to_tree(action.input, console=self.console)
                input_node = action_node.add("[cyan]📥 Input:[/cyan]")
                for child in input_tree.children:
                    input_node.add(child.label)

            if action.output:
                output_tree = dict_to_tree(action.output, console=self.console)
                output_node = action_node.add("[green]📤 Output:[/green]")
                for child in output_tree.children:
                    output_node.add(child.label)

        self.console.print(tree)

        # Show complete final reasoning result with full SQL query
        final_action = actions[-1] if actions else None
        if final_action and final_action.output and isinstance(final_action.output, dict):
            self._display_final_reasoning_result(final_action)

    def _display_final_reasoning_result(self, final_action: ActionHistory) -> None:
        """Display the complete final reasoning result with proper SQL formatting"""
        self.console.print("\n[bold green]🔧 Complete Final Reasoning Result:[/]")

        # Create a panel for the final reasoning result
        content_parts = []

        # Add the messages
        if final_action.messages:
            content_parts.append(Text(f"💬 {final_action.messages}", style="bold blue"))

        # Add status and timing
        status_info = f"Status: {final_action.status.upper()}"
        if final_action.end_time and final_action.start_time:
            duration = (final_action.end_time - final_action.start_time).total_seconds()
            status_info += f" (Duration: {duration:.2f}s)"
        content_parts.append(Text(f"📊 {status_info}", style="bold yellow"))

        # Add the complete output using rich_util formatting
        if final_action.output:
            output_tree = dict_to_tree(final_action.output, console=self.console)
            content_parts.append(Text("📤 Complete Output:", style="bold green"))
            content_parts.append(output_tree)

        # Display content parts separately since they contain mixed types
        if content_parts:
            for part in content_parts:
                if isinstance(part, Text):
                    self.console.print(part)
                else:
                    # This is a Tree object, print it directly
                    self.console.print(part)

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


def create_action_display(console: Optional[Console] = None) -> ActionHistoryDisplay:
    """Factory function to create ActionHistoryDisplay"""
    return ActionHistoryDisplay(console)
