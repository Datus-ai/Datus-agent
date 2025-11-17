# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Workflow screen module for Datus CLI.
Provides an interactive screen for agent workflow visualization and control.
"""

import asyncio
import os
import time

from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Static

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.cli.screen.base_app import BaseApp
from datus.schemas.node_models import BaseInput, BaseResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Configure debug logging based on environment variable
WORKFLOW_SCREEN_DEBUG = os.environ.get("DATUS_WORKFLOW_DEBUG", "0").lower() in ("1", "true", "yes")

# debug logging setup
if WORKFLOW_SCREEN_DEBUG:
    import logging

    file_logger = logging.getLogger("workflow-screen")
    if not any(isinstance(h, logging.FileHandler) for h in file_logger.handlers):
        file_handler = logging.FileHandler("workflow_screen_debug.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        file_logger.addHandler(file_handler)
        file_logger.setLevel(logging.DEBUG)
else:
    # Create a null logger when debug is disabled
    import logging

    file_logger = logging.getLogger("workflow-screen")
    file_logger.addHandler(logging.NullHandler())
    file_logger.setLevel(logging.WARNING)


class WorkflowScreen(Screen):
    """Interactive screen for agent workflow visualization and control."""

    BINDINGS = [
        Binding("escape", "exit", "Exit"),
        Binding("q", "exit", "Exit"),
        Binding("p", "pause", "Pause"),
        Binding("r", "resume", "Resume"),
        Binding("j", "cursor_down", "Down"),
        Binding("k", "cursor_up", "Up"),
        Binding("e", "edit", "Edit"),
        Binding("c", "toggle_context", "Toggle Context"),
        Binding("s", "select_node", "Select Node details"),
        # ("space", "select_node", "Select Node details"),
        # Binding("enter", "select_node", "Select Node details"),
    ]

    selected_node_index = reactive(0)
    is_paused = reactive(False)
    show_context = reactive(True)

    def __init__(self, workflow: Workflow, exit_callback=None):
        """
        Initialize the workflow screen.

        Args:
            workflow: The agent workflow to visualize
            exit_callback: Callback for when the screen exits
        """
        super().__init__()
        self.workflow = workflow
        self.exit_callback = exit_callback
        self.update_task = None
        self.selected_zone = "node-list"  # node-list, node-details, context

    def compose(self):
        """Compose the layout of the screen."""
        yield Header(show_clock=True)

        yield Container(
            Label(
                f"Workflow: {self.workflow.task.task if hasattr(self.workflow, 'task') else 'Untitled Workflow'}",
                id="workflow-title",
            ),
            Label(
                f"Database: {self.workflow.task.database_name if hasattr(self.workflow, 'task') else 'Unknown'}",
                id="workflow-database",
            ),
            Horizontal(
                # Left panel: Node list
                Vertical(
                    Label("Workflow Steps", classes="panel-title"),
                    DataTable(id="node-list"),
                    id="node-list-panel",
                ),
                # Right panel: Node details with scrolling
                Vertical(
                    Label("Node Details", classes="panel-title"),
                    VerticalScroll(Static(id="node-details"), id="node-details-scroll"),
                    id="node-details-panel",
                ),
                id="main-container",
            ),
            # Context panel (collapsible)
            Container(
                Label("Context", classes="panel-title"),
                VerticalScroll(Static(id="context-panel"), id="context-scroll"),
                id="context-container",
                classes="hidden" if not self.show_context else "",
            ),
            id="workflow-container",
        )

        yield Footer()

    def on_mount(self):
        """Initialize the screen on mount."""
        # Setup node list table
        node_table = self.query_one("#node-list", DataTable)
        node_table.add_column("Step", width=4)
        node_table.add_column("Type", width=15)
        node_table.add_column("Status", width=10)
        node_table.add_column("Start", width=10)
        node_table.add_column("Duration", width=10)

        # Populate node list
        self.update_node_list()

        # Setup context panel
        self.update_context_panel()

        # Create background task for updates
        self.update_task = asyncio.create_task(self.update_periodically())

    async def update_periodically(self):
        """Update the screen periodically in the background."""
        try:
            while True:
                if not self.is_paused:
                    # save current cursor position
                    node_table = self.query_one("#node-list", DataTable)
                    current_cursor = node_table.cursor_coordinate

                    # Update all components with latest data
                    self.update_node_list()
                    self.update_node_details()
                    self.update_context_panel()

                    # restore cursor position
                    if current_cursor is not None:
                        node_table = self.query_one("#node-list", DataTable)
                        if current_cursor.row < node_table.row_count:
                            node_table.cursor_coordinate = current_cursor

                    # Force UI refresh
                    self.refresh()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Handle normal cancellation
            logger.info("Update task cancelled")
        except Exception as e:
            # Log any errors
            logger.error(f"Error in update_periodically: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

    def update_node_list(self):
        """Update the node list table with current workflow state."""
        try:
            # Use file logger
            file_logger.info(f"Workflow node_order: {self.workflow.node_order}")
            file_logger.info(f"Workflow nodes keys: {list(self.workflow.nodes.keys())}")
            file_logger.info(f"Update node list Node, current node index: {self.workflow.current_node_index}")

            # Get node table widget
            node_table = self.query_one("#node-list", DataTable)
            # Clear current rows
            node_table.clear()

            # Check each node
            for i, node_id in enumerate(self.workflow.node_order):
                node = self.workflow.nodes[node_id]

                # Skip if node is not a Node object
                if not isinstance(node, Node):
                    file_logger.warning(f"Skipping non-Node object: {type(node)}")
                    continue

                # Modify the display text to highlight current node
                step_text = str(i + 1)
                type_text = node.type
                status_text = node.status
                start_time_text = time.strftime("%H:%M:%S", time.localtime(node.start_time)) if node.start_time else ""
                duration_text = f"{node.end_time - node.start_time:.2f} s" if node.end_time and node.start_time else ""

                # If this is the current node, add a marker to the type name (◉)
                if i == self.workflow.current_node_index:
                    type_text = f"▶ {type_text}"

                try:
                    # Add row for this node without style parameter
                    node_table.add_row(step_text, type_text, status_text, start_time_text, duration_text, key=i)
                except Exception as e:
                    file_logger.error(f"Error adding row for node {i}: {str(e)}")

            # Check final table - don't auto-select current node
            file_logger.info(f"Final table row count: {node_table.row_count}")
        except Exception as e:
            logger.error(f"Error in update_node_list: {e}")

    def update_node_details(self):
        """Update the node details panel with selected node information."""
        try:
            file_logger.info(f"update_node_details called with selected_node_index: {self.selected_node_index}")

            if self.selected_node_index >= len(self.workflow.node_order):
                file_logger.warning(f"Selected node index {self.selected_node_index} is out of range")
                return

            node_id = self.workflow.node_order[self.selected_node_index]

            node = self.workflow.nodes[node_id]
            file_logger.info(f"Node retrieved: {node is not None}, type: {node.type}")

            # Skip if node is not a Node object
            if not isinstance(node, Node):
                return

            from rich.text import Text
            from rich.tree import Tree

            tree = Tree(f"[bold]{node.type.capitalize()}[/bold]: {node.description}")

            status_node = tree.add("📊 [bold]Status[/bold]")
            status_node.add(f"State: {node.status}")
            if node.start_time:
                status_node.add(f"Started: {time.strftime('%H:%M:%S', time.localtime(node.start_time))}")
            if node.end_time:
                status_node.add(f"Ended: {time.strftime('%H:%M:%S', time.localtime(node.end_time))}")
                if node.start_time:
                    duration = node.end_time - node.start_time
                    status_node.add(f"Duration: {duration:.2f} seconds")

            # Add input information as a tree
            if hasattr(node, "input") and node.input:
                input_node = tree.add("📥 [bold]Input[/bold]")

                if isinstance(node.input, BaseInput):
                    # Try to get input as dictionary
                    input_dict = node.input.to_dict()
                    self._add_dict_to_tree(input_node, input_dict)

                else:
                    input_str = str(node.input)
                    input_node.add(Text(input_str[:500] + "..." if len(input_str) > 500 else input_str))

            # Add result information as a tree
            if hasattr(node, "result") and node.result:
                result_node = tree.add("📤 [bold]Result[/bold]")

                if isinstance(node.result, BaseResult):
                    # Add success/failure status
                    result_node.add(f"Success: {'✅' if node.result.success else '❌'}")

                    # Add error information if present
                    if not node.result.success and hasattr(node.result, "error") and node.result.error:
                        error_node = result_node.add("[bold red]Error[/bold red]")
                        error_str = str(node.result.error)
                        error_node.add(Text(error_str[:500] + "..." if len(error_str) > 500 else error_str))

                    # Try to get result as dictionary and display as tree
                    result_dict = node.result.to_dict()
                    self._add_dict_to_tree(result_node, result_dict)

                else:
                    result_str = str(node.result)
                    result_node.add(Text(result_str[:500] + "..." if len(result_str) > 500 else result_str))

            self.query_one("#node-details", Static).update(tree)
            file_logger.info("Node details panel updated successfully")
        except Exception as e:
            import traceback

            error_message = f"Error in update_node_details: {str(e)}\n{traceback.format_exc()}"
            file_logger.error(error_message)

    def _add_dict_to_tree(self, parent_node, data, max_depth=3, current_depth=0):
        """Add dictionary data to a Rich tree node with depth limiting.

        Args:
            parent_node: The parent tree node to add items to
            data: Dictionary or other data to add
            max_depth: Maximum depth to recurse into nested dictionaries
            current_depth: Current recursion depth
        """
        from rich.text import Text

        # Stop recursion if we're too deep
        if current_depth >= max_depth:
            parent_node.add("[dim]Max depth reached...[/dim]")
            return

        # Handle different data types
        if isinstance(data, dict):
            # Process dictionary
            for key, value in data.items():
                key_str = str(key)

                if isinstance(value, (dict, list)):
                    # Create a branch node for complex types
                    branch = parent_node.add(f"[bold blue]{key_str}[/bold blue]")
                    self._add_dict_to_tree(branch, value, max_depth, current_depth + 1)
                else:
                    # Format value based on type
                    if value is None:
                        value_str = "[dim]None[/dim]"
                    elif isinstance(value, bool):
                        value_str = "[green]True[/green]" if value else "[red]False[/red]"
                    elif isinstance(value, (int, float)):
                        value_str = f"[cyan]{value}[/cyan]"
                    else:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."

                    # Add leaf node with formatted value
                    parent_node.add(f"[bold blue]{key_str}[/bold blue]: {value_str}")

        elif isinstance(data, list):
            # Process list
            if len(data) == 0:
                parent_node.add("[dim]Empty list[/dim]")
            else:
                # Show limited number of items for large lists
                max_items = 10
                for i, item in enumerate(data[:max_items]):
                    if isinstance(item, (dict, list)):
                        branch = parent_node.add(f"[bold green]{i}[/bold green]")
                        self._add_dict_to_tree(branch, item, max_depth, current_depth + 1)
                    else:
                        item_str = str(item)
                        if len(item_str) > 100:
                            item_str = item_str[:97] + "..."
                        parent_node.add(f"[bold green]{i}[/bold green]: {item_str}")

                # Show message if list was truncated
                if len(data) > max_items:
                    parent_node.add(f"[dim]... and {len(data) - max_items} more items[/dim]")

        else:
            # Handle any other type as string
            data_str = str(data)
            if len(data_str) > 500:
                data_str = data_str[:497] + "..."
            parent_node.add(Text(data_str))

    def update_context_panel(self):
        """Update the context panel with current workflow context."""
        try:
            if not hasattr(self.workflow, "context"):
                return

            context = self.workflow.context

            # Build context summary
            content = "## Context Summary\n\n"

            # Add schemas
            if hasattr(context, "table_schemas") and context.table_schemas:
                content += "### Table Schemas\n"
                for schema in context.table_schemas[:5]:  # Limit to 5
                    content += f"- {schema.database_name}.{schema.schema_name}.{schema.table_name}\n"
                if len(context.table_schemas) > 5:
                    content += f"... and {len(context.table_schemas) - 5} more\n"

            # Add SQL contexts
            if hasattr(context, "sql_contexts") and context.sql_contexts:
                content += "\n### Reference SQL\n"
                for i, sql_ctx in enumerate(context.sql_contexts[-3:]):  # Show last 3
                    if hasattr(sql_ctx, "sql_query"):
                        query = sql_ctx.sql_query
                        if len(query) > 100:
                            query = query[:100] + "..."
                        content += f"- SQL {len(context.sql_contexts) - 3 + i + 1}: `{query}`\n"

            # Add metrics
            if hasattr(context, "metrics") and context.metrics:
                content += "\n### Metrics\n"
                for metric in context.metrics[:5]:  # Limit to 5
                    content += f"- {metric.name}: {metric.description} {metric.constraint} {metric.sql_query}\n"
                if len(context.metrics) > 5:
                    content += f"... and {len(context.metrics) - 5} more\n"

            # Update the context panel
            self.query_one("#context-panel", Static).update(content)
        except Exception:
            import traceback

            traceback.print_exc()

    def on_data_table_row_selected(self, event):
        """Handle node selection."""
        logger.info(f"on_data_table_row_selected called with event: {event}")
        self.selected_node_index = event.cursor_row
        self.update_node_details()

    def watch_show_context(self, show_context: bool) -> None:
        """Toggle the context panel visibility."""
        try:
            context_container = self.query_one("#context-container", Container)
            main_container = self.query_one("#main-container", Horizontal)

            if show_context:
                context_container.remove_class("hidden")
                main_container.remove_class("full-height")
            else:
                context_container.add_class("hidden")
                main_container.add_class("full-height")
        except Exception:
            # ignore this error, because it may be because the UI is not fully rendered
            pass

    def action_exit(self):
        """Exit the workflow screen."""
        if self.update_task:
            self.update_task.cancel()

        if self.exit_callback:
            self.exit_callback()

        # self.app.pop_screen()
        self.app.exit()

    def action_cursor_down(self):
        """Move cursor down in the tree."""
        if self.selected_zone == "node-list":
            node_table = self.query_one("#node-list")
            node_table.action_cursor_down()
        elif self.selected_zone == "node-details":
            tree = self.query_one("#node-details")
            tree.action_cursor_down()
        elif self.selected_zone == "context":
            pass

    def action_cursor_up(self):
        """Move cursor up in the tree."""
        if self.selected_zone == "node-list":
            node_table = self.query_one("#node-list")
            node_table.action_cursor_up()
        elif self.selected_zone == "node-details":
            tree = self.query_one("#node-details")
            tree.action_cursor_up()
        elif self.selected_zone == "context":
            pass

    def action_pause(self):
        """Pause the workflow execution."""
        self.is_paused = True
        self.workflow.pause()

    def action_resume(self):
        """Resume the workflow execution."""
        self.is_paused = False
        self.workflow.resume()

    def action_edit(self):
        """Edit the selected node input."""
        if self.selected_node_index >= len(self.workflow.nodes):
            return

        # TODO: Implement node editing

    def action_toggle_context(self):
        """Toggle the context panel visibility."""
        self.show_context = not self.show_context

    def action_select_node(self):
        """Select the current node and show its details."""
        try:
            # Get the data table
            node_table = self.query_one("#node-list", DataTable)

            # If there's a cursor row, use it as the selected index
            file_logger.info(f"node_table.cursor_coordinate: {node_table.cursor_coordinate}")

            if node_table.cursor_coordinate is not None:
                self.selected_node_index = node_table.cursor_coordinate.row

                # Log the selection
                file_logger.info(f"Node selected via action: index={self.selected_node_index}")

                # Update the node details panel
                self.update_node_details()
        except Exception as e:
            logger.error(f"Error in action_select_node: {e}")


class WorkflowApp(BaseApp):
    """Textual app for workflow screen."""

    # Add the CSS to the app
    CSS = """
    #workflow-database {
        border-bottom: solid $primary-lighten-1;
    }

    .panel-title {
        background: $panel;
        text-align: center;
        color: $text;
    }

    #node-list-panel {
        width: 50%;
        height: 100%;
    }

    #node-details-panel {
        width: 50%;
        height: 100%;
    }

    #node-details-scroll {
        height: 100%;
        border: solid $panel-lighten-1;
        padding: 1;
    }

    #main-container {
        height: 70%;
    }

    #workflow-container {
        margin-bottom: 1;
        height: 100%;
    }

    #main-container.full-height {
        height: 100%;
    }

    #context-container {
        height: 30%;
        border-top: solid $panel-lighten-1;
    }

    .hidden {
        display: none;
    }
    """

    def __init__(self, workflow: Workflow, exit_callback=None):
        """
        Initialize the workflow app.

        Args:
            workflow: The agent workflow to visualize
            exit_callback: Callback for when the screen exits
        """
        super().__init__()
        self.workflow = workflow
        self.exit_callback = exit_callback

    def on_mount(self):
        """Push the workflow screen when the app is mounted."""
        self.push_screen(WorkflowScreen(self.workflow, self.exit_callback))


def show_workflow_screen(workflow: Workflow, run_new_loop=True, exit_callback=None):
    """
    Show the workflow screen.

    Args:
        workflow: The agent workflow to visualize
        exit_callback: Callback for when the screen exits
    """
    if run_new_loop:
        # Rich already runs in a separate loop, so we need to create a new one
        import asyncio

        new_loop = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Don't create a new loop if one is already running
                logger.info(f"Using existing running event loop: {loop}")
                app = WorkflowApp(workflow, exit_callback)
                app.run()
                return
            else:
                logger.info(f"Got existing event loop: {loop}")
        except RuntimeError:
            new_loop = True
            loop = asyncio.new_event_loop()
            logger.info(f"Created new event loop: {loop}")
            asyncio.set_event_loop(loop)

        app = WorkflowApp(workflow, exit_callback)

        try:
            app.run()
        finally:
            if new_loop:
                # Ensure proper cleanup before closing the loop
                try:
                    # Cancel any remaining tasks
                    pending_tasks = asyncio.all_tasks(loop)
                    for task in pending_tasks:
                        task.cancel()
                    if pending_tasks:
                        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                except Exception as e:
                    logger.warning(f"Error during task cleanup: {e}")
                finally:
                    loop.close()
    else:
        app = WorkflowApp(workflow, exit_callback)
        app.run()
