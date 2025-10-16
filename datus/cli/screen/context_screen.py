# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Context screen module for Datus CLI.
Provides interactive screens for database exploration.
"""

from typing import Any, Dict

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual.widgets import Tree as TextualTree

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ContextScreen(Screen):
    """Base screen for context exploration."""

    def __init__(self, title: str, context_data: Dict, inject_callback=None, **kwargs):
        """
        Initialize the context screen.

        Args:
            title: Title of the screen
            context_data: Data to display in the context screen
            inject_callback: Callback for injecting data into the workflow
        """
        super().__init__(**kwargs)
        self.title = title
        self.context_data = context_data
        self.inject_callback = inject_callback


class WorkloadContextScreen(ContextScreen):
    """Screen for displaying workload context."""

    BINDINGS = [
        Binding("escape", "exit", "Exit"),
        Binding("q", "exit", "Exit"),
        Binding("j", "cursor_down", "Down"),
        Binding("k", "cursor_up", "Up"),
        Binding("h", "left", "Collapse"),
        Binding("l", "right", "Expand"),
        Binding("e", "edit", "Edit"),
        Binding("space", "toggle_node", "Toggle"),
        Binding("enter", "select_node", "Select"),
    ]

    def __init__(self, title: str, context_data: Dict, inject_callback=None):
        """
        Initialize the workload context screen.

        Args:
            title: Title of the screen
            context_data: Data to display in the context screen
            inject_callback: Callback for injecting data into the workflow
        """
        super().__init__(title, context_data, inject_callback)
        self.current_details = None

    class SelectNodeMessage(Message):
        """Message sent when a tree node is selected."""

        def __init__(self, node_data: Dict[str, Any]):
            self.node_data = node_data
            super().__init__()

    def compose(self):
        """Compose the layout of the screen."""
        yield Header(show_clock=True)

        with Horizontal():
            # Left side: Context tree
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("# Context Explorer", id="tree-help")

                # Create trees for each context type
                context_tree = TextualTree("Workflow Context", id="context-tree")
                yield context_tree

            # Right side: Details panel
            with Vertical(id="details-container", classes="details-panel"):
                yield Static("# Details\n\nSelect a node to view details", id="details-panel")

        yield Footer()

    async def on_mount(self):
        self._build_context_tree()

    def _build_context_tree(self):
        """Build the context tree from the context data."""
        # Create main category nodes
        tree = self.query_one("#context-tree", TextualTree)
        sql_node = tree.root.add("SQL Contexts")
        schema_node = tree.root.add("Table Schemas")
        schema_values_node = tree.root.add("Table Values")
        metrics_node = tree.root.add("Metrics")

        # Add SQL contexts
        if "sql_contexts" in self.context_data and self.context_data["sql_contexts"]:
            for i, item in enumerate(self.context_data["sql_contexts"]):
                node = sql_node.add(f"SQL Context {i + 1}", data={"type": "sql", "index": i, "content": item})
                if "sql_query" in item:
                    node.add_leaf(f"Query: {item['sql_query'][:50]}...")

        # Add table schemas
        if "table_schemas" in self.context_data and self.context_data["table_schemas"]:
            for i, item in enumerate(self.context_data["table_schemas"]):
                table_name = item.get("table_name", f"Table {i + 1}")
                node = schema_node.add(table_name, data={"type": "schema", "index": i, "content": item})
                if "columns" in item and isinstance(item["columns"], list):
                    for col in item["columns"][:3]:  # Show first 3 columns only in tree
                        col_name = col.get("name", "")
                        col_type = col.get("type", "")
                        node.add_leaf(f"{col_name} ({col_type})")
                    if len(item["columns"]) > 3:
                        node.add_leaf(f"... {len(item['columns']) - 3} more columns")

        # Add table values
        if "table_values" in self.context_data and self.context_data["table_values"]:
            for i, item in enumerate(self.context_data["table_values"]):
                table_name = item.get("table_name", f"Values {i + 1}")
                schema_values_node.add(table_name, data={"type": "schema_values", "index": i, "content": item})

        # Add metrics
        if "metrics" in self.context_data and self.context_data["metrics"]:
            for i, item in enumerate(self.context_data["metrics"]):
                metric_name = item.get("name", f"Metric {i + 1}")
                metrics_node.add(metric_name, data={"type": "metrics", "index": i, "content": item})

    def on_tree_node_selected(self, event):
        """Handle selection of a tree node."""
        node = event.node
        if hasattr(node, "data") and node.data:
            # Update the details panel with the selected node's data
            self.current_details = node.data
            self._update_details()
            # Post a message that can be handled by parent components if needed
            self.post_message(self.SelectNodeMessage(node.data))

    def _update_details(self):
        """Update the details panel with the selected node's data."""
        if not self.current_details:
            return

        details_panel = self.query_one("#details-panel")
        node_type = self.current_details.get("type", "")
        content_data = self.current_details.get("content", {})

        renderable_text = Text()

        if node_type == "sql":
            if "sql_query" in content_data:
                sql_query = content_data["sql_query"]
                renderable_text.append("# SQL Context\n\n")
                renderable_text.append(f"```sql\n{sql_query}\n```\n\n")

                for key, value in content_data.items():
                    if key != "sql_query":
                        renderable_text.append(key, style="bold")
                        renderable_text.append(": ")
                        renderable_text.append(str(value))
                        renderable_text.append("\n")
            details_panel.update(renderable_text)

        elif node_type == "schema":
            table_name = content_data.get("table_name", "Unknown Table")
            renderable_text.append(f"# Table Schema: {table_name}\n\n")
            if "columns" in content_data and isinstance(content_data["columns"], list):
                renderable_text.append("## Columns\n\n")
                for col in content_data["columns"]:
                    col_name = col.get("name", "Unknown")
                    col_type = col.get("type", "Unknown")
                    nullable = "NULL" if col.get("nullable", False) else "NOT NULL"
                    renderable_text.append("- ")
                    renderable_text.append(col_name, style="bold")
                    renderable_text.append(f" ({col_type}) {nullable}\n")
            details_panel.update(renderable_text)

        elif node_type == "schema_values":
            table_name = content_data.get("table_name", "Unknown Table")
            renderable_text.append(f"# Table Values: {table_name}\n\n")
            if "sample_data" in content_data:
                renderable_text.append("## Sample Data\n\n")
                renderable_text.append(str(content_data["sample_data"]))
            details_panel.update(renderable_text)

        elif node_type == "metrics":
            metric_name = content_data.get("name", "Unknown Metric")
            renderable_text.append(f"# Metric: {metric_name}\n\n")
            for key, value in content_data.items():
                renderable_text.append(key, style="bold")
                renderable_text.append(": ")
                renderable_text.append(str(value))
                renderable_text.append("\n")
            details_panel.update(renderable_text)

    def action_exit(self):
        """Exit the screen."""
        # self.app.pop_screen()
        self.app.exit()

    def action_cursor_down(self):
        """Move cursor down in the tree."""
        tree = self.query_one("#context-tree")
        tree.action_cursor_down()
        # if self.selected_zone == "node-list":
        #    pass
        # elif self.selected_zone == "node-details":
        #    pass
        # elif self.selected_zone == "context":
        #    pass

    def action_cursor_up(self):
        """Move cursor up in the tree."""
        tree = self.query_one("#context-tree")
        tree.action_cursor_up()

    def action_left(self):
        """Collapse the current node."""
        tree = self.query_one("#context-tree")
        if tree.cursor_node and tree.cursor_node.is_expanded:
            tree.cursor_node.collapse()

    def action_right(self):
        """Expand the current node."""
        tree = self.query_one("#context-tree")
        if tree.cursor_node and not tree.cursor_node.is_expanded and tree.cursor_node.children:
            tree.cursor_node.expand()

    def action_toggle_node(self):
        """Toggle the current node."""
        tree = self.query_one("#context-tree")
        if tree.cursor_node:
            if tree.cursor_node.children:
                tree.cursor_node.toggle()

    def action_select_node(self):
        """Select the current node."""
        tree = self.query_one("#context-tree")
        if tree.cursor_node:
            tree.select_node(tree.cursor_node)

    # ToDo: move set context table schema, sqlcontext, tablevalues, metrics into this screen
    def action_edit(self):
        """Edit the table schema."""
