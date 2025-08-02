"""
Context screen module for Datus CLI.
Provides interactive screens for database exploration.
"""

from typing import Any, Dict

from rich import box
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Static
from textual.widgets import Tree as TextualTree
from textual.widgets._tree import TreeNode

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ContextScreen(Screen):
    """Base screen for context exploration."""

    def __init__(self, title: str, context_data: Dict, inject_callback=None):
        """
        Initialize the context screen.

        Args:
            title: Title of the screen
            context_data: Data to display in the context screen
            inject_callback: Callback for injecting data into the workflow
        """
        super().__init__()
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
                self._build_context_tree()
                yield context_tree

            # Right side: Details panel
            with Vertical(id="details-container", classes="details-panel"):
                yield Static("# Details\n\nSelect a node to view details", id="details-panel")

        yield Footer()

    def _build_context_tree(self, tree):
        """Build the context tree from the context data."""
        # Create main category nodes
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


class CatalogScreen(ContextScreen):
    """Screen for displaying database catalogs."""

    BINDINGS = [
        ("escape", "exit", "Exit"),
        ("i", "inject", "Inject"),
    ]

    def compose(self):
        """Compose the layout of the screen."""
        yield Header(show_clock=True)

        # Create a tree for the catalog data
        tree = TextualTree("Database Catalog", expand_guide_chars="├─")
        self.populate_tree(tree, self.context_data)

        yield Container(
            Static(f"# {self.title}\n\nUse arrow keys to navigate, press `i` to inject into workflow, `Esc` to exit."),
            tree,
            id="catalog-container",
        )

        yield Footer()

    def populate_tree(self, tree, data):
        """
        Populate the tree with catalog data.

        Args:
            tree: Textual tree widget
            data: Catalog data to display
        """
        # Group data by database and schema
        grouped_data = {}
        for item in data:
            db_name = item.get("database_name", "")
            schema_name = item.get("schema_name", "")
            table_name = item.get("table_name", "")

            if db_name not in grouped_data:
                grouped_data[db_name] = {}

            if schema_name not in grouped_data[db_name]:
                grouped_data[db_name][schema_name] = []

            grouped_data[db_name][schema_name].append(table_name)

        # Build the tree
        for db_name, schemas in grouped_data.items():
            db_node = tree.root.add(db_name, expand=True)
            for schema_name, tables in schemas.items():
                schema_node = db_node.add(schema_name, expand=True)
                for table_name in sorted(tables):
                    schema_node.add_leaf(
                        table_name,
                        {
                            "type": "table",
                            "name": table_name,
                            "schema": schema_name,
                            "database": db_name,
                        },
                    )

    def on_tree_node_selected(self, event):
        """Handle tree node selection."""
        node = event.node
        if node.data:
            # Display info about the selected item
            self.query_one(Static).update(
                f"# {self.title}\n\nSelected: {node.data['database']}.{node.data['schema']}.{node.data['name']}\n\n"
                "Press `i` to inject this table into the workflow."
            )

    def action_exit(self):
        """Exit the screen."""
        self.app.pop_screen()

    def action_inject(self):
        """Inject the selected item into the workflow."""
        selected_nodes = self.query_one(TextualTree).selected_nodes
        if selected_nodes and selected_nodes[0].data:
            node = selected_nodes[0]
            data = node.data

            if self.inject_callback:
                self.inject_callback(data)

            self.query_one(Static).update(
                f"# {self.title}\n\nInjected {data['database']}.{data['schema']}.{data['name']} into workflow!"
            )


class TableScreen(ContextScreen):
    """Screen for displaying table details."""

    BINDINGS = [
        ("escape", "exit", "Exit"),
        ("i", "inject", "Inject"),
    ]

    def compose(self):
        """Compose the layout of the screen."""
        yield Header(show_clock=True)

        yield Container(
            Static(
                f"#{self.title}\n\nTable details here.\n\n"
                "Use arrow keys to navigate, press `i` to inject into workflow, `Esc` to exit."
            ),
            id="table-container",
        )
        yield Footer()

    def action_exit(self):
        """Exit the screen."""
        self.app.pop_screen()

    def action_inject(self):
        """Inject the table schema into the workflow."""
        if self.inject_callback:
            self.inject_callback(self.context_data)

        self.query_one(Static).update(f"# {self.title}\n\nInjected table into workflow!")


class MetricsScreen(ContextScreen):
    """Screen for displaying metrics."""

    BINDINGS = [
        ("escape", "exit", "Exit"),
    ]

    def compose(self):
        """Compose the layout of the screen."""
        yield Header(show_clock=True)

        yield Container(
            Static(f"# {self.title}\n\nMetrics data here.\n\nPress `Esc` to exit."),
            id="metrics-container",
        )

        yield Footer()

    def action_exit(self):
        """Exit the screen."""
        self.app.pop_screen()


class NavigationHelpScreen(ModalScreen):
    """Modal screen to display navigation help."""

    def compose(self) -> ComposeResult:
        """Compose the navigation help modal."""
        yield Container(
            Static(
                "# Navigation Help\n\n"
                "## Arrow Key Navigation:\n"
                "• ↑ - Move cursor up\n"
                "• ↓ - Move cursor down\n"
                "• → - Expand current node\n"
                "• ← - Collapse current node\n\n"
                "## Other Keys:\n"
                "• F1 - Toggle fullscreen\n"
                "• F2 - Toggle this help\n"
                # "• F3 - Preview details\n"
                "• F4 - Show path\n"
                "• F5 - Select and exit\n"
                "• Esc - Exit without selection\n\n"
                "Press any key to close this help.",
                id="navigation-help-content",
            ),
            id="navigation-help-container",
        )

    def on_key(self, event) -> None:
        """Close the modal on any key press."""
        self.dismiss()


class CatalogsScreen(ContextScreen):
    """Screen for displaying database catalogs."""

    BINDINGS = [
        Binding("f1", "toggle_fullscreen", "Fullscreen"),
        Binding("f2", "show_navigation_help", "Help"),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("right", "expand_node", "Expand", show=False),
        Binding("left", "collapse_node", "Collapse", show=False),
        # Binding("f3", "preview_details", "Preview"),
        Binding("f4", "show_path", "Show Path"),
        Binding("f5", "exit_with_selection", "Select"),
        Binding("escape", "exit_without_selection", "Exit", show=False),
    ]

    def __init__(self, title: str, context_data: Dict, inject_callback=None):
        """
        Initialize the catalogs screen.

        Args:
            schema_store: SchemaStorage instance
            inject_callback: Callback for injecting data into the CLI
        """
        super().__init__(title=title, context_data=context_data, inject_callback=inject_callback)
        self.db_type = context_data["db_type"]
        self.inject_callback = inject_callback
        self.selected_path = ""
        self.selected_data = {}
        self.tree_data = {}
        self.current_node_data = None
        self.is_fullscreen = False

    class SelectPathMessage(Message):
        """Message sent when a path is selected."""

        def __init__(self, path: str):
            self.path = path
            super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="Catalogs")

        with Horizontal():
            # Left side: Catalog tree
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("", id="tree-help")
                yield TextualTree(label="Database Catalogs", id="catalogs-tree")

            # Right side: Details panel
            with Vertical(id="details-container", classes="details-panel"):
                yield Static("Details Area (F3 Preview Details)", id="details-panel")

        yield Footer()

    def on_mount(self) -> None:
        self._build_catalog_tree()

    def _build_catalog_tree(self) -> None:
        """Load catalog data from LanceDB and populate the tree."""
        try:
            tree = self.query_one("#catalogs-tree", TextualTree)
            # Load all schema metadata
            schemas = self.context_data["schemas"]  # pyarrow.Table
            # Build tree structure
            tree.root.expand()

            # Group data by catalog, database, schema, and table
            tree_data = {}

            # Determine the hierarchy levels based on db_type
            # Process data efficiently without converting entire table to list
            catalog_names = schemas["catalog_name"]
            database_names = schemas["database_name"]
            schema_names = schemas["schema_name"]
            table_names = schemas["table_name"]
            definitions = schemas["definition"]
            table_types = schemas["table_type"]
            identifiers = schemas["identifier"]

            # Iterate through rows efficiently
            for i in range(schemas.num_rows):
                schema_info = {
                    "identifier": identifiers[i].as_py(),
                    "catalog_name": catalog_names[i].as_py() or "",
                    "database_name": database_names[i].as_py() or "",
                    "schema_name": schema_names[i].as_py() or "",
                    "table_name": table_names[i].as_py() or "",
                    "definition": definitions[i].as_py(),
                    "table_type": table_types[i].as_py(),
                }

                catalog_name = schema_info["catalog_name"]
                database_name = schema_info["database_name"]
                schema_name = schema_info["schema_name"]
                table_name = schema_info["table_name"]

                # Adjust hierarchy based on database type
                if self.db_type == DBType.SQLITE:
                    # SQLite: database -> table
                    if database_name not in tree_data:
                        tree_data[database_name] = {}
                    tree_data[database_name][table_name] = schema_info
                elif self.db_type in [DBType.DUCKDB, DBType.POSTGRES, DBType.POSTGRESQL]:
                    # DuckDB/PostgreSQL: database -> schema -> table
                    if database_name not in tree_data:
                        tree_data[database_name] = {}
                    if schema_name not in tree_data[database_name]:
                        tree_data[database_name][schema_name] = {}
                    tree_data[database_name][schema_name][table_name] = schema_info
                elif self.db_type == DBType.MYSQL:
                    # MySQL: database -> table
                    if database_name not in tree_data:
                        tree_data[database_name] = {}
                    tree_data[database_name][table_name] = schema_info
                elif self.db_type == DBType.STARROCKS:
                    # StarRocks: catalog -> database -> table
                    if catalog_name not in tree_data:
                        tree_data[catalog_name] = {}
                    if database_name not in tree_data[catalog_name]:
                        tree_data[catalog_name][database_name] = {}
                    tree_data[catalog_name][database_name][table_name] = schema_info
                elif self.db_type in [DBType.MSSQL, DBType.SQLSERVER]:
                    # SQL Server: database -> schema -> table
                    if database_name not in tree_data:
                        tree_data[database_name] = {}
                    if schema_name not in tree_data[database_name]:
                        tree_data[database_name][schema_name] = {}
                    tree_data[database_name][schema_name][table_name] = schema_info
                elif self.db_type == DBType.ORACLE:
                    # Oracle: database -> schema -> table
                    if database_name not in tree_data:
                        tree_data[database_name] = {}
                    if schema_name not in tree_data[database_name]:
                        tree_data[database_name][schema_name] = {}
                    tree_data[database_name][schema_name][table_name] = schema_info
                elif self.db_type == DBType.BIGQUERY:
                    # BigQuery: catalog(project) -> dataset(schema) -> table
                    if catalog_name not in tree_data:
                        tree_data[catalog_name] = {}
                    if schema_name not in tree_data[catalog_name]:
                        tree_data[catalog_name][schema_name] = {}
                    tree_data[catalog_name][schema_name][table_name] = schema_info
                else:
                    # Snowflake: catalog -> database -> schema -> table
                    if self.db_type == DBType.SNOWFLAKE and not catalog_name:
                        catalog_name = "default"
                    # Default handling for other databases
                    if catalog_name not in tree_data:
                        tree_data[catalog_name] = {}
                    if database_name not in tree_data[catalog_name]:
                        tree_data[catalog_name][database_name] = {}
                    if schema_name not in tree_data[catalog_name][database_name]:
                        tree_data[catalog_name][database_name][schema_name] = {}
                    tree_data[catalog_name][database_name][schema_name][table_name] = schema_info

            self.tree_data = tree_data
            self.populate_tree(tree, tree_data)

        except Exception as e:
            logger.error(f"Failed to load catalog data: {str(e)}")
            details_panel = self.query_one("#details-panel", Static)
            details_panel.update(f"[bold red]Error:[/] Failed to load catalog data: {str(e)}")

    def populate_tree(self, tree: TextualTree, data: Dict) -> None:
        """Populate the tree with catalog data."""
        tree.clear()

        for first_name, first_data in data.items():
            first_node = tree.root.add(first_name)

            for second_name, second_data in first_data.items():
                if "definition" in second_data:
                    first_node.add_leaf(second_name, data=second_data)
                    continue
                second_node = first_node.add(second_name)

                for third_name, third_data in second_data.items():
                    if "definition" in third_data:
                        second_node.add_leaf(third_name, data=third_data)
                        continue
                    third_node = second_node.add(third_name)

                    for fourth_name, fourth_data in third_data.items():
                        third_node.add_leaf(fourth_name, data=fourth_data)

    def on_tree_node_selected(self, event: TextualTree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        self.current_node_data = node.data

        # Update path display
        self.update_path_display(node)

        # Show details if it's a table
        if node.data and node.data.get("table_type"):
            self.show_table_details(node.data)

    def on_tree_node_highlighted(self, event: TextualTree.NodeHighlighted) -> None:
        """Handle tree node highlighting."""
        node = event.node
        self.current_node_data = node.data

        # Update path display
        self.update_path_display(node)

        # Show details if it's a table
        if node.data and node.data.get("table_type"):
            self.show_table_details(node.data)

    def update_path_display(self, node: TreeNode) -> None:
        """Update the header with the current path."""
        path_parts = []
        current = node
        if node.data:
            self.selected_data = node.data

        # Build path from current node up to root
        while current and str(current.label) != "Database Catalogs":
            name = str(current.label)
            if name:
                path_parts.insert(0, name)
            current = current.parent

        # If we're at the root or have no data, clear the path
        if path_parts:
            self.selected_path = ".".join(path_parts)
            # Update header with only the path
            header = self.query_one(Header)
            header._name = self.selected_path
        else:
            self.selected_path = ""
            header = self.query_one(Header)
            header._name = "Catalogs"

    def show_table_details(self, table_info: Dict[str, Any]) -> None:
        """Show table details in the details panel."""
        try:
            # Parse the table definition to extract column information
            definition = table_info.get("definition", "")
            details_panel = self.query_one("#details-panel", Static)
            if not definition:
                details_panel.update("No details available")
                return
            table = Table(title="Catalogs Detail", show_header=False, box=box.ROUNDED)
            table.add_column("Property", justify="right", style="cyan", no_wrap=True)
            table.add_column("Value", justify="left", style="cyan", no_wrap=False)
            # Add all available properties dynamically
            property_labels = {
                "identifier": "Identifier",
                "catalog_name": "Catalog",
                "database_name": "Database",
                "schema_name": "Schema",
                "table_name": "Table",
                "table_type": "Type",
                "definition": "Definition",
            }

            # Add properties to the table if they exist
            for key, label in property_labels.items():
                value = table_info.get(key, "")
                if not value:
                    continue
                if key != "definition":
                    col_value = str(value)
                else:
                    col_value = Syntax(
                        value,
                        "sql",
                        theme="dracula",
                        line_numbers=True,
                        word_wrap=True,
                        background_color="default",
                    )
                table.add_row(label, col_value)
            details_panel.update(table)

        except Exception as e:
            logger.error(f"Failed to show table details: {str(e)}")
            details_panel = self.query_one("#details-panel", Static)
            details_panel.update(f"[bold red]Error:[/] Failed to display table details: {str(e)}")

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        tree = self.query_one("#catalogs-tree", TextualTree)
        tree.action_cursor_down()
        self.clear_header()

    def clear_header(self):
        self.query_one("#tree-help", Static).update("")

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        tree = self.query_one("#catalogs-tree", TextualTree)
        tree.action_cursor_up()
        self.clear_header()

    def action_expand_node(self) -> None:
        """Expand the current node."""
        tree = self.query_one("#catalogs-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        """Collapse the current node."""
        tree = self.query_one("#catalogs-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.collapse()

    def action_show_navigation_help(self) -> None:
        """Toggle navigation help popup."""
        # Check if navigation help is currently the top screen
        current_screen = self.app.screen_stack[-1] if self.app.screen_stack else None

        if isinstance(current_screen, NavigationHelpScreen):
            # Close the navigation help if it's already open
            self.app.pop_screen()
        else:
            # Show navigation help
            self.app.push_screen(NavigationHelpScreen())

    def action_toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        self.is_fullscreen = not self.is_fullscreen
        tree_container = self.query_one("#tree-container")
        details_container = self.query_one("#details-container")

        if self.is_fullscreen:
            tree_container.styles.width = "100%"
            details_container.styles.width = "100%"
        else:
            tree_container.styles.width = "50%"
            details_container.styles.width = "50%"

    def action_preview_details(self) -> None:
        """Preview details of the selected node."""
        if self.current_node_data and self.current_node_data.get("type") == "table":
            self.show_table_details(self.current_node_data)

    def action_show_path(self) -> None:
        """Show the current selected full path in details panel."""
        if self.selected_path:
            tree_header = self.query_one("#tree-help", Static)
            tree_header.update(f"Selected Path: {self.selected_path}")

    def action_exit_with_selection(self) -> None:
        """Exit screen and send selected path to CLI."""
        if self.selected_path and self.inject_callback:
            # Send the selected path to the CLI
            self.inject_callback(self.selected_path, self.selected_data)
        # Exit the screen
        self.app.exit()

    def action_exit_without_selection(self) -> None:
        """Exit screen without selection and clear selected path."""
        # Clear the selected path when ESC is pressed
        self.selected_path = ""
        self.selected_data = {}
        # Exit the screen
        self.app.exit()
