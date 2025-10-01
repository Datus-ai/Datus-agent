import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static
from textual.widgets import Tree as TextualTree
from textual.widgets._tree import TreeNode
from textual.worker import get_current_worker

from datus.cli.screen.context_screen import ContextScreen
from datus.storage.lancedb_conditions import and_, eq
from datus.storage.metric.store import SemanticModelStorage
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=100)
def _fetch_schema_with_cache(
    db_connector: BaseSqlConnector,
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
) -> list:
    """Fetch schema with LRU cache support."""
    try:
        return (
            db_connector.get_schema(
                catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_name=table_name
            )
            or []
        )
    except Exception as e:
        logger.error(f"Schema fetch failed: {str(e)}")
        return []


class CatalogScreen(ContextScreen):
    """Screen for displaying database catalogs."""

    CSS = """
        /* Main layout containers */
        #tree-container {
            width: 35%;
            height: 100%;
            background: $surface;
            overflow: hidden;
        }

        #details-container {
            height: 100%;
            background: $surface-lighten-1;
            overflow: hidden;
        }

        #columns-panel-container {
            width: 100%;
            height: 65%;
            background: $surface;
            color: $text;
            overflow-y: auto;
            overflow-x: hidden;
        }

        #semantic-panel-container {
            width: 100%;
            height: 35%;
            background: $surface;
            color: $text;
            overflow-y: auto;
            overflow-x: hidden;
        }

        #semantic-model-panel {
            width: 100%;
            padding: 1 1;
        }

        #panel-divider {
            height: 1;
            background: $surface-darken-1;
            margin: 0;
        }

        /* Tree styling - minimal borders */
        #catalogs-tree {
            width: 100%;
            height: 1fr;
            background: $surface;
            border: none;
            overflow-y: auto;
        }

        #catalogs-tree > .tree--guides {
            color: $primary-lighten-2;
        }

        #catalogs-tree:focus {
            border: none;
        }

        /* Loading states */
        .loading {
            color: $text-muted;
            text-style: italic;
        }

        .loading-spinner {
            color: $accent;
            text-style: italic;
        }

        .error {
            color: $error;
            text-style: bold;
        }

        /* Enhanced loading indicator */
        .loading-pulse {
            color: $accent;
            text-style: italic;
        }

        /* Table styling enhancements */
        #columns-panel {
            width: 100%;
            height: auto;
            background: $surface;
            color: $text;
            padding: 1 1;
        }

        /* Fullscreen mode */
        .fullscreen #tree-container {
            width: 100%;
        }

        .fullscreen #details-container {
            width: 100%;
        }

        /* Tree node styling */
        .tree--cursor {
            background: $accent-darken-1;
            color: $text;
        }

        .tree--highlighted {
            background: $accent-lighten-1;
            color: $text;
        }

        #tree-help {
            width: 100%;
            height: 1;
            background: $surface-darken-1;
            color: $text-muted;
            text-align: center;
        }
    """

    BINDINGS = [
        Binding("f1", "toggle_fullscreen", "Fullscreen"),
        Binding("f2", "show_navigation_help", "Help"),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("right", "expand_node", "Expand", show=False),
        Binding("left", "collapse_node", "Collapse", show=False),
        # Binding("enter", "load_details", "Load Details", show=False),
        # Binding("f3", "preview_details", "Preview"),
        Binding("f4", "show_path", "Show Path"),
        Binding("f5", "exit_with_selection", "Select"),
        Binding("escape", "exit_without_selection", "Exit", show=False),
        Binding("r", "retry_current_node", "Retry", show=False),
    ]

    def __init__(self, title: str, context_data: Dict, inject_callback=None):
        """
        Initialize the catalogs screen.

        Args:
            context_data: Dictionary containing database connection info
                - namespace: Database namespace
                - db_type: Database type
                - database_name: Specific database name (optional)
            inject_callback: Callback for injecting data into the CLI
        """
        super().__init__(title=title, context_data=context_data, inject_callback=inject_callback)
        self.db_type: DBType = context_data.get("db_type", DBType.SQLITE)
        self.catalog_name = context_data.get("catalog_name", "")
        self.database_name = context_data.get("database_name", "")
        self.inject_callback = inject_callback
        self.selected_path = ""
        self.selected_data = {}
        self.tree_data = {}
        self.current_node_data = None
        self.is_fullscreen = False
        self.db_connector: BaseSqlConnector = context_data.get("db_connector")
        rag = context_data.get("rag")
        semantic_storage = context_data.get("semantic_model_storage")
        if not semantic_storage and rag is not None:
            semantic_storage = getattr(rag, "semantic_model_storage", None)
        self.semantic_model_storage: Optional[SemanticModelStorage] = semantic_storage
        self.loading_nodes = set()  # Track which nodes are currently loading
        self._current_loading_task = None  # Track current async task
        self.timeout_seconds = context_data.get("timeout_seconds", 30)  # Default 30 seconds timeout

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="Catalogs")

        with Horizontal():
            # Left side: Catalog tree
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("", id="tree-help")
                yield TextualTree(label="Database Catalogs", id="catalogs-tree")

            # Right side: Details panel with split layout
            with Vertical(id="details-container", classes="details-panel"):
                # Upper section: Columns
                yield ScrollableContainer(
                    Static(
                        "Select a table and press Enter to view columns",
                        id="columns-panel",
                    ),
                    id="columns-panel-container",
                )

                # Divider between panels
                yield Static(id="panel-divider")

                # Lower section: Semantic model details
                yield ScrollableContainer(
                    Static(
                        "Select a table to view semantic model information",
                        id="semantic-model-panel",
                    ),
                    id="semantic-panel-container",
                )

        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._build_catalog_tree()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter" or event.key == "right":
            self.action_load_details()
        elif event.key == "escape":
            self.action_exit_without_selection()
        else:
            await super()._on_key(event)

    def on_unmount(self):
        self.clear_cache()

    def _build_catalog_tree(self) -> None:
        """Load catalog data from database connectors and populate the tree with lazy loading."""
        try:
            tree = self.query_one("#catalogs-tree", TextualTree)
            tree.root.expand()
            if not self.db_connector:
                self.query_one("#semantic-model-panel", Static).update("[red]Error:[/] No database connection selected")
                return

            # Clear existing tree
            tree.clear()

            # Show loading state
            tree_helper = self.query_one("#tree-help", Static)
            tree_helper.update("[dim]Loading database structure...[/dim]")

            # Get top-level items based on database type - only load first level
            if self.db_type == DBType.SQLITE:
                # SQLite: show database node with lazy loading for tables
                db_node = tree.root.add(self.database_name, data={"type": "database", "name": self.database_name})
                db_node.add_leaf("📁 Loading tables...", data={"type": "loading"})

            elif self.db_type == DBType.MYSQL:
                # MySQL: show databases with lazy loading for tables
                self._load_databases_lazy(tree)
            elif self.db_type == DBType.DUCKDB:
                self._add_db_name(tree, self.database_name)
            elif self.db_type in [DBType.POSTGRES, DBType.POSTGRESQL]:
                # DuckDB/PostgreSQL: show databases with lazy loading for schemas
                self._load_databases_lazy(tree)

            elif self.db_type == DBType.SNOWFLAKE:
                # Snowflake: show databases with lazy loading for schemas
                self._load_databases_lazy(tree)

            elif self.db_type == DBType.STARROCKS:
                # StarRocks: show catalogs with lazy loading for databases
                self._load_catalogs_lazy(tree)

            else:
                # Generic: show databases with lazy loading for tables
                self._load_databases_lazy(tree)

            # Clear loading message
            tree_helper.update("")

        except Exception as e:
            logger.error(f"Failed to load catalog data: {str(e)}")
            self.query_one("#tree-help", Static).update(f"[red]Error:[/] Failed to load catalog data: {str(e)}")
            self.query_one("#semantic-model-panel", Static).update(
                f"[red]Error:[/] Failed to load catalog data: {str(e)}"
            )

    def populate_tree(self, tree: TextualTree, data: Dict) -> None:
        """Populate the tree with catalog data."""
        tree.clear()

        for first_name, first_data in data.items():
            first_node = tree.root.add(first_name)

            for second_name, second_data in first_data.items():
                if "identifier" in second_data:
                    first_node.add_leaf(second_name, data=second_data)
                    continue
                second_node = first_node.add(second_name)

                for third_name, third_data in second_data.items():
                    if "identifier" in third_data:
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

        # No automatic loading - user must press Enter or Right arrow

    def on_tree_node_highlighted(self, event: TextualTree.NodeHighlighted) -> None:
        """Handle tree node highlighting."""
        node = event.node
        self.current_node_data = node.data

        # Update path display
        self.update_path_display(node)

        # No automatic loading - user must press Enter or Right arrow

    def update_path_display(self, node: TreeNode) -> None:
        """Update the header with the current path."""
        path_parts = []
        current = node
        if node.data:
            self.selected_data = node.data

        # Build path from current node up to root
        while current and str(current.label) != "Database Catalogs":
            name = str(current.data.get("name", ""))
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
        """Show table details using database connector with async loading and performance optimization."""
        # Cancel any existing loading task
        if (
            hasattr(self, "_current_loading_task")
            and self._current_loading_task
            and not self._current_loading_task.is_finished
        ):
            self._current_loading_task.cancel()

        # Start async loading
        self._current_loading_task = self.load_table_details_async(table_info)

    @work(thread=True)
    async def load_table_details_async(self, table_info: Dict[str, Any]) -> None:
        """Async worker to load table details and semantic model without blocking UI."""
        get_current_worker()  # Get worker context

        semantic_panel = self.query_one("#semantic-model-panel", Static)
        columns_panel = self.query_one("#columns-panel", Static)

        try:
            if not self.db_connector:
                self.app.call_from_thread(semantic_panel.update, "[red]No database connection available[/red]")
                self.app.call_from_thread(columns_panel.update, "")
                return

            # Extract identifiers for downstream lookups
            catalog_name = table_info.get("catalog_name", "")
            database_name = table_info.get("database_name", "")
            schema_name = table_info.get("schema_name", "")
            table_name = table_info.get("name", "")
            full_name = self.db_connector.full_name(catalog_name, database_name, schema_name, table_name)

            # Show loading state
            self.app.call_from_thread(
                semantic_panel.update,
                f"[bold cyan]⏳ Loading[/bold cyan] [yellow]{full_name}[/yellow]\n\n"
                f"[dim]Fetching semantic model information...[/dim]",
            )
            self.app.call_from_thread(
                columns_panel.update,
                "[bold cyan]⏳ Loading[/bold cyan] [yellow]Columns[/yellow]\n\n[dim]Fetching column schema...[/dim]",
            )

            # Load schema information (cached)
            columns_renderable: Any

            table_schema = self._get_cached_schema(
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_name=table_name,
            )

            if len(table_schema) == 0:
                columns_renderable = f"[yellow]No schema information available for {full_name}[/yellow]"
            else:
                columns_renderable = self._create_optimized_table(table_schema, full_name)

            self.app.call_from_thread(columns_panel.update, columns_renderable)

            # Load semantic model details, if storage is available
            semantic_records: List[Dict[str, Any]] = []
            semantic_message: Optional[str] = None
            semantic_message_style = "dim"

            if not self.semantic_model_storage:
                semantic_message = "Semantic model storage is not configured."
            else:
                try:
                    semantic_records = self._fetch_semantic_model_record(
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                        table_name=table_name,
                    )
                    if not semantic_records:
                        semantic_message = "No semantic model found for this table."
                except Exception as storage_error:  # pragma: no cover - defensive logging
                    semantic_message = f"Failed to load semantic model: {storage_error}"
                    semantic_message_style = "red"
                    logger.error(
                        (
                            f"Failed to load semantic model: catalog_name={catalog_name}, "
                            f"database_name={database_name}, schema_name={schema_name}, table_name={table_name}, "
                            f"error_msg = {storage_error}"
                        )
                    )

            semantic_renderable = self._build_semantic_panel_content(
                semantic_records=semantic_records,
                semantic_message=semantic_message,
                semantic_message_style=semantic_message_style,
            )

            self.app.call_from_thread(semantic_panel.update, semantic_renderable)

        except Exception as e:  # pragma: no cover - defensive logging for UI thread
            logger.error(f"Failed to load table details: {str(e)}")
            error_msg = str(e)
            message = (
                (
                    "[yellow]⏱️ Timeout:[/] Failed to load table details (press 'r' to retry)\n\n[dim]"
                    f"Error: {error_msg}[/dim]"
                )
                if "timeout" in error_msg.lower()
                else f"[red]❌ Error:[/] Failed to display table details: {error_msg}"
            )
            self.app.call_from_thread(semantic_panel.update, message)
            self.app.call_from_thread(columns_panel.update, "")

    def action_load_details(self) -> None:
        """Load table details when Enter or Right arrow is pressed on a table."""
        tree = self.query_one("#catalogs-tree", TextualTree)
        if tree.cursor_node is None:
            return

        node = tree.cursor_node
        if node.data and node.data.get("type") == "table":
            self.show_table_details(node.data)
        else:
            # For non-table nodes, expand/collapse them
            if node.is_expanded:
                node.collapse()
            else:
                node.expand()

    def _get_cached_schema(self, catalog_name: str, database_name: str, schema_name: str, table_name: str) -> list:
        """Get cached schema data using LRU cache."""
        return _fetch_schema_with_cache(self.db_connector, catalog_name, database_name, schema_name, table_name)

    def _create_optimized_table(self, table_schema: list, full_name: str) -> Table:
        """Create optimized Rich table with performance considerations."""
        # Use simpler styling for performance
        table = Table(
            title=f"[bold cyan]📊 {full_name}[/bold cyan]",
            show_header=True,
            box=box.SIMPLE,
            border_style="blue",
            header_style="bold cyan",
            expand=True,
            padding=(0, 1),
        )

        table.add_column("#", style="dim", width=3, justify="left")
        table.add_column("Column", style="bright_cyan", min_width=15, max_width=25)
        table.add_column("Type", style="bright_magenta", min_width=8, max_width=15)
        table.add_column("Null", style="yellow", width=3, justify="left")
        table.add_column("Default", style="green", min_width=5, max_width=12)
        table.add_column("PK", style="red", width=2, justify="left")

        # Batch add rows for better performance
        # max_columns = min(len(table_schema), 100)  # Limit for very large schemas
        for idx, column in enumerate(table_schema, 1):
            if not isinstance(column, dict):
                continue

            # Skip invalid columns with no name or 'unknown' name
            col_name = str(column.get("name", "")).strip()
            if not col_name or col_name.lower() == "unknown":
                continue

            col_type = str(column.get("type", "Unknown"))[:15]
            nullable = "✓" if column.get("nullable", True) else "✗"
            default = str(column.get("default_value", ""))[:12] or "-"
            is_key = "✓" if column.get("pk", False) else "-"

            table.add_row(str(idx), col_name, col_type, nullable, default, is_key)

        # Add note if truncated
        # if len(table_schema) > max_columns:
        #     table.add_row("...", f"+{len(table_schema) - max_columns} more", "", "", "", "")

        return table

    def _fetch_semantic_model_record(
        self,
        *,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Fetch all semantic model records for the given table identifiers."""
        if not self.semantic_model_storage:
            return []

        results = self.semantic_model_storage._search_all(
            where=and_(
                eq("catalog_name", catalog_name or ""),
                eq("database_name", database_name or ""),
                eq("schema_name", schema_name or ""),
                eq("table_name", table_name or ""),
            ),
            select_fields=[
                "semantic_model_name",
                "domain",
                "layer1",
                "layer2",
                "semantic_model_desc",
                "identifiers",
                "dimensions",
                "measures",
                "semantic_file_path",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
            ],
        )
        if results is None or results.num_rows == 0:
            return []

        try:
            return results.to_pylist()
        except AttributeError:
            return []

    def _build_semantic_panel_content(
        self,
        *,
        semantic_records: List[Dict[str, Any]],
        semantic_message: Optional[str],
        semantic_message_style: str,
    ) -> Group:
        """Build the semantic model panel content."""
        from rich import box

        sections: List[Table] = []

        if semantic_records:
            total = len(semantic_records)
            for idx, record in enumerate(semantic_records, 1):
                table = Table(
                    title=(
                        f"[bold cyan]📋 Semantic Model #{idx} of {total}: "
                        f"{record.get('semantic_model_name', 'Unnamed')}[/]"
                    ),
                    show_header=False,
                    box=box.SIMPLE,
                    border_style="blue",
                    expand=True,
                    padding=(0, 1),
                )

                table.add_column("Key", style="bright_cyan", ratio=1)
                table.add_column("Value", style="yellow", justify="left", ratio=3, no_wrap=False)

                table.add_row("Semantic Model Name", record.get("semantic_model_name", "") or "[dim]N/A[/dim]")
                table.add_row("Domain", record.get("domain", "") or "[dim]N/A[/dim]")
                table.add_row("Layer1", record.get("layer1", "") or "[dim]N/A[/dim]")
                table.add_row("Layer2", record.get("layer2", "") or "[dim]N/A[/dim]")
                table.add_row(
                    "Semantic File",
                    record.get("semantic_file_path", "") or "[dim]N/A[/dim]",
                )
                table.add_row("Description", record.get("semantic_model_desc", "") or "[dim]N/A[/dim]")
                table.add_row("Identifiers", self._create_nested_table_for_json(record.get("identifiers")))
                table.add_row("Dimensions", self._create_nested_table_for_json(record.get("dimensions")))
                table.add_row("Measures", self._create_nested_table_for_json(record.get("measures")))

                sections.append(table)
        else:
            table = Table(
                title="[bold cyan]📋 Semantic Models[/bold cyan]",
                show_header=False,
                box=box.SIMPLE,
                border_style="blue",
                expand=True,
                padding=(0, 1),
            )
            table.add_column("Semantic Model", style="bright_cyan")
            table.add_row(
                Text(
                    semantic_message or "No semantic model information available for this table.",
                    style=semantic_message_style or "dim",
                )
            )
            sections.append(table)

        return Group(*sections)

    def _create_nested_table_for_json(self, field_value: Any) -> Any:
        """Create nested table for JSON field values."""
        from rich import box

        if not field_value or field_value == "N/A":
            return "[dim]N/A[/dim]"

        # Parse JSON string if needed
        parsed_data = field_value
        if isinstance(field_value, str):
            try:
                parsed_data = json.loads(field_value)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, return as simple colored text
                if field_value.startswith("[") and field_value.endswith("]"):
                    return f"[bright_green]{field_value}[/bright_green]"
                if field_value.startswith("{") and field_value.endswith("}"):
                    return f"[bright_blue]{field_value}[/bright_blue]"
                return str(field_value)

        # Create nested table based on data type
        if isinstance(parsed_data, list):
            if not parsed_data:
                return "[dim]Empty list[/dim]"

            nested_table = Table(show_header=True, box=box.ROUNDED, border_style="dim", padding=(0, 0), expand=True)
            if parsed_data and isinstance(parsed_data[0], dict):
                for key in parsed_data[0].keys():
                    nested_table.add_column(str(key), style="dim cyan", justify="center")

                for item in parsed_data:
                    values = [str(value) for value in item.values()]
                    nested_table.add_row(*values)
            else:
                nested_table.add_column("Value", style="dim cyan")
                for item in parsed_data:
                    nested_table.add_row(str(item))

            return nested_table

        if isinstance(parsed_data, dict):
            if not parsed_data:
                return "[dim]Empty object[/dim]"

            nested_table = Table(show_header=True, box=box.ROUNDED, border_style="dim", padding=(0, 1), expand=True)
            nested_table.add_column("Property", style="bright_cyan", width=15)
            nested_table.add_column("Value", style="bright_yellow")

            for key, value in parsed_data.items():
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=1, ensure_ascii=False)
                else:
                    value_str = str(value)
                nested_table.add_row(str(key), value_str)

            return nested_table

        # For simple values, return as formatted string
        return f"[bright_white]{str(parsed_data)}[/bright_white]"

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
            details_container.styles.width = "0%"
            tree_container.add_class("fullscreen")
            details_container.remove_class("fullscreen")
        else:
            tree_container.styles.width = "50%"
            details_container.styles.width = "50%"
            tree_container.remove_class("fullscreen")
            details_container.add_class("fullscreen")

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

    def action_retry_current_node(self) -> None:
        """Retry loading the current node if it failed with timeout."""
        tree = self.query_one("#catalogs-tree", TextualTree)
        if tree.cursor_node is None:
            return

        node = tree.cursor_node
        if node.data and node.data.get("type") in ["timeout", "error"]:
            # Find the parent node to retry loading
            parent = node.parent
            if parent:
                # Remove timeout/error node
                node.remove()
                # Retry loading the parent node
                self._retry_load_node(parent)

    def _retry_load_node(self, node: TreeNode) -> None:
        """Retry loading a specific node."""
        if not node.data:
            return

        node_type = node.data.get("type")

        # Clear any existing children by removing them individually
        for child in list(node.children):
            child.remove()

        # Add loading indicator
        if node_type == "database":
            if DBType.support_schema(self.db_type):
                node.add_leaf("⏳ Loading schemas...", data={"type": "loading"})
            else:
                node.add_leaf("⏳ Loading tables...", data={"type": "loading"})
        elif node_type == "catalog":
            node.add_leaf("⏳ Loading databases...", data={"type": "loading"})
        elif node_type == "schema":
            node.add_leaf("⏳ Loading tables...", data={"type": "loading"})

        # Trigger reload
        node.expand()

    def _load_databases_lazy(self, tree: TextualTree) -> None:
        """Lazy load databases for MySQL, PostgreSQL, DuckDB, etc."""
        try:
            databases = self.db_connector.get_databases()
            if not databases:
                tree.root.add_leaf("📂 No databases found", data={"type": "empty"})
                return

            for db_name in databases:
                self._add_db_name(tree, db_name)
        except Exception as e:
            logger.error(f"Failed to load databases: {str(e)}")
            if "timeout" in str(e).lower():
                tree.root.add_leaf(
                    "⏱️ Timeout loading databases (press 'r' to retry)",
                    data={"type": "timeout", "operation": "databases"},
                )
            else:
                tree.root.add_leaf("❌ Error loading databases", data={"type": "error"})

    def _add_db_name(self, tree: TextualTree, db_name: str):
        db_node = tree.root.add(f"📁 {db_name}", data={"type": "database", "name": db_name})
        support_schema = DBType.support_schema(self.db_type)
        db_node.add_leaf(
            f"⏳ Loading {'schemas' if support_schema else 'tables'}...",
            data={"type": "loading"},
        )

    def _load_catalogs_lazy(self, tree: TextualTree) -> None:
        """Lazy load catalogs for StarRocks and similar systems."""
        try:
            catalogs = [self.catalog_name or self.db_connector.catalog_name]
            if not catalogs:
                self._load_databases_lazy(tree)
            else:
                for catalog_name in catalogs:
                    catalog_node = tree.root.add(f"📁 {catalog_name}", data={"type": "catalog", "name": catalog_name})
                    catalog_node.add_leaf("⏳ Loading databases...", data={"type": "loading"})
        except Exception as e:
            logger.error(f"Failed to load catalogs: {str(e)}")
            if "timeout" in str(e).lower():
                tree.root.add_leaf(
                    "⏱️ Timeout loading catalogs (press 'r' to retry)",
                    data={"type": "timeout", "operation": "catalogs"},
                )
            else:
                tree.root.add_leaf("❌ Error loading catalogs", data={"type": "error"})

    def on_tree_node_expanded(self, event: TextualTree.NodeExpanded) -> None:
        """Handle tree node expansion for lazy loading - optimized for speed."""
        node = event.node
        if not node.data:
            return

        node_type = node.data.get("type")

        # Check for loading placeholders
        loading_children = [child for child in node.children if child.data and child.data.get("type") == "loading"]
        for loading_child in loading_children:
            loading_child.remove()

        # Check if this node has already been loaded or is currently loading
        node_key = str(node.label)
        if node_key in self.loading_nodes:
            return

        # Skip if node already has children (except loading placeholders)
        if node.children and not loading_children:
            return

        # Mark as loading
        self.loading_nodes.add(node_key)

        try:
            if node_type == "database":
                if DBType.support_schema(self.db_type):
                    self._load_schemas_for_database(node)
                else:
                    self._load_tables_for_schema(node)
            elif node_type == "catalog":
                if DBType.support_database(self.db_type):
                    self._load_databases_for_catalog(node)
                else:
                    self._load_schemas_for_database(node)
            elif node_type == "schema":
                self._load_tables_for_schema(node)
        except Exception as e:
            logger.error(f"Error loading node {node_key}: {str(e)}")
            node.add_leaf("❌ Error loading data", data={"type": "error"})
        finally:
            # Remove from loading set
            self.loading_nodes.discard(node_key)

    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        _fetch_schema_with_cache.cache_clear()

    def _load_schemas_for_database(self, db_node: TreeNode) -> None:
        """Load schemas for a database node."""
        db_name = str(db_node.label).replace("📁 ", "")
        try:
            self.db_connector.switch_context(database_name=db_name)
            schemas = self.db_connector.get_schemas()

            if not schemas:
                db_node.add_leaf("📂 No schemas found", data={"type": "empty"})
                return

            for schema_name in schemas:
                schema_node = db_node.add(
                    f"📂 {schema_name}", data={"type": "schema", "name": schema_name, "database": db_name}
                )
                schema_node.add_leaf("⏳ Loading tables...", data={"type": "loading"})
        except DatusException as e:
            logger.error(f"Failed to load schemas for database {db_name}: {str(e)}")
            if e.code == ErrorCode.DB_EXECUTION_TIMEOUT.code:
                db_node.add_leaf(
                    "⏱️ Timeout loading schemas (press 'r' to retry)",
                    data={"type": "timeout", "operation": "schemas", "parent": db_name},
                )
            else:
                db_node.add_leaf("❌ Error loading schemas", data={"type": "error"})

    def _load_databases_for_catalog(self, catalog_node: TreeNode) -> None:
        """Load databases for a catalog node."""
        catalog_name = str(catalog_node.label).replace("📁 ", "")
        try:
            databases = self.db_connector.get_databases(catalog_name=catalog_name)

            if not databases:
                catalog_node.add_leaf("📂 No databases found", data={"type": "empty"})
                return

            for db_name in databases:
                db_node = catalog_node.add(
                    f"📁 {db_name}", data={"type": "database", "name": db_name, "catalog": catalog_name}
                )
                db_node.add_leaf("⏳ Loading schemas...", data={"type": "loading"})
        except DatusException as e:
            logger.error(f"Failed to load databases for catalog {catalog_name}: {str(e)}")
            if e.code == ErrorCode.DB_EXECUTION_TIMEOUT.code:
                catalog_node.add_leaf(
                    "⏱️ Timeout loading databases (press 'r' to retry)",
                    data={"type": "timeout", "operation": "databases", "parent": catalog_name},
                )
            else:
                catalog_node.add_leaf("❌ Error loading databases", data={"type": "error"})

    def _load_tables_for_schema(self, schema_node: TreeNode) -> None:
        """Load tables for a schema node."""
        if not DBType.support_schema(self.db_type):
            schema_name = ""
            db_name = schema_node.data.get("name")
            if not DBType.support_catalog(self.db_type):
                catalog_name = ""
            else:
                catalog_name = "" if not schema_node.parent else schema_node.parent.data.get("name")

        else:
            schema_name = str(schema_node.data.get("name", ""))
            parent = schema_node.parent
            if not parent:
                return
            if DBType.support_database(self.db_type):
                db_name = parent.data.get("name")
                if DBType.support_catalog(self.db_type):
                    parent = parent.parent
                    catalog_name = "" if not parent or not parent.data else parent.data.get("name")
                else:
                    catalog_name = ""
            else:
                db_name = ""
                catalog_name = "" if not DBType.support_catalog(self.db_type) else parent.data.get("name")

        try:
            tables = self.db_connector.get_tables(
                catalog_name=catalog_name, database_name=db_name, schema_name=schema_name
            )

            if not tables:
                schema_node.add_leaf("📂 No tables found", data={"type": "empty"})
                return

            for table_name in sorted(tables):
                table_data = {
                    "type": "table",
                    "name": table_name,
                    "table_type": "table",
                    "schema_name": schema_name,
                    "database_name": db_name,
                    "catalog_name": catalog_name,
                    "identifier": self.db_connector.identifier(catalog_name, db_name, schema_name, table_name),
                }
                schema_node.add_leaf(f"📋 {table_name}", data=table_data)
        except DatusException as e:
            logger.error(f"Failed to load tables for schema {schema_name}: {str(e)}")
            if e.code == ErrorCode.DB_EXECUTION_TIMEOUT.code:
                schema_node.add_leaf(
                    "⏱️ Timeout loading tables (press 'r' to retry)",
                    data={"type": "timeout", "operation": "tables", "parent": schema_name},
                )
            else:
                schema_node.add_leaf("❌ Error loading tables", data={"type": "error"})

    def _clear_table_details(self):
        self.query_one("#columns-panel", Static).update("Select a table and press Enter to view columns")
        self.query_one("#semantic-model-panel", Static).update("Select a table to view semantic model information")


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
                "• Enter - Load table details\n"
                "• F4 - Show path\n"
                "• F5 - Select and exit\n"
                "• R - Retry loading (on timeout/error)\n"
                "• Esc - Exit without selection\n\n"
                "Press any key to close this help.",
                id="navigation-help-content",
            ),
            id="navigation-help-container",
        )

    def on_key(self, event) -> None:
        """Close the modal on any key press."""
        self.dismiss()
