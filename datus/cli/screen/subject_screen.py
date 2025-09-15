import json
from functools import lru_cache
from typing import Any, Dict

from rich.syntax import Syntax
from rich.table import Table
from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Label, Static
from textual.widgets import Tree as TextualTree
from textual.widgets._tree import TreeNode
from textual.worker import get_current_worker

from datus.cli.screen.context_screen import ContextScreen
from datus.storage.metric.store import SemanticMetricsRAG
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=100)
def _fetch_metrics_with_cache(
    rag: SemanticMetricsRAG, domain: str = "", layer1: str = "", layer2: str = "", semantic_model_name: str = ""
) -> list:
    """Fetch schema with LRU cache support."""
    try:
        # search from semantic_mode
        # group by domain layer1 layer2 name
        logger.info(f"Fetching schema with cache for{domain}, {layer1}, {layer2} ,{semantic_model_name}")
        return rag.get_metrics(
            domain,
            layer1,
            layer2,
            semantic_model_name,
            selected_fields=["name", "description", "constraint", "sql_query"],
        )
    except Exception as e:
        logger.error(f"Metrics fetch failed: {str(e)}")
        return []


class SubjectScreen(ContextScreen):
    """Screen for displaying database catalogs."""

    CSS = """
        /* Main layout containers */
        #tree-container {
            height: 100%;
            background: $surface;
            # overflow-y: scroll
        }

        #details-container {
            width: 50%;
            height: 100%;
            background: $surface-lighten-1;
            overflow-y: auto;
            overflow-x: hidden;
        }

        /* Tree styling - minimal borders */
        #semantic-tree {
            width: 100%;
            background: $surface;
            border: none;
            # padding: 1;
        }

        #semantic-tree > .tree--guides {
            color: $primary-lighten-2;
        }

        #semantic-tree:focus {
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
        #properties-container {
            background: $surface;
            height: 40%;
            width: 100%;
        }
        #properties-panel {
            background: $surface;
            color: $text;
            height: auto;
            overflow-y: auto;
        }

        #metrics-panel {
            width: 100%;
            height: auto;
            background: $surface;
            color: $text;
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
        Binding("f4", "show_path", "Show Path"),
        Binding("f5", "exit_with_selection", "Select"),
        Binding("escape", "exit_without_selection", "Exit", show=False),
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
        self.rag: SemanticMetricsRAG = context_data.get("rag")
        self.database_name = context_data.get("database_name")
        self.inject_callback = inject_callback
        self.selected_path = ""
        self.selected_data = {}
        self.tree_data = {}
        self.current_node_data = None
        self.is_fullscreen = False
        self.loading_nodes = set()  # Track which nodes are currently loading
        self._current_loading_task = None  # Track current async task

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="Semantic Models")

        with Horizontal():
            # Left side: Catalog tree
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("[dim]Loading Semantic Models...[/dim]", id="tree-help")
                yield TextualTree(label="Semantic Models", id="semantic-tree")

            # Right side: Details panel with split layout
            with Vertical(id="details-container", classes="details-panel"):
                # Upper section: Semantic model properties
                yield ScrollableContainer(
                    Static("Select a semantic model and press Enter to view details", id="properties-panel"),
                    id="properties-container",
                )

                # Lower section: Metrics
                yield ScrollableContainer(Label("", id="metrics-panel"))

        yield Footer()

    async def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._build_tree()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter" or event.key == "right":
            self.action_load_details()
        elif event.key == "escape":
            self.action_exit_without_selection()
        else:
            super()._on_key(event)

    def on_unmount(self):
        self.clear_cache()

    def _build_tree(self) -> None:
        """Load catalog data from database connectors and populate the tree with lazy loading."""
        try:
            tree = self.query_one("#semantic-tree", TextualTree)
            tree.root.expand()
            # Clear existing tree
            tree.clear()
            self.run_worker(self._load_semantics_lazy, thread=True)

        except Exception as e:
            logger.error(f"Failed to load catalog data: {str(e)}")
            self.query_one("#tree-help", Static).update(f"[red]Error:[/] Failed to load catalog data: {str(e)}")
            self.query_one("#properties-panel", Static).update(f"[red]Error:[/] Failed to load catalog data: {str(e)}")

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
        while current and str(current.label) != "Semantic Models":
            name = str(current.data.get("name", "")) if current.data else str(current.label)
            # Remove icons from the name
            name = name.replace("📁 ", "").replace("📂 ", "").replace("📋 ", "")
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
            header._name = "Semantic Models"

    def show_metrics(self, semantic_info: Dict[str, Any]) -> None:
        """Show semantic model details and metrics with async loading and performance optimization."""
        # Cancel any existing loading task
        if (
            hasattr(self, "_current_loading_task")
            and self._current_loading_task
            and not self._current_loading_task.is_finished
        ):
            self._current_loading_task.cancel()

        # Start async loading
        self._current_loading_task = self.load_semantic_details_async(semantic_info)

    @work(thread=True)
    async def load_semantic_details_async(self, semantic_info: Dict[str, Any]) -> None:
        """Async worker to load semantic model details and metrics without blocking UI."""
        get_current_worker()  # Get worker context

        try:
            properties_panel = self.query_one("#properties-panel", Static)
            metrics_panel = self.query_one("#metrics-panel", Label)

            if not self.rag:
                self.app.call_from_thread(properties_panel.update, "[red]No RAG system available[/red]")
                self.app.call_from_thread(metrics_panel.update, "")
                return

            # Clear previous details and show loading animation
            domain = semantic_info.get("domain", "")
            layer1 = semantic_info.get("layer1", "")
            layer2 = semantic_info.get("layer2", "")
            semantic_model_name = semantic_info.get("name", "")
            semantic_details = semantic_info.get("details", {})

            # Show loading state
            self.app.call_from_thread(
                properties_panel.update,
                f"[bold cyan]⏳ Loading[/bold cyan] [yellow]{semantic_model_name}[/yellow]\n\n"
                f"[dim]Fetching semantic model properties...[/dim]",
            )
            self.app.call_from_thread(
                metrics_panel.update,
                "[bold cyan]⏳ Loading[/bold cyan] [yellow]Metrics[/yellow]\n\n[dim]Fetching available metrics...[/dim]",
            )

            # Build properties display efficiently
            properties_table = self._build_properties_content(semantic_details)
            self.app.call_from_thread(properties_panel.update, properties_table)

            # Get metrics for this semantic model
            metrics = self._get_cached_metrics(domain, layer1, layer2, semantic_model_name)

            # Build metrics display
            if not metrics:
                self.app.call_from_thread(metrics_panel.update, "[dim]No metrics found for this semantic model[/dim]")
                return

            # Use optimized table rendering for metrics
            metrics_table = self._create_metrics_table(metrics, semantic_model_name)
            self.app.call_from_thread(metrics_panel.update, metrics_table)

        except Exception as e:
            logger.error(f"Failed to load semantic details: {str(e)}")
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                self.app.call_from_thread(
                    self.query_one("#properties-panel", Static).update,
                    f"[yellow]⏱️ Error:[/] Failed to load Metrics\n\n" f"[dim]Error: {error_msg}[/dim]",
                )
            else:
                self.app.call_from_thread(
                    self.query_one("#properties-panel", Static).update,
                    f"[red]❌ Error:[/] Failed to display semantic details: {error_msg}",
                )
            self.app.call_from_thread(self.query_one("#metrics-panel", Static).update, "")

    def action_load_details(self) -> None:
        """Load semantic model details when Enter or Right arrow is pressed."""
        tree = self.query_one("#semantic-tree", TextualTree)
        if tree.cursor_node is None:
            return

        node = tree.cursor_node
        if node.data and node.data.get("type") == "semantic_model":
            self.show_metrics(node.data)
        else:
            # For non-semantic model nodes, expand/collapse them
            if node.is_expanded:
                node.collapse()
            else:
                node.expand()

    def _get_cached_metrics(self, domain: str, layer1: str, layer2: str, semantic_model_name: str) -> list:
        """Get cached metrics data using LRU cache."""
        return _fetch_metrics_with_cache(self.rag, domain, layer1, layer2, semantic_model_name)

    def _create_nested_table_for_json(self, field_value: Any, title: str = "") -> Any:
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
                elif field_value.startswith("{") and field_value.endswith("}"):
                    return f"[bright_blue]{field_value}[/bright_blue]"
                else:
                    return str(field_value)

        # Create nested table based on data type
        if isinstance(parsed_data, list):
            if not parsed_data:
                return "[dim]Empty list[/dim]"

            # Create table for list items
            nested_table = Table(show_header=True, box=box.ROUNDED, border_style="dim", padding=(0, 0), expand=True)
            if parsed_data and isinstance(parsed_data[0], dict):
                for k in parsed_data[0].keys():
                    nested_table.add_column(k, style="dim cyan", justify="center")

                for item in parsed_data:
                    values = [str(v) for v in item.values()]
                    nested_table.add_row(*values)
            else:
                nested_table.add_column("Value", style="dim cyan")
                for item in parsed_data:
                    nested_table.add_row(str(item))

            return nested_table

        elif isinstance(parsed_data, dict):
            if not parsed_data:
                return "[dim]Empty object[/dim]"

            # Create table for dict key-value pairs
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

        else:
            # For simple values, return as formatted string
            return f"[bright_white]{str(parsed_data)}[/bright_white]"

    def _build_properties_content(
        self,
        semantic_details: Dict[str, Any],
    ) -> Table:
        """Build semantic model properties display as a Rich table."""
        from rich import box

        table = Table(
            title="[bold cyan]📋 Semantic Model Properties[/bold cyan]",
            show_header=False,
            box=box.SIMPLE,
            border_style="blue",
            expand=True,
            padding=(0, 1),
        )

        table.add_column("Key", style="bright_cyan", ratio=1)
        table.add_column("Value", style="yellow", justify="left", ratio=3, no_wrap=False)

        # Add basic properties
        table.add_row("Semantic Model Name", semantic_details.get("semantic_model_name", "N/A"))
        table.add_row("Domain", semantic_details.get("domain", "N/A"))
        table.add_row("Layer1", semantic_details.get("layer1", "N/A"))
        table.add_row("Layer2", semantic_details.get("layer2", "N/A"))
        table.add_row("Catalog Name", semantic_details.get("catalog_name", "") or "[dim]N/A[/dim]")
        table.add_row("Database Name", semantic_details.get("database_name", "") or "[dim]N/A[/dim]")
        table.add_row("Schema Name", semantic_details.get("schema_name", "") or "[dim]N/A[/dim]")
        table.add_row("Table Name", semantic_details.get("table_name", "") or "[dim]N/A[/dim]")
        table.add_row("Semantic File", semantic_details.get("semantic_file_path", "N/A"))

        # Create nested tables for JSON fields
        dimensions_table = self._create_nested_table_for_json(semantic_details.get("dimensions"), "Dimensions")
        measures_table = self._create_nested_table_for_json(semantic_details.get("measures"), "Measures")
        identifiers_table = self._create_nested_table_for_json(semantic_details.get("identifiers"), "Identifiers")

        table.add_row("Dimensions", dimensions_table)
        table.add_row("Measures", measures_table)
        table.add_row("Identifiers", identifiers_table)
        table.add_row("Description", semantic_details.get("semantic_model_desc", "N/A"))

        return table

    def _create_metrics_table(self, metrics: list, semantic_model_name: str) -> Table:
        """Create optimized Rich table for metrics display."""
        from rich import box

        table = Table(
            title=f"[bold cyan]📊 Metrics for {semantic_model_name}[/bold cyan]",
            show_header=True,
            show_lines=True,
            box=box.SIMPLE,
            border_style="blue",
            header_style="bold cyan",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=3, justify="center")
        table.add_column("Name", style="bright_cyan", min_width=15, max_width=25)
        table.add_column("Description", style="bright_magenta", min_width=10, max_width=30)
        table.add_column("Constraint", style="yellow", min_width=15)
        table.add_column("SQL", min_width=10, max_width=30, overflow="fold")

        # Batch add rows for better performance
        for idx, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue

            # Extract metric information
            metric_name = str(metric.get("name", ""))
            if not metric_name or metric_name.lower() == "unknown":
                continue

            description = str(metric.get("description", ""))
            constraint = str(metric.get("constraint", ""))

            table.add_row(
                str(idx + 1),
                metric_name,
                description,
                constraint,
                Syntax(metric.get("sql_query", ""), lexer="sql", line_numbers=True, start_line=1, word_wrap=True),
            )

        return table

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        tree = self.query_one("#semantic-tree", TextualTree)
        tree.action_cursor_down()
        self.clear_header()

    def clear_header(self):
        self.query_one("#tree-help", Static).update("")

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        tree = self.query_one("#semantic-tree", TextualTree)
        tree.action_cursor_up()
        self.clear_header()

    def action_expand_node(self) -> None:
        """Expand the current node."""
        tree = self.query_one("#semantic-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        """Collapse the current node."""
        tree = self.query_one("#semantic-tree", TextualTree)
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

    def _load_semantics_lazy(self):
        """Load semantic data into the tree structure."""
        tree = self.query_one("#semantic-tree", TextualTree)
        query_semantic_data = self.rag.search_all_semantic_models(self.database_name)
        semantic_data = {}
        for item in query_semantic_data:
            domain = item["domain"]
            layer1 = item["layer1"]
            layer2 = item["layer2"]
            semantic_model_name = item["semantic_model_name"]

            # Initialize domain if not exists
            if domain not in semantic_data:
                semantic_data[domain] = {}

            # Initialize layer1 if not exists
            if layer1 not in semantic_data[domain]:
                semantic_data[domain][layer1] = {}

            # Initialize layer2 if not exists
            if layer2 not in semantic_data[domain][layer1]:
                semantic_data[domain][layer1][layer2] = {}

            # Add semantic model
            semantic_data[domain][layer1][layer2][semantic_model_name] = item

        try:
            if not semantic_data:
                tree.root.add_leaf("📂 No semantic data found", data={"type": "empty"})
                return

            # Build the tree structure: domain -> layer1 -> layer2 -> semantic_model_name -> semantic_details
            for domain, layer1_data in semantic_data.items():
                domain_node = tree.root.add(f"📁 {domain}", data={"type": "domain", "name": domain})

                for layer1, layer2_data in layer1_data.items():
                    layer1_node = domain_node.add(
                        f"📂 {layer1}", data={"type": "layer1", "name": layer1, "domain": domain}
                    )

                    for layer2, semantic_models in layer2_data.items():
                        layer2_node = layer1_node.add(
                            f"📂 {layer2}", data={"type": "layer2", "name": layer2, "layer1": layer1, "domain": domain}
                        )

                        for semantic_model_name, semantic_details in semantic_models.items():
                            model_data = {
                                "type": "semantic_model",
                                "name": semantic_model_name,
                                "layer2": layer2,
                                "layer1": layer1,
                                "domain": domain,
                                "details": semantic_details,
                            }
                            layer2_node.add_leaf(f"📋 {semantic_model_name}", data=model_data)

            self.query_one("#tree-help", Static).update("")
        except Exception as e:
            logger.error(f"Failed to load semantic data: {str(e)}")
            tree.root.add_leaf("❌ Error loading semantic data", data={"type": "error"})

    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        _fetch_metrics_with_cache.cache_clear()

    def _clear_table_details(self):
        self.query_one("#properties-panel", Static).update("Select a semantic model and press Enter to view details")
        self.query_one("#metrics-panel", Static).update("")


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
                "• Enter - Load Metrics\n"
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
