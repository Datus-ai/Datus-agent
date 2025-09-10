from typing import Dict, List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static
from textual.widgets import Tree as TextualTree
from textual.widgets._tree import TreeNode

from datus.cli.screen.context_screen import ContextScreen
from datus.storage.metric.store import MetricStorage, SemanticModelStorage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MetricsScreen(ContextScreen):
    """Screen for displaying semantic models and metrics in interactive tree."""

    CSS = """
        /* Main layout containers */
        #tree-container {
            height: 100%;
            background: $surface;
        }

        #details-container {
            height: 100%;
            background: $surface-lighten-1;
            overflow-y: auto;
            overflow-x: hidden;
        }

        #details-panel {
            width: 100%;
            height: 100%;
            overflow-y: auto;
            overflow-x: auto;
            padding: 1;
        }

        /* Tree styling */
        #metrics-tree {
            width: 100%;
            background: $surface;
            border: none;
        }

        #metrics-tree > .tree--guides {
            color: $primary-lighten-2;
        }

        #metrics-tree:focus {
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

        /* Tree node styling */
        .tree--cursor {
            background: $accent-darken-1;
            color: $text;
        }

        .tree--highlighted {
            background: $accent-lighten-1;
            color: $text;
        }
    """

    BINDINGS = [
        Binding("down", "cursor_down", "Down", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("right", "expand_node", "Expand", show=False),
        Binding("left", "collapse_node", "Collapse", show=False),
        Binding("enter", "load_details", "Load Details", show=False),
        Binding("f2", "show_navigation_help", "Help"),
        Binding("escape", "exit_screen", "Exit", show=False),
    ]

    def __init__(self, title: str, context_data: Dict, inject_callback=None):
        """Initialize metrics screen."""
        super().__init__(title=title, context_data=context_data, inject_callback=inject_callback)
        self.semantic_model_storage: SemanticModelStorage = context_data.get("semantic_model_storage")
        self.metric_storage: MetricStorage = context_data.get("metric_storage")
        self.current_path = ""
        self.selected_data = {}
        self.loading_nodes = set()  # Track which nodes are currently loading

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="Metrics")

        with Horizontal():
            # Left side: Metrics tree
            with Vertical(id="tree-container"):
                yield TextualTree(label="Metrics Catalog", id="metrics-tree")

            # Right side: Details panel
            with Vertical(id="details-container"):
                yield Static("Select a node and press Enter to view details", id="details-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._build_metrics_tree()

    def _build_metrics_tree(self) -> None:
        """Build metrics tree from storage data."""
        try:
            tree = self.query_one("#metrics-tree", TextualTree)
            tree.root.expand()

            if not self.semantic_model_storage:
                self.query_one("#details-panel", Static).update("[red]Error:[/] No semantic model storage available")
                return

            semantic_models = self.semantic_model_storage.search_all("")

            grouped_models = self._group_semantic_models(semantic_models)

            for domain, layer1_groups in grouped_models.items():
                domain_node = tree.root.add(f"📁 {domain}", data={"type": "domain", "name": domain})

                for layer1, layer2_groups in layer1_groups.items():
                    layer1_node = domain_node.add(f"📂 {layer1}", data={"type": "layer1", "name": layer1})

                    for layer2, models in layer2_groups.items():
                        layer2_node = layer1_node.add(f"📂 {layer2}", data={"type": "layer2", "name": layer2})

                        # Add semantic model nodes
                        for model in models:
                            model_node = layer2_node.add(
                                f"📊 {model['semantic_model_name']}",
                                data={"type": "semantic_model", "semantic_model_data": model},
                            )

                            # Add metrics loading placeholder
                            model_node.add_leaf("⏳ Loading metrics...", data={"type": "loading_metrics"})

        except Exception as e:
            logger.error(f"Failed to build metrics tree: {str(e)}")
            self.query_one("#details-panel", Static).update(f"[red]Error:[/] Failed to build metrics tree: {str(e)}")

    def _group_semantic_models(self, semantic_models: List[Dict]) -> Dict:
        """Group semantic models by domain -> layer1 -> layer2."""
        grouped = {}

        for model in semantic_models:
            domain = model.get("domain", "unknown")
            catalog_db_schema = model.get("catalog_database_schema", "")

            parts = catalog_db_schema.split("_") if catalog_db_schema else []
            layer1 = parts[0] if len(parts) > 0 else "default"
            layer2 = parts[1] if len(parts) > 1 else "default"

            if domain not in grouped:
                grouped[domain] = {}
            if layer1 not in grouped[domain]:
                grouped[domain][layer1] = {}
            if layer2 not in grouped[domain][layer1]:
                grouped[domain][layer1][layer2] = []

            grouped[domain][layer1][layer2].append(model)

        return grouped

    def on_tree_node_selected(self, event: TextualTree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        self.selected_data = node.data or {}
        self.update_path_display(node)

    def on_tree_node_highlighted(self, event: TextualTree.NodeHighlighted) -> None:
        """Handle tree node highlighting."""
        node = event.node
        self.selected_data = node.data or {}
        self.update_path_display(node)

    def update_path_display(self, node: TreeNode) -> None:
        """Update the header with the current path."""
        path_parts = []
        current = node

        # Build path from current node up to root
        while current and hasattr(current, "data") and current.data:
            name = str(current.data.get("name", ""))
            if not name and current.data.get("type") == "semantic_model":
                semantic_data = current.data.get("semantic_model_data", {})
                name = semantic_data.get("semantic_model_name", "")
            if not name and current.data.get("type") == "metric":
                metric_data = current.data.get("metric_data", {})
                name = metric_data.get("metric_name", "")

            if name:
                path_parts.insert(0, name)
            current = current.parent

        # Update header with the path
        if path_parts:
            self.current_path = ".".join(path_parts)
            header = self.query_one(Header)
            header._name = self.current_path
        else:
            self.current_path = ""
            header = self.query_one(Header)
            header._name = "Metrics"

    def on_tree_node_expanded(self, event: TextualTree.NodeExpanded) -> None:
        """Handle tree node expansion for lazy loading metrics."""
        node = event.node
        if not node.data:
            return

        node_type = node.data.get("type")

        # Handle loading placeholders
        loading_children = [
            child for child in node.children if child.data and child.data.get("type") == "loading_metrics"
        ]
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
            # Load metrics for semantic model
            if node_type == "semantic_model":
                semantic_model_data = node.data.get("semantic_model_data", {})
                semantic_model_name = semantic_model_data.get("semantic_model_name")
                if semantic_model_name:
                    self._load_metrics_for_model(node, semantic_model_name)
        except Exception as e:
            logger.error(f"Error loading node {node_key}: {str(e)}")
            node.add_leaf("❌ Error loading data", data={"type": "error"})
        finally:
            # Remove from loading set
            self.loading_nodes.discard(node_key)

    def _load_metrics_for_model(self, model_node: TreeNode, semantic_model_name: str) -> None:
        """Load metrics for a semantic model."""
        try:
            if not self.metric_storage:
                model_node.add_leaf("❌ Metrics storage not available", data={"type": "error"})
                return

            metrics = self.metric_storage.search_all(semantic_model_name)

            # Add metric nodes
            for metric in metrics:
                model_node.add_leaf(f"• {metric['metric_name']}", data={"type": "metric", "metric_data": metric})

            # If no metrics found
            if not metrics:
                model_node.add_leaf("No metrics found", data={"type": "empty"})

        except Exception as e:
            logger.error(f"Failed to load metrics for {semantic_model_name}: {str(e)}")
            model_node.add_leaf("❌ Error loading metrics", data={"type": "error"})

    def action_load_details(self) -> None:
        """Load details when Enter is pressed."""
        tree = self.query_one("#metrics-tree", TextualTree)
        if tree.cursor_node is None:
            return

        node = tree.cursor_node
        node_data = node.data

        if not node_data:
            return

        node_type = node_data.get("type")
        details_panel = self.query_one("#details-panel", Static)

        if node_type == "semantic_model":
            self._show_semantic_model_details(node_data.get("semantic_model_data", {}), details_panel)
        elif node_type == "metric":
            self._show_metric_details(node_data.get("metric_data", {}), details_panel)
        else:
            details_panel.update("Select a semantic model or metric and press Enter to view details")

    def _show_semantic_model_details(self, semantic_model_data: Dict, details_panel: Static) -> None:
        """Show semantic model details."""
        try:
            import json

            content = "[bold cyan]📊 Semantic Model Details[/bold cyan]\n\n"
            content += f"[bold]Name:[/] {semantic_model_data.get('semantic_model_name', 'N/A')}\n"
            content += f"[bold]Description:[/] {semantic_model_data.get('semantic_model_desc', 'N/A')}\n"
            content += f"[bold]Domain:[/] {semantic_model_data.get('domain', 'N/A')}\n"
            content += f"[bold]Catalog:[/] {semantic_model_data.get('catalog_name', 'N/A')}\n"
            content += f"[bold]Database:[/] {semantic_model_data.get('database_name', 'N/A')}\n"
            content += f"[bold]Schema:[/] {semantic_model_data.get('schema_name', 'N/A')}\n"
            content += f"[bold]Table:[/] {semantic_model_data.get('table_name', 'N/A')}\n\n"

            # Show identifiers
            content += "[bold]Identifiers:[/]\n"
            try:
                identifiers = json.loads(semantic_model_data.get("identifiers", "[]"))
                for identifier in identifiers:
                    content += f"  - {identifier.get('name', 'N/A')}: {identifier.get('type', 'N/A')}\n"
            except Exception:
                content += "  - Failed to parse identifiers\n"

            content += "\n[bold]Dimensions:[/]\n"
            try:
                dimensions = json.loads(semantic_model_data.get("dimensions", "[]"))
                for dimension in dimensions:
                    content += f"  - {dimension.get('name', 'N/A')}: {dimension.get('type', 'N/A')}\n"
            except Exception:
                content += "  - Failed to parse dimensions\n"

            content += "\n[bold]Measures:[/]\n"
            try:
                measures = json.loads(semantic_model_data.get("measures", "[]"))
                for measure in measures:
                    content += f"  - {measure.get('name', 'N/A')}: {measure.get('agg', 'N/A')}\n"
            except Exception:
                content += "  - Failed to parse measures\n"

            details_panel.update(content)
        except Exception as e:
            logger.error(f"Failed to show semantic model details: {str(e)}")
            details_panel.update(f"[red]Error:[/] Failed to load semantic model details: {str(e)}")

    def _show_metric_details(self, metric_data: Dict, details_panel: Static) -> None:
        """Show metric details."""
        try:
            content = "[bold cyan]📈 Metric Details[/bold cyan]\n\n"
            content += f"[bold]Name:[/] {metric_data.get('metric_name', 'N/A')}\n"
            content += f"[bold]Description:[/] {metric_data.get('metric_value', 'N/A')}\n"
            content += f"[bold]Type:[/] {metric_data.get('metric_type', 'N/A')}\n\n"
            content += "[bold]SQL Query:[/]\n"
            content += metric_data.get("metric_sql_query", "N/A")

            details_panel.update(content)
        except Exception as e:
            logger.error(f"Failed to show metric details: {str(e)}")
            details_panel.update(f"[red]Error:[/] Failed to load metric details: {str(e)}")

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        tree = self.query_one("#metrics-tree", TextualTree)
        tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        tree = self.query_one("#metrics-tree", TextualTree)
        tree.action_cursor_up()

    def action_expand_node(self) -> None:
        """Expand the current node."""
        tree = self.query_one("#metrics-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        """Collapse the current node."""
        tree = self.query_one("#metrics-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.collapse()

    def action_exit_screen(self) -> None:
        """Exit the screen."""
        self.app.pop_screen()

    def action_show_navigation_help(self) -> None:
        """Show navigation help."""
        current_screen = self.app.screen_stack[-1] if self.app.screen_stack else None

        if isinstance(current_screen, NavigationHelpScreen):
            self.app.pop_screen()
        else:
            self.app.push_screen(NavigationHelpScreen())


class NavigationHelpScreen(ModalScreen):
    """Modal screen to display navigation help."""

    def compose(self) -> ComposeResult:
        """Compose the navigation help modal."""
        yield Container(
            Static(
                "# Metrics Navigation Help\n\n"
                "## Arrow Key Navigation:\n"
                "• ↑ - Move cursor up\n"
                "• ↓ - Move cursor down\n"
                "• → - Expand current node\n"
                "• ← - Collapse current node\n\n"
                "## Other Keys:\n"
                "• Enter - Load details for selected node\n"
                "• F2 - Toggle this help\n"
                "• Esc - Exit screen\n\n"
                "Press any key to close this help.",
                id="navigation-help-content",
            ),
            id="navigation-help-container",
        )

    def on_key(self, event) -> None:
        """Close the modal on any key press."""
        self.dismiss()
