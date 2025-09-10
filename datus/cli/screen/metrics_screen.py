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
            width: 50%;
            background: $surface;
        }

        #details-container {
            height: 100%;
            width: 50%;
            background: $surface-lighten-1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0;
        }

        #details-panel {
            width: 100%;
            height: 100%;
            overflow-y: auto;
            overflow-x: auto;
            padding: 1;
            box-sizing: border-box;
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
        Binding("f1", "show_navigation_help", "Help"),
        Binding("f2", "exit_with_selection", "Select"),
        Binding("escape", "exit_without_selection", "Exit", show=False),
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
                yield Static("Select a semantic model or metric to view details", id="details-panel")

        yield Footer()

    def on_mount(self) -> None:
        self._build_metrics_tree()

    def _build_metrics_tree(self) -> None:
        """Build metrics tree from storage data.

        构建指标目录树，按照 domain -> layer1 -> layer2 -> semantic_model -> metrics 的层级结构组织。
        """
        try:
            tree = self.query_one("#metrics-tree", TextualTree)
            tree.root.expand()

            if not self.semantic_model_storage:
                self.query_one("#details-panel", Static).update("[red]Error:[/] No semantic model storage available")
                return

            # Get all semantic models
            semantic_models = self.semantic_model_storage.search_all("")

            # 如果没有数据，显示明确的提示
            if not semantic_models:
                tree.root.add_leaf("📂 No semantic models found in storage", data={"type": "empty_info"})
                self.query_one("#details-panel", Static).update("No semantic models found in the storage.")
                return

            # Group by domain -> layer1 -> layer2 -> semantic_model
            grouped_models = self._group_semantic_models(semantic_models)

            # 如果分组后没有数据，显示原始数据结构
            if not grouped_models:
                tree.root.add_leaf("📂 Error in data grouping", data={"type": "grouping_error"})
                self.query_one("#details-panel", Static).update(
                    f"Found {len(semantic_models)} models but failed to group them."
                )
                return

            # Build tree structure
            for domain, layer1_groups in grouped_models.items():
                domain_node = tree.root.add(f"📁 {domain}", data={"type": "domain", "name": domain})

                for layer1, layer2_groups in layer1_groups.items():
                    layer1_node = domain_node.add(f"📂 {layer1}", data={"type": "layer1", "name": layer1})

                    for layer2, models in layer2_groups.items():
                        layer2_node = layer1_node.add(f"📂 {layer2}", data={"type": "layer2", "name": layer2})

                        # Add semantic model nodes
                        for model in models:
                            model_name = model.get("semantic_model_name", "Unknown Model")
                            model_node = layer2_node.add(
                                f"📊 {model_name}", data={"type": "semantic_model", "semantic_model_data": model}
                            )

                            # Add metrics loading placeholder
                            model_node.add_leaf("⏳ Loading metrics...", data={"type": "loading_metrics"})

        except Exception as e:
            logger.error(f"Failed to build metrics tree: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            self.query_one("#details-panel", Static).update(f"[red]Error:[/] Failed to build metrics tree: {str(e)}")

    def _group_semantic_models(self, semantic_models: List[Dict]) -> Dict:
        """Group semantic models by domain -> layer1 -> layer2."""
        grouped = {}

        # 首先获取所有不同的 domain, layer1, layer2 组合
        if self.metric_storage:
            try:
                # 获取所有 metrics 数据以提取分层结构
                metrics_table = self.metric_storage.search_all(
                    select_fields=["domain", "layer1", "layer2", "semantic_model_name"]
                )

                # 创建一个字典来映射 semantic_model 到其分层结构
                semantic_model_hierarchy = {}
                if metrics_table and metrics_table.num_rows > 0:
                    domains = metrics_table["domain"].to_pylist()
                    layers1 = metrics_table["layer1"].to_pylist()
                    layers2 = metrics_table["layer2"].to_pylist()
                    semantic_names = metrics_table["semantic_model_name"].to_pylist()

                    for i in range(len(domains)):
                        semantic_name = semantic_names[i]
                        # 为每个 semantic_model_name 保留其分层信息
                        if semantic_name not in semantic_model_hierarchy:
                            semantic_model_hierarchy[semantic_name] = {
                                "domain": domains[i] or "unknown_domain",
                                "layer1": layers1[i] or "default_layer1",
                                "layer2": layers2[i] or "default_layer2",
                            }
            except Exception as e:
                logger.error(f"Failed to fetch hierarchy from metrics: {str(e)}")
                semantic_model_hierarchy = {}

        for model in semantic_models:
            semantic_model_name = model.get("semantic_model_name", "Unknown Model")

            # 尝试从 metrics 中获取分层信息
            hierarchy_info = semantic_model_hierarchy.get(semantic_model_name, {})

            # 获取 domain，如果没有则使用默认值
            domain = hierarchy_info.get("domain") or model.get("domain") or "unknown_domain"

            # 解析分层结构
            layer1 = hierarchy_info.get("layer1") or "default_layer1"
            layer2 = hierarchy_info.get("layer2") or "default_layer2"

            # 确保字典结构存在
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
        self.action_load_details()  # 自动加载详情

    def on_tree_node_highlighted(self, event: TextualTree.NodeHighlighted) -> None:
        """Handle tree node highlighting."""
        node = event.node
        self.selected_data = node.data or {}
        self.update_path_display(node)
        self.action_load_details()  # 自动加载详情

    def update_path_display(self, node: TreeNode) -> None:
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
                # 使用正确的字段名 'name' 而不是 'metric_name'
                name = metric_data.get("name", "")

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

            # 明确指定需要的字段，确保返回所有需要的数据
            required_fields = ["name", "description", "constraint", "sql_query", "semantic_model_name"]
            metrics_table = self.metric_storage.search_all(semantic_model_name, select_fields=required_fields)

            # 转换 PyArrow Table 为字典列表
            metrics = []
            if metrics_table and metrics_table.num_rows > 0:
                # 获取所有字段名
                field_names = metrics_table.schema.names
                for i in range(metrics_table.num_rows):
                    row_dict = {}
                    for field_name in field_names:
                        value = metrics_table[field_name][i]
                        # 处理PyArrow值
                        if hasattr(value, "as_py"):
                            row_dict[field_name] = value.as_py()
                        else:
                            row_dict[field_name] = value
                    metrics.append(row_dict)

            # Add metric nodes
            for metric in metrics:
                # 使用正确的字段名 'name' 而不是 'metric_name'
                metric_name = metric.get("name", "Unknown Metric")
                model_node.add_leaf(f"• {metric_name}", data={"type": "metric", "metric_data": metric})

            # If no metrics found
            if not metrics:
                model_node.add_leaf("No metrics found", data={"type": "empty"})

        except Exception as e:
            logger.error(f"Failed to load metrics for {semantic_model_name}: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            model_node.add_leaf("❌ Error loading metrics", data={"type": "error"})

    def _show_semantic_model_details(self, semantic_model_data: Dict, details_panel: Static) -> None:
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
            # 调试信息
            logger.debug(f"Metric data received: {metric_data}")

            content = "[bold cyan]📈 Metric Details[/bold cyan]\n\n"

            # 检查并显示字段值
            name = metric_data.get("name", "N/A")
            description = metric_data.get("description", "N/A")
            sql_query = metric_data.get("sql_query", "N/A")

            content += f"[bold]Name:[/] {name}\n"
            content += f"[bold]Description:[/] {description}\n"
            content += f"[bold]Type:[/] {metric_data.get('constraint', 'N/A')}\n\n"
            content += "[bold]SQL Query:[/]\n"
            content += sql_query

            details_panel.update(content)
        except Exception as e:
            logger.error(f"Failed to show metric details: {str(e)}")
            logger.error(f"Metric data: {metric_data}")
            details_panel.update(f"[red]Error:[/] Failed to load metric details: {str(e)}")

    def action_load_details(self) -> None:
        details_panel = self.query_one("#details-panel", Static)

        if not self.selected_data:
            details_panel.update("Select a semantic model or metric to view details")
            return

        node_type = self.selected_data.get("type")

        if node_type == "semantic_model":
            semantic_model_data = self.selected_data.get("semantic_model_data", {})
            self._show_semantic_model_details(semantic_model_data, details_panel)
        elif node_type == "metric":
            metric_data = self.selected_data.get("metric_data", {})
            self._show_metric_details(metric_data, details_panel)
        else:
            details_panel.update("Select a semantic model or metric to view details")

    def action_cursor_down(self) -> None:
        tree = self.query_one("#metrics-tree", TextualTree)
        tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        tree = self.query_one("#metrics-tree", TextualTree)
        tree.action_cursor_up()

    def action_expand_node(self) -> None:
        tree = self.query_one("#metrics-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        tree = self.query_one("#metrics-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.collapse()

    def action_exit_screen(self) -> None:
        """Exit the screen."""
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        else:
            # 如果是最后一个屏幕，正常退出应用
            self.app.exit()

    def action_show_navigation_help(self) -> None:
        """Show navigation help."""
        current_screen = self.app.screen_stack[-1] if self.app.screen_stack else None

        if isinstance(current_screen, NavigationHelpScreen):
            self.app.pop_screen()
        else:
            self.app.push_screen(NavigationHelpScreen())

    def action_exit_with_selection(self) -> None:
        """Exit screen and send selected data to CLI."""
        if self.selected_data and self.inject_callback:
            # 可以根据需要传递适当的数据
            self.inject_callback(self.current_path, self.selected_data)
        self.app.exit()

    def action_exit_without_selection(self) -> None:
        """Exit screen without selection."""
        self.selected_data = {}
        self.current_path = ""
        self.app.exit()


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
                "• F1 - Toggle this help\n"
                "• F2 - Select\n"
                "• Esc - Exit screen\n\n"
                "Press any key to close this help.",
                id="navigation-help-content",
            ),
            id="navigation-help-container",
        )

    def on_key(self, event) -> None:
        """Close the modal on any key press."""
        self.dismiss()
