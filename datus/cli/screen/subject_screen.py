from functools import lru_cache
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
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
from datus.cli.subject_rich_utils import build_historical_sql_tags
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.sql_history import SqlHistoryRAG
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=128)
def _fetch_metrics_with_cache(
    metrics_rag: SemanticMetricsRAG, domain: str, layer1: str, layer2: str, name: str
) -> List[Dict[str, Any]]:
    try:
        table = metrics_rag.get_metrics_detail(
            domain=domain,
            layer1=layer1,
            layer2=layer2,
            name=name,
        )
        return table if table is not None else []
    except Exception as e:
        logger.error(f"Metrics fetch failed: {str(e)}")
        return []


@lru_cache(maxsize=128)
def _sql_details_cache(
    sql_rag: SqlHistoryRAG, domain: str, layer1: str, layer2: str, name: str
) -> List[Dict[str, Any]]:
    try:
        table = sql_rag.get_sql_history_detail(
            domain,
            layer1,
            layer2,
            name,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "Failed to fetch SQL details for %s/%s/%s/%s: %s",
            domain,
            layer1,
            layer2,
            name,
            exc,
        )
        return []

    return table if table is not None else []


class SubjectScreen(ContextScreen):
    """Screen for browsing domain metrics alongside SQL history."""

    CSS = """
        #tree-container {
            width: 35%;
            height: 100%;
            background: $surface;
            overflow: hidden;
        }

        #details-container {
            width: 65%;
            height: 100%;
            background: $surface-lighten-1;
            overflow: hidden;
        }

        #subject-tree {
            width: 100%;
            height: 1fr;
            background: $surface;
            border: none;
            overflow-y: auto;
        }

        #subject-tree:focus {
            border: none;
        }

        #subject-tree > .tree--guides {
            color: $primary-lighten-2;
        }

        #metrics-panel-container,
        #sql-panel-container {
            width: 100%;
            height: 50%;
            background: $surface;
            color: $text;
            overflow-y: auto;
            overflow-x: hidden;
        }

        #metrics-panel,
        #sql-panel {
            width: 100%;
            padding: 1 1;
        }

        #panel-divider {
            height: 1;
            background: $surface-darken-1;
            margin: 0;
        }

        .hidden {
            display: none;
        }

        .fullscreen #tree-container {
            width: 100%;
        }

        .fullscreen #details-container {
            display: none;
        }

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
                - metrics_rag
                - sql_rag
            inject_callback: Callback for injecting data into the CLI
        """
        super().__init__(title=title, context_data=context_data, inject_callback=inject_callback)
        self.metrics_rag: Optional[SemanticMetricsRAG] = context_data.get("metrics_rag") or context_data.get("rag")
        self.sql_rag: Optional[SqlHistoryRAG] = context_data.get("sql_rag")
        self.inject_callback = inject_callback
        self.selected_path = ""
        self.selected_data: Dict[str, Any] = {}
        self.tree_data: Dict[str, Any] = {}
        self.is_fullscreen = False
        self._current_loading_task = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="Metrics & SQL")

        with Horizontal():
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("[dim]Loading subjects...[/dim]", id="tree-help")
                yield TextualTree(label="Metrics & SQL", id="subject-tree")

            with Vertical(id="details-container", classes="details-panel"):
                yield ScrollableContainer(
                    Static("Select a node to view metrics", id="metrics-panel"),
                    id="metrics-panel-container",
                )
                yield Static(id="panel-divider", classes="hidden")
                yield ScrollableContainer(
                    Static("Select a node to view SQL history", id="sql-panel"),
                    id="sql-panel-container",
                )

        yield Footer()

    async def on_mount(self) -> None:
        self._build_tree()

    def on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "right"}:
            self.action_load_details()
        elif event.key == "escape":
            self.action_exit_without_selection()
        else:
            super()._on_key(event)

    def _build_tree(self) -> None:
        tree = self.query_one("#subject-tree", TextualTree)
        tree.clear()
        tree.root.expand()
        tree.root.add_leaf("⏳ Loading...", data={"type": "loading"})
        if self._current_loading_task and not self._current_loading_task.is_finished:
            self._current_loading_task.cancel()
        self._current_loading_task = self.run_worker(self._load_subject_data, thread=True)

    @work(thread=True)
    def _load_subject_data(self) -> None:
        get_current_worker()
        tree_data: Dict[str, Any] = {}

        try:
            metrics_table = self.metrics_rag.search_all_metrics(select_fields=["domain", "layer1", "layer2", "name"])
            metrics_rows = metrics_table if metrics_table is not None else []
        except Exception as exc:
            logger.error(f"Failed to load metric taxonomy for subject screen: {exc}")
            metrics_rows = []

        for item in metrics_rows:
            domain = item.get("domain") or "Uncategorized"
            layer1 = item.get("layer1") or "General"
            layer2 = item.get("layer2") or "General"
            name = item.get("name") or "Unnamed Metric"
            bucket = (
                tree_data.setdefault(domain, {})
                .setdefault(layer1, {})
                .setdefault(layer2, {})
                .setdefault(name, {"metrics_count": 0, "sql_count": 0})
            )
            bucket["metrics_count"] += 1

        try:
            sql_table = self.sql_rag.search_all_sql_history(selected_fields=["domain", "layer1", "layer2", "name"])
            sql_rows = sql_table if sql_table is not None else []
        except Exception as exc:
            logger.error(f"Failed to load SQL taxonomy for subject screen: {exc}")
            sql_rows = []

        for item in sql_rows:
            domain = item.get("domain") or "Uncategorized"
            layer1 = item.get("layer1") or "General"
            layer2 = item.get("layer2") or "General"
            name = item.get("name") or "Unnamed SQL"
            bucket = (
                tree_data.setdefault(domain, {})
                .setdefault(layer1, {})
                .setdefault(layer2, {})
                .setdefault(name, {"metrics_count": 0, "sql_count": 0})
            )
            bucket["sql_count"] += 1

        self.tree_data = tree_data
        self.app.call_from_thread(self._populate_tree, tree_data)

    def _populate_tree(self, tree_data: Dict[str, Any]) -> None:
        tree = self.query_one("#subject-tree", TextualTree)
        tree.clear()
        tree.root.expand()

        if not tree_data:
            tree.root.add_leaf("📂 No metrics or SQL history found", data={"type": "empty"})
            return

        for domain in sorted(tree_data.keys()):
            domain_node = tree.root.add(f"📁 {domain}", data={"type": "domain", "name": domain})
            for layer1 in sorted(tree_data[domain].keys()):
                layer1_node = domain_node.add(f"📂 {layer1}", data={"type": "layer1", "name": layer1, "domain": domain})
                for layer2 in sorted(tree_data[domain][layer1].keys()):
                    layer2_node = layer1_node.add(
                        f"📂 {layer2}",
                        data={"type": "layer2", "name": layer2, "layer1": layer1, "domain": domain},
                    )
                    for name in sorted(tree_data[domain][layer1][layer2].keys()):
                        payload = tree_data[domain][layer1][layer2][name]
                        type_tokens: List[str] = []
                        if payload.get("metrics_count"):
                            type_tokens.append("metrics")
                        if payload.get("sql_count"):
                            type_tokens.append("sql")
                        type_hint = f" ({'/'.join(type_tokens)})" if type_tokens else ""
                        label = f"📋 {name}{type_hint}"
                        node_data = {
                            "type": "subject_entry",
                            "name": name,
                            "domain": domain,
                            "layer1": layer1,
                            "layer2": layer2,
                            "metrics_count": payload.get("metrics_count", 0),
                            "sql_count": payload.get("sql_count", 0),
                        }
                        layer2_node.add_leaf(label, data=node_data)

    def on_tree_node_selected(self, event: TextualTree.NodeSelected) -> None:
        self.update_path_display(event.node)

    def on_tree_node_highlighted(self, event: TextualTree.NodeHighlighted) -> None:
        self.update_path_display(event.node)

    def update_path_display(self, node: TreeNode) -> None:
        path_parts: List[str] = []
        current = node
        if node.data:
            self.selected_data = node.data

        while current and str(current.label) != "Metrics & SQL":
            name = str(current.data.get("name", "")) if current.data else str(current.label)
            name = name.replace("📁 ", "").replace("📂 ", "").replace("📋 ", "")
            if name:
                path_parts.insert(0, name)
            current = current.parent

        header = self.query_one(Header)
        if path_parts:
            self.selected_path = ".".join(path_parts)
            header._name = self.selected_path
        else:
            self.selected_path = ""
            header._name = "Metrics & SQL"

    def action_load_details(self) -> None:
        tree = self.query_one("#subject-tree", TextualTree)
        if not tree.cursor_node:
            return

        node = tree.cursor_node
        if node.data and node.data.get("type") == "subject_entry":
            self._show_subject_details(node.data)
        else:
            if node.is_expanded:
                node.collapse()
            else:
                node.expand()

    def _show_subject_details(self, subject_info: Dict[str, Any]) -> None:
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        metrics_panel = self.query_one("#metrics-panel", Static)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)
        sql_panel = self.query_one("#sql-panel", Static)
        divider = self.query_one("#panel-divider", Static)

        metrics = []
        sql_entries = []

        metrics_count = subject_info.get("metrics_count", 0)
        sql_count = subject_info.get("sql_count", 0)

        if metrics_count and self.metrics_rag:
            metrics = self._fetch_metrics_details(
                subject_info.get("domain", ""),
                subject_info.get("layer1", ""),
                subject_info.get("layer2", ""),
                subject_info.get("name", ""),
            )

        if sql_count and self.sql_rag:
            sql_entries = self._fetch_sql_details(
                subject_info.get("domain", ""),
                subject_info.get("layer1", ""),
                subject_info.get("layer2", ""),
                subject_info.get("name", ""),
            )

        if metrics:
            metrics_panel.update(self._create_metrics_panel_content(metrics, subject_info.get("name", "")))
            self._toggle_visibility(metrics_container, True)
        else:
            metrics_panel.update("[dim]No metrics for this item[/dim]")
            self._toggle_visibility(metrics_container, False)

        if sql_entries:
            sql_panel.update(self._create_sql_panel_content(sql_entries))
            self._toggle_visibility(sql_container, True)
        else:
            sql_panel.update("[dim]No SQL history for this item[/dim]")
            self._toggle_visibility(sql_container, False)

        metrics_visible = bool(metrics)
        sql_visible = bool(sql_entries)

        if metrics_visible and sql_visible:
            metrics_container.styles.height = "50%"
            sql_container.styles.height = "50%"
        elif metrics_visible:
            metrics_container.styles.height = "100%"
            sql_container.styles.height = "0%"
        elif sql_visible:
            sql_container.styles.height = "100%"
            metrics_container.styles.height = "0%"
        else:
            metrics_container.styles.height = "50%"
            sql_container.styles.height = "50%"

        self._toggle_visibility(divider, metrics_visible and sql_visible)

    def _toggle_visibility(self, widget: Any, visible: bool) -> None:
        if visible:
            widget.remove_class("hidden")
        else:
            widget.add_class("hidden")

    def _create_metrics_panel_content(self, metrics: List[Dict[str, Any]], group_name: str) -> Group:
        sections: List[Table] = []
        for idx, metric in enumerate(metrics, 1):
            if not isinstance(metric, dict):
                continue

            metric_name = str(metric.get("name", "Unnamed Metric"))
            description = str(metric.get("description", ""))
            constraint = str(metric.get("constraint", ""))
            sql_query = str(metric.get("sql_query", ""))

            table = Table(
                title=f"[bold cyan]📊 Metric #{idx}: {metric_name}[/bold cyan]",
                show_header=False,
                box=box.SIMPLE,
                border_style="blue",
                expand=True,
                padding=(0, 1),
            )
            table.add_column("Key", style="bright_cyan", width=16)
            table.add_column("Value", style="yellow", ratio=1)

            if group_name:
                table.add_row("Group", group_name)
            if description:
                table.add_row("Description", description)
            if constraint:
                table.add_row("Constraint", constraint)
            if sql_query:
                table.add_row(
                    "SQL",
                    Syntax(sql_query, "sql", theme="monokai", word_wrap=True, line_numbers=False),
                )

            sections.append(table)

        return Group(*sections) if sections else Group()

    def _create_sql_panel_content(self, sql_entries: List[Dict[str, Any]]) -> Group:
        sections: List[Table] = []
        for idx, sql_entry in enumerate(sql_entries, 1):
            details = Table(
                title=f"[bold cyan]📝 SQL #{idx}: {sql_entry.get('name', 'Unnamed')}[/bold cyan]",
                show_header=False,
                box=box.SIMPLE,
                border_style="blue",
                expand=True,
                padding=(0, 1),
            )
            details.add_column("Key", style="bright_cyan", width=12)
            details.add_column("Value", style="yellow", ratio=1)

            if summary := sql_entry.get("summary"):
                details.add_row("Summary", summary)
            if comment := sql_entry.get("comment"):
                details.add_row("Comment", comment)
            if tags := sql_entry.get("tags"):
                details.add_row("Tags", build_historical_sql_tags(tags))

            details.add_row(
                "SQL",
                Syntax(str(sql_entry.get("sql", "")), "sql", theme="monokai", word_wrap=True, line_numbers=True),
            )

            sections.append(details)

        return Group(*sections)

    def action_cursor_down(self) -> None:
        self.query_one("#subject-tree", TextualTree).action_cursor_down()
        self.query_one("#tree-help", Static).update("")

    def action_cursor_up(self) -> None:
        self.query_one("#subject-tree", TextualTree).action_cursor_up()
        self.query_one("#tree-help", Static).update("")

    def action_expand_node(self) -> None:
        tree = self.query_one("#subject-tree", TextualTree)
        if tree.cursor_node:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        tree = self.query_one("#subject-tree", TextualTree)
        if tree.cursor_node:
            tree.cursor_node.collapse()

    def action_show_navigation_help(self) -> None:
        current_screen = self.app.screen_stack[-1] if self.app.screen_stack else None
        if isinstance(current_screen, NavigationHelpScreen):
            self.app.pop_screen()
        else:
            self.app.push_screen(NavigationHelpScreen())

    def action_toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        self.query_one("#tree-container").set_class(self.is_fullscreen, "fullscreen")
        self.query_one("#details-container").set_class(self.is_fullscreen, "fullscreen")

    def action_show_path(self) -> None:
        if self.selected_path:
            self.query_one("#tree-help", Static).update(f"Selected Path: {self.selected_path}")

    def action_exit_with_selection(self) -> None:
        if self.selected_path and self.inject_callback:
            self.inject_callback(self.selected_path, self.selected_data)
        self.app.exit()

    def action_exit_without_selection(self) -> None:
        self.selected_path = ""
        self.selected_data = {}
        self.app.exit()

    def on_unmount(self) -> None:
        self.clear_cache()

    def clear_cache(self) -> None:
        _fetch_metrics_with_cache.cache_clear()
        _sql_details_cache.cache_clear()

    def _fetch_metrics_details(self, domain: str, layer1: str, layer2: str, name: str) -> List[Dict[str, Any]]:
        return _fetch_metrics_with_cache(
            self.metrics_rag, domain or "default", layer1 or "default", layer2 or "default", name or ""
        )

    def _fetch_sql_details(self, domain: str, layer1: str, layer2: str, name: str) -> List[Dict[str, Any]]:
        return _sql_details_cache(
            self.sql_rag, domain or "default", layer1 or "default", layer2 or "default", name or ""
        )


class NavigationHelpScreen(ModalScreen):
    """Modal screen to display navigation help."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                "# Navigation Help\n\n"
                "## Arrow Key Navigation:\n"
                "• ↑ - Move cursor up\n"
                "• ↓ - Move cursor down\n"
                "• → - Expand node / Load details\n"
                "• ← - Collapse node\n\n"
                "## Other Keys:\n"
                "• F1 - Toggle fullscreen\n"
                "• F2 - Toggle this help\n"
                "• Enter - Load details\n"
                "• F4 - Show path\n"
                "• F5 - Select and exit\n"
                "• Esc - Exit without selection\n\n"
                "Press any key to close this help.",
                id="navigation-help-content",
            ),
            id="navigation-help-container",
        )

    def on_key(self, event) -> None:
        self.dismiss()
