import json
from typing import Any, Dict

from rich import box
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
from datus.storage.sql_history import SqlHistoryRAG
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class HistoricalSqlScreen(ContextScreen):
    """Screen for displaying historical SQL queries."""

    CSS = """
        #tree-container {
            height: 100%;
            background: $surface;
        }

        #details-container {
            width: 50%;
            height: 100%;
            background: $surface-lighten-1;
            overflow-y: auto;
            overflow-x: hidden;
        }

        #history-tree {
            width: 100%;
            background: $surface;
            border: none;
        }

        #history-tree:focus {
            border: none;
        }

        #details-panel {
            background: $surface;
            color: $text;
            height: 100%;
            overflow-y: auto;
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

        #tree-help {
            width: 100%;
            height: 1;
            background: $surface-darken-1;
            color: $text-muted;
            text-align: center;
        }

        #navigation-help-container {
            padding: 2 4;
            width: auto;
            height: auto;
            max-width: 80%;
            max-height: 80%;
            background: $surface-darken-1;
            border: thick $primary-lighten-2;
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
        super().__init__(title=title, context_data=context_data, inject_callback=inject_callback)
        self.rag: SqlHistoryRAG = context_data.get("rag")
        self.inject_callback = inject_callback
        self.selected_path = ""
        self.selected_data = {}
        self.is_fullscreen = False
        self._current_loading_task = None

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="SQL History")

        with Horizontal():
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("", id="tree-help")
                yield TextualTree(label="SQL History", id="history-tree")

            with Vertical(id="details-container", classes="details-panel"):
                yield ScrollableContainer(
                    Static("Select a SQL entry and press Enter to view details", id="details-panel"),
                )

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

    def _build_tree(self) -> None:
        """Load history data and populate the tree."""
        tree = self.query_one("#history-tree", TextualTree)
        try:
            tree.clear()
            tree.root.expand()
            tree.root.add_leaf("⏳ Loading...", data={"type": "loading"})
            self.run_worker(self._load_data_lazy, thread=True)
        except Exception as e:
            logger.error(f"Failed to start loading SQL history: {str(e)}")
            tree.clear()
            tree.root.add_leaf(f"❌ Error: {str(e)}", data={"type": "error"})

    def on_tree_node_selected(self, event: TextualTree.NodeSelected) -> None:
        """Handle tree node selection."""
        self.update_path_display(event.node)

    def on_tree_node_highlighted(self, event: TextualTree.NodeHighlighted) -> None:
        """Handle tree node highlighting."""
        self.update_path_display(event.node)

    def update_path_display(self, node: TreeNode) -> None:
        """Update the header with the current path."""
        path_parts = []
        current = node
        if node.data:
            self.selected_data = node.data

        while current and str(current.label) != "SQL History":
            name = str(current.data.get("name", "")) if current.data else str(current.label)
            name = name.replace("📁 ", "").replace("📂 ", "").replace("📋 ", "")
            if name:
                path_parts.insert(0, name)
            current = current.parent

        if path_parts:
            self.selected_path = ".".join(path_parts)
            header = self.query_one(Header)
            header._name = self.selected_path
        else:
            self.selected_path = ""
            header = self.query_one(Header)
            header._name = "SQL History"

    def show_sql_details(self, history_info: Dict[str, Any]) -> None:
        """Show SQL history details with async loading."""
        if (
            hasattr(self, "_current_loading_task")
            and self._current_loading_task
            and not self._current_loading_task.is_finished
        ):
            self._current_loading_task.cancel()
        self._current_loading_task = self.load_sql_details_async(history_info)

    @work(thread=True)
    async def load_sql_details_async(self, history_info: Dict[str, Any]) -> None:
        """Async worker to load SQL details."""
        get_current_worker()

        try:
            details_panel = self.query_one("#details-panel", Static)
            if not self.rag:
                self.app.call_from_thread(details_panel.update, "[red]No RAG system available[/red]")
                return

            self.app.call_from_thread(
                details_panel.update, f"[dim]Loading details for {history_info.get('name', 'N/A')}...[/dim]"
            )

            details_table = self._build_details_content(history_info)
            self.app.call_from_thread(details_panel.update, details_table)

        except Exception as e:
            logger.error(f"Failed to load SQL details: {str(e)}")
            self.app.call_from_thread(
                self.query_one("#details-panel", Static).update,
                f"[red]❌ Error:[/] Failed to display SQL details: {str(e)}",
            )

    def _build_details_content(self, history_details: Dict[str, Any]) -> Table:
        """Build SQL history details display as a Rich table."""
        table = Table(
            title=f"[bold cyan]📋 SQL Details: {history_details.get('name', 'N/A')}[/bold cyan]",
            show_header=False,
            box=box.SIMPLE,
            border_style="blue",
            expand=True,
            padding=(1, 0),
        )

        table.add_column("Key", style="bright_cyan", ratio=1)
        table.add_column("Value", style="yellow", justify="left", ratio=3, no_wrap=False)

        # Display all fields from the history_details dictionary
        for key, value in history_details.items():
            display_key = key.replace("_", " ").title()

            if key in ("sql_query", "sql"):
                table.add_row(
                    display_key, Syntax(str(value), "sql", theme="monokai", line_numbers=True, word_wrap=True)
                )
            elif key == "tags" and value:
                table.add_row(display_key, build_historical_sql_tags(value))
            elif isinstance(value, (dict, list)):
                table.add_row(display_key, json.dumps(value, indent=2, ensure_ascii=False))
            else:
                table.add_row(display_key, str(value))

        return table

    def action_load_details(self) -> None:
        """Load details when Enter or Right arrow is pressed."""
        tree = self.query_one("#history-tree", TextualTree)
        if tree.cursor_node is None:
            return

        node = tree.cursor_node
        if node.data and node.data.get("type") == "sql_history":
            self.show_sql_details(node.data.get("details", {}))
        else:
            if node.is_expanded:
                node.collapse()
            else:
                node.expand()

    @work(thread=True)
    def _load_data_lazy(self):
        """Load semantic data into the tree structure."""
        tree = self.query_one("#history-tree", TextualTree)
        try:
            history_data_list = self.rag.search_all_sql_history()

            history_data_tree = {}
            for item in history_data_list:
                domain = item.get("domain") or "Uncategorized"
                layer1 = item.get("layer1") or "Default"
                layer2 = item.get("layer2") or "Default"
                name = item.get("name") or "Untitled"

                if domain not in history_data_tree:
                    history_data_tree[domain] = {}
                if layer1 not in history_data_tree[domain]:
                    history_data_tree[domain][layer1] = {}
                if layer2 not in history_data_tree[domain][layer1]:
                    history_data_tree[domain][layer1][layer2] = {}

                history_data_tree[domain][layer1][layer2][name] = item

            self.app.call_from_thread(tree.clear)

            if not history_data_tree:
                self.app.call_from_thread(tree.root.add_leaf, "📂 No SQL history found", data={"type": "empty"})
                return

            for domain, layer1_data in history_data_tree.items():
                domain_node = self.app.call_from_thread(
                    tree.root.add, f"📁 {domain}", data={"type": "domain", "name": domain}
                )
                for layer1, layer2_data in layer1_data.items():
                    layer1_node = self.app.call_from_thread(
                        domain_node.add, f"📂 {layer1}", data={"type": "layer1", "name": layer1, "domain": domain}
                    )
                    for layer2, history_items in layer2_data.items():
                        layer2_node = self.app.call_from_thread(
                            layer1_node.add,
                            f"📂 {layer2}",
                            data={"type": "layer2", "name": layer2, "layer1": layer1, "domain": domain},
                        )
                        for name, details in history_items.items():
                            node_data = {
                                "type": "sql_history",
                                "name": name,
                                "layer2": layer2,
                                "layer1": layer1,
                                "domain": domain,
                                "details": details,
                            }
                            self.app.call_from_thread(layer2_node.add_leaf, f"📋 {name}", data=node_data)

        except Exception as e:
            logger.error(f"Failed to load SQL history tree: {str(e)}")
            self.app.call_from_thread(tree.clear)
            self.app.call_from_thread(tree.root.add_leaf, f"❌ Error loading data: {str(e)}", data={"type": "error"})

    def action_cursor_down(self) -> None:
        self.query_one("#history-tree", TextualTree).action_cursor_down()
        self.clear_header()

    def clear_header(self):
        self.query_one("#tree-help", Static).update("")

    def action_cursor_up(self) -> None:
        self.query_one("#history-tree", TextualTree).action_cursor_up()
        self.clear_header()

    def action_expand_node(self) -> None:
        tree = self.query_one("#history-tree", TextualTree)
        if tree.cursor_node is not None:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        tree = self.query_one("#history-tree", TextualTree)
        if tree.cursor_node is not None:
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
        """Close the modal on any key press."""
        self.dismiss()
