# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

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
from textual.widget import Widget
from textual.widgets import Footer, Header, Label, Static
from textual.widgets import Tree as TextualTree
from textual.widgets._tree import TreeNode
from textual.worker import get_current_worker

from datus.cli.screen.base_widgets import EditableTree, FocusableStatic, InputWithLabel, ParentSelectionTree
from datus.cli.screen.context_screen import ContextScreen
from datus.cli.subject_rich_utils import build_historical_sql_tags
from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.subject_manager import SubjectUpdater
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


TREE_VALIDATION_RULES: Dict[str, Dict[str, str]] = {
    "domain": {
        "pattern": r"^[a-zA-Z0-9_\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+$",
    },
    "layer1": {
        "pattern": r"^[a-zA-Z0-9_\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+$",
    },
    "layer2": {
        "pattern": r"^[a-zA-Z0-9_\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+$",
    },
}


class TreeEditDialog(ModalScreen[Optional[Dict[str, Any]]]):
    """Dialog for editing a tree node name and reparenting within siblings."""

    BINDINGS = [
        Binding("ctrl+w", "save_exist", "Save and Exit", show=True, priority=True),
        Binding("escape", "action_cancel_exist", "Exist", show=True, priority=True),
    ]

    CSS = """
    InputWithLabel{
        height: 10%;
    }
    #tree-edit-dialog {
        layout: vertical;
        width: 60%;
        height: 100%;
        # background: $panel;
        border: tall $primary;
        padding: 1;
        align: center middle;
    }
    #tree-edit-name-input {
        margin-bottom: 1;
    }
    #tree-edit-parent-label {
        margin-top: 1;
        margin-bottom: 1;
    }
    #tree-parent-selector {
        overflow-y: auto;
    }
    """

    def __init__(
        self,
        *,
        level: str,
        current_name: str,
        current_parent: Optional[Dict[str, Any]],
        parent_tree: Optional[List[Dict[str, Any]]],
        parent_selection_type: Optional[str],
        pattern: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.level = level
        self.label = self.level.title() if self.level.title() != "Subject_Entry" else "Name"
        self.current_name = current_name
        self.current_parent = current_parent
        self.parent_tree = parent_tree or []
        self.parent_selection_type = parent_selection_type
        self._pattern = pattern
        self.parent_selector: Optional[ParentSelectionTree] = None

    def compose(self) -> ComposeResult:
        with Container():
            with Vertical(id="tree-edit-dialog"):
                input_widget = InputWithLabel(
                    label=self.label,
                    value=self.current_name,
                    readonly=False,
                    regex=self._pattern,
                    lines=1,
                    id="tree-edit-name-input",
                )
                yield input_widget

                if self.parent_tree and self.parent_selection_type:
                    yield Label("Parent", id="tree-edit-parent-label")
                    self.parent_selector = ParentSelectionTree(
                        self.parent_tree,
                        allowed_type=self.parent_selection_type,
                        current_selection=self.current_parent,
                    )
                    yield self.parent_selector

                yield Footer()

    def on_key(self, event: events.Key) -> None:
        """Handle key events at the screen level to ensure global shortcuts work"""
        if hasattr(event, "ctrl") and event.ctrl and event.key == "w":
            self.action_save_exist()
            event.prevent_default()
            event.stop()
            return

        # ESC cancels dialog
        if event.key == "escape" or (hasattr(event, "ctrl") and event.ctrl and event.key == "q"):
            self.cancel_and_close()
            event.prevent_default()
            event.stop()
            return

    def on_mount(self) -> None:
        name_input = self.query_one("#tree-edit-name-input", InputWithLabel)
        self.set_focus(name_input)
        name_input.cursor_position = len(self.current_name)

    def action_save_exist(self):
        name_input = self.query_one("#tree-edit-name-input", InputWithLabel)
        new_name = name_input.get_value().strip()
        parent_value = self.current_parent
        if self.parent_selector:
            parent_value = self.parent_selector.get_selected() or self.current_parent

        if not new_name:
            self.app.notify("Name cannot be empty", severity="warning")
            return

        self.dismiss({"name": new_name, "parent": parent_value})

    def action_cancel_exist(self):
        self.dismiss(None)
        return

    def cancel_and_close(self) -> None:
        """Restore dialog state and close without saving."""
        # Restore name input
        name_input = self.query_one("#tree-edit-name-input", InputWithLabel)
        name_input.set_value(self.current_name)
        # Restore parent selection (if any)
        if self.parent_selector and self.current_parent:
            try:
                # Reset internal selection then focus the original
                self.parent_selector._selected = self.current_parent
                self.parent_selector._focus_current_selection()
            except Exception as e:
                logger.warning(f"Failed to restore parent selection: {e}")
        self.dismiss(None)


class MetricsPanel(Vertical):
    """
    A panel for displaying and editing metric details using DetailField components.
    """

    can_focus = True

    def __init__(self, metric: Dict[str, Any], readonly: bool = True) -> None:
        super().__init__()
        self.entry = metric
        self.readonly = readonly
        self.fields: List[InputWithLabel] = []

    def compose(self) -> ComposeResult:
        metric_name = self.entry.get("name", "Unnamed Metric")
        yield Label(f"📊 [bold cyan]Metric: {metric_name}[/]")
        semantic_model_field = InputWithLabel(
            "Semantic Model Name",
            self.entry.get("semantic_model_name", ""),
            lines=1,
            readonly=self.readonly,
            language="markdown",
        )
        self.fields.append(semantic_model_field)
        yield semantic_model_field

        llm_text_field = InputWithLabel(
            "LLM Text",
            self.entry.get("llm_text", ""),
            lines=10,
            readonly=self.readonly,
            language="markdown",
        )
        self.fields.append(llm_text_field)
        yield llm_text_field

    def _fill_data(self):
        self.fields[0].set_value(self.entry.get("semantic_model_name", ""))
        self.fields[1].set_value(self.entry.get("llm_text", ""))

    def set_readonly(self, readonly: bool) -> None:
        """
        Toggle the read-only mode for all fields in this panel.
        """
        self.readonly = readonly
        for field in self.fields:
            field.set_readonly(readonly)

    def is_modified(self) -> bool:
        """
        Return True if any field has been modified.
        """
        return any(field.is_modified() for field in self.fields)

    def get_value(self) -> Dict[str, str]:
        """
        Return a dictionary mapping field labels to their current values.

        Maps field labels to their storage keys.
        """
        values: Dict[str, str] = {}
        for field in self.fields:
            key = field.label_text.lower()
            if key == "semantic model name":
                key = "semantic_model_name"
            if key == "llm text":
                key = "llm_text"
            values[key] = field.get_value()
        return values

    def restore(self):
        for field in self.fields:
            field.restore()

    def update_data(self, summary_data: Dict[str, Any]):
        self.entry.update(summary_data)
        self._fill_data()

    def focus_first_input(self) -> bool:
        for field in self.fields:
            if field.focus_input():
                return True
        return False


class ReferenceSqlPanel(Vertical):
    """
    A panel for displaying and editing reference SQL details using DetailField components.
    """

    can_focus = True  # Allow this panel to receive focus

    def __init__(self, entry: Dict[str, Any], readonly: bool = True) -> None:
        super().__init__()
        self.entry = entry
        self.readonly = readonly
        self.fields: List[InputWithLabel] = []

    def compose(self) -> ComposeResult:
        sql_name = self.entry.get("name", "Unnamed")
        yield Label(f"📝 [bold cyan]SQL: {sql_name}[/]")
        summary_field = InputWithLabel(
            "Summary", self.entry.get("summary", ""), lines=2, readonly=self.readonly, language="markdown"
        )
        self.fields.append(summary_field)
        yield summary_field
        comment_field = InputWithLabel(
            "Comment", self.entry.get("comment", ""), lines=2, readonly=self.readonly, language="markdown"
        )
        self.fields.append(comment_field)
        yield comment_field
        tags_field = InputWithLabel(
            "Tags",
            self.entry.get("tags", ""),
            lines=2,
            readonly=self.readonly,
        )
        self.fields.append(tags_field)
        yield tags_field
        sql_field = InputWithLabel("SQL", self.entry.get("sql", ""), lines=5, readonly=self.readonly, language="sql")
        self.fields.append(sql_field)
        yield sql_field

    def _fill_data(self):
        self.fields[0].set_value(self.entry.get("summary", ""))
        self.fields[1].set_value(self.entry.get("comment", ""))
        self.fields[2].set_value(self.entry.get("tags", ""))
        self.fields[3].set_value(self.entry.get("sql", ""))

    def update_data(self, summary_data: Dict[str, Any]):
        self.entry.update(summary_data)
        self._fill_data()

    def focus_first_input(self) -> bool:
        for field in self.fields:
            if field.focus_input():
                return True
        return False

    def set_readonly(self, readonly: bool) -> None:
        """
        Toggle the read-only mode for all fields in this panel.
        """
        self.readonly = readonly
        for field in self.fields:
            field.set_readonly(readonly)

    def is_modified(self) -> bool:
        """
        Return True if any field has been modified.
        """
        return any(field.is_modified() for field in self.fields)

    def get_value(self) -> Dict[str, str]:
        """
        Return a dictionary mapping field labels to their current values.
        """
        return {field.label_text.lower(): field.get_value() for field in self.fields}

    def restore(self):
        for field in self.fields:
            field.restore()


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
    sql_rag: ReferenceSqlRAG, domain: str, layer1: str, layer2: str, name: str
) -> List[Dict[str, Any]]:
    try:
        table = sql_rag.get_reference_sql_detail(
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
    """Screen for browsing domain metrics alongside reference SQL."""

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
        # Binding("f6", "change_edit_mode", "Change to edit/readonly mode "),
        Binding("q", "quit_if_idle", "Quit", show=False),
        Binding("ctrl+e", "start_edit", "Edit", show=True, priority=True),
        Binding("ctrl+w", "save_edit", "Save", show=True, priority=True),
        Binding("ctrl+q", "cancel_or_exit", "Exit", show=True, priority=True),
    ]

    def __init__(self, title: str, context_data: Dict, inject_callback=None):
        """
        Initialize the subject screen.

        Args:
            context_data: Dictionary containing database connection info
                - metrics_rag
                - sql_rag
            inject_callback: Callback for injecting data into the CLI
        """
        super().__init__(title=title, context_data=context_data, inject_callback=inject_callback)
        self.agent_config: AgentConfig = context_data.get("agent_config")
        self.metrics_rag: SemanticMetricsRAG = SemanticMetricsRAG(self.agent_config)
        self.sql_rag: ReferenceSqlRAG = ReferenceSqlRAG(self.agent_config)
        self.inject_callback = inject_callback
        self.selected_path = ""
        self.readonly = True
        self.selected_data: Dict[str, Any] = {}
        self.tree_data: Dict[str, Any] = {}
        self.is_fullscreen = False
        self._current_loading_task = None
        self._editing_component: Optional[str] = None
        self._last_tree_selection: Optional[Dict[str, Any]] = None
        self._active_dialog: TreeEditDialog | None = None
        self._subject_updater: SubjectUpdater | None = None

    @property
    def subject_updater(self) -> SubjectUpdater:
        if self._subject_updater is None:
            self._subject_updater = SubjectUpdater(self.agent_config)
        return self._subject_updater

    def compose(self) -> ComposeResult:
        header = Header(show_clock=True, name="Metrics & SQL")
        yield header

        with Horizontal():
            with Vertical(id="tree-container", classes="tree-panel"):
                yield Static("", id="tree-help")
                yield EditableTree(label="Metrics & SQL", id="subject-tree")

            with Vertical(id="details-container", classes="details-panel"):
                yield ScrollableContainer(
                    Static("Select a node to view metrics", id="metrics-panel"),
                    id="metrics-panel-container",
                    can_focus=False,
                )
                yield Static(id="panel-divider", classes="hidden")
                yield ScrollableContainer(
                    Label("Select a node to view reference SQL", id="sql-panel"),
                    id="sql-panel-container",
                    can_focus=False,
                )

        yield Footer()

    async def on_mount(self) -> None:
        self._build_tree()

    def on_key(self, event: events.Key) -> None:
        ctrl_pressed = getattr(event, "ctrl", False)
        if event.key in {"enter", "right"}:
            self.action_load_details()
        elif event.key == "escape":
            self.action_cancel_or_exit()
        elif event.key == "q" and ctrl_pressed:
            self.action_cancel_or_exit()
            event.prevent_default()
            event.stop()
            return
        elif event.key == "q":
            self.action_quit_if_idle()
            event.prevent_default()
            event.stop()
            return
        else:
            super()._on_key(event)

    def _resolve_focus_component(self) -> tuple[Optional[str], Optional[Widget]]:
        """Determine which major component currently has focus."""
        widget = self.focused
        while widget is not None:
            widget_id = widget.id or ""
            if widget_id in {"metrics-panel-container", "metrics-panel"}:
                return "metrics", widget
            if widget_id in {"sql-panel-container", "sql-panel"}:
                return "sql", widget
            if isinstance(widget, EditableTree):
                return "tree", widget
            if isinstance(widget, MetricsPanel):
                return "metrics", widget
            if isinstance(widget, ReferenceSqlPanel):
                return "sql", widget
            widget = widget.parent
        return None, None

    def _update_edit_indicator(self, component: Optional[str]) -> None:
        """Update header subtitle to reflect active editing context."""
        header = self.query_one(Header)
        messages = {
            "tree": "✏️ Editing tree node",
            "metrics": "✏️ Editing metrics details",
            "sql": "✏️ Editing reference SQL",
        }
        header.sub_title = messages.get(component, "")
        header.refresh()

    def _build_tree(self) -> None:
        tree = self.query_one("#subject-tree", EditableTree)
        tree.clear()
        tree.root.expand()
        tree.root.add_leaf("⏳ Loading...", data={"type": "loading"})
        if self._current_loading_task and not self._current_loading_task.is_finished:
            self._current_loading_task.cancel()
        self._current_loading_task = self.run_worker(self._load_subject_tree_data, thread=True)

    @work(thread=True)
    def _load_subject_tree_data(self) -> None:
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
            sql_table = self.sql_rag.search_all_reference_sql(selected_fields=["domain", "layer1", "layer2", "name"])
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
        tree = self.query_one("#subject-tree", EditableTree)
        tree.clear()
        tree.root.expand()

        if not tree_data:
            tree.root.add_leaf("📂 No metrics or reference SQL found", data={"type": "empty"})
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
                            type_tokens.append("📈")
                        if payload.get("sql_count"):
                            type_tokens.append("💻")

                        label = f"{''.join(type_tokens)} {name}"
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
        tree = self.query_one("#subject-tree", EditableTree)
        if not tree.has_focus or not tree.cursor_node:
            return

        node = tree.cursor_node
        if node.data and node.data.get("type") == "subject_entry":
            self._show_subject_details(node.data)
        else:
            if node.is_expanded:
                node.collapse()
            else:
                node.expand()

    def action_change_edit_mode(self):
        self.action_start_edit()

    def action_start_edit(self) -> None:
        component, widget = self._resolve_focus_component()
        if component is None:
            self.app.notify("No component focused for editing", severity="warning")
            return

        if self._editing_component and self._editing_component != component:
            self.app.notify("Finish the active edit before starting a new one", severity="warning")
            return

        if component == "tree":
            assert isinstance(widget, EditableTree)
            widget.request_edit()
            return
        self._begin_panel_edit(component, widget)

    def action_save_edit(self) -> None:
        if self._editing_component in {"metrics", "sql"}:
            self._save_panel_edit(self._editing_component)
            return

        self.app.notify("Nothing to save", severity="warning")

    def _begin_panel_edit(self, component: str, current_widget: Optional[Widget]) -> None:
        if component not in {"metrics", "sql"}:
            return

        if self._editing_component == component:
            return

        if not self.selected_data:
            self.app.notify("Select a subject entry before editing", severity="warning")
            return

        # Switch layout to editable panels if we're currently in read-only mode.
        panel: Optional[MetricsPanel | ReferenceSqlPanel] = None
        if self.readonly:
            self.readonly = False
            self._show_subject_details(self.selected_data)

        if isinstance(current_widget, (MetricsPanel, ReferenceSqlPanel)):
            panel = current_widget
        else:
            panel = self._get_panel(component)

        if panel is None:
            self.app.notify("No details available to edit", severity="warning")
            self.readonly = True
            self._update_edit_indicator(None)
            self._show_subject_details(self.selected_data)
            return

        panel.set_readonly(False)
        self._editing_component = component
        self._update_edit_indicator(component)

        def focus_panel_inputs() -> None:
            if hasattr(panel, "focus_first_input") and panel.focus_first_input():
                return
            self.set_focus(panel)

        self.app.call_after_refresh(focus_panel_inputs)

    def _save_panel_edit(self, component: str) -> None:
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)

        panel: Optional[MetricsPanel | ReferenceSqlPanel] = None
        if component == "metrics":
            if query := metrics_container.query(MetricsPanel):
                panel = query.first()
        elif component == "sql":
            if query := sql_container.query(ReferenceSqlPanel):
                panel = query.first()

        if panel is None:
            self.app.notify("No panel available to save", severity="warning")
            return

        data = panel.get_value()
        modified = panel.is_modified()
        panel.set_readonly(True)
        self._editing_component = None
        self._update_edit_indicator(None)
        self.readonly = True

        if modified:
            panel.update_data(data)
            if component == "sql":
                self.subject_updater.update_historical_sql(self.selected_data, data)
                _sql_details_cache.cache_clear()
            elif component == "metrics":
                self.subject_updater.update_metrics_detail(self.selected_data, data)
                _fetch_metrics_with_cache.cache_clear()
        else:
            self.app.notify(f"No changes detected in {component}.", severity="warning")

        if self.selected_data:
            self._show_subject_details(self.selected_data)

    def on_editable_tree_edit_requested(self, message: EditableTree.EditRequested) -> None:
        message.stop()
        if self._editing_component and self._editing_component != "tree":
            self.app.notify("Finish the active edit before editing the tree", severity="warning")
            return

        self._start_tree_edit_for_node(message.node)

    def _start_tree_edit_for_node(self, node: TreeNode) -> None:
        node_data = node.data or {}
        node_type = node_data.get("type")

        if node_type not in {"domain", "layer1", "layer2", "subject_entry"}:
            self.app.notify("Selected item cannot be edited", severity="warning")
            self._update_edit_indicator(None)
            return

        path = self._derive_path_from_node(node_type, node_data)
        if path is None:
            self.app.notify("Unable to resolve node path for editing", severity="error")
            self._update_edit_indicator(None)
            return

        current_parent, parent_tree, selection_type = self._build_parent_selection_tree(node_type, node_data)
        if selection_type is None:
            self.app.notify("This node cannot change its parent", severity="warning")
            self._update_edit_indicator(None)
            return

        current_name = node_data.get("name") or str(node.label)

        self._editing_component = "tree"
        self._update_edit_indicator("tree")
        self._last_tree_selection = {
            "node_type": node_type,
            "path": path,
            "node_data": dict(node_data),
        }
        dialog = TreeEditDialog(
            level=node_type,
            current_name=current_name,
            current_parent=current_parent,
            parent_tree=parent_tree,
            parent_selection_type=selection_type,
            pattern=TREE_VALIDATION_RULES.get(node_type, {}).get("pattern"),
        )
        self._active_dialog = dialog
        self.app.push_screen(dialog, callback=self._on_tree_edit_finished)

    def _render_readonly_panels(
        self,
        subject_info: Dict[str, Any],
        metrics: List[Dict[str, Any]],
        sql_entries: List[Dict[str, Any]],
    ) -> None:
        """Render static (read-only) details using the `_create_*_content` helpers."""

        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)

        # Clear existing panels
        for child in list(metrics_container.children):
            child.remove()
        for child in list(sql_container.children):
            child.remove()

        if metrics:
            name = self._create_metrics_panel_content(metrics, subject_info.get("name", ""))
            metrics_container.mount(FocusableStatic(name))
            self._toggle_visibility(metrics_container, True)
        else:
            metrics_container.mount(Static("[dim]No metrics for this item[/dim]"))
            self._toggle_visibility(metrics_container, False)

        if sql_entries:
            group = self._create_sql_panel_content(sql_entries)
            sql_container.mount(FocusableStatic(group))
            self._toggle_visibility(sql_container, True)
        else:
            sql_container.mount(Static("[dim]No reference SQL for this item[/dim]"))
            self._toggle_visibility(sql_container, False)

    def _render_editable_panels(self, metrics: List[Dict[str, Any]], sql_entries: List[Dict[str, Any]]) -> None:
        """Render editable panels (MetricsPanel / ReferenceSqlPanel)."""
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)

        # Clear existing panels
        for child in list(metrics_container.children):
            child.remove()
        for child in list(sql_container.children):
            child.remove()

        if metrics:
            metrics_panel = MetricsPanel(metrics[0], readonly=False)
            metrics_container.mount(metrics_panel)
            self._toggle_visibility(metrics_container, True)
        else:
            metrics_container.mount(Static("[dim]No metrics for this item[/dim]"))
            self._toggle_visibility(metrics_container, False)

        if sql_entries:
            sql_panel = ReferenceSqlPanel(sql_entries[0], readonly=False)
            sql_container.mount(sql_panel)
            self._toggle_visibility(sql_container, True)
        else:
            sql_container.mount(Static("[dim]No reference SQL for this item[/dim]"))
            self._toggle_visibility(sql_container, False)

    def _build_parent_selection_tree(
        self, node_type: str, node_data: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        if node_type == "domain":
            current = {"selection_type": "root"}
            nodes = [
                {
                    "label": "Root",
                    "data": {"selection_type": "root"},
                    "expand": False,
                }
            ]
            return current, nodes, "root"

        if node_type == "layer1":
            domains = sorted(self.tree_data.keys())
            nodes = [
                {
                    "label": domain,
                    "data": {"selection_type": "domain", "domain": domain},
                    "expand": False,
                }
                for domain in domains
            ]
            current_domain = node_data.get("domain")
            current = {"selection_type": "domain", "domain": current_domain} if current_domain else None
            return current, nodes, "domain"

        if node_type == "layer2":
            domain = node_data.get("domain")
            if not domain or domain not in self.tree_data:
                return None, [], None
            nodes: List[Dict[str, Any]] = []
            for domain_name, layer1_map in sorted(self.tree_data.items()):
                layer1_children = []
                for layer1_name in sorted(layer1_map.keys()):
                    layer1_children.append(
                        {
                            "label": layer1_name,
                            "data": {
                                "selection_type": "layer1",
                                "domain": domain_name,
                                "layer1": layer1_name,
                            },
                        }
                    )

                node_entry: Dict[str, Any] = {
                    "label": domain_name,
                    "data": {"selection_type": "domain", "domain": domain_name},
                    "expand": domain_name == domain,
                }
                if layer1_children:
                    node_entry["children"] = layer1_children
                nodes.append(node_entry)
            current_layer1 = node_data.get("layer1")
            current = (
                {
                    "selection_type": "layer1",
                    "domain": domain,
                    "layer1": current_layer1,
                }
                if current_layer1
                else None
            )
            return current, nodes, "layer1"

        if node_type == "subject_entry":
            domain = node_data.get("domain")
            if not domain:
                return None, [], None
            choices: List[Dict[str, Any]] = []
            for domain_name, layer1_map in sorted(self.tree_data.items()):
                layer1_children: List[Dict[str, Any]] = []
                for layer1_name, layer2_map in sorted(layer1_map.items()):
                    layer2_children = [
                        {
                            "label": layer2_name,
                            "data": {
                                "selection_type": "layer2",
                                "domain": domain_name,
                                "layer1": layer1_name,
                                "layer2": layer2_name,
                            },
                        }
                        for layer2_name in sorted(layer2_map.keys())
                    ]

                    layer1_children.append(
                        {
                            "label": layer1_name,
                            "data": {
                                "selection_type": "layer1-context",
                                "domain": domain_name,
                                "layer1": layer1_name,
                            },
                            "expand": domain_name == domain and layer1_name == node_data.get("layer1"),
                            "children": layer2_children,
                        }
                    )

                choices.append(
                    {
                        "label": domain_name,
                        "data": {
                            "selection_type": "domain",
                            "domain": domain_name,
                        },
                        "expand": domain_name == domain,
                        "children": layer1_children,
                    }
                )

            current = (
                {
                    "selection_type": "layer2",
                    "domain": domain,
                    "layer1": node_data.get("layer1"),
                    "layer2": node_data.get("layer2"),
                }
                if node_data.get("layer2")
                else None
            )
            return current, choices, "layer2"

        return None, [], None

    def _derive_path_from_node(self, node_type: str, node_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        try:
            if node_type == "domain":
                return {"domain": node_data["name"]}
            if node_type == "layer1":
                return {"domain": node_data["domain"], "layer1": node_data["name"]}
            if node_type == "layer2":
                return {
                    "domain": node_data["domain"],
                    "layer1": node_data["layer1"],
                    "layer2": node_data["name"],
                }
            if node_type == "subject_entry":
                return {
                    "domain": node_data["domain"],
                    "layer1": node_data["layer1"],
                    "layer2": node_data["layer2"],
                    "name": node_data["name"],
                }
        except KeyError:
            return None
        return None

    def _on_tree_edit_finished(self, result: Optional[Dict[str, Any]]) -> None:
        context = self._last_tree_selection or {}
        node_type = context.get("node_type")
        old_path = context.get("path")

        self._editing_component = None
        self._update_edit_indicator(None)
        self._last_tree_selection = None

        if not node_type or not old_path:
            return

        if not result:
            return

        new_name = result.get("name", "").strip()
        if not new_name:
            self.app.notify("Name cannot be empty", severity="warning")
            return

        parent_value = result.get("parent")
        new_path = dict(old_path)

        if node_type == "domain":
            new_path["domain"] = new_name
        elif node_type == "layer1":
            new_path["layer1"] = new_name
            if parent_value and parent_value.get("selection_type") == "domain":
                new_path["domain"] = parent_value.get("domain", new_path.get("domain", ""))
        elif node_type == "layer2":
            new_path["layer2"] = new_name
            if parent_value and parent_value.get("selection_type") == "layer1":
                new_path["layer1"] = parent_value.get("layer1", new_path.get("layer1", ""))
                new_path["domain"] = parent_value.get("domain", new_path.get("domain", ""))
        elif node_type == "subject_entry":
            new_path["name"] = new_name
            if parent_value and parent_value.get("selection_type") == "layer2":
                new_path["domain"] = parent_value.get("domain", new_path.get("domain", ""))
                new_path["layer1"] = parent_value.get("layer1", new_path.get("layer1", ""))
                new_path["layer2"] = parent_value.get("layer2", new_path.get("layer2", ""))
        if node_type == "subject_entry":
            if old_path == new_path:
                self.app.notify("No changes")
                return
        else:
            if old_path[node_type] == new_path[node_type]:
                self.app.notify("No changes")
                return
        self.subject_updater.update_domain_layers(old_path, update_values=new_path)
        self._show_subject_details(new_path)

        moved_payload = self._apply_tree_edit(node_type, old_path, new_path)
        if moved_payload is None:
            self.app.notify("Failed to update tree data", severity="error")
            return

        self._populate_tree(self.tree_data)
        tree = self.query_one("#subject-tree", EditableTree)
        self._focus_tree_path(tree, new_path, node_type)

        if tree.cursor_node:
            self.update_path_display(tree.cursor_node)

        new_selected_data = self._build_selected_data(node_type, new_path, moved_payload)
        if new_selected_data:
            self.selected_data = new_selected_data

        # self._notify_tree_save(node_type, old_path, new_path)

        # if node_type == "subject_entry" and new_selected_data:
        #     self._show_subject_details(new_selected_data)

    def _apply_tree_edit(self, node_type: str, old_path: Dict[str, str], new_path: Dict[str, str]) -> Optional[Any]:
        if node_type == "domain":
            subtree = self.tree_data.pop(old_path["domain"], None)
            if subtree is None:
                return None
            self.tree_data[new_path["domain"]] = subtree
            return subtree

        if node_type == "layer1":
            source_domain = self.tree_data.get(old_path["domain"], {})
            subtree = source_domain.pop(old_path["layer1"], None)
            if subtree is None:
                return None
            self.tree_data.setdefault(new_path["domain"], {})[new_path["layer1"]] = subtree
            if not source_domain:
                self.tree_data.pop(old_path["domain"], None)
            return subtree

        if node_type == "layer2":
            domain_bucket = self.tree_data.get(old_path["domain"], {})
            layer1_bucket = domain_bucket.get(old_path["layer1"], {})
            subtree = layer1_bucket.pop(old_path["layer2"], None)
            if subtree is None:
                return None
            self.tree_data.setdefault(new_path["domain"], {}).setdefault(new_path["layer1"], {})[
                new_path["layer2"]
            ] = subtree
            if not layer1_bucket:
                domain_bucket.pop(old_path["layer1"], None)
            if not domain_bucket:
                self.tree_data.pop(old_path["domain"], None)
            return subtree

        if node_type == "subject_entry":
            domain_bucket = self.tree_data.get(old_path["domain"], {})
            layer1_bucket = domain_bucket.get(old_path["layer1"], {})
            layer2_bucket = layer1_bucket.get(old_path["layer2"], {})
            payload = layer2_bucket.pop(old_path["name"], None)
            if payload is None:
                return None
            self.tree_data.setdefault(new_path["domain"], {}).setdefault(new_path["layer1"], {}).setdefault(
                new_path["layer2"], {}
            )[new_path["name"]] = payload
            if not layer2_bucket:
                layer1_bucket.pop(old_path["layer2"], None)
            if not layer1_bucket:
                domain_bucket.pop(old_path["layer1"], None)
            if not domain_bucket:
                self.tree_data.pop(old_path["domain"], None)
            return payload

        return None

    def _focus_tree_path(self, tree: EditableTree, path: Dict[str, str], node_type: str) -> None:
        node = tree.root
        traversal_order = [
            ("domain", path.get("domain")),
            ("layer1", path.get("layer1")),
            ("layer2", path.get("layer2")),
            ("subject_entry", path.get("name")),
        ]

        for level_type, target_name in traversal_order:
            if not target_name:
                break
            for child in node.children:
                data = child.data or {}
                if data.get("type") == level_type and data.get("name") == target_name:
                    node = child
                    node.expand()
                    break
            else:
                return

        tree.move_cursor(node)

    def _build_selected_data(self, node_type: str, path: Dict[str, str], payload: Any) -> Optional[Dict[str, Any]]:
        if node_type == "domain":
            return {"type": "domain", "name": path.get("domain", "")}
        if node_type == "layer1":
            return {"type": "layer1", "name": path.get("layer1", ""), "domain": path.get("domain", "")}
        if node_type == "layer2":
            return {
                "type": "layer2",
                "name": path.get("layer2", ""),
                "layer1": path.get("layer1", ""),
                "domain": path.get("domain", ""),
            }
        if node_type == "subject_entry":
            return {
                "type": "subject_entry",
                "name": path.get("name", ""),
                "layer2": path.get("layer2", ""),
                "layer1": path.get("layer1", ""),
                "domain": path.get("domain", ""),
                "metrics_count": payload.get("metrics_count", 0) if isinstance(payload, dict) else 0,
                "sql_count": payload.get("sql_count", 0) if isinstance(payload, dict) else 0,
            }
        return None

    def _show_subject_details(self, subject_info: Dict[str, Any]) -> None:
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)

        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)
        divider = self.query_one("#panel-divider", Static)

        metrics: List[Dict[str, Any]] = []
        sql_entries: List[Dict[str, Any]] = []

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

        # Render depending on mode
        if self.readonly:
            self._render_readonly_panels(subject_info, metrics, sql_entries)
        else:
            self._render_editable_panels(metrics, sql_entries)

        # Layout sizing + divider logic (same for both modes)
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

    def _create_metrics_panel_content(self, metrics: List[Dict[str, Any]], metrics_name: str) -> Group:
        sections: List[Table] = []
        for idx, metric in enumerate(metrics, 1):
            if not isinstance(metric, dict):
                continue

            metric_name = str(metric.get("name", ""))
            semantic_model_name = str(metric.get("semantic_model_name", ""))
            llm_text = str(metric.get("llm_text", ""))

            table = Table(
                title=f"[bold cyan]📊 Metric #{idx}: {metric_name}[/bold cyan]",
                show_header=False,
                box=box.SIMPLE,
                border_style="blue",
                expand=True,
                padding=(0, 1),
            )
            table.add_column("Key", style="bright_cyan", width=20)
            table.add_column("Value", style="yellow", ratio=1)

            if metrics_name:
                table.add_row("Name", metrics_name)
            if semantic_model_name:
                table.add_row("Semantic Model Name", semantic_model_name)
            if llm_text:
                table.add_row("LLM Text", llm_text)

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
        tree = self.query_one("#subject-tree", EditableTree)
        if not tree.has_focus:
            return
        tree.action_cursor_down()
        self.query_one("#tree-help", Static).update("")

    def action_cursor_up(self) -> None:
        tree = self.query_one("#subject-tree", EditableTree)
        if not tree.has_focus:
            return
        tree.action_cursor_up()
        self.query_one("#tree-help", Static).update("")

    def action_expand_node(self) -> None:
        tree = self.query_one("#subject-tree", EditableTree)
        if tree.has_focus and tree.cursor_node:
            tree.cursor_node.expand()

    def action_collapse_node(self) -> None:
        tree = self.query_one("#subject-tree", EditableTree)
        if tree.has_focus and tree.cursor_node:
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

    def action_quit_if_idle(self) -> None:
        """Exit quickly when there is no active edit or dialog."""
        if self._active_dialog is not None and self._active_dialog.is_active:
            return
        if self._editing_component in {"metrics", "sql", "tree"}:
            return
        if not self.readonly:
            return
        self.action_exit_without_selection()

    def _get_panel(self, component: str) -> Optional[MetricsPanel | ReferenceSqlPanel]:
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)
        if component == "metrics":
            query = metrics_container.query(MetricsPanel)
            return query.first() if query else None
        if component == "sql":
            query = sql_container.query(ReferenceSqlPanel)
            return query.first() if query else None
        return None

    def action_cancel_or_exit(self) -> None:
        """
        ESC / Ctrl+Q behavior:
        1) If a dialog is open: restore dialog state and close it.
        2) If in edit mode: restore snapshot and leave edit mode without saving.
        3) Otherwise: perform the original exit behavior.
        """
        # Case 1: active dialog
        if self._active_dialog is not None:
            try:
                self._active_dialog.cancel_and_close()
            except Exception as e:
                logger.warning(f"Failed to restore dialog state: {e}")
                # Fall back to closing the dialog without extra restore
                try:
                    self._active_dialog.dismiss(None)
                except Exception as e2:
                    logger.warning(f"Failed to dismiss dialog: {e2}")
            self._active_dialog = None
            return

        # Case 2: in-panel editing
        if self._editing_component in {"metrics", "sql"}:
            component = self._editing_component
            panel = self._get_panel(component)
            if panel is not None:
                try:
                    panel.restore()
                except Exception as e:
                    logger.warning(f"Failed to restore panel state: {e}")
                panel.set_readonly(True)
            # Reset edit state
            self._editing_component = None
            self._update_edit_indicator(None)
            self.readonly = True
            if self.selected_data:
                self._show_subject_details(self.selected_data)
            return

        # Case 3: default behavior (exit)
        try:
            self.action_exit_without_selection()
        except Exception as e:
            logger.warning(f"Failed to exit without selection: {e}")
            try:
                self.app.pop_screen()
            except Exception as e2:
                logger.warning(f"Failed to pop screen: {e2}")

    def on_unmount(self) -> None:
        self.clear_cache()
        self._subject_updater = None
        self.agent_config = None
        self.sql_rag = None
        self.metrics_rag = None

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
                "• Ctrl+e - Enter edit mode\n"
                "• Ctrl+w - Save and exit edit mode\n"
                "• Esc - Exit editing mode or application\n\n"
                "Press any key to close this help.",
                id="navigation-help-content",
            ),
            id="navigation-help-container",
        )

    def on_key(self, event) -> None:
        self.dismiss()
