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
from datus.storage.ext_knowledge.store import ExtKnowledgeRAG
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.subject_manager import SubjectUpdater
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


TREE_VALIDATION_RULES: Dict[str, Dict[str, str]] = {
    "subject_node": {
        "pattern": r"^[a-zA-Z0-9_\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+$",
        "description": "Subject tree node component",
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

        description_field = InputWithLabel(
            "Description",
            self.entry.get("description", ""),
            lines=10,
            readonly=self.readonly,
            language="markdown",
        )
        self.fields.append(description_field)
        yield description_field

        # dimensions is a list, convert to comma-separated string for display
        dimensions_value = self.entry.get("dimensions", [])
        dimensions_str = (
            ", ".join(dimensions_value) if isinstance(dimensions_value, list) else str(dimensions_value or "")
        )
        dimensions_field = InputWithLabel(
            "Dimensions",
            dimensions_str,
            lines=2,
            readonly=self.readonly,
            language="markdown",
        )
        self.fields.append(dimensions_field)
        yield dimensions_field

        # SQL field
        sql_field = InputWithLabel(
            "SQL",
            self.entry.get("sql", ""),
            lines=5,
            readonly=self.readonly,
            language="sql",
        )
        self.fields.append(sql_field)
        yield sql_field

    def _fill_data(self):
        self.fields[0].set_value(self.entry.get("semantic_model_name", ""))
        self.fields[1].set_value(self.entry.get("description", ""))
        dimensions_value = self.entry.get("dimensions", [])
        dimensions_str = (
            ", ".join(dimensions_value) if isinstance(dimensions_value, list) else str(dimensions_value or "")
        )
        self.fields[2].set_value(dimensions_str)
        self.fields[3].set_value(self.entry.get("sql", ""))

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
            # description remains as "description"
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
        search_text_field = InputWithLabel(
            "Search Text", self.entry.get("search_text", ""), lines=2, readonly=self.readonly, language="markdown"
        )
        self.fields.append(search_text_field)
        yield search_text_field
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
        self.fields[1].set_value(self.entry.get("search_text", ""))
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

        Maps field labels to their storage keys.
        """
        values: Dict[str, str] = {}
        for field in self.fields:
            key = field.label_text.lower()
            if key == "search text":
                key = "search_text"
            values[key] = field.get_value()
        return values

    def restore(self):
        for field in self.fields:
            field.restore()


class ExtKnowledgePanel(Vertical):
    """
    A panel for displaying and editing external knowledge details using DetailField components.
    """

    can_focus = True

    def __init__(self, entry: Dict[str, Any], readonly: bool = True) -> None:
        super().__init__()
        self.entry = entry
        self.readonly = readonly
        self.fields: List[InputWithLabel] = []

    def compose(self) -> ComposeResult:
        knowledge_name = self.entry.get("name", "Unnamed Knowledge")
        yield Label(f"📚 [bold cyan]Knowledge: {knowledge_name}[/]")
        search_text_field = InputWithLabel(
            "search_text",
            self.entry.get("search_text", ""),
            lines=2,
            readonly=self.readonly,
            language="markdown",
        )
        self.fields.append(search_text_field)
        yield search_text_field

        explanation_field = InputWithLabel(
            "Explanation",
            self.entry.get("explanation", ""),
            lines=10,
            readonly=self.readonly,
            language="markdown",
        )
        self.fields.append(explanation_field)
        yield explanation_field

    def _fill_data(self):
        self.fields[0].set_value(self.entry.get("search_text", ""))
        self.fields[1].set_value(self.entry.get("explanation", ""))

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

    def update_data(self, summary_data: Dict[str, Any]):
        self.entry.update(summary_data)
        self._fill_data()

    def focus_first_input(self) -> bool:
        for field in self.fields:
            if field.focus_input():
                return True
        return False


@lru_cache(maxsize=128)
def _fetch_metrics_with_cache(metric_rag: MetricRAG, subject_path_tuple: tuple, name: str) -> List[Dict[str, Any]]:
    """Fetch metrics with caching. subject_path_tuple is a tuple to allow hashing for lru_cache."""
    try:
        subject_path = list(subject_path_tuple) if subject_path_tuple else []
        table = metric_rag.get_metrics_detail(
            subject_path=subject_path,
            name=name,
        )
        return table if table is not None else []
    except Exception as e:
        logger.error(f"Metrics fetch failed: {str(e)}")
        return []


@lru_cache(maxsize=128)
def _sql_details_cache(sql_rag: ReferenceSqlRAG, subject_path_tuple: tuple, name: str) -> List[Dict[str, Any]]:
    """Fetch SQL details with caching. subject_path_tuple is a tuple to allow hashing for lru_cache."""
    try:
        subject_path = list(subject_path_tuple) if subject_path_tuple else []
        table = sql_rag.get_reference_sql_detail(
            subject_path,
            name,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "Failed to fetch SQL details for %s/%s: %s", "/".join(subject_path) if subject_path else "[]", name, exc
        )
        return []

    return table if table is not None else []


class SubjectScreen(ContextScreen):
    """Screen for browsing metrics alongside reference SQL."""

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
        Binding("f7", "create_node", "New Node", show=True),
        # Binding("f8", "delete_node", "Delete Node", show=True), todo lancedb delete
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
        self.metrics_rag: MetricRAG = MetricRAG(self.agent_config)
        self.sql_rag: ReferenceSqlRAG = ReferenceSqlRAG(self.agent_config)
        self.ext_knowledge_rag: ExtKnowledgeRAG = ExtKnowledgeRAG(self.agent_config)
        self.subject_tree_store = self.metrics_rag.storage.subject_tree
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
        self._pending_focus_path: Optional[List[str]] = None  # Path to focus after tree reload

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
            if isinstance(widget, ExtKnowledgePanel):
                return "ext_knowledge", widget
            widget = widget.parent
        return None, None

    def _update_edit_indicator(self, component: Optional[str]) -> None:
        """Update header subtitle to reflect active editing context."""
        header = self.query_one(Header)
        messages = {
            "tree": "✏️ Editing tree node",
            "metrics": "✏️ Editing metrics details",
            "sql": "✏️ Editing reference SQL",
            "ext_knowledge": "✏️ Editing external knowledge",
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
        """Load subject tree data from SubjectTreeStore and attach metrics/SQL counts."""
        get_current_worker()

        try:
            # Step 1: Load complete subject tree from SubjectTreeStore
            tree_structure = self.subject_tree_store.get_tree_structure()

            # Step 2: Initialize tree_data with node metadata
            tree_data = self._init_tree_data_from_structure(tree_structure)

            # Step 3: Fetch and attach subject_entry (metrics/SQL names) as virtual leaf nodes
            self._attach_subject_entries(tree_data)

            self.tree_data = tree_data
            self.app.call_from_thread(self._populate_tree, tree_data)

        except Exception as exc:
            logger.error(f"Failed to load subject tree data: {exc}")
            self.tree_data = {}
            self.app.call_from_thread(self._populate_tree, {})

    def _init_tree_data_from_structure(self, tree_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively initialize tree_data from SubjectTreeStore structure.

        Args:
            tree_structure: Nested dict from subject_tree.get_tree_structure()
                          Format: {"name": {"node_id": int, "name": str, "children": {...}}}

        Returns:
            tree_data with initialized metadata for each node
        """
        tree_data = {}

        for name, node_info in tree_structure.items():
            tree_data[name] = {
                "node_id": node_info["node_id"],
                "name": name,
                "children": {},
                "subject_entries": {},  # Will be populated later
            }

            # Recursively process children
            if "children" in node_info and node_info["children"]:
                tree_data[name]["children"] = self._init_tree_data_from_structure(node_info["children"])

        return tree_data

    def _attach_subject_entries(self, tree_data: Dict[str, Any]) -> None:
        """Dynamically fetch and attach subject_entry (metrics/SQL names) as virtual leaf nodes.
        Each entry is separated by type (metric or sql) with entry_type field.

        Args:
            tree_data: Tree data structure to attach entries to
        """

        def attach_entries_for_node(node_data: Dict[str, Any], subject_path: List[str]) -> None:
            """Recursively attach entries for a node and its children."""
            node_id = node_data.get("node_id")
            if not node_id:
                return

            # Get metrics for this exact node (not descendants)
            metrics_results = self.metrics_rag.storage.list_entries(node_id)
            for metric in metrics_results:
                name = metric.get("name", "")
                if name:
                    # Create separate entry for each metric with unique key
                    entry_key = f"metric:{name}"
                    node_data["subject_entries"][entry_key] = {"name": name, "entry_type": "metric"}

            # Get SQL for this exact node (not descendants)
            sql_results = self.sql_rag.reference_sql_storage.list_entries(node_id)
            for sql_entry in sql_results:
                name = sql_entry.get("name", "")
                if name:
                    # Create separate entry for each SQL with unique key
                    entry_key = f"sql:{name}"
                    node_data["subject_entries"][entry_key] = {"name": name, "entry_type": "sql"}

            # Get ext_knowledge for this exact node (not descendants)
            ext_knowledge_results = self.ext_knowledge_rag.store.list_entries(node_id)
            for ext_knowledge_entry in ext_knowledge_results:
                name = ext_knowledge_entry.get("name", "")
                if name:
                    # Create separate entry for each ext_knowledge with unique key
                    entry_key = f"ext_knowledge:{name}"
                    node_data["subject_entries"][entry_key] = {"name": name, "entry_type": "ext_knowledge"}

            # Recursively attach entries for children
            for child_name, child_data in node_data.get("children", {}).items():
                child_path = subject_path + [child_name]
                attach_entries_for_node(child_data, child_path)

        # Attach entries for all root nodes
        for name, node_data in tree_data.items():
            attach_entries_for_node(node_data, [name])

    def _populate_tree(self, tree_data: Dict[str, Any]) -> None:
        """Populate tree with subject nodes and entries from tree_data."""
        tree = self.query_one("#subject-tree", EditableTree)
        tree.clear()
        tree.root.expand()

        if not tree_data:
            tree.root.add_leaf("📂 No entry found", data={"type": "empty"})
            return

        # Recursively build tree for each root node
        for name, node_data in sorted(tree_data.items()):
            self._add_tree_node_recursive(tree.root, name, node_data)

        # Focus on pending path if set (after tree is fully populated)
        if self._pending_focus_path:
            self._focus_tree_path_by_name(tree, self._pending_focus_path)
            self._pending_focus_path = None

    def _add_tree_node_recursive(self, parent_node: TreeNode, name: str, node_data: Dict[str, Any]) -> TreeNode:
        """Recursively add subject_node and its children to the tree.

        Args:
            parent_node: Parent TreeNode to attach to
            name: Name of the current node
            node_data: Node data containing node_id, children, subject_entries, etc.

        Returns:
            The created TreeNode
        """
        icon = "📁"
        label = f"{icon} {name}"

        # Create subject_node
        tree_node = parent_node.add(
            label,
            data={
                "type": "subject_node",
                "node_id": node_data["node_id"],
                "name": name,
            },
        )

        # Recursively add children (other subject nodes)
        for child_name, child_data in sorted(node_data.get("children", {}).items()):
            self._add_tree_node_recursive(tree_node, child_name, child_data)

        # Add subject_entries as leaf nodes
        for _entry_key, entry_data in sorted(node_data.get("subject_entries", {}).items()):
            # Get entry type and name from entry_data
            entry_type = entry_data.get("entry_type", "")
            entry_name = entry_data.get("name", "")

            # Generate icon based on entry_type
            if entry_type == "metric":
                icon = "📈"
            elif entry_type == "sql":
                icon = "💻"
            elif entry_type == "ext_knowledge":
                icon = "📚"
            else:
                icon = "📋"  # Fallback icon

            entry_label = f"{icon} {entry_name}"
            tree_node.add_leaf(
                entry_label,
                data={
                    "type": "subject_entry",
                    "entry_type": entry_type,  # Add entry_type to node data
                    "name": entry_name,
                    "node_id": node_data["node_id"],  # Parent subject node_id
                },
            )

        return tree_node

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
            name = name.replace("📂 ", "").replace("📋 ", "")
            if name:
                path_parts.insert(0, name)
            current = current.parent

        header = self.query_one(Header)
        if path_parts:
            self.selected_path = "/".join(path_parts)
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
        if self._editing_component in {"metrics", "sql", "ext_knowledge"}:
            self._save_panel_edit(self._editing_component)
            return

        self.app.notify("Nothing to save", severity="warning")

    def _begin_panel_edit(self, component: str, current_widget: Optional[Widget]) -> None:
        if component not in {"metrics", "sql", "ext_knowledge"}:
            return

        if self._editing_component == component:
            return

        if not self.selected_data:
            self.app.notify("Select a subject entry before editing", severity="warning")
            return

        # Switch layout to editable panels if we're currently in read-only mode.
        panel: Optional[MetricsPanel | ReferenceSqlPanel | ExtKnowledgePanel] = None
        if self.readonly:
            self.readonly = False
            self._show_subject_details(self.selected_data)

        if isinstance(current_widget, (MetricsPanel, ReferenceSqlPanel, ExtKnowledgePanel)):
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

        panel: Optional[MetricsPanel | ReferenceSqlPanel | ExtKnowledgePanel] = None
        if component == "metrics":
            if query := metrics_container.query(MetricsPanel):
                panel = query.first()
        elif component == "sql":
            if query := sql_container.query(ReferenceSqlPanel):
                panel = query.first()
        elif component == "ext_knowledge":
            if query := metrics_container.query(ExtKnowledgePanel):
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

            # Extract subject_path and name from selected_data
            node_id = self.selected_data.get("node_id")
            name = self.selected_data.get("name")

            if not node_id or not name:
                self.app.notify("Missing node_id or name in selected data", severity="error")
                return

            subject_path = self.subject_tree_store.get_full_path(node_id)

            if component == "sql":
                logger.info(f"Updating SQL: subject_path={subject_path}, name={name}, data={data}")
                self.subject_updater.update_historical_sql(subject_path, name, data)
                _sql_details_cache.cache_clear()
            elif component == "metrics":
                logger.info(f"Updating metrics: subject_path={subject_path}, name={name}, data={data}")
                self.subject_updater.update_metrics_detail(subject_path, name, data)
                _fetch_metrics_with_cache.cache_clear()
            elif component == "ext_knowledge":
                logger.info(f"Updating ext_knowledge: subject_path={subject_path}, name={name}, data={data}")
                self.subject_updater.update_ext_knowledge(subject_path, name, data)
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

        if node_type not in {"subject_node", "subject_entry"}:
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
        ext_knowledge_entries: Optional[List[Dict[str, Any]]] = None,
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
        elif ext_knowledge_entries:
            # Display ext_knowledge in metrics container when no metrics
            group = self._create_ext_knowledge_panel_content(ext_knowledge_entries)
            metrics_container.mount(FocusableStatic(group))
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

    def _render_editable_panels(
        self,
        metrics: List[Dict[str, Any]],
        sql_entries: List[Dict[str, Any]],
        ext_knowledge_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Render editable panels (MetricsPanel / ReferenceSqlPanel / ExtKnowledgePanel)."""
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
        elif ext_knowledge_entries:
            # Display ext_knowledge panel in metrics container when no metrics
            ext_knowledge_panel = ExtKnowledgePanel(ext_knowledge_entries[0], readonly=False)
            metrics_container.mount(ext_knowledge_panel)
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
        """Build parent selection tree for editing nodes.

        Args:
            node_type: Type of node being edited ("subject_node" or "subject_entry")
            node_data: Data of the node being edited

        Returns:
            Tuple of (current_parent, parent_tree_nodes, selection_type)
        """
        if node_type == "subject_entry":
            # subject_entry can only move to subject_node (not root)
            # For subject_entry, node_id is the parent node's ID (not the entry itself)
            parent_node_id = node_data.get("node_id")
            return self._build_subject_node_selection_tree(
                current_node_id=None,  # Don't look for parent of parent
                current_parent_id=parent_node_id,  # Directly specify current parent
                allow_root=False,
            )

        elif node_type == "subject_node":
            # subject_node can move to any other subject_node or root
            # Exclude self and all descendants
            current_node_id = node_data.get("node_id")
            return self._build_subject_node_selection_tree(
                current_node_id=current_node_id, exclude_node_id=current_node_id, allow_root=True
            )

        return None, [], None

    def _build_subject_node_selection_tree(
        self,
        current_node_id: Optional[int] = None,
        exclude_node_id: Optional[int] = None,
        allow_root: bool = True,
        current_parent_id: Optional[int] = None,
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        """Build selection tree from subject_tree, excluding descendants of exclude_node_id.

        Args:
            current_node_id: ID of the current node (to determine current parent)
            exclude_node_id: ID of node to exclude along with its descendants
            allow_root: Whether to allow selection of root (None parent)
            current_parent_id: Direct ID of current parent (for subject_entry where node_id is the parent)

        Returns:
            Tuple of (current_parent, selection_nodes, selection_type)
        """

        # Get current parent
        current_parent = None
        if current_parent_id is not None:
            # For subject_entry, current_parent_id is directly specified
            current_parent = {"selection_type": "subject_node", "node_id": current_parent_id}
        elif current_node_id:
            try:
                node = self.subject_tree_store.get_node(current_node_id)
                if node and node.get("parent_id"):
                    current_parent = {"selection_type": "subject_node", "node_id": node["parent_id"]}
                elif allow_root:
                    current_parent = {"selection_type": "root"}
            except Exception as e:
                logger.warning(f"Failed to get current node parent: {e}")

        # Build excluded IDs set
        excluded_ids = set()
        if exclude_node_id:
            excluded_ids.add(exclude_node_id)
            try:
                descendants = self.subject_tree_store.get_descendants(exclude_node_id)
                excluded_ids.update(d["node_id"] for d in descendants)
            except Exception as e:
                logger.warning(f"Failed to get descendants for exclusion: {e}")

        # Build selection tree
        nodes = []
        if allow_root:
            nodes.append(
                {
                    "label": "Root",
                    "data": {"selection_type": "root"},
                    "expand": current_parent and current_parent.get("selection_type") == "root",
                }
            )

        # Add all subject nodes except excluded ones
        for _name, node_info in sorted(self.tree_data.items()):
            if node_info["node_id"] not in excluded_ids:
                selection_node = self._build_selection_node_recursive(node_info, excluded_ids, current_parent)
                nodes.append(selection_node)

        return current_parent, nodes, "subject_node"

    def _build_selection_node_recursive(
        self, node_info: Dict, excluded_ids: set, current_parent: Optional[Dict]
    ) -> Dict[str, Any]:
        """Recursively build selection tree node.

        Args:
            node_info: Node info from tree_data
            excluded_ids: Set of node IDs to exclude
            current_parent: Current parent selection for expansion

        Returns:
            Selection node dictionary
        """
        node_data = {"selection_type": "subject_node", "node_id": node_info["node_id"]}

        should_expand = (
            current_parent
            and current_parent.get("selection_type") == "subject_node"
            and current_parent.get("node_id") == node_info["node_id"]
        )

        result = {"label": node_info["name"], "data": node_data, "expand": should_expand}

        # Recursively add children
        children = []
        for _child_name, child_info in sorted(node_info.get("children", {}).items()):
            if child_info["node_id"] not in excluded_ids:
                child_node = self._build_selection_node_recursive(child_info, excluded_ids, current_parent)
                children.append(child_node)

        if children:
            result["children"] = children

        return result

    def _derive_path_from_node(self, node_type: str, node_data: Dict[str, Any]) -> Optional[List[str]]:
        """Derive full path from node data.

        Args:
            node_type: Type of node ("subject_node" or "subject_entry")
            node_data: Node data containing node_id and other fields

        Returns:
            List of path components (e.g., ["Finance", "Revenue", "Q1"])
        """
        try:
            if node_type == "subject_node":
                # Get path from subject_tree using node_id
                node_id = node_data.get("node_id")
                if not node_id:
                    return None
                return self.subject_tree_store.get_full_path(node_id)

            elif node_type == "subject_entry":
                # Path = subject_node_path + entry_name
                node_id = node_data.get("node_id")  # Parent subject node's ID
                if not node_id:
                    return None

                subject_path = self.subject_tree_store.get_full_path(node_id)
                entry_name = node_data.get("name")
                if not entry_name:
                    return None

                return subject_path + [entry_name]

        except Exception as e:
            logger.error(f"Failed to derive path from node: {e}")
            return None

        return None

    def _on_tree_edit_finished(self, result: Optional[Dict[str, Any]]) -> None:
        """Handle tree edit completion with new path-based structure.

        Args:
            result: Dict with "name" (new name) and "parent" (new parent selection)
        """
        logger.info(f"edit:{result}")
        context = self._last_tree_selection or {}
        node_type = context.get("node_type")
        old_path = context.get("path")  # List[str]

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

        # Build new path based on node type and parent selection
        parent_value = result.get("parent")
        new_path = self._build_new_path_from_result(node_type, old_path, new_name, parent_value)

        if new_path is None:
            self.app.notify("Failed to build new path", severity="error")
            return

        # Check if there are actual changes
        if old_path == new_path:
            self.app.notify("No changes")
            return

        try:
            # Apply the edit to SubjectTreeStore or LanceDB

            if node_type == "subject_node":
                # Rename/move subject node in tree
                self.subject_tree_store.rename(old_path, new_path)

            elif node_type == "subject_entry":
                # Rename subject_entry in LanceDB (storage)
                # Get entry_type from context to determine which storage to update
                node_data = context.get("node_data", {})
                entry_type = node_data.get("entry_type", "")

                if entry_type == "metric":
                    # Only rename in metric storage
                    try:
                        self.metrics_rag.storage.rename(old_path, new_path)
                        _fetch_metrics_with_cache.cache_clear()
                    except Exception as e:
                        logger.warning(f"Failed to rename metric entry: {e}")
                elif entry_type == "sql":
                    # Only rename in SQL storage
                    try:
                        self.sql_rag.reference_sql_storage.rename(old_path, new_path)
                        _sql_details_cache.cache_clear()
                    except Exception as e:
                        logger.warning(f"Failed to rename SQL entry: {e}")
                elif entry_type == "ext_knowledge":
                    # Only rename in ext_knowledge storage
                    try:
                        self.ext_knowledge_rag.store.rename(old_path, new_path)
                    except Exception as e:
                        logger.warning(f"Failed to rename ext_knowledge entry: {e}")
                else:
                    # Fallback: rename in all storages for backward compatibility
                    try:
                        self.metrics_rag.storage.rename(old_path, new_path)
                        _fetch_metrics_with_cache.cache_clear()
                    except Exception as e:
                        logger.warning(f"Failed to rename metric entry: {e}")

                    try:
                        self.sql_rag.reference_sql_storage.rename(old_path, new_path)
                        _sql_details_cache.cache_clear()
                    except Exception as e:
                        logger.warning(f"Failed to rename SQL entry: {e}")

                    try:
                        self.ext_knowledge_rag.store.rename(old_path, new_path)
                    except Exception as e:
                        logger.warning(f"Failed to rename ext_knowledge entry: {e}")

            # Store path to focus after tree reload, then reload
            self._pending_focus_path = new_path
            self._current_loading_task = self.run_worker(self._load_subject_tree_data, thread=True)

            self.app.notify(f"Successfully renamed to {'/'.join(new_path)}", severity="success")

        except ValueError as e:
            self.app.notify(f"Failed to rename: {str(e)}", severity="error")
            logger.error(f"Rename failed: {e}")
        except Exception as e:
            self.app.notify("Failed to apply changes", severity="error")
            logger.error(f"Unexpected error during rename: {e}")

    def _build_new_path_from_result(
        self, node_type: str, old_path: List[str], new_name: str, parent_value: Optional[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """Build new path from edit result.

        Args:
            node_type: Type of node being edited ("subject_node" or "subject_entry")
            old_path: Original path as List[str]
            new_name: New name for the node
            parent_value: Selected parent from dialog (may be None)

        Returns:
            New path as List[str], or None if invalid
        """
        try:
            if node_type == "subject_node":
                # Build parent path from parent_value
                if parent_value is None or parent_value.get("selection_type") == "root":
                    # Root level node
                    return [new_name]
                elif parent_value.get("selection_type") == "subject_node":
                    # Get parent path from parent node_id
                    parent_node_id = parent_value.get("node_id")
                    if parent_node_id:
                        parent_path = self.subject_tree_store.get_full_path(parent_node_id)
                        return parent_path + [new_name]

            elif node_type == "subject_entry":
                # For subject_entry, parent is the subject_node
                if parent_value and parent_value.get("selection_type") == "subject_node":
                    parent_node_id = parent_value.get("node_id")
                    if parent_node_id:
                        parent_path = self.subject_tree_store.get_full_path(parent_node_id)
                        return parent_path + [new_name]
                else:
                    # No parent change, keep same parent path, just change name
                    return old_path[:-1] + [new_name]

            return None

        except Exception as e:
            logger.error(f"Failed to build new path: {e}")
            return None

    def _focus_tree_path_by_name(self, tree: EditableTree, path: List[str]) -> None:
        """Focus tree node by traversing the path using name matching.

        Args:
            tree: The EditableTree widget
            path: List of path components (e.g., ["Finance", "Revenue", "Q1"])
        """
        if not path:
            return

        node = tree.root
        for target_name in path:
            # Remove icon prefixes from label for matching
            found = False
            for child in node.children:
                child_name = child.data.get("name", "") if child.data else ""
                # Strip icon prefixes
                child_name = child_name.replace("📁", "").replace("📈", "").replace("💻", "").replace("📚", "").strip()
                if child_name == target_name:
                    node = child
                    node.expand()
                    found = True
                    break
            if not found:
                logger.warning(f"Could not find node '{target_name}' in path")
                return

        tree.move_cursor(node)

    def _show_subject_details(self, subject_info: Dict[str, Any]) -> None:
        """Show metrics, SQL, and ext_knowledge details for the selected subject.

        Args:
            subject_info: Dict containing node data with new structure:
                - For subject_node: {"type": "subject_node", "node_id": int, "name": str, ...}
                - For subject_entry: {"type": "subject_entry", "node_id": int, "name": str, ...}
        """
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)
        divider = self.query_one("#panel-divider", Static)

        metrics: List[Dict[str, Any]] = []
        sql_entries: List[Dict[str, Any]] = []
        ext_knowledge_entries: List[Dict[str, Any]] = []

        node_type = subject_info.get("type")
        node_id = subject_info.get("node_id")
        name = subject_info.get("name", "")
        entry_type = subject_info.get("entry_type", "")

        if node_type == "subject_entry" and node_id and name:
            # Fetch only the specific type based on entry_type
            if entry_type == "metric":
                metrics = self._fetch_metrics_by_path_and_name(node_id, name)
            elif entry_type == "sql":
                sql_entries = self._fetch_sql_by_path_and_name(node_id, name)
            elif entry_type == "ext_knowledge":
                ext_knowledge_entries = self._fetch_ext_knowledge_by_path_and_name(node_id, name)

        # Render depending on mode
        if self.readonly:
            self._render_readonly_panels(subject_info, metrics, sql_entries, ext_knowledge_entries)
        else:
            self._render_editable_panels(metrics, sql_entries, ext_knowledge_entries)

        # Layout sizing + divider logic (same for both modes)
        metrics_visible = bool(metrics)
        sql_visible = bool(sql_entries)
        ext_knowledge_visible = bool(ext_knowledge_entries)

        # Calculate visible count for layout
        visible_count = sum([metrics_visible, sql_visible, ext_knowledge_visible])

        if visible_count == 0:
            metrics_container.styles.height = "50%"
            sql_container.styles.height = "50%"
        elif visible_count == 1:
            if metrics_visible:
                metrics_container.styles.height = "100%"
                sql_container.styles.height = "0%"
            elif sql_visible:
                sql_container.styles.height = "100%"
                metrics_container.styles.height = "0%"
            elif ext_knowledge_visible:
                # Use metrics container for ext_knowledge display
                metrics_container.styles.height = "100%"
                sql_container.styles.height = "0%"
        else:
            metrics_container.styles.height = "50%"
            sql_container.styles.height = "50%"

        self._toggle_visibility(divider, visible_count >= 2)

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
            description = str(metric.get("description", ""))
            dimensions_value = metric.get("dimensions", [])
            dimensions_str = (
                ", ".join(dimensions_value) if isinstance(dimensions_value, list) else str(dimensions_value or "")
            )

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
            if description:
                table.add_row("Description", description)
            if dimensions_str:
                table.add_row("Dimensions", dimensions_str)
            sql_value = metric.get("sql", "")
            if sql_value:
                table.add_row(
                    "SQL",
                    Syntax(sql_value, "sql", theme="monokai", word_wrap=True, line_numbers=True),
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
            if search_text := sql_entry.get("search_text"):
                details.add_row("Search Text", search_text)
            if tags := sql_entry.get("tags"):
                details.add_row("Tags", build_historical_sql_tags(tags))

            details.add_row(
                "SQL",
                Syntax(str(sql_entry.get("sql", "")), "sql", theme="monokai", word_wrap=True, line_numbers=True),
            )

            sections.append(details)

        return Group(*sections)

    def _create_ext_knowledge_panel_content(self, ext_knowledge_entries: List[Dict[str, Any]]) -> Group:
        sections: List[Table] = []
        for idx, entry in enumerate(ext_knowledge_entries, 1):
            details = Table(
                title=f"[bold cyan]📚 Knowledge #{idx}: {entry.get('name', 'Unnamed')}[/bold cyan]",
                show_header=False,
                box=box.SIMPLE,
                border_style="blue",
                expand=True,
                padding=(0, 1),
            )
            details.add_column("Key", style="bright_cyan", width=12)
            details.add_column("Value", style="yellow", ratio=1)

            if search_text := entry.get("search_text"):
                details.add_row("SearchText", search_text)
            if explanation := entry.get("explanation"):
                details.add_row("Explanation", explanation)

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

    def action_create_node(self) -> None:
        """Create a new subject node as child of the currently selected node or at root level."""
        tree = self.query_one("#subject-tree", EditableTree)

        # Determine parent node and ID
        parent_node_id = None
        current_parent = {"selection_type": "root"}

        if tree.cursor_node:
            parent_node = tree.cursor_node
            parent_data = parent_node.data or {}

            # Only subject_node can have children (not subject_entry or other types)
            if parent_data.get("type") == "subject_node":
                parent_node_id = parent_data.get("node_id")
                if parent_node_id:
                    current_parent = {"selection_type": "subject_node", "node_id": parent_node_id}
            elif parent_data.get("type") in {"subject_entry", "loading", "empty"}:
                # If cursor is on non-subject_node, create at root level
                pass

        # Build parent selection tree (with root option)
        nodes = []
        # Add root option
        nodes.append(
            {
                "label": "Root",
                "data": {"selection_type": "root"},
                "expand": current_parent.get("selection_type") == "root",
            }
        )

        # Add all subject nodes as potential parents
        for _name, node_info in sorted(self.tree_data.items()):
            selection_node = self._build_selection_node_recursive(node_info, set(), current_parent)
            nodes.append(selection_node)

        # Set editing state and show dialog
        self._editing_component = "tree"
        self._update_edit_indicator("tree")
        self._last_tree_selection = {
            "action": "create",
            "parent_node_id": parent_node_id,
        }

        dialog = TreeEditDialog(
            level="subject_node",
            current_name="",
            current_parent=current_parent,
            parent_tree=nodes,
            parent_selection_type="subject_node",
            pattern=TREE_VALIDATION_RULES.get("subject_node", {}).get("pattern"),
        )
        self._active_dialog = dialog
        self.app.push_screen(dialog, callback=self._on_create_node_finished)

    def _on_create_node_finished(self, result: Optional[Dict[str, Any]]) -> None:
        """Handle create node dialog completion."""
        self._editing_component = None
        self._update_edit_indicator(None)
        self._last_tree_selection = None
        self._active_dialog = None

        if not result:
            return

        new_name = result.get("name", "").strip()
        if not new_name:
            self.app.notify("Name cannot be empty", severity="warning")
            return

        parent_value = result.get("parent")
        if not parent_value:
            self.app.notify("Invalid parent selection", severity="error")
            return

        # Determine parent_node_id and path based on selection type
        parent_node_id = None
        new_path = [new_name]

        if parent_value.get("selection_type") == "root":
            # Create at root level (parent_node_id = None)
            parent_node_id = None
            new_path = [new_name]
        elif parent_value.get("selection_type") == "subject_node":
            # Create under a subject node
            parent_node_id = parent_value.get("node_id")
            if not parent_node_id:
                self.app.notify("Invalid parent node", severity="error")
                return
            try:
                parent_path = self.subject_tree_store.get_full_path(parent_node_id)
                new_path = parent_path + [new_name]
            except Exception as e:
                self.app.notify("Failed to get parent path", severity="error")
                logger.error(f"Failed to get parent path: {e}")
                return
        else:
            self.app.notify("Invalid parent selection type", severity="error")
            return

        try:
            # Create the new node (parent_node_id can be None for root level)
            self.subject_tree_store.create_node(parent_node_id, new_name)

            # Reload tree
            self._current_loading_task = self.run_worker(self._load_subject_tree_data, thread=True)

            self.app.notify(f"Created node: {'/'.join(new_path)}", severity="success")

        except ValueError as e:
            self.app.notify(f"Failed to create node: {str(e)}", severity="error")
            logger.error(f"Create node failed: {e}")
        except Exception as e:
            self.app.notify("Failed to create node", severity="error")
            logger.error(f"Unexpected error during create: {e}")

    def action_delete_node(self) -> None:
        """Delete the currently selected node."""
        tree = self.query_one("#subject-tree", EditableTree)
        if not tree.cursor_node:
            self.app.notify("Select a node to delete", severity="warning")
            return

        node = tree.cursor_node
        node_data = node.data or {}
        node_type = node_data.get("type")

        if node_type != "subject_node":
            self.app.notify("Can only delete subject nodes", severity="warning")
            return

        node_id = node_data.get("node_id")
        node_name = node_data.get("name", "")

        if not node_id:
            self.app.notify("Invalid node", severity="error")
            return

        # Check if node has children
        try:
            children = self.subject_tree_store.get_children(node_id)
            if children:
                self.app.notify(f"Cannot delete: '{node_name}' has {len(children)} child(ren).", severity="warning")
                return
        except Exception:
            pass

        # Store path for potential refresh
        try:
            node_path = self.subject_tree_store.get_full_path(node_id)
        except Exception:
            node_path = [node_name] if node_name else []

        # Confirm and delete
        self._editing_component = "tree"
        self._update_edit_indicator("tree")
        self._last_tree_selection = {
            "action": "delete",
            "node_id": node_id,
            "node_path": node_path,
        }

        # For simplicity, just delete directly (could add confirmation dialog)
        try:
            self.subject_tree_store.delete_node(node_id)

            # Clear caches
            _fetch_metrics_with_cache.cache_clear()
            _sql_details_cache.cache_clear()

            # Reload tree
            self._current_loading_task = self.run_worker(self._load_subject_tree_data, thread=True)

            self.app.notify(f"Deleted node: {'/'.join(node_path)}", severity="success")

        except ValueError as e:
            self.app.notify(f"Failed to delete: {str(e)}", severity="error")
            logger.error(f"Delete failed: {e}")
        except Exception as e:
            self.app.notify("Failed to delete node", severity="error")
            logger.error(f"Unexpected error during delete: {e}")
        finally:
            self._editing_component = None
            self._update_edit_indicator(None)
            self._last_tree_selection = None

    def _get_panel(self, component: str) -> Optional[MetricsPanel | ReferenceSqlPanel | ExtKnowledgePanel]:
        metrics_container = self.query_one("#metrics-panel-container", ScrollableContainer)
        sql_container = self.query_one("#sql-panel-container", ScrollableContainer)
        if component == "metrics":
            query = metrics_container.query(MetricsPanel)
            return query.first() if query else None
        if component == "sql":
            query = sql_container.query(ReferenceSqlPanel)
            return query.first() if query else None
        if component == "ext_knowledge":
            query = metrics_container.query(ExtKnowledgePanel)
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
        if self._editing_component in {"metrics", "sql", "ext_knowledge"}:
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

    def _fetch_metrics_by_path_and_name(self, node_id: int, name: str) -> List[Dict[str, Any]]:
        """Fetch metrics by subject node ID and entry name.

        Args:
            node_id: Subject node ID
            name: Entry name

        Returns:
            List of metric entries matching the criteria
        """
        return self.metrics_rag.storage.list_entries(node_id=node_id, name=name)

    def _fetch_sql_by_path_and_name(self, node_id: int, name: str) -> List[Dict[str, Any]]:
        """Fetch SQL entries by subject node ID and entry name.

        Args:
            node_id: Subject node ID
            name: Entry name

        Returns:
            List of SQL entries matching the criteria
        """
        return self.sql_rag.reference_sql_storage.list_entries(node_id=node_id, name=name)

    def _fetch_ext_knowledge_by_path_and_name(self, node_id: int, name: str) -> List[Dict[str, Any]]:
        """Fetch ext_knowledge entries by subject node ID and entry name.

        Args:
            node_id: Subject node ID
            name: Entry name

        Returns:
            List of ext_knowledge entries matching the criteria
        """
        return self.ext_knowledge_rag.store.list_entries(node_id=node_id, name=name)


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
                "• F7 - Create new node\n"
                # "• F8 - Delete selected node\n"
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
