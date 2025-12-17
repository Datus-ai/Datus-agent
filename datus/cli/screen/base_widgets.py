# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Input, Label, Static, TextArea, Tree
from textual.widgets._tree import TreeNode

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SelectableTree(Tree):
    """Custom tree component, only leaf nodes are allowed to be selected"""

    BINDINGS = [
        Binding("enter", "toggle_or_select", "Choose", show=True),
        Binding("right", "toggle_node", "Toggle", show=True),
        Binding("left", "toggle_node", "Toggle", show=True),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_leaf = None  # Store the currently selected leaf node

    def is_leaf_node(self, node: TreeNode) -> bool:
        """Determine whether it is a selectable leaf node"""
        if node is None:
            return False

        # Branch nodes (allow_expand=True) are not leaves even if not populated yet
        if getattr(node, "allow_expand", True):
            return False

        return self._get_node_level(node) == self._get_max_level()

    def _get_node_level(self, node: TreeNode) -> int:
        """Return the depth level of a node relative to the root"""
        level = 0
        current = node
        while current is not None and current.parent is not None:
            level += 1
            current = current.parent
        return level

    def _get_max_level(self) -> int:
        """Compute the deepest level currently present in the tree"""
        max_level = 0
        stack: list[tuple[TreeNode, int]] = [(self.root, 0)]
        while stack:
            current, level = stack.pop()
            if level > max_level:
                max_level = level
            for child in current.children:
                stack.append((child, level + 1))
        return max_level

    def _ensure_selected_leaf_valid(self) -> None:
        """Remove selection marker if cached node is no longer selectable"""
        if self.selected_leaf and not self.is_leaf_node(self.selected_leaf):
            old_label = str(self.selected_leaf.label).replace("✓ ", "")
            self.selected_leaf.set_label(old_label)
            self.selected_leaf = None

    def action_toggle_or_select(self):
        # Only leaf nodes can be selected
        if self.cursor_node is None:
            return

        if not self.is_leaf_node(self.cursor_node):
            self.cursor_node.toggle()
            return
        return self.action_select_node()

    def action_select_node(self) -> None:
        """Select the node where the current cursor is located (leaf node only)"""
        if self.cursor_node is None:
            return

        # Ensure previous selection is still valid before proceeding
        self._ensure_selected_leaf_valid()

        # Only leaf nodes can be selected
        if not self.is_leaf_node(self.cursor_node):
            if self.app:
                self.app.notify("Only deepest leaf nodes can be selected", severity="warning")
            return

        # Cancel the previously selected node
        if self.selected_leaf:
            old_label = str(self.selected_leaf.label).replace("✓ ", "")
            self.selected_leaf.set_label(old_label)

        # Select a new node
        self.selected_leaf = self.cursor_node
        new_label = f"✓ {self.cursor_node.label}"
        self.cursor_node.set_label(new_label)

        if self.app:
            self.app.notify(f"Selected: {self.cursor_node.label}", severity="information")

    def set_default_selection(self, node_path: list[str]) -> None:
        """Set the default selected node

        Args:
            node_path: Node path list, such as ["root node", "child node 1", "leaves 1"]
        """
        current = self.root

        # Traverse the path to find the target node
        for i, label in enumerate(node_path):
            if i == 0:  # Skip the root node
                continue

            found = False
            for child in current.children:
                if str(child.label) == label:
                    current = child
                    found = True
                    break

            if not found:
                if self.app:
                    self.app.notify(f"Node not found: {label}", severity="error")
                return

        # Make sure it is a leaf node
        if not self.is_leaf_node(current):
            self.app.notify("The default selected must be a leaf node!", severity="error")
            return

        # Settings are selected
        self.selected_leaf = current
        current.set_label(f"✓ {current.label}")

        # Expand parent node path
        node = current.parent
        while node and node != self.root:
            node.expand()
            node = node.parent


class _NodeNamePrompt(ModalScreen[Optional[str]]):
    """Simple modal prompt that asks user for a node name"""

    DEFAULT_CSS = """
    _NodeNamePrompt {
        align: center middle;
    }
    _NodeNamePrompt Container {
        width: 60%;
        max-width: 60;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    _NodeNamePrompt Label {
        padding-bottom: 1;
    }
    _NodeNamePrompt Input {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Please enter the node name")
            yield Input(placeholder="After entering, press Enter to confirm", id="node-name-input")

    def on_mount(self) -> None:
        input_widget = self.query_one("#node-name-input", Input)
        self.call_after_refresh(input_widget.focus)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            if self.app:
                self.app.notify("Node name cannot be empty", severity="warning")
            return
        self.dismiss(value)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)


class InputWithLabel(Widget):
    """
    A horizontal layout containing a label and an editable component (Input or TextArea).
    Tracks the original value for change detection and supports read-only mode.
    """

    DEFAULT_CSS = """
    InputWithLabel {
        layout: horizontal;
        min-height: 4;
        height: auto;
        # align: center ;
    }
    InputWithLabel Label {
        padding: 1;
        width: 10%;
        text-align: right;
    }
    InputWithLabel TextArea {
        width: 1fr;
    }
    InputWithLabel Input {
        width: 1fr;
        border: round $accent;
        padding-left: 1;
        padding-right: 1;
    }
    """

    def __init__(
        self,
        label: str,
        value: str,
        lines: int = 1,
        readonly: bool = False,
        language: str | None = None,
        label_color: str = "cyan",
        regex: str | re.Pattern | None = None,
        **kwargs,
    ) -> None:
        """
        :param label: The text displayed for the field label.
        :param value: The initial value for the input component.
        :param multiline: Whether to use a TextArea instead of Input.
        :param readonly: If True, disables editing of the input.
        :param label_color: The default colour applied to the label text.
        """
        super().__init__(**kwargs)
        self.label_text = label
        self.original_value = value
        self.lines = lines
        self.readonly = readonly
        self.language = language
        self.label_color = label_color
        if regex:
            self.regex = re.compile(regex) if isinstance(regex, str) else regex
        else:
            self.regex = None
        self._last_valid = value
        self._last_cursor_location = len(value)
        self.input_widget: Optional[Input | TextArea] = None

    def compose(self) -> ComposeResult:
        label_widget = Label(Text(f"{self.label_text}:", style=self.label_color))
        yield label_widget

        text_area = TextArea(
            text=self.original_value,
            language=self.language,
            show_line_numbers=False,
            compact=True,
            read_only=self.readonly,
        )
        text_area.styles.margin = 0
        text_area.styles.height = self.lines * 2
        self.input_widget = text_area
        yield text_area

    def set_readonly(self, readonly: bool) -> None:
        """
        Toggle the read-only mode for this field.
        """
        self.readonly = readonly
        if self.input_widget:
            self.input_widget.read_only = readonly

    def is_modified(self) -> bool:
        """
        Return True if the value has been changed since initialization.
        """
        return self.get_value() != self.original_value

    def get_value(self) -> str:
        """
        Return the current value from the input widget.
        """
        if self.input_widget:
            return self.input_widget.text if isinstance(self.input_widget, TextArea) else self.input_widget.value
        return self.original_value

    # # --- Regex validation handlers ---
    # async def on_input_changed(self, event: Input.Changed) -> None:
    #     """
    #     Handle changes to the single‑line Input. If a regex is provided,
    #     revert the change when the new value doesn’t match.
    #     """
    #     if event.input is not self.input_widget or not self.regex:
    #         return
    #
    #     # If entire value matches, record it; otherwise revert to last valid
    #     if self.regex.fullmatch(event.value) or event.value == "":
    #         self._last_valid = event.value
    #         self._last_cursor_location = event.input.cursor_position
    #     else:
    #         event.input.value = self._last_valid
    #         event.input.cursor_position = min(self._last_cursor_location, len(self._last_valid))
    #     event.stop()

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """
        Handle changes to the multi‑line TextArea. Works similarly to the Input handler.
        """
        if event.text_area is not self.input_widget or not self.regex:
            return
        last_cursor = event.text_area.cursor_location
        current_text = event.text_area.text
        current_cursor = event.text_area.cursor_location
        if self.regex.fullmatch(current_text) or current_text == "":
            self._last_valid = current_text
            self._last_cursor_location = current_cursor
        else:
            event.text_area.text = self._last_valid
            if hasattr(event.text_area, "cursor_location"):
                event.text_area.cursor_location = last_cursor or (0, 0)
        event.stop()

    def restore(self) -> None:
        if isinstance(self.input_widget, TextArea):
            self.input_widget.text = self.original_value
        elif isinstance(self.input_widget, Input):
            self.input_widget.value = self.original_value
        self._last_valid = self.original_value
        self._last_cursor_location = len(self.original_value)

    def set_value(self, value: str):
        self.original_value = value

        self.input_widget.text = value
        self._last_valid = value
        self._last_cursor_location = len(value)

    def focus_input(self) -> bool:
        if self.input_widget is None or self.app is None:
            return False
        self.app.call_after_refresh(self.input_widget.focus)
        return True

    @property
    def cursor_position(self) -> int:
        row, col = self.input_widget.cursor_location
        lines = self.input_widget.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + col

    @cursor_position.setter
    def cursor_position(self, position: int) -> None:
        # Convert 1D position to 2D cursor location
        lines = self.input_widget.text.split("\n")
        current_pos = 0
        for row, line in enumerate(lines):
            if current_pos + len(line) >= position:
                col = position - current_pos
            self.input_widget.cursor_location = (row, col)
            current_pos += len(line) + 1  # +1 for newline


class FocusableStatic(Static):
    can_focus = True
    DEFAULT_CSS = """
    FocusableStatic:focus,
    FocusableStatic:focus-within {
        background: $foreground 10%;
        color: $text;
    }
    """


class EditableTree(Tree):
    """Textual Tree with helper hooks for edit requests."""

    class EditRequested(Message):
        """Message emitted when an edit is requested for the current node."""

        def __init__(self, tree: "EditableTree", node: TreeNode) -> None:
            self.tree = tree
            self.node = node
            super().__init__()

    def __init__(self, *args, editable: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.editable = editable

    def request_edit(self) -> None:
        """Emit an edit request for the current cursor node."""
        if not self.editable:
            return
        node = self.cursor_node
        if node and node.data:
            self.post_message(self.EditRequested(self, node))

    def action_start_edit(self) -> None:
        """Textual action hook for triggering an edit."""
        self.request_edit()


class ParentSelectionTree(SelectableTree):
    """Hierarchical selector used inside the tree edit dialog."""

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        allowed_type: str,
        current_selection: Optional[Dict[str, Any]] = None,
        label: str = "Change Parent",
        tree_id: str = "tree-parent-selector",
    ) -> None:
        super().__init__(label=label, id=tree_id)
        self.nodes = nodes
        self.allowed_type = allowed_type
        self._selected = current_selection or {}
        self.current_selection = current_selection or {}

    def on_mount(self) -> None:
        """Mount the tree and populate it"""
        self.root.expand()
        self._populate(self.root, self.nodes)
        self._focus_current_selection()

    def _populate(self, parent: TreeNode, children: List[Dict[str, Any]]) -> None:
        """Recursively populate the tree structure"""
        for child in children:
            label = child.get("label", "")
            data = child.get("data", {})
            children_payload = child.get("children")
            force_branch = child.get("force_branch", False)

            # Check if this node has children to determine if it should be a branch or leaf
            has_children = bool(children_payload) or force_branch

            if has_children:
                node = parent.add(label, data=data)
            else:
                node = parent.add_leaf(label, data=data)

            if child.get("expand", False):
                node.expand()

            if children_payload:
                self._populate(node, children_payload)

    def _focus_current_selection(self) -> None:
        """Focus on the currently selected node"""
        target = self._selected
        if not target:
            return

        def matches(node: TreeNode) -> bool:
            data = node.data or {}
            for key, value in target.items():
                if key not in data or data[key] != value:
                    return False
            return True

        stack = [self.root]
        while stack:
            node = stack.pop()
            if matches(node):
                # Use set_default_selection logic
                if self.is_leaf_node(node):
                    self.selected_leaf = node
                    node.set_label(f"✓ {node.label}")

                    # Expand parent path
                    parent_node = node.parent
                    while parent_node and parent_node != self.root:
                        parent_node.expand()
                        parent_node = parent_node.parent

                self.move_cursor(node)
                return
            stack.extend(reversed(node.children))

    def get_selected(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected node data"""
        return self._selected

    def is_selectable_node(self, node: TreeNode) -> bool:
        """Check if a node is selectable based on allowed_type"""
        # Check if the selection_type matches allowed_type
        data = node.data or {}
        return data.get("selection_type") == self.allowed_type

    def action_select_node(self) -> None:
        """Override select action to check allowed_type"""
        if self.cursor_node is None:
            return

        # Check if the selection_type matches allowed_type
        data = self.cursor_node.data or {}
        if data.get("selection_type") != self.allowed_type:
            if self.app:
                self.app.notify(f"Only nodes of type '{self.allowed_type}' can be selected!", severity="warning")
            return

        # Cancel the previously selected node
        if self.selected_leaf:
            old_label = str(self.selected_leaf.label).replace("✓ ", "")
            self.selected_leaf.set_label(old_label)

        # Select the new node
        self.selected_leaf = self.cursor_node
        self._selected = data
        new_label = f"✓ {self.cursor_node.label}"
        self.cursor_node.set_label(new_label)

    def on_tree_node_selected(self, event) -> None:
        """Handle tree node selection event (for mouse clicks)"""
        data = event.node.data or {}
        if data.get("selection_type") != self.allowed_type:
            if self.app:
                self.app.notify(f"Only nodes of type '{self.allowed_type}' can be selected!", severity="warning")
            return

        # Trigger the selection action
        self.move_cursor(event.node)
        self.focus()
        self.action_select_node()

    def _collect_context(self, node: TreeNode) -> Dict[str, Any]:
        """Collect context from ancestor nodes.
        For subject_tree architecture, context includes node_id.
        """
        context: Dict[str, Any] = {}
        current = node
        while current and current is not self.root:
            data = current.data or {}
            # Collect relevant context data for subject_tree
            if "node_id" in data and "node_id" not in context:
                context["node_id"] = data["node_id"]
            if "selection_type" in data and "selection_type" not in context:
                context["selection_type"] = data["selection_type"]
            current = current.parent
        return context

    def _label_path(self, node: TreeNode) -> List[str]:
        path: List[str] = []
        current = node
        while current and current is not self.root:
            path.append(str(current.label))
            current = current.parent
        path.reverse()
        return path

    def _upsert_node_metadata(
        self,
        parent: TreeNode,
        label: str,
        data: Dict[str, Any],
        is_leaf: bool,
    ) -> None:
        path = self._label_path(parent)
        bucket = self.nodes

        for ancestor_label in path:
            entry = next((child for child in bucket if child.get("label") == ancestor_label), None)
            if entry is None:
                entry = {"label": ancestor_label, "data": {}, "children": []}
                bucket.append(entry)
            bucket = entry.setdefault("children", [])

        existing = next((child for child in bucket if child.get("label") == label), None)
        entry_payload: Dict[str, Any] = {"label": label, "data": data}

        if not is_leaf:
            entry_payload["children"] = existing.get("children", []) if existing else []
            entry_payload["force_branch"] = True
        elif existing:
            entry_payload.update({k: existing[k] for k in ("children",) if k in existing})

        if existing:
            existing.update(entry_payload)
            if is_leaf:
                existing.pop("children", None)
                existing.pop("force_branch", None)
        else:
            if is_leaf:
                entry_payload.pop("force_branch", None)
            else:
                entry_payload.setdefault("children", [])
            bucket.append(entry_payload)
            bucket.sort(key=lambda item: str(item.get("label")))
