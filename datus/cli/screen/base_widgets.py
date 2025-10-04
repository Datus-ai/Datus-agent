import re
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
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
        """Determine whether it is a leaf node"""
        return not node.children

    def action_toggle_or_select(self):
        # Only leaf nodes can be selected
        if not self.is_leaf_node(self.cursor_node):
            self.cursor_node.toggle()
            return
        return self.action_select_node()

    def action_select_node(self) -> None:
        """Select the node where the current cursor is located (leaf node only)"""
        if self.cursor_node is None:
            return

        # Only leaf nodes can be selected
        if not self.is_leaf_node(self.cursor_node):
            return

        # Cancel the previously selected node
        if self.selected_leaf:
            old_label = str(self.selected_leaf.label).replace("✓ ", "")
            self.selected_leaf.set_label(old_label)

        # Select a new node
        self.selected_leaf = self.cursor_node
        new_label = f"✓ {self.cursor_node.label}"
        self.cursor_node.set_label(new_label)

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


class InputWithLabel(Widget):
    """
    A horizontal layout containing a label and an editable component (Input or TextArea).
    Tracks the original value for change detection and supports read-only mode.
    """

    DEFAULT_CSS = """
    InputWithLabel {
        layout: horizontal;
        height: auto;
        align: center middle;
    }
    InputWithLabel Label {
        padding: 1;
        min-width: 12;
        max-width: 20;
        text-align: right;
    }
    InputWithLabel TextArea {
        width: 1fr;
        # border: round $accent;
        # padding-left: 1;
        # padding-right: 1;
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
        if self.lines <= 1:
            input_widget = Input(value=self.original_value)
            input_widget.disabled = self.readonly
            input_widget.styles.margin = 0
            self.input_widget = input_widget
            yield input_widget
        else:
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
            if isinstance(self.input_widget, TextArea):
                self.input_widget.read_only = readonly
            else:
                self.input_widget.disabled = readonly

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

    # --- Regex validation handlers ---
    async def on_input_changed(self, event: Input.Changed) -> None:
        """
        Handle changes to the single‑line Input. If a regex is provided,
        revert the change when the new value doesn’t match.
        """
        if event.input is not self.input_widget or not self.regex:
            return

        # If entire value matches, record it; otherwise revert to last valid
        if self.regex.fullmatch(event.value) or event.value == "":
            self._last_valid = event.value
            self._last_cursor_location = event.input.cursor_position
        else:
            event.input.value = self._last_valid
            event.input.cursor_position = min(self._last_cursor_location, len(self._last_valid))
        event.stop()

    async def on_textarea_changed(self, event: TextArea.Changed) -> None:
        """
        Handle changes to the multi‑line TextArea. Works similarly to the Input handler.
        """
        if event.text_area is not self.input_widget or not self.regex:
            return

        current_text = event.text_area.text
        current_cursor = event.text_area.cursor_location
        if self.regex.fullmatch(current_text) or current_text == "":
            self._last_valid = current_text
            self._last_cursor_location = current_cursor
        else:
            event.text_area.text = self._last_valid
            if hasattr(event.text_area, "cursor_location"):
                event.text_area.cursor_location = min(self._last_cursor_location, len(self._last_valid))
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

        if isinstance(self.input_widget, TextArea):
            self.input_widget.text = value
        else:
            self.input_widget.value = value
        self._last_valid = value
        self._last_cursor_location = len(value)

    def focus_input(self) -> bool:
        if self.input_widget is None or self.app is None:
            return False
        self.app.call_after_refresh(self.input_widget.focus)
        return True

    @property
    def cursor_position(self) -> int:
        if isinstance(self.input_widget, Input):
            return self.input_widget.cursor_position
        return 0

    @cursor_position.setter
    def cursor_position(self, position: int) -> None:
        if isinstance(self.input_widget, Input):
            self.input_widget.cursor_position = position


class FocusableStatic(Static):
    can_focus = True


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

            # Check if this node has children to determine if it should be a branch or leaf
            has_children = bool(child.get("children"))

            if has_children:
                node = parent.add(label, data=data)
            else:
                node = parent.add_leaf(label, data=data)

            if child.get("expand", False):
                node.expand()

            if has_children:
                self._populate(node, child.get("children", []))

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
        if not self.is_leaf_node(node):
            return False

        data = node.data or {}
        return data.get("selection_type") == self.allowed_type

    def action_select_node(self) -> None:
        """Override select action to check allowed_type"""
        if self.cursor_node is None:
            return

        # Check if it's a leaf node
        if not self.is_leaf_node(self.cursor_node):
            self.app.notify("Only leaf nodes can be selected!", severity="warning")
            return

        # Check if the selection_type matches allowed_type
        data = self.cursor_node.data or {}
        if data.get("selection_type") != self.allowed_type:
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
            self.app.notify(f"Only nodes of type '{self.allowed_type}' can be selected!", severity="warning")
            return

        if not self.is_leaf_node(event.node):
            return

        # Trigger the selection action
        self.focus(event.node)
        self.action_select_node()
