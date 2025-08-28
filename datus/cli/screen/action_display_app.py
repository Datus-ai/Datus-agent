import json
import sys
from typing import List, Optional

import pyperclip
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Collapsible, DataTable, Footer, Header, Label, Static, TextArea

from datus.cli.action_history_display import BaseActionContentGenerator
from datus.cli.screen.base_app import BaseApp
from datus.schemas.action_history import ActionHistory, ActionRole
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CollapsibleActionContentGenerator(BaseActionContentGenerator):
    def __init__(self):
        super().__init__()
        self.sql_dict = {}

    """Generates Collapsible content for action history display - optimized for Textual"""

    def format_collapsible_title(self, action: ActionHistory) -> str:
        """Format title for Collapsible widget based on ActionContentGenerator.format_streaming_action"""
        dot = self._get_action_dot(action)

        if action.role == ActionRole.TOOL:
            # For Tool role: "🔧 Tool call - function_name"
            function_name = "unknown"
            if action.input and isinstance(action.input, dict):
                function_name = action.input.get("function_name", "unknown")
            return f"{dot} Tool call - {function_name}"
        else:
            # For other roles: use original format
            if ":" in action.messages and (i := action.messages.index(":")) > 0:
                title = action.messages[0:i]
            else:
                title = action.messages
            return f"{dot} {title}"

    def create_action_collapsible(self, action: ActionHistory, index: int) -> Optional[Widget]:
        """Create a Collapsible widget for an action"""

        if action.role == ActionRole.ASSISTANT and action.messages == "Generating response with tools...":
            return None
        title = self.format_collapsible_title(action)

        # Create content widgets
        content_widgets = []

        if action.role == ActionRole.TOOL:
            content_widgets.extend(self._create_tool_content(action, index))
        elif action.role == ActionRole.USER:
            start = action.messages.index(":")
            content_widgets.append(TextArea(f"{title}:\n{action.messages[start + 2:]}", read_only=True))
        else:
            content_widgets.extend(self._create_non_tool_content(action, index))

        # Create vertical container with all content
        content_container = Vertical(*content_widgets)

        return Collapsible(content_container, title=title, collapsed=True)  # Default collapsed

    def _create_tool_content(self, action: ActionHistory, index: int) -> List:
        """Create content for Tool role actions"""
        widgets = []

        # 1. Parameters as table
        if action.input and isinstance(action.input, dict):
            params_table = self._create_params_table(action.input, index)
            if params_table:
                widgets.extend(params_table)

        # 3. Output as table
        if action.output:
            output_table = self._create_output_table(action, action.role, index)
            if output_table:
                widgets.extend(output_table)

        return widgets

    def _create_non_tool_content(self, action: ActionHistory, action_index: int) -> List[Widget]:
        """Create content for non-Tool role actions"""
        widgets = []

        # Display output
        if action.output:
            if self._is_json_like(action.output):
                # JSON highlighting
                json_display = self._create_json_display(action.output, action_index)
                widgets.extend(json_display)
            else:
                # Plain text
                widgets.append(TextArea(str(action.output)))
        return widgets

    def _create_sql_widget(self, sql: str) -> Widget:
        return TextArea(sql, read_only=True, language="sql", line_number_start=0, theme="monokai")

    def _create_params_table(self, input_data: dict, action_index: int) -> List[Widget]:
        """Create table for parameters"""
        if "arguments" not in input_data:
            return []
        args = input_data["arguments"]
        if isinstance(args, str):
            args = args.strip()
            if args.startswith("{"):
                json_args = json.loads(args)
                return self._do_create_params_table(json_args, action_index)
            elif args.startswith("["):
                return []
        return self._do_create_params_table(args, action_index)

    def _do_create_params_table(self, args: dict, action_index: int) -> List[Widget]:
        result = []
        if "sql" in args and len(args) == 1:
            return self._create_sql_widgets(args, action_index)
        result.append(Static("[bold]Parameters[/bold]", classes="section-title"))
        table = DataTable(show_header=True, name="Parameters")
        table.add_column("Parameter")
        table.add_column("Value")
        has_sql = False
        for key, value in args.items():
            # Perhaps determine whether it is SQL based on str.
            if key.lower() == "sql":  # Skip SQL, handle separately
                has_sql = True
                continue
            value_str = str(value)
            table.add_row(key, value_str)
        if table.row_count > 0:
            result.append(table)
        if has_sql:
            result.extend(self._create_sql_widgets(args, action_index))
        return result

    def _create_sql_widgets(self, input_data: dict, action_index: int) -> List:
        """Create SQL display with syntax highlighting and copy button"""
        widgets = []

        # Look for SQL in arguments
        if "arguments" in input_data and isinstance(input_data["arguments"], dict):
            args = input_data["arguments"]
        else:
            args = input_data

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return [TextArea(args, read_only=True, language="markdown", line_number_start=0, theme="monokai")]

        sql_content = args.get("sql") or args.get("query")

        if sql_content:
            # SQL syntax highlighting

            sql_widget = TextArea(sql_content, read_only=True, language="sql", line_number_start=0, theme="monokai")
            widgets.append(sql_widget)
            # Copy button
            copy_button = Button("📋copy", id=f"copy_sql_{action_index}")
            button_container = Horizontal(
                Static(""), copy_button, classes="button-container"  # Placeholder, push the button to the right
            )
            self.sql_dict[str(action_index)] = sql_content
            widgets.append(button_container)

        return widgets

    def get_sql(self, action_id) -> str:
        return self.sql_dict.get(action_id, "")

    def _create_output_table(self, action: ActionHistory, role: ActionRole, action_index: int) -> List[Widget]:
        output_data = action.output
        """Create table for output data"""
        if not output_data:
            return []

        result: List[Widget] = [Static("[bold]Output[/bold]", classes="section-title")]
        # Normalize output_data to dict format
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except Exception:
                result.append(
                    TextArea(output_data, read_only=True, language="markdown", line_number_start=0, theme="monokai")
                )
                return result
        if not isinstance(output_data, dict):
            if isinstance(output_data, list):
                result.append(TextArea(json.dumps(output_data, indent=2), language="json", theme="monokai"))
            else:
                result.append(TextArea(str(output_data), language="markdown", theme="monokai"))
            return result
        if "sql" in output_data:
            result.extend(self._create_sql_widgets(output_data, action_index))
            return result

        # Use raw_output if available
        data = output_data.get("raw_output", output_data)

        # Parse text field if present
        if data and isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                result.append(TextArea(data, language="markdown", theme="monokai"))
                return result

        if role == ActionRole.TOOL:
            function_name = action.input.get("function_name", "unknown")
            if function_name == "read_query":
                #  original_rows, original_columns, is_compressed, and compressed_data
                if data.get("success", 0) == 1:
                    data = data["result"]
                    is_compressed = data.get("is_compressed", False)
                    compressed_data = str(data.get("compressed_data"))
                    if not is_compressed:
                        result.append(TextArea(compressed_data, language="markdown", theme="monokai"))
                    else:
                        result.append(
                            TextArea(
                                f"""
                    {compressed_data}

                    ---

                    **Total Rows**: {data.get("original_rows")}

                    ---

                    **Columns**: {', '.join(data.get("original_columns", []))}
                    """,
                                language="markdown",
                                theme="monokai",
                            )
                        )
                elif data.get("error"):
                    result.append(Label(f"[bold red]Execute Failed: {data['error']}[/]"))
                else:
                    result.append(TextArea(json.dumps(data, indent=2), language="json", theme="monokai"))

                return result

        # Extract result items
        items = None
        if "text" in data and isinstance(data["text"], str):
            try:
                cleaned_text = data["text"].replace("'", '"').replace("None", "null")
                items = json.loads(cleaned_text)
            except Exception:
                result.append(TextArea(data["text"], language="markdown", theme="monokai"))
                return result
        elif "result" in data and isinstance(data["result"], list):
            items = data["result"]

        if not items or len(items) == 0:
            result.append(TextArea(json.dumps(data, indent=2), language="json", theme="monokai"))
            return result
        if not isinstance(items, list):
            result.append(TextArea(json.dumps(items, indent=2), language="json", theme="monokai"))
            return result

        # Create table
        first_item = items[0]
        if not isinstance(first_item, dict):
            return [TextArea(json.dumps(items, indent=2), language="json", theme="monokai")]

        table = DataTable(show_header=True, name="Output")

        # Add columns
        for key in first_item.keys():
            table.add_column(str(key).title())

        # Add rows (limit to 10)
        # max_rows = min(len(items), 10)
        for item in items:
            if isinstance(item, dict):
                row_values = []
                for key in first_item.keys():
                    value = item.get(key, "")
                    row_values.append(str(value))
                table.add_row(*row_values)
        result.append(table)
        return result

    def _create_json_display(self, data, action_index: int) -> List[Widget]:
        """Create JSON syntax highlighting"""
        try:
            if isinstance(data, str):
                json_obj = json.loads(data)
            elif isinstance(data, dict) or isinstance(data, list):
                json_obj = data
            else:
                return [TextArea(str(data), language="markdown", theme="monokai", line_number_start=0)]
        except Exception as e:
            logger.debug(f"parse json failed, use markdown to show: reason={str(e)}, data={data}")
            return [TextArea(str(data), language="markdown", theme="monokai", line_number_start=0)]

        if "raw_output" in json_obj:
            output_data = json_obj["raw_output"]
            try:
                output_data = json.loads(output_data)
                if "sql" in output_data:
                    return self._create_sql_widgets(json_obj["sql"], action_index)
                else:
                    return [
                        TextArea(
                            json.dumps(output_data, indent=2), language="json", theme="monokai", line_number_start=0
                        )
                    ]
            except Exception:
                return [TextArea(str(output_data), language="markdown", theme="monokai", line_number_start=0)]
        result = []
        if "response" not in json_obj and "sql" not in json_obj:
            return [TextArea(json.dumps(json_obj, indent=2), language="json", theme="monokai", line_number_start=0)]
        if "response" in json_obj:
            result.append(TextArea(json_obj["response"], language="markdown", theme="monokai", line_number_start=0))
        if "sql" in json_obj:
            result.extend(self._create_sql_widgets(json_obj, action_index))
        return result

    def _is_json_like(self, data) -> bool:
        """Check if data is JSON-like"""
        if isinstance(data, (dict, list)):
            return True
        if isinstance(data, str):
            return data.strip().startswith("[") or data.strip().startswith("{")
        return False

    def generate_collapsible_actions(self, actions: List[ActionHistory]) -> List[Widget]:
        """Generate list of Collapsible widgets for actions"""
        result = []

        for i, action in enumerate(actions):
            widget = self.create_action_collapsible(action, i)
            if widget:
                result.append(widget)
        return result


class ChatActionScreen(Screen):
    CSS = """
    ScrollableContainer {
        border: round $primary;
        height: auto;
        overflow-y: auto;
    }

    .section-title {
        height: auto;
        text-style: bold;
    }
    Button {
        margin-top: 1;
        height: auto;
        border: round $primary;
        width: auto;
    }
    Button:focus {
        outline: none;
    }

    Button:hover {
        outline: none;
    }

    .button-container {
        height: auto;
        min-height: 5;
        width: 100%;
        content-align: right middle;
    }

    .button-container Static {
        width: 1fr;
    }

    .button-container Button {
        dock: right;
        margin-right: 1;
    }

    Collapsible {
        height: auto;
    }

    Label {
        padding: 1;
        background: $surface;
        height: auto;
    }

    VerticalGroup {
        height: auto;
    }
    Vertical {
        height: auto;
    }
    Static {
        height: auto;
    }
    TextArea {
        border: round $primary;
    }
    """
    # Define different key bindings based on operating system
    if sys.platform == "darwin":  # macOS
        BINDINGS = [
            Binding("cmd+c", "copy_text", "Copy Selection", priority=True),
            Binding("escape", "exit", "Exit"),
            Binding("ctrl+q", "exit", "Exit"),
            Binding("e", "toggle_all", "Toggle All"),
        ]
    else:  # Windows/Linux
        BINDINGS = [
            Binding("ctrl+shift+c", "copy_text", "Selection", priority=True),
            Binding("escape", "exit", "Exit"),
            Binding("ctrl+q", "exit", "Exit"),
            Binding("e", "toggle_all", "Toggle All"),
        ]

    def __init__(
        self,
        action_list: List[ActionHistory],
        # action_queue: Queue,
        **kwargs,
    ):
        super().__init__(name="Execution Details", **kwargs)
        self.collapsible_generator = CollapsibleActionContentGenerator()
        self.action_list = action_list
        # self.action_queue = action_queue
        self.displayed_count = 0  # Track displayed message count

    def compose(self):
        yield Header(show_clock=True)
        with VerticalScroll(id="actions_scroll"):
            yield VerticalGroup(id="actions_container")
        yield Footer()

    def action_copy_text(self) -> None:
        if sys.platform != "darwin":
            return super().action_copy_text()
        else:
            focused = self.focused
            if isinstance(focused, DataTable):
                self._copy_datatable_cell(focused)
            elif isinstance(focused, TextArea):
                self._copy_textarea_selection(focused)

    def action_toggle_all(self) -> None:
        collapsible_widgets = self.query(Collapsible)
        any_expanded = any(not c.collapsed for c in collapsible_widgets)

        new_state = any_expanded
        for collapsible in collapsible_widgets:
            collapsible.collapsed = new_state

    def _copy_datatable_cell(self, table: DataTable) -> None:
        if table.cursor_row is not None and table.cursor_column is not None:
            cell_value = table.get_cell(table.cursor_row, table.cursor_column)
            pyperclip.copy(str(cell_value))

    def _copy_textarea_selection(self, text_area: TextArea) -> None:
        selected = text_area.selected_text
        if selected:
            pyperclip.copy(selected)

    async def on_mount(self):
        # Show welcome message and exit instructions
        container = self.query_one("#actions_container", VerticalGroup)
        if not self.action_list:
            container.mount(Static("[dim]Waiting for message updates...[/dim]"))
        else:
            self.display_actions(self.action_list, container)
        # Record displayed message count
        self.displayed_count = 0

    def display_actions(self, actions: List[ActionHistory], container: Widget):
        """Display action messages using Collapsible widgets"""
        # Generate Collapsible widgets for actions
        collapsible_widgets = self.collapsible_generator.generate_collapsible_actions(actions)

        # Mount all widgets to container
        for widget in collapsible_widgets:
            container.mount(widget)

        self.displayed_count += len(actions)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle copy button press events"""
        button_id = event.button.id
        if button_id and button_id.startswith("copy_sql_"):
            # Extract SQL content from the corresponding action
            # This is a simplified version - in practice you'd store SQL content
            # in the button's metadata or find it through the widget hierarchy
            try:
                # Find the SQL content from the action that corresponds to this button
                sql_content = self._get_sql_content_for_button(button_id)
                if sql_content:
                    pyperclip.copy(sql_content)
                    self.notify("SQL copied to clipboard!", severity="information")
                else:
                    self.notify("No SQL content found", severity="warning")
            except ImportError:
                self.notify("pyperclip not available - cannot copy to clipboard", severity="error")
            except Exception as e:
                self.notify(f"Failed to copy: {e}", severity="error")

    def _get_sql_content_for_button(self, button_id: str) -> Optional[str]:
        """Get SQL content for a specific copy button"""
        # Extract the action ID from button ID
        action_id = button_id.replace("copy_sql_", "")
        # Find the action with this ID and extract SQL
        # to put in dict
        return self.collapsible_generator.get_sql(action_id)

    # def check_new_actions(self):
    #     """Check and append new action messages from queue or actions list"""
    #     log_widget = self.query_one("#actions", RichLog)

    #     # If we have a queue, try to get new actions from it
    #     if self.action_queue:
    #         new_actions = []
    #         try:
    #             # Get all available actions from queue without blocking
    #             while True:
    #                 action = self.action_queue.get_nowait()
    #                 new_actions.append(action)
    #         except Empty:
    #             pass  # No more actions in queue

    #         self.display_actions(new_actions, log_widget)

    def action_exit(self):
        """Exit application"""
        self.app.exit()

    async def action_switch_mode(self, mode=None) -> None:
        """Switch display mode via Ctrl+R"""
        self.app.exit("mode_switch")


class ChatApp(BaseApp):
    BINDINGS = [
        Binding("escape", "exit", "Exit"),
        Binding("q", "exit", "Exit"),
        Binding("ctrl+q", "exit", "Exit"),
    ]

    def __init__(self, actions_list: List[ActionHistory], **kwargs):
        super().__init__(**kwargs)
        self.actions_list = actions_list
        # self.action_queue = Queue()
        # for action in actions_list:
        #     self.action_queue.put(action)

    def on_mount(self):
        self.push_screen(ChatActionScreen(self.actions_list))

    def action_exit(self):
        """Exit application"""
        self.exit()
