"""
Enhanced MCP (Model Context Protocol) server management screen for Datus CLI.
Provides elegant interactive interface for browsing and selecting MCP servers and tools.
"""

from typing import Any, Dict, List

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, Static

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MCPServerListScreen(Screen):
    """Screen for displaying and selecting MCP servers."""

    CSS = """
    #mcp-container {
        align: center middle;
        height: 100%;
        background: $surface;
    }

    #mcp-main-panel {
        width: 90%;
        max-width: 120;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1;
    }

    #mcp-title {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #server-list {
        width: 100%;
        height: auto;
        margin: 1 0;
    }

    .server-item {
        width: 100%;
        height: 1;
        padding: 0 1;
    }

    .server-item:hover {
        background: $accent 15%;
    }

    .server-item:focus {
        background: $accent;
    }

    .server-name {
        color: $text;
        text-style: bold;
    }

    .server-status {
        margin-left: 2;
    }

    .status-connected {
        color: $success;
    }

    .status-failed {
        color: $error;
    }

    .server-tip {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "exit", "Exit"),
        Binding("q", "exit", "Exit"),
    ]

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the MCP server list screen.

        Args:
            data: Dictionary containing mcp_servers configuration
        """
        super().__init__()
        self.mcp_servers: Dict[str, Any] = data.get("mcp_servers", {})
        self.pre_index = None

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="MCP Servers")

        with Container(id="mcp-container"):
            with Container(id="mcp-main-panel"):
                yield Static("Manage MCP servers", id="mcp-title")
                yield ListView(id="server-list")
                # cache_path = os.path.expanduser("~/.datus")
                yield Static(
                    "Tip: View log files in logs",
                    id="mcp-tip",
                    classes="server-tip",
                )

        yield Footer()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.action_select_server()
            event.stop()
        elif event.key == "down":
            self.action_cursor_down()
        elif event.key == "up":
            self.action_cursor_up()
        else:
            super()._on_key(event)

    def on_mouse_up(self, event: events.MouseDown) -> None:
        server_list = self.query_one("#server-list", ListView)
        self._switch_list_cursor(server_list, self.pre_index, server_list.index)
        self.pre_index = None

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse click events on list items."""
        # Check if we clicked on a list item
        server_list = self.query_one("#server-list", ListView)
        self.pre_index = server_list.index

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        server_list = self.query_one("#server-list", ListView)
        for i, (server_name, server_config) in enumerate(self.mcp_servers.items()):
            # Determine status based on configuration
            status_symbol = "✔"
            status_text = "connected"
            status_class = "status-connected"

            # In a real implementation, check actual connection status
            if server_config.get("disabled") or not server_config.get("command"):
                status_symbol = "✘"
                status_text = "failed"
                status_class = "status-failed"

            # Create rich server item
            item_label = Label(f"{'> ' if i == 0 else '  '}{i+1}. {server_name}", classes="server-name")
            status_label = Label(
                f"{status_symbol} {status_text} · Enter to view details", classes=f"server-status {status_class}"
            )

            # Create horizontal layout for server item
            item_container = Horizontal(item_label, status_label, classes="server-item")
            list_item = ListItem(item_container)

            # Store server data
            list_item.server_data = {
                "name": server_name,
                "config": server_config,
                "type": server_config.get("type", "unknown"),
                "command": server_config.get("command", ""),
                "args": server_config.get("args", []),
                "env": server_config.get("env", {}),
                "cwd": server_config.get("cwd", ""),
            }
            server_list.append(list_item)
        server_list.index = 0
        server_list.focus()

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        list_view = self.query_one("#server-list", ListView)
        if list_view.index == len(list_view.children) - 1:
            return
        self._switch_list_cursor(list_view, list_view.index, list_view.index + 1)

    def _switch_list_cursor(self, list_view: ListView, pre_index: int, new_index: int):
        if pre_index == new_index:
            return
        previous_item = list_view.children[pre_index]
        previous_label = previous_item.query_one(Label)
        content = previous_label.renderable
        if content.startswith("> "):
            previous_label.update("  " + content[2:])

        current_item = list_view.children[new_index]
        current_label = current_item.query_one(Label)
        content = current_label.renderable
        if content.startswith("  "):
            current_label.update("> " + content[2:])

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        list_view = self.query_one("#server-list", ListView)
        if list_view.index == 0:
            return
        self._switch_list_cursor(list_view, list_view.index, list_view.index - 1)

    def action_select_server(self) -> None:
        """Select the current server and show detailed view."""
        list_view = self.query_one("#server-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(list_view.children):
            selected_item = list_view.children[list_view.index]
            server_data = getattr(selected_item, "server_data", {})
            self.app.push_screen(MCPServerDetailScreen(server_data))

    def action_exit(self) -> None:
        """Exit the screen."""
        self.app.exit()


class MCPServerDetailScreen(Screen):
    """Screen for displaying detailed information about an MCP server."""

    CSS = """
    #detail-container {
        align: center middle;
        height: 100%;
        background: $surface;
    }

    #detail-panel {
        width: 90%;
        max-width: 120;
        height: auto;
        background: $surface;
        border: round $primary;
    }

    .server-header {
        text-align: center;
        text-style: bold;
        color: $text;
    }

    .info-row {
        height: 1;
    }

    .info-label {
        color: $text-muted;
        width: 12;
    }

    .info-value {
        color: $text;
    }

    .command-value {
        color: $text;
        text-style: italic;
    }

    .args-value {
        color: $text;
        text-style: italic;
    }

    .capability-tag {
        color: $success;
        text-style: bold;
    }

    #view-tools-option {
        text-align: center;
        color: $text;
    }

    .option-highlight {
        color: $accent;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", "Back"),
        Binding("q", "back", "Back"),
        # Binding("enter", "view_tools", "View Tools"),
    ]

    def __init__(self, server_data: Dict[str, Any]):
        """
        Initialize the MCP server detail screen.

        Args:
            server_data: MCP server configuration data
        """
        super().__init__()
        self.server_data = server_data
        self.server_name = server_data.get("name", "Unknown Server")
        self.server_config = server_data.get("config", {})

        # Mock tools for demonstration - updated to match your example
        self.tools = [
            {"name": "list_metrics", "description": "List all metrics available"},
            {"name": "get_dimensions", "description": "Get dimensions for metrics"},
            {"name": "get_entities", "description": "Get all entities"},
            {"name": "query_metrics", "description": "Query metrics with filters"},
            {"name": "validate_configs", "description": "Validate configuration files"},
            {"name": "create_metric", "description": "Create a new metric"},
        ]

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        config = self.server_config
        command = config.get("command", "")
        args = config.get("args", [])
        env = config.get("env", {})
        # cwd = config.get("cwd", "")

        # Format command and arguments
        # full_command = f"{command} {' '.join(args)}" if args else command
        args_str = " ".join(args) if args else ""

        yield Header(show_clock=True, name=f"{self.server_name} - MCP Server")

        with Container(id="detail-container"):
            with Container(id="detail-panel"):
                yield Static(f"{self.server_name} MCP Server", classes="server-header")

                # Server information
                with Container(classes="server-info"):
                    yield Static("Status: ✔ connected", classes="info-row info-value")
                    yield Static(f"Command: {command}", classes="info-row command-value")
                    if args_str:
                        yield Static(f"Args: {args_str}", classes="info-row args-value")
                    yield Static("Capabilities: tools", classes="info-row capability-tag")
                    yield Static(f"Tools: {len(self.tools)} tools", classes="info-row info-value")
                    if env:
                        yield Static(f"Env: {env}", classes="info-row info-value")
                    # if cwd:
                    #     yield Static(f"Cwd: {cwd}", classes="info-row info-value")

                yield Button("❯ 1. View tools", id="view-tools-option", classes="option-highlight")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # if event.button.id == "option-highlight":
        self.action_view_tools()

    def action_view_tools(self) -> None:
        """View the tools provided by this server."""
        self.app.push_screen(MCPToolsScreen(self.server_data, self.tools))

    def action_back(self) -> None:
        """Go back to the server list."""
        self.app.pop_screen()


class MCPToolsScreen(Screen):
    """Screen for displaying tools provided by an MCP server."""

    CSS = """
    #tools-container {
        align: center middle;
        height: 100%;
        background: $surface;
    }

    #tools-panel {
        width: 90%;
        max-width: 80;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1;
    }

    .tools-header {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #tools-list {
        width: 100%;
        height: auto;
    }

    .tool-item {
        width: 100%;
        height: 1;
        padding: 0 1;
    }

    .tool-name {
        color: $text;
    }

    .tool-item:hover {
        background: $accent 15%;
    }

    .tool-item:focus {
        background: $accent;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", "Back"),
        Binding("q", "back", "Back"),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("up", "cursor_up", "Up", show=False),
    ]

    def __init__(self, server_data: Dict[str, Any], tools: List[Dict[str, str]]):
        """
        Initialize the MCP tools screen.

        Args:
            server_data: MCP server configuration data
            tools: List of available tools
        """
        super().__init__()
        self.server_data = server_data
        self.server_name = server_data.get("name", "Unknown Server")
        self.tools = tools

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name=f"Tools for {self.server_name}")

        with Container(id="tools-container"):
            with Container(id="tools-panel"):
                yield Static(f"Tools for {self.server_name} ({len(self.tools)} tools)", classes="tools-header")
                yield ListView(id="tools-list")

        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        tools_list = self.query_one("#tools-list", ListView)
        for i, tool in enumerate(self.tools):
            tool_name = tool.get("name", f"tool_{i+1}")
            tool_label = Label(f"{i+1}. {tool_name}", classes="tool-name")
            list_item = ListItem(tool_label)
            tools_list.append(list_item)
        tools_list.index = 0
        tools_list.focus()

    def action_back(self) -> None:
        """Go back to the server detail screen."""
        self.app.pop_screen()


class MCPServerApp(App):
    """Main application for MCP server management."""

    def __init__(self, mcp_servers: Dict[str, Any]):
        """
        Initialize the MCP server app.

        Args:
            mcp_servers: List of available MCP servers
        """
        super().__init__()
        self.mcp_servers = mcp_servers
        self.theme = "textual-dark"

    def on_mount(self):
        """Push the server list screen on mount."""
        self.push_screen(MCPServerListScreen({"mcp_servers": self.mcp_servers}))
