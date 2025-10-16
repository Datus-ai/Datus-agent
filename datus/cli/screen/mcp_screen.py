# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Enhanced MCP (Model Context Protocol) server management screen for Datus CLI.
Provides elegant interactive interface for browsing and selecting MCP servers and tools.
"""

from typing import Any, Dict, List, Optional

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from datus.cli.screen.base_app import BaseApp
from datus.tools.mcp_tools import MCPServerType, MCPTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MCPServerListScreen(Screen):
    """Screen for displaying and selecting MCP servers."""

    CSS = """
    #mcp-container {
        align: left middle;
        height: 100%;
        background: $surface;
    }

    #mcp-main-panel {
        width: 80%;
        max-width: 140;
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

    .status-checking {
        color: $warning;
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

    def __init__(self, mcp_tool: MCPTool, data: Dict[str, Any]):
        """
        Initialize the MCP server list screen.

        Args:
            data: Dictionary containing servers list from MCPTool.list_servers
        """
        super().__init__()
        self.mcp_tool = mcp_tool
        # Handle both old format (mcp_servers dict) and new format (servers list)
        if "servers" in data:
            self.servers = data["servers"]  # New format: list of server dicts
        else:
            # Old format: convert dict to list
            self.servers = [{"name": name, **config} for name, config in data.get("mcp_servers", {}).items()]
        self.pre_index = None

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name="MCP Servers")

        with Container(id="mcp-container"):
            with Container(id="mcp-main-panel"):
                list_items = []
                for i, server in enumerate(self.servers):
                    server_name = server.get("name", "Unknown")

                    # Initial status while checking connectivity
                    status_symbol = "●"
                    status_text = "checking status"
                    status_class = "status-checking"

                    # Create rich server item
                    item_label = Label(f"{'> ' if i == 0 else '  '}{i+1}. {server_name}", classes="server-name")
                    status_label = Label(
                        f"{status_symbol} {status_text} · Enter to view details",
                        classes=f"server-status {status_class}",
                    )

                    # Create horizontal layout for server item
                    item_container = Horizontal(item_label, status_label, classes="server-item")
                    list_item = ListItem(item_container)

                    # Store server data in new format
                    list_item.server_data = server
                    list_item.server_status_label = status_label  # Keep reference to status label for updates
                    list_items.append(list_item)
                yield ListView(*list_items, id="server-list")

                # cache_path = os.path.expanduser("~/.datus")
                yield Static(
                    "Tip: View log files in logs",
                    id="mcp-tip",
                    classes="server-tip",
                )

        yield Footer()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.action_select_server()
            event.stop()
        elif event.key == "down":
            self.action_cursor_down()
        elif event.key == "up":
            self.action_cursor_up()
        else:
            await super()._on_key(event)

    def on_mouse_up(self, event: events.MouseDown) -> None:
        server_list = self.query_one("#server-list", ListView)
        _switch_list_cursor(server_list, self.pre_index, server_list.index)
        self.pre_index = None

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse click events on list items."""
        # Check if we clicked on a list item
        server_list = self.query_one("#server-list", ListView)
        self.pre_index = server_list.index

    async def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Start workers for each server independently
        server_list = self.query_one("#server-list", ListView)
        for i, server in enumerate(self.servers):
            if i < len(server_list.children):
                list_item = server_list.children[i]
                status_label = list_item.server_status_label
                # Run a thread worker for each server check (since check_connectivity is blocking)
                self.run_worker(
                    lambda s=server, lbl=status_label, idx=i: self._check_single_server_sync(s, lbl),
                    thread=True,
                    group="connectivity_checks",
                )

    async def on_unmount(self) -> None:
        """Called when the screen is unmounted."""
        try:
            self.workers.cancel_all()
        except Exception as e:
            logger.warning(f"Failed to cancel workers on unmount: {str(e)}")

    def _check_single_server_sync(self, server: Dict[str, Any], status_label: Label) -> None:
        """Synchronous check for a single server (runs in thread)."""
        server_name = server.get("name", "Unknown")
        try:
            result = self.mcp_tool.check_connectivity(server_name)
            if result.success and result.result.get("connectivity", False):
                status_symbol = "✔"
                status_text = "connected"
                status_class = "status-connected"
            else:
                status_symbol = "✘"
                status_text = "failed"
                status_class = "status-failed"
        except Exception as e:
            logger.error(f"Check mcp server failed {e}")
            status_symbol = "✘"
            status_text = "failed"
            status_class = "status-failed"
        # Update UI from thread safely
        try:
            self.app.call_from_thread(
                self._update_server_status, status_label, status_symbol, status_text, status_class
            )
        except Exception as e:
            logger.warning(f"Update UI failed: {str(e)}")

    def _update_server_status(self, status_label, status_symbol, status_text, status_class):
        """Update server status label with new connectivity status."""
        status_label.update(f"{status_symbol} {status_text} · Enter to view details")
        status_label.set_class(False, "status-checking")
        status_label.set_class(True, status_class)

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        list_view = self.query_one("#server-list", ListView)
        if list_view.index is None or len(list_view.children) == 0:
            return
        if list_view.index == len(list_view.children) - 1:
            return
        _switch_list_cursor(list_view, list_view.index, list_view.index + 1)

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        list_view = self.query_one("#server-list", ListView)
        if list_view.index is None or len(list_view.children) == 0:
            return
        if list_view.index == 0:
            return
        _switch_list_cursor(list_view, list_view.index, list_view.index - 1)

    def action_select_server(self) -> None:
        """Select the current server and show detailed view."""
        list_view = self.query_one("#server-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(list_view.children):
            selected_item = list_view.children[list_view.index]
            server_data = getattr(selected_item, "server_data", {})
            self.app.push_screen(MCPServerDetailScreen(self.mcp_tool, server_data))

    def action_exit(self) -> None:
        """Exit the screen."""
        self.app.exit()


class MCPServerDetailScreen(Screen):
    """Screen for displaying detailed information about an MCP server."""

    CSS = """
    #detail-container {
        align: left middle;
        height: 100%;
        background: $surface;
    }

    #detail-panel {
        width: 80%;
        max-width: 140;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1;
    }

    .server-header {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #server-info-display {
        width: 100%;
        height: auto;
        layout: grid;
        grid-size: 2;
        grid-columns: 15 1fr;
    }

    Label.label {
        text-style: bold;
        color: $text-muted;
    }

    #view-tools-option {
        margin-top: 0;
    }

    ListView {
        height: 25%;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", "Back"),
        Binding("q", "back", "Back"),
    ]

    def __init__(self, mcp_tool: MCPTool, server_data: Dict[str, Any]):
        """
        Initialize the MCP server detail screen.

        Args:
            server_data: MCP server configuration data in new format
        """
        super().__init__()
        self.mcp_tool = mcp_tool
        self.server_data = server_data
        self.server_name = server_data.get("name", "Unknown Server")
        self.server_type = server_data.get("type", "unknown")
        self.title = f"{self.server_name} MCP Server"
        self.connected = False

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""

        yield Header(show_clock=True, name=f"{self.server_name} - MCP Server")

        with Container(id="detail-container"):
            with Container(id="detail-panel"):
                # Server information
                with ScrollableContainer(classes="server-info"):
                    # Render the info using Grid with Labels
                    with Grid(id="server-info-display"):
                        yield Label("Type:", classes="label")
                        yield Label(f"[cyan]{self.server_type}[/cyan]")

                        # Type-specific configuration
                        if self.server_type == MCPServerType.STDIO:
                            command = self.server_data.get("command", "")
                            args = self.server_data.get("args", [])
                            env = self.server_data.get("env", {})

                            yield Label("Command:", classes="label")
                            yield Label(f"[green]{command}[/green]")
                            if args:
                                args_str = " ".join(args)
                                yield Label("Args:", classes="label")
                                yield Label(f"[yellow]{args_str}[/yellow]")
                            if env:
                                env_str = ", ".join([f"{k}={v}" for k, v in env.items()])
                                yield Label("Env:", classes="label")
                                yield Label(f"[magenta]{env_str}[/magenta]")

                        elif self.server_type in [MCPServerType.SSE, MCPServerType.HTTP]:
                            url = self.server_data.get("url", "")
                            headers = self.server_data.get("headers", {})
                            timeout = self.server_data.get("timeout")

                            yield Label("URL:", classes="label")
                            yield Label(f"[blue]{url}[/blue]")
                            if headers:
                                headers_str = ", ".join([f"{k}: {v}" for k, v in headers.items()])
                                yield Label("Headers:", classes="label")
                                yield Label(f"[magenta]{headers_str}[/magenta]")
                            if timeout:
                                yield Label("Timeout:", classes="label")
                                yield Label(f"[yellow]{timeout}s[/yellow]")

                        # Capabilities row
                        yield Label("Capabilities:", classes="label")
                        yield Label("tools")

                        # Tools row with loading status
                        yield Label("Tools:", classes="label")
                        yield Label("[dim]Loading...[/dim]", id="tools-value")
                yield ListView(ListItem(Label("> View Tools")), id="view-tools-option")

        yield Footer()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            if not self.connected:
                return
            self.action_view_tools()
            event.stop()
        else:
            await super()._on_key(event)

    async def on_mount(self) -> None:
        """Called when the screen is mounted."""
        view_tools = self.query_one("#view-tools-option", ListView)
        view_tools.disabled = True
        view_tools.has_focus = True
        self.run_worker(self._fetch_tools_sync, thread=True)

    async def on_unmount(self) -> None:
        try:
            self.workers.cancel_all()
        except Exception as e:
            logger.warning(f"Failed to cancel workers on unmount: {str(e)}")

    def _fetch_tools_sync(self) -> None:
        """Synchronous fetch for tools (runs in thread)."""
        try:
            tools_result = self.mcp_tool.list_tools(self.server_name)
            logger.info(f"Tools: {tools_result}")
            self.app.call_from_thread(self._update_tools_ui, tools_result)
        except Exception as e:
            logger.error(f"Error loading tools for server {self.server_name}: {str(e)}")
            self.app.call_from_thread(self._update_tools_ui, None, str(e))

    def _update_tools_ui(self, tools_result=None, error=None) -> None:
        """Update UI in main thread with fetched tools."""
        tools_value = self.query_one("#tools-value", Label)
        view_tools = self.query_one("#view-tools-option", ListView)

        if error:
            tools_value.update(f"[red]Error loading tools ({error})[/red]")
            self.tools = []
            self.connected = False
            return

        if tools_result.success:
            self.tools = tools_result.result["tools"]
            tools_count = len(self.tools)
            tools_value.update(f"[green]{tools_count}[/green] tools available")
            view_tools.disabled = False
            self.connected = True
        else:
            self.connected = False
            tools_value.update("[red]Failed to load tools[/red]")
            logger.error(f"Failed to load tools for server {self.server_name}: {tools_result.message}")
            self.tools = []

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
        align: left middle;
        height: 100%;
        background: $surface;
    }

    #tools-panel {
        width: 80%;
        max-width: 140;
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
        height: 2;
        padding: 0 1;
    }

    .tool-name {
        color: $text;
        text-style: bold;
    }

    .tool-description {
        color: $text-muted;
        margin-top: 0;
        margin-left: 2;
    }

    .tool-item:hover {
        background: $accent 15%;
    }

    .tool-item:focus {
        background: $accent;
    }

    #tool-params {
        width: 100%;
        height: auto;
        margin-top: 1;
    }

    .param-label {
        color: $text-muted;
        text-style: bold;
    }

    .param-value {
        color: $text;
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
            server_data: MCP server configuration data in new format
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

    async def on_mount(self) -> None:
        """Called when the screen is mounted."""
        tools_list = self.query_one("#tools-list", ListView)
        for i, tool in enumerate(self.tools):
            tool_name = tool.get("name", f"tool_{i+1}")
            tool_description = tool.get("description", "No description available")

            # Create tool item with name and description
            tool_label = Label(f"{i+1}. {tool_name}", classes="tool-name")
            description_label = Label(f"{tool_description}", classes="tool-description")

            # Create horizontal layout for tool item
            item_container = Horizontal(tool_label, description_label, classes="tool-item")
            list_item = ListItem(item_container)

            # Store tool data in new format
            list_item.tool_data = tool
            tools_list.append(list_item)
        tools_list.index = 0
        tools_list.has_focus = True

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.action_view_tool_details()
            event.stop()
        else:
            super()._on_key(event)

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        list_view = self.query_one("#tools-list", ListView)
        if list_view.index == 0:
            return
        # _switch_list_cursor(list_view, list_view.index, list_view.index - 1)

    def action_view_tool_details(self) -> None:
        """View the details of the current tool."""
        list_view = self.query_one("#tools-list", ListView)
        if list_view.index is not None and 0 <= list_view.index < len(list_view.children):
            selected_item = list_view.children[list_view.index]
            tool_data = getattr(selected_item, "tool_data", {})
            self.app.push_screen(MCPToolDetailScreen(self.server_data, tool_data))

    def action_back(self) -> None:
        """Go back to the server detail screen."""
        self.app.pop_screen()


def _switch_list_cursor(list_view: ListView, pre_index: Optional[int] = None, new_index: Optional[int] = None):
    if pre_index == new_index:
        return
    if pre_index is not None:
        previous_item = list_view.children[pre_index]
        previous_label = previous_item.query_one(Label)
        content = previous_label.renderable
        if content.startswith("> "):
            previous_label.update("  " + content[2:])

    if new_index is not None and 0 <= new_index < len(list_view.children):
        current_item = list_view.children[new_index]
        current_label = current_item.query_one(Label)
        content = current_label.renderable
        if content.startswith("  "):
            current_label.update("> " + content[2:])


class MCPToolDetailScreen(Screen):
    """Screen for displaying detailed information about an MCP tool."""

    CSS = """
    #detail-container {
        align: left middle;
        height: 100%;
        background: $surface;
    }

    #detail-panel {
        width: 80%;
        max-width: 140;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1;
    }

    .tool-header {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #tool-info-display {
        width: 100%;
        height: auto;
        layout: grid;
        grid-size: 2;
        grid-columns: 15 1fr;
    }

    Label.label {
        text-style: bold;
        color: $text-muted;
    }

    #tool-params {
        width: 100%;
        height: auto;
        margin-top: 1;
    }

    .param-label {
        color: $text-muted;
        text-style: bold;
    }

    .param-value {
        color: $text;
    }
    """
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", "Back"),
        Binding("q", "back", "Back"),
    ]

    def __init__(self, server_data: Dict[str, Any], tool_data: Dict[str, Any]):
        """
        Initialize the MCP tool detail screen.

        Args:
            server_data: MCP server configuration data in new format
            tool_data: Tool configuration data
        """
        super().__init__()
        self.server_data = server_data
        self.tool_data = tool_data
        self.tool_name = tool_data.get("name", "Unknown Tool")
        self.title = f"{self.tool_name} - Tool Details"

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield Header(show_clock=True, name=f"{self.tool_name} - Tool Details")

        with Container(id="detail-container"):
            with Container(id="detail-panel"):
                # Tool information
                with ScrollableContainer(classes="tool-info"):
                    # Render the info using Grid with Labels
                    with Grid(id="tool-info-display"):
                        yield Label("Name:", classes="label")
                        yield Label(f"[cyan]{self.tool_name}[/cyan]")

                        yield Label("Description:", classes="label")
                        yield Label(f"{self.tool_data.get('description', 'No description available')}")

                        input_schema = self.tool_data.get("inputSchema", {})
                        properties = input_schema.get("properties", {})
                        required = input_schema.get("required", [])

                        if properties:
                            yield Label("Parameters:", classes="label")
                            with Container(id="tool-params"):
                                for param_name, param_info in properties.items():
                                    param_type = param_info.get("type", "unknown")
                                    param_description = param_info.get("description", "No description available")
                                    is_required = param_name in required

                                    yield Label(f"{param_name} (Required: {is_required})", classes="param-label")
                                    yield Label(
                                        f"Type: {param_type}\nDescription: {param_description}", classes="param-value"
                                    )

        yield Footer()

    def action_back(self) -> None:
        """Go back to the tools list screen."""
        self.app.pop_screen()


class MCPServerApp(BaseApp):
    """Main application for MCP server management."""

    def __init__(self, servers: List[Dict[str, Any]], mcp_tool: MCPTool):
        """
        Initialize the MCP server app.

        Args:
            servers: List of available MCP servers from MCPTool.list_servers
        """
        super().__init__()
        self.title = "MCP Servers"
        self.servers = servers
        self.theme = "textual-dark"
        self.mcp_tool = mcp_tool

    def on_mount(self):
        """Push the server list screen on mount."""
        self.push_screen(MCPServerListScreen(self.mcp_tool, {"servers": self.servers}))

    def _on_exit_app(self) -> None:
        super()._on_exit_app()
