"""
Enhanced MCP (Model Context Protocol) server management screen for Datus CLI.
Provides elegant interactive interface for browsing and selecting MCP servers and tools.
"""

from typing import Any, Dict, List

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

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
        #
        # server_list.index = 0
        # server_list.focus()

        # Schedule async task to check server connectivity
        self.app.call_later(self._check_servers_connectivity)

    async def _check_servers_connectivity(self) -> None:
        """Asynchronously check connectivity for all servers in parallel and update UI."""
        try:
            server_list = self.query_one("#server-list", ListView)

            # Create async tasks for parallel connectivity checking
            import asyncio

            async def check_single_server(server, index):
                """Check connectivity for a single server and update its status."""
                server_name = server.get("name", "Unknown")

                # Get the corresponding list item
                if index < len(server_list.children):
                    list_item = server_list.children[index]
                    status_label = list_item.server_status_label

                    try:
                        # Run connectivity check in thread pool to prevent blocking
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, self.mcp_tool.check_connectivity, server_name
                        )
                        logger.info(f"$$$check {server_name} result: {result}")

                        if result.success and result.result.get("connectivity", False):
                            # Server is reachable
                            status_symbol = "✔"
                            status_text = "connected"
                            status_class = "status-connected"
                        else:
                            # Server is not reachable
                            status_symbol = "✘"
                            status_text = "failed"
                            status_class = "status-failed"

                    except Exception as e:
                        # Handle exception during connectivity check
                        logger.error(f"Check mcp server failed {e}")
                        status_symbol = "✘"
                        status_text = "failed"
                        status_class = "status-failed"

                    # Update the status label on the main thread
                    self.app.call_later(
                        lambda: self._update_server_status(status_label, status_symbol, status_text, status_class)
                    )

            # Create and run all connectivity checks in parallel
            tasks = [check_single_server(server, i) for i, server in enumerate(self.servers)]

            # Wait for all tasks to complete (but they update UI as they finish)
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error checking server connectivity: {e}")

    def _update_server_status(self, status_label, status_symbol, status_text, status_class):
        """Update server status label with new connectivity status."""
        status_label.update(f"{status_symbol} {status_text} · Enter to view details")
        status_label.set_class(False, "status-checking")
        status_label.set_class(True, status_class)

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

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            if not self.connected:
                return
            self.action_view_tools()
            event.stop()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        """Asynchronously fetch tool list and update UI when complete"""
        self.query_one("#view-tools-option", ListView).has_focus = True
        # info_grid = self.query_one("#server-info-display", Grid)
        #
        # # Add server type row
        # info_grid.mount(Label("Type:", classes="label"))
        # info_grid.mount(Label(f"[cyan]{self.server_type}[/cyan]"))
        #
        # # Type-specific configuration
        # if self.server_type == "stdio":
        #     command = self.server_data.get("command", "")
        #     args = self.server_data.get("args", [])
        #     env = self.server_data.get("env", {})
        #
        #     info_grid.mount(Label("Command:", classes="label"))
        #     info_grid.mount(Label(f"[green]{command}[/green]"))
        #     if args:
        #         args_str = " ".join(args)
        #         info_grid.mount(Label("Args:", classes="label"))
        #         info_grid.mount(Label(f"[yellow]{args_str}[/yellow]"))
        #     if env:
        #         env_str = ", ".join([f"{k}={v}" for k, v in env.items()])
        #         info_grid.mount(Label("Env:", classes="label"))
        #         info_grid.mount(Label(f"[magenta]{env_str}[/magenta]"))
        #
        # elif self.server_type in ["sse", "http"]:
        #     url = self.server_data.get("url", "")
        #     headers = self.server_data.get("headers", {})
        #     timeout = self.server_data.get("timeout")
        #
        #     info_grid.mount(Label("URL:", classes="label"))
        #     info_grid.mount(Label(f"[blue]{url}[/blue]"))
        #     if headers:
        #         headers_str = ", ".join([f"{k}: {v}" for k, v in headers.items()])
        #         info_grid.mount(Label("Headers:", classes="label"))
        #         info_grid.mount(Label(f"[magenta]{headers_str}[/magenta]"))
        #     if timeout:
        #         info_grid.mount(Label("Timeout:", classes="label"))
        #         info_grid.mount(Label(f"[yellow]{timeout}s[/yellow]"))
        #
        # # Capabilities row
        # info_grid.mount(Label("Capabilities:", classes="label"))
        # info_grid.mount(Label("tools"))
        #
        # # Tools row with loading status
        # info_grid.mount(Label("Tools:", classes="label"))
        # info_grid.mount(Label("[dim]Loading...[/dim]", id="tools-value"))

        # Schedule async task to fetch tools
        self.app.call_later(self._fetch_tools_async)

    async def _fetch_tools_async(self) -> None:
        tools_value = self.query_one("#tools-value", Label)
        try:
            # Asynchronously fetch tool list
            tools = self.mcp_tool.list_tools(self.server_name)
            logger.info(f"Tools: {tools}")

            # Update tools list
            self.tools = [] if not tools.success else tools.result["tools"]

            # Update tools row with count or error message
            tools_count = len(self.tools)
            if tools.success:
                tools_value.update(f"[green]{tools_count}[/green] tools available")
                view_tools = self.query_one("#view-tools-option", ListView)
                view_tools.disabled = False
                self.connected = True

            else:
                self.connected = False
                tools_value.update("[red]Failed to load tools[/red]")
                logger.error(f"Failed to load tools for server {self.server_name}: {tools.message}")

        except Exception as e:
            # Handle exception case
            logger.error(f"Error loading tools for server {self.server_name}: {str(e)}")

            tools_value.update(f"[red]Error loading tools ({str(e)})[/red]")

            self.tools = []

    # def on_button_pressed(self, _event: Button.Pressed) -> None:
    #     self.action_view_tools()

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

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        tools_list = self.query_one("#tools-list", ListView)
        for i, tool in enumerate(self.tools):
            tool_name = tool.get("name", f"tool_{i+1}")
            # tool_description = tool.get("description", "No description available")

            # Create tool item with name and description
            tool_label = Label(f"{i+1}. {tool_name}", classes="tool-name tool-description")
            list_item = ListItem(tool_label)
            tools_list.append(list_item)
        tools_list.index = 0
        tools_list.focus()

    def action_back(self) -> None:
        """Go back to the server detail screen."""
        self.app.pop_screen()


class MCPServerApp(App):
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
