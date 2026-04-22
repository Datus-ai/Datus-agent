# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MCP-related commands for the Datus CLI.
This module provides commands to list and manage MCP configurations.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List

from rich.table import Table

from datus.cli.cli_styles import (
    TABLE_HEADER_STYLE,
    print_empty_set,
    print_error,
    print_info,
    print_status,
    print_success,
    print_usage,
    print_warning,
)
from datus.cli.screen.mcp_screen import MCPServerApp
from datus.tools.mcp_tools import MCPTool, parse_command_string
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class MCPCommands:
    """Handles all MCP-related commands."""

    def __init__(self, cli_instance: "DatusCLI"):
        """Initialize with reference to the CLI instance for shared resources."""
        self.cli = cli_instance
        self.console = cli_instance.console
        self.mcp_tool = MCPTool()

    def cmd_mcp(self, args: str):
        if args == "list":
            self.cmd_mcp_list()
        elif args.startswith("add"):
            self.cmd_mcp_add(args[3:].strip())
        elif args.startswith("remove"):
            self.cmd_mcp_remove(args[6:].strip())
        elif args.startswith("check"):
            self.cmd_mcp_check(args[5:].strip())
        elif args.startswith("call"):
            self.cmd_call_tool(args[4:].strip())
        elif args.startswith("filter"):
            self.cmd_mcp_filter(args[6:].strip())
        else:
            print_error(self.console, "Invalid MCP command", prefix=False)

    def cmd_mcp_list(self):
        mcp_servers = self.mcp_tool.list_servers()
        if not mcp_servers.success:
            print_error(self.console, f"Error listing MCP servers: {mcp_servers.message}")
            return
        if not mcp_servers.result:
            print_empty_set(self.console, "No MCP servers found")
            return
        servers = mcp_servers.result["servers"]
        try:
            screen = MCPServerApp(servers, self.mcp_tool)
            screen.run()
        except Exception as e:
            print_warning(self.console, f"Interactive mode error: {str(e)}")
            print_info(self.console, "Falling back to table display mode...")
            self._display_servers_table(servers)

    def _display_servers_table(self, servers: List[Dict[str, Any]]):
        """Display servers in a formatted table."""
        table = Table(title="MCP Servers", header_style=TABLE_HEADER_STYLE)
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Command", style="green")
        table.add_column("Args", style="yellow")

        for server in servers:
            name = server.get("name", "unknown")
            server_type = server.get("type", "unknown")
            status = "[green]Available[/green]" if server_type == "builtin" else "[yellow]User[/yellow]"

            table.add_row(
                name,
                status,
                server_type,
                server.get("command", ""),
                " ".join(server.get("args", [])),
            )

        self.console.print(table)

    def cmd_mcp_add(self, args: str):
        """Add a new MCP configuration."""
        try:
            transport_type, server_name, config_params = parse_command_string(args)
            # Call the add_server method
            result = self.mcp_tool.add_server(name=server_name, type=transport_type, **config_params)

            if result.success:
                print_success(self.console, f"Successfully added MCP server: {server_name}", symbol=True)
                self.console.print(f"Type: {transport_type}")
            else:
                print_error(self.console, f"Error adding MCP server: {result.message}")

        except Exception as e:
            logger.error(f"Error in cmd_mcp_add: {e}")
            print_error(self.console, str(e))

    def cmd_mcp_remove(self, args: str):
        """Remove an MCP configuration."""
        server_name = args.strip()
        if not server_name:
            print_error(self.console, "Please specify the name of the MCP server to remove", prefix=False)
            return
        remove_result = self.mcp_tool.remove_server(server_name)
        if remove_result.success:
            print_success(self.console, f"Successfully removed MCP server: {server_name}", symbol=True)
        else:
            print_error(self.console, f"Error removing MCP server: {remove_result.message}")

    def cmd_mcp_check(self, args: str):
        server_name = args.strip()
        if not server_name:
            print_error(self.console, "Please specify the name of the MCP server to check", prefix=False)
            return

        result = self.mcp_tool.check_connectivity(server_name)

        if result.success:
            connectivity = result.result.get("connectivity", False)
            details = result.result.get("details", {})

            if connectivity:
                print_status(self.console, f"Server '{server_name}' is reachable", ok=True)
                self.console.print(f"  Type: {details.get('type', 'unknown')}")
                if "tools_count" in details:
                    self.console.print(f"  Available tools: {details['tools_count']}")
            else:
                print_status(self.console, f"Server '{server_name}' is not reachable", ok=False)
                if "error" in details:
                    self.console.print(f"  Error: {details['error']}")
        else:
            print_status(self.console, f"Error: {result.message}", ok=False)

    def cmd_call_tool(self, args: str):
        """Call a tool on a MCP server."""
        params = args.strip().split()
        server_tool = params[0].split(".")
        if len(server_tool) != 2:
            print_error(self.console, "Invalid server.tool format")
            return
        server_name, tool_name = server_tool
        tool_params = None
        if len(params) >= 2:
            arguments = " ".join(params[1:])
            if arguments:
                try:
                    tool_params = json.loads(arguments)
                except Exception as e:
                    print_error(self.console, f"The parameters for calling the tool should be in json format: {e}")
                    return
        # parse arguments to dict
        result = self.mcp_tool.call_tool(server_name, tool_name, tool_params)
        if not result.success:
            print_error(self.console, f"Error calling tool: {result.message}")
            return
        if not (result := result.result["result"]):
            print_empty_set(self.console, "No result returned")
            return
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                self.console.print(result)
                return
        elif not isinstance(result, dict):
            self.console.print(result)
            return
        if result.get("isError") or False:
            print_error(self.console, f"Call Tool Error: {result['content']}")
            return

        self.console.print(result)

    def cmd_mcp_filter(self, args: str):
        """Handle MCP filter subcommands."""
        if args.startswith("set"):
            self.cmd_mcp_filter_set(args[3:].strip())
        elif args.startswith("get"):
            self.cmd_mcp_filter_get(args[3:].strip())
        elif args.startswith("remove"):
            self.cmd_mcp_filter_remove(args[6:].strip())
        else:
            print_error(self.console, "Invalid filter command. Use: set, get, or remove", prefix=False)

    def cmd_mcp_filter_set(self, args: str):
        """Set tool filter for a server."""
        params = args.strip().split()
        if len(params) < 2:
            print_usage(
                self.console,
                "/mcp filter set <server_name> [--allowed tool1,tool2] [--blocked tool3,tool4] [--enabled true/false]",
            )
            return

        server_name = params[0]
        allowed_tools = None
        blocked_tools = None
        enabled = True

        i = 1
        while i < len(params):
            if params[i] == "--allowed" and i + 1 < len(params):
                allowed_tools = [tool.strip() for tool in params[i + 1].split(",") if tool.strip()]
                i += 2
            elif params[i] == "--blocked" and i + 1 < len(params):
                blocked_tools = [tool.strip() for tool in params[i + 1].split(",") if tool.strip()]
                i += 2
            elif params[i] == "--enabled" and i + 1 < len(params):
                enabled = params[i + 1].lower() in ("true", "1", "yes", "on")
                i += 2
            else:
                i += 1

        result = self.mcp_tool.set_tool_filter(
            server_name=server_name, allowed_tools=allowed_tools, blocked_tools=blocked_tools, enabled=enabled
        )

        if result.success:
            print_status(self.console, f"Tool filter set for server '{server_name}'", ok=True)
            if allowed_tools:
                self.console.print(f"  Allowed tools: {', '.join(allowed_tools)}")
            if blocked_tools:
                self.console.print(f"  Blocked tools: {', '.join(blocked_tools)}")
            self.console.print(f"  Filter enabled: {enabled}")
        else:
            print_status(self.console, f"Error setting tool filter: {result.message}", ok=False)

    def cmd_mcp_filter_get(self, args: str):
        """Get tool filter configuration for a server."""
        server_name = args.strip()
        if not server_name:
            print_error(self.console, "Please specify the name of the MCP server", prefix=False)
            return

        result = self.mcp_tool.get_tool_filter(server_name)

        if result.success:
            if result.result["has_filter"]:
                filter_config = result.result["filter_config"]
                self.console.print(f"[cyan]Tool filter for server '{server_name}':[/cyan]")
                self.console.print(f"  Enabled: {filter_config.get('enabled', False)}")

                allowed = filter_config.get("allowed_tool_names")
                if allowed:
                    self.console.print(f"  Allowed tools: {', '.join(allowed)}")

                blocked = filter_config.get("blocked_tool_names")
                if blocked:
                    self.console.print(f"  Blocked tools: {', '.join(blocked)}")

                if not allowed and not blocked:
                    self.console.print("  No specific tools configured")
            else:
                print_warning(self.console, f"No tool filter configured for server '{server_name}'")
        else:
            print_status(self.console, f"Error getting tool filter: {result.message}", ok=False)

    def cmd_mcp_filter_remove(self, args: str):
        """Remove tool filter for a server."""
        server_name = args.strip()
        if not server_name:
            print_error(self.console, "Please specify the name of the MCP server", prefix=False)
            return

        result = self.mcp_tool.remove_tool_filter(server_name)

        if result.success:
            print_status(self.console, f"Tool filter removed for server '{server_name}'", ok=True)
        else:
            print_status(self.console, f"Error removing tool filter: {result.message}", ok=False)
