"""
MCP-related commands for the Datus CLI.
This module provides commands to list and manage MCP configurations.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

from datus.cli.screen.mcp_screen import MCPServerApp
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
        # builtin MCP servers
        mcp_server_json = {
            "mcpServers": {
                "filesystem": {
                    "type": "builtin",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        "/Users/username/Desktop",
                        "/path/to/other/allowed/dir",
                    ],
                },
                "sqlite": {
                    "type": "builtin",
                    "command": "uv",
                    "args": [
                        "--directory",
                        "mcp/mcp-sqlite-server/src/mcp_server_sqlite/sqlite",
                        "run",
                        "mcp-server-sqlite",
                        "--db-path",
                        "<your path>",
                    ],
                },
                "jetbrains": {"type": "builtin", "command": "npx", "args": ["-y", "@jetbrains/mcp-proxy"]},
            }
        }
        self.mcp_servers = mcp_server_json["mcpServers"]

    def cmd_mcp(self, args: str):
        if args == "list":
            self.cmd_mcp_list(args)
        elif args.startswith("add"):
            self.cmd_mcp_add(args[3:].strip())
        else:
            self.console.print("[red]Invalid MCP command[/red]")

    def cmd_mcp_list(self, args: str):
        user_mcp_config_path = Path.home() / ".datus" / "mcp.json"
        user_mcps = {}
        if user_mcp_config_path.exists():
            try:
                with open(user_mcp_config_path, "r") as f:
                    user_config = json.load(f)
                    user_mcps = user_config.get("mcpServers", {})
            except Exception as e:
                self.console.print(f"[red]Error reading user MCP config: {e}[/red]")
        else:
            self.console.print("[yellow]No user MCP configuration found at ~/.datus/mcp.json[/yellow]")

        for mcp_name, mcp_config in user_mcps.items():
            mcp_config["type"] = "user"
            self.mcp_servers[mcp_name] = mcp_config

        try:
            screen = MCPServerApp(self.mcp_servers)
            screen.run()
        except Exception as e:
            self.console.print(f"[yellow]Interactive mode error: {e}[/yellow]")
            self._display_servers_table()

    def _display_servers_table(self):
        """Display servers in a formatted table."""
        table = Table(title="MCP Servers")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Command", style="green")
        table.add_column("Args", style="yellow")

        for name, config in self.mcp_servers:
            server_type = config.get("type", "unknown")
            status = "[green]Available[/green]" if server_type == "builtin" else "[yellow]User[/yellow]"

            table.add_row(
                name,
                status,
                server_type,
                config.get("command", ""),
                " ".join(config.get("args", [])),
            )

        self.console.print(table)

    def cmd_mcp_add(self, args: str):
        """Add a new MCP configuration."""

        datus_dir = Path.home() / ".datus"
        datus_dir.mkdir(exist_ok=True)

        mcp_config_path = datus_dir / "mcp.json"

        if not mcp_config_path.exists():
            base_config = {"mcpServers": {}}
            with open(mcp_config_path, "w") as f:
                json.dump(base_config, f, indent=2)
            self.console.print(f"[green]Created new MCP configuration file at {mcp_config_path}[/green]")

        try:
            with open(mcp_config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            self.console.print(f"[red]Error reading MCP config: {e}[/red]")
            return

        mcp_name = input("Enter MCP name: ").strip()
        if not mcp_name:
            self.console.print("[red]MCP name cannot be empty[/red]")
            return

        command = input("Enter command to run MCP server (e.g., uvx, python): ").strip()
        if not command:
            self.console.print("[red]Command cannot be empty[/red]")
            return

        args_input = input("Enter command arguments (space-separated): ").strip()
        args_list = args_input.split() if args_input else []

        new_mcp_config = {"command": command, "args": args_list}

        config["mcpServers"][mcp_name] = new_mcp_config

        try:
            with open(mcp_config_path, "w") as f:
                json.dump(config, f, indent=2)
            self.console.print(f"[green]Successfully added MCP configuration for '{mcp_name}'[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving MCP config: {e}[/red]")
            return
