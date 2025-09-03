"""
Datus-CLI Context Commands
This module provides context-related commands for the Datus CLI.
"""

from typing import TYPE_CHECKING

from rich.table import Table
from rich.tree import Tree

from datus.utils.loggings import get_logger
from datus.utils.rich_util import dict_to_tree

if TYPE_CHECKING:
    from datus.cli import DatusCLI

logger = get_logger(__name__)


class ContextCommands:
    """Handles all context-related commands in the CLI."""

    def __init__(self, cli: "DatusCLI"):
        """Initialize with a reference to the CLI instance."""
        self.cli = cli
        self.console = cli.console

    def cmd_catalogs(self, args: str):
        """Display database catalogs using Textual tree interface."""
        try:
            # Import here to avoid circular imports

            if not self.cli.db_connector and not self.cli.agent_config:
                self.console.print("[bold red]Error:[/] No database connection or configuration.")
                return

            from datus.cli.screen import show_catalogs_screen

            # Push the catalogs screen
            show_catalogs_screen(
                title="Database Catalogs",
                data={
                    "db_type": self.cli.agent_config.db_type,
                    "catalog_name": self.cli.current_catalog,
                    "database_name": self.cli.current_db_name,
                    "db_connector": self.cli.db_connector,
                },
                inject_callback=self.cli.catalogs_callback,
            )

        except Exception as e:
            logger.error(f"Catalog display error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to display catalog: {str(e)}")

    def cmd_metrics(self, args: str):
        """Display metrics."""
        self.console.print("[yellow]Feature not implemented in MVP.[/]")
