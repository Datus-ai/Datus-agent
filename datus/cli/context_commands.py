# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Datus-CLI Context Commands
This module provides context-related commands for the Datus CLI.
"""

from typing import TYPE_CHECKING

from datus.cli.screen import show_subject_screen
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli import DatusCLI

logger = get_logger(__name__)


class ContextCommands:
    """Handles all context-related commands in the CLI."""

    def __init__(self, cli: "DatusCLI"):
        """Initialize with a reference to the CLI instance."""
        self.cli = cli
        self.console = cli.console

    def cmd_catalog(self, args: str):
        """Display database catalogs using Textual tree interface."""
        try:
            # Import here to avoid circular imports

            if not self.cli.db_connector and not self.cli.agent_config:
                self.console.print("[bold red]Error:[/] No database connection or configuration.")
                return

            from datus.cli.screen import show_catalog_screen

            # Push the catalogs screen
            show_catalog_screen(
                title="Database Catalogs",
                data={
                    "db_type": self.cli.agent_config.db_type,
                    "catalog_name": self.cli.cli_context.current_catalog,
                    "database_name": self.cli.cli_context.current_db_name,
                    "db_connector": self.cli.db_connector,
                    "agent_config": self.cli.agent_config,
                },
                inject_callback=self.cli.catalogs_callback,
            )

        except Exception as e:
            logger.error(f"Catalog display error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to display catalog: {str(e)}")

    def cmd_subject(self, args: str):
        """Display metrics."""

        show_subject_screen(
            title="Subject",
            data={
                "agent_config": self.cli.agent_config,
            },
        )
