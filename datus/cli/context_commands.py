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

    def cmd_context(self, args: str = "sql"):
        """Display the current context."""
        try:
            if not self.cli.agent.workflow:
                self.console.print("[yellow]No workflow available. Please start a new workflow using !dastart[/]")
                return

            context_type = args.lower().strip() if args else "sql"
            query_map = {
                "sql": self.cli.agent.workflow.context.sql_contexts,
                "schema": self.cli.agent.workflow.context.table_schemas,
                "schema_values": self.cli.agent.workflow.context.table_values,
                "metrics": self.cli.agent.workflow.context.metrics,
            }

            if context_type in query_map:
                for i, item in enumerate(query_map[context_type]):
                    tree = dict_to_tree(item.to_dict(), tree=Tree(f"{i + 1}."))
                    self.console.print(tree)
            elif context_type == "lastsql":
                self.console.print(dict_to_tree(self.cli.agent.workflow.context.sql_contexts[-1].to_dict()))
            elif context_type == "task":
                self.console.print(self.cli.agent.workflow.task.to_dict())
            else:
                self.console.print(dict_to_tree(self.cli.agent.workflow.context.to_dict()))
        except Exception as e:
            logger.error(f"Context display error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to display context: {str(e)}")

    def cmd_context_screen(self, args: str):
        """Display the current context in a screen."""
        if not self.cli.agent.workflow:
            self.console.print("[yellow]No workflow available. Please start a new workflow using !dastart[/]")
            return

        context_type = args.strip().lower()

        try:
            from datus.cli.screen import show_workflow_context_screen

            # Get all context data from the workflow
            context_data = {
                "sql_contexts": [item.to_dict() for item in self.cli.agent.workflow.context.sql_contexts],
                "table_schemas": [item.to_dict() for item in self.cli.agent.workflow.context.table_schemas],
                "table_values": [item.to_dict() for item in self.cli.agent.workflow.context.table_values],
                "metrics": [item.to_dict() for item in self.cli.agent.workflow.context.metrics],
            }

            # Enhanced debugging
            logger.debug("Context data summary:")
            logger.debug(f"  SQL contexts: {len(context_data['sql_contexts'])}")
            logger.debug(f"  Table schemas: {len(context_data['table_schemas'])}")
            logger.debug(f"  Table values: {len(context_data['table_values'])}")
            logger.debug(f"  Metrics: {len(context_data['metrics'])}")

            # Debug table schemas specifically
            if context_data["table_schemas"]:
                logger.debug(f"First table schema keys: {list(context_data['table_schemas'][0].keys())}")
                first_schema = context_data["table_schemas"][0]
                if "table_name" in first_schema:
                    logger.debug(f"First table name: {first_schema['table_name']}")
                if "columns" in first_schema:
                    logger.debug(f"Columns type: {type(first_schema['columns'])}")
                    if isinstance(first_schema["columns"], list) and first_schema["columns"]:
                        logger.debug(f"First column: {first_schema['columns'][0]}")
                    else:
                        logger.debug(f"Columns content: {first_schema['columns']}")

            logger.debug(f"Full context data: {context_data}")

            if context_type:
                # If a specific context type is specified, filter to only show that type
                if context_type in ["sql", "schema", "schema_values", "metrics"]:
                    # Map command arg to context data key
                    context_map = {
                        "sql": "sql_contexts",
                        "schema": "table_schemas",
                        "schema_values": "table_values",
                        "metrics": "metrics",
                    }

                    # Create a new context_data with only the specified type
                    filtered_data = {context_map[context_type]: context_data[context_map[context_type]]}
                    logger.debug(f"Filtered data for {context_type}: {filtered_data}")
                    show_workflow_context_screen(f"Context: {context_type.capitalize()}", filtered_data)
                else:
                    self.console.print(f"[bold yellow]Warning:[/] Unknown context type: {context_type}")
                    self.console.print("Valid types: sql, schema, schema_values, metrics")
            else:
                # Show all context types
                show_workflow_context_screen("Workflow Context", context_data)

        except Exception as e:
            logger.error(f"Context screen error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] Failed to display context screen: {str(e)}")

    def cmd_tables(self, args: str):
        """List all tables in the current database with context information."""
        if not self.cli.db_connector:
            self.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            # For SQLite, query the sqlite_master table
            result = self.cli.db_connector.get_tables(
                catalog_name=self.cli.current_catalog,
                database_name=self.cli.current_db_name,
                schema_name=self.cli.current_schema,
            )
            self.cli.last_result = result

            if result:
                # Display results
                table = Table(title="Show tables", show_header=True, header_style="bold green")
                # Add columns
                table.add_column("Table Name")
                # Add rows
                for row in result:
                    table.add_row(row)
                self.console.print(table)
            else:
                self.console.print("Empty set.")

        except Exception as e:
            logger.error(f"Table listing error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_metrics(self, args: str):
        """Display metrics."""
        self.console.print("[yellow]Feature not implemented in MVP.[/]")
