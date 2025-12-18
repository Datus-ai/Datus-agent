# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Metadata-related CLI commands for database introspection.
This module handles database, table, and schema listing/switching functionality.
"""

from typing import TYPE_CHECKING

import numpy as np
from rich.box import SIMPLE_HEAD
from rich.panel import Panel
from rich.table import Table

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class MetadataCommands:
    """Handler for metadata-related CLI commands (.databases, .tables, etc.)."""

    def __init__(self, cli_instance: "DatusCLI"):
        """Initialize with reference to the main CLI instance."""
        self.cli = cli_instance

    def cmd_list_databases(self, args: str = ""):
        """List all databases in the current connection."""
        try:
            # For SQLite, this is simply the current database file
            namespace = self.cli.agent_config.current_namespace
            database_config_dict = self.cli.agent_config.namespaces[namespace]
            result = []
            show_uri = False
            if len(database_config_dict) > 1:
                # Multi-Database
                show_uri = True
                for _, db_config in database_config_dict.items():
                    logic_name = db_config.logic_name
                    is_current = logic_name == self.cli.cli_context.current_logic_db_name
                    result.append(
                        {
                            "logic_name": logic_name if not is_current else f"[bold green]{logic_name}[/]",
                            "name": db_config.database,
                            "uri": db_config.uri,
                        }
                    )
            else:
                # single database
                db_config = list(database_config_dict.values())[0]
                db_type = db_config.type
                if db_type in (DBType.SQLITE, DBType.DUCKDB):
                    show_uri = True
                    is_current = db_config.logic_name == self.cli.cli_context.current_logic_db_name
                    result.append(
                        {
                            "logic_name": db_config.logic_name
                            if not is_current
                            else f"[bold green]{db_config.logic_name}[/]",
                            "name": db_config.database,
                            "uri": db_config.uri,
                        }
                    )
                else:
                    for db_name in self.cli.db_connector.get_databases(
                        catalog_name=self.cli.cli_context.current_catalog
                    ):
                        result.append(
                            {
                                "name": db_name
                                if db_name != self.cli.cli_context.current_db_name
                                else f"[bold green]{db_name}[/]"
                            }
                        )

            self.cli.last_result = result
            # Display results
            table = Table(title="Databases", show_header=True, header_style="bold green")
            if show_uri:
                table.add_column("Logic Name(Used for switch)")
                table.add_column("Database Name")
                table.add_column("URI")
                for db_config in result:
                    table.add_row(db_config["logic_name"], db_config["name"], db_config["uri"])
            else:
                table.add_column("Name")
                for db_config in result:
                    table.add_row(db_config["name"])
            self.cli.console.print(table)

        except Exception as e:
            logger.error(f"Database listing error: {str(e)}")
            self.cli.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_switch_database(self, args: str = ""):
        """Switch current database."""
        new_db = args.strip()
        if not new_db:
            self.cli.console.print("[bold red]Error:[/] Database name is required")
            self.cmd_list_databases()
            return
        if (
            self.cli.db_connector.dialect in (DBType.SQLITE, DBType.DUCKDB)
            and self.cli.cli_context.current_logic_db_name == new_db
        ):
            self.cli.console.print(
                f"[yellow]It's now under the database [bold]{new_db}[/] and doesn't need to be switched[/]"
            )
            return
        if new_db == self.cli.cli_context.current_db_name:
            self.cli.console.print(
                f"[yellow]It's now under the database [bold]{new_db}[/] and doesn't need to be switched[/]"
            )
            return

        self.cli.agent_config.current_database = new_db
        self.cli.cli_context.current_logic_db_name = new_db
        if self.cli.agent_config.db_type in (DBType.SQLITE, DBType.DUCKDB):
            if new_db not in self.cli.agent_config.current_db_configs():
                self.cli.console.print(f"[bold yellow]No corresponding database was found: {new_db}[/]")
                return
            # Logic database name
            self.cli.db_connector = self.cli.db_manager.get_conn(self.cli.agent_config.current_namespace, new_db)
            # use real database name
            self.cli.cli_context.update_database_context(
                db_name=self.cli.db_connector.database_name, db_logic_name=new_db
            )
            self.cli.agent_config.current_database = new_db
            self.cli.reset_session()
        else:
            self.cli.db_connector.switch_context(database_name=new_db)
            self.cli.agent_config.current_database = new_db
            self.cli.cli_context.update_database_context(db_name=new_db)

        if self.cli.agent_config.db_type in (DBType.SQLITE, DBType.DUCKDB):
            self.cli.chat_commands.update_chat_node_tools()

        self.cli.console.print(f"[bold green]Database switched to: {new_db}[/]")

    def cmd_tables(self, args: str):
        """List all tables in the current database (internal command)."""
        # Reuse functionality from context commands, but with internal command styling
        if not self.cli.db_connector:
            self.cli.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            # For SQLite, query the sqlite_master table
            result = self.cli.db_connector.get_tables(
                catalog_name=self.cli.cli_context.current_catalog,
                database_name=self.cli.cli_context.current_db_name,
                schema_name=self.cli.cli_context.current_schema,
            )
            self.cli.last_result = result
            if result:
                # Display results
                table = Table(
                    show_header=True,
                    header_style="bold green",
                )
                # Add columns
                table.add_column("Table Name")
                for row in result:
                    table.add_row(row)
                if self.cli.cli_context.current_schema:
                    if self.cli.cli_context.current_db_name:
                        show_name = f"{self.cli.cli_context.current_db_name}.{self.cli.cli_context.current_schema}"
                    else:
                        show_name = self.cli.cli_context.current_schema
                else:
                    show_name = self.cli.cli_context.current_db_name
                panel = Panel(table, title=f"Tables in Database {show_name}", title_align="left", box=SIMPLE_HEAD)
                self.cli.console.print(panel)
            else:
                # For other database types, execute the appropriate query
                self.cli.console.print("[yellow]Empty set.[/]")

        except Exception as e:
            logger.error(f"Table listing error: {str(e)}")
            self.cli.console.print(f"[bold red]Error:[/] {str(e)}")

    def cmd_schemas(self, args: str):
        """List all schemas in the current database."""
        dialect = self.cli.db_connector.dialect
        if not DBType.support_schema(dialect):
            self.cli.console.print(f"[bold red]The {dialect} database does not support schema[/]")
            return
        result = self.cli.db_connector.get_schemas(
            catalog_name=self.cli.cli_context.current_catalog, database_name=self.cli.cli_context.current_db_name
        )
        self.cli.last_result = result
        if result:
            # Display results
            table = Table(
                show_header=True,
                header_style="bold green",
            )
            # Add columns
            table.add_column("Schema Name")
            for row in result:
                table.add_row(row)
            if self.cli.cli_context.current_catalog:
                if self.cli.cli_context.current_db_name:
                    show_name = f"{self.cli.cli_context.current_catalog}.{self.cli.cli_context.current_db_name}"
                else:
                    show_name = self.cli.cli_context.current_catalog
            else:
                show_name = self.cli.cli_context.current_db_name
            panel = Panel(table, title=f"Schema in Database {show_name}", title_align="left", box=SIMPLE_HEAD)
            self.cli.console.print(panel)
        else:
            # For other database types, execute the appropriate query
            self.cli.console.print("[yellow]Empty set.[/]")

    def cmd_switch_schema(self, args: str):
        """Switch current schema."""
        dialect = self.cli.db_connector.dialect
        if not DBType.support_schema(dialect):
            self.cli.console.print(f"[bold red]The {dialect} database does not support schema[/]")
            return
        schema_name = args.strip()
        if not schema_name:
            self.cli.console.print("[yellow]You need to give the name of the schema you want to switch to[/]")
            return
        self.cli.db_connector.switch_context(
            catalog_name=self.cli.cli_context.current_catalog,
            database_name=self.cli.cli_context.current_db_name,
            schema_name=schema_name,
        )
        self.cli.cli_context.current_schema = schema_name
        self.cli.console.print(f"[bold green]Schema switched to: {self.cli.cli_context.current_schema}[/]")

    def cmd_table_schema(self, args: str):
        """Show schema information for tables."""
        if not self.cli.db_connector:
            self.cli.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            if args.strip():
                table_name = args.strip()
                result = self.cli.db_connector.get_schema(
                    catalog_name=self.cli.cli_context.current_db_name,
                    database_name=self.cli.cli_context.current_db_name,
                    schema_name=self.cli.cli_context.current_schema,
                    table_name=table_name,
                )
                self.cli.last_result = result

                # Display schema for the specific table
                schema_table = Table(
                    title=f"Schema for {table_name}",
                    show_header=True,
                    header_style="bold green",
                )
                schema_table.add_column("Column Position")
                schema_table.add_column("Name")
                schema_table.add_column("Type")
                schema_table.add_column("Nullable")
                schema_table.add_column("Default")
                schema_table.add_column("PK")

                for row in result:
                    schema_table.add_row(
                        str(row.get("cid", "")),
                        str(row.get("name", "")),
                        str(row.get("type", "")),
                        str(row.get("nullable", "")),
                        str(row.get("default_value", "")) if row.get("default_value") is not None else "",
                        str(row.get("pk", "")),
                    )

                self.cli.console.print(schema_table)
            else:
                # List all tables with basic schema info
                table_names = self.cli.db_connector.get_tables(
                    catalog_name=self.cli.cli_context.current_catalog,
                    database_name=self.cli.cli_context.current_db_name,
                    schema_name=self.cli.cli_context.current_schema,
                )
                self.cli.last_result = table_names

                # Display list of tables
                self.cli.console.print("[bold green]Available tables:[/]")
                # Display table list
                for idx, table_name in enumerate(table_names):
                    self.cli.console.print(f"{idx + 1}. {table_name}")

                self.cli.console.print("\n[dim]Use .schemas [table_name] to view detailed schema.[/]")

        except Exception as e:
            logger.error(f"Schema listing error: {str(e)}")
            self.cli.console.print(f"[bold red]Error:[/] {str(e)}")
            if "result" in locals():
                logger.debug(f"Result object structure: {dir(result)}")
                for key in dir(result):
                    if not key.startswith("_"):
                        try:
                            value = getattr(result, key)
                            logger.debug(f"  {key}: {value}")
                        except Exception as e:
                            logger.debug(f"  {key}: Error accessing - {e}")
                if hasattr(result, "__dict__"):
                    logger.debug(f"Result __dict__: {result.__dict__}")
                logger.debug(f"Result type: {type(result)}")

    def cmd_indexes(self, args: str):
        """Show indexes for a table."""
        table_name = args.strip()
        if not table_name:
            self.cli.console.print("[bold red]Error:[/] Table name required")
            return

        if not self.cli.db_connector:
            self.cli.console.print("[bold red]Error:[/] No database connection.")
            return

        try:
            # For SQLite, query the sqlite_master table
            if self.cli.db_connector.get_type() == DBType.SQLITE:
                sql = f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'"
                result = self.cli.db_connector.execute_pandas(sql)

                if result is None or not result.success:
                    self.cli.console.print("[bold red]Error:[/] Query failed")
                    return

                indexes = result.sql_return.replace({np.nan: None}).to_dict(orient="records")
                if indexes:
                    index_table = Table(title=f"Indexes for {table_name}")
                    index_table.add_column("Index Name")
                    for idx in indexes:
                        # Handle both dict and tuple formats
                        if isinstance(idx, dict):
                            index_name = idx.get("name", str(idx))
                        elif isinstance(idx, (list, tuple)) and len(idx) > 0:
                            index_name = idx[0]
                        else:
                            index_name = str(idx)
                        index_table.add_row(index_name)
                    self.cli.console.print(index_table)
                else:
                    self.cli.console.print(f"[yellow]Table {table_name} has no indexes[/]")
            else:
                # For other database types, use information schema or equivalent
                # This is a placeholder for future database type support
                self.cli.console.print(
                    f"[yellow]Index listing not yet supported for {self.cli.db_connector.get_type()}[/]"
                )

        except Exception as e:
            logger.error(f"Index listing error: {str(e)}")
            self.cli.console.print(f"[bold red]Error:[/] {str(e)}")
