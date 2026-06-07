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

from datus.cli._render_utils import build_row_table
from datus.cli.cli_styles import TABLE_HEADER_STYLE, print_empty_set, print_error, print_success, print_warning
from datus.tools.db_tools import connector_registry
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
        """List the databases of the current datasource."""
        try:
            datasource = self.cli.agent_config.current_datasource
            db_config = self.cli.agent_config.datasource_configs[datasource]
            result = []
            show_uri = False
            db_type = getattr(db_config, "type", "")
            is_file_based = bool(getattr(db_config, "path_pattern", "")) or db_type in (DBType.SQLITE, DBType.DUCKDB)
            if is_file_based:
                # File datasource: one database per matched file (glob), or the single
                # configured file. Enumerate from config rather than the connector, whose
                # SQLite/DuckDB ``get_databases`` only reports the internal ``main`` schema.
                show_uri = True
                uris = self.cli.db_manager.get_db_uris(datasource)
                for db_name in self.cli.agent_config.list_databases(datasource):
                    is_current = db_name == self.cli.cli_context.current_db_name
                    result.append(
                        {
                            "name": db_name if not is_current else f"[green]{db_name}[/]",
                            "uri": uris.get(db_name, ""),
                        }
                    )
            else:
                for db_name in self.cli.db_connector.get_databases(catalog_name=self.cli.cli_context.current_catalog):
                    result.append(
                        {
                            "name": (
                                db_name if db_name != self.cli.cli_context.current_db_name else f"[green]{db_name}[/]"
                            )
                        }
                    )

            self.cli.last_result = result
            if not result:
                print_empty_set(self.cli.console)
                return

            # Display results via the shared row-table helper so styling
            # matches ``.<service>.list_*`` output.
            if show_uri:
                columns = [("name", "Database(Used for switch)"), ("uri", "URI")]
            else:
                columns = [("name", "Name")]
            table = build_row_table(result, title="Databases", columns=columns)
            if table is not None:
                self.cli.console.print(table)

        except Exception as e:
            logger.error(f"Database listing error: {str(e)}")
            print_error(self.cli.console, str(e))

    def cmd_switch_database(self, args: str = ""):
        """Switch the active database within the current datasource."""
        new_db = args.strip()
        if not new_db:
            print_error(self.cli.console, "Database name is required")
            self.cmd_list_databases()
            return
        if new_db == self.cli.cli_context.current_db_name:
            print_warning(
                self.cli.console,
                f"It's now under the database {new_db} and doesn't need to be switched",
            )
            return

        datasource = self.cli.agent_config.current_datasource
        db_config = self.cli.agent_config.datasource_configs[datasource]
        # File datasources (glob path_pattern or a single embedded SQLite/DuckDB file): the
        # target must be one of the datasource's configured databases.
        is_file_based = bool(getattr(db_config, "path_pattern", "")) or getattr(db_config, "type", "") in (
            DBType.SQLITE,
            DBType.DUCKDB,
        )
        if is_file_based and new_db not in self.cli.agent_config.list_databases(datasource):
            print_warning(self.cli.console, f"No corresponding database was found: {new_db}")
            return
        try:
            # Reconnect bound to (datasource, new_db) — file → that file; server → that database.
            self.cli.db_connector = self.cli.db_manager.get_conn(datasource, new_db)
        except Exception as e:
            print_error(self.cli.console, str(e))
            return
        self.cli.cli_context.update_database_context(db_name=self.cli.db_connector.database_name)
        # ``reset_session()`` already refreshes the chat node tools, so don't rebuild them twice.
        self.cli.reset_session()
        print_success(self.cli.console, f"Database switched to: {new_db}")

    def cmd_tables(self, args: str):
        """List all tables in the current database (internal command)."""
        # Reuse functionality from context commands, but with internal command styling
        if not self.cli.db_connector:
            print_error(self.cli.console, "No database connection.")
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
                # Display results via the shared row-table helper. The
                # connector returns a list of names; wrap as single-field
                # dicts so ``build_row_table`` can label the column.
                table = build_row_table(
                    [{"name": row} for row in result],
                    columns=[("name", "Table Name")],
                )
                assert table is not None  # non-empty input, helper returns Table
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
                print_empty_set(self.cli.console)

        except Exception as e:
            logger.error(f"Table listing error: {str(e)}")
            print_error(self.cli.console, str(e))

    def cmd_schemas(self, args: str):
        """List all schemas in the current database."""
        dialect = self.cli.db_connector.dialect
        if not connector_registry.support_schema(dialect):
            print_error(self.cli.console, f"The {dialect} database does not support schema", prefix=False)
            return
        result = self.cli.db_connector.get_schemas(
            catalog_name=self.cli.cli_context.current_catalog, database_name=self.cli.cli_context.current_db_name
        )
        self.cli.last_result = result
        if result:
            # Display results via the shared row-table helper.
            table = build_row_table(
                [{"name": row} for row in result],
                columns=[("name", "Schema Name")],
            )
            assert table is not None
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
            print_empty_set(self.cli.console)

    def cmd_switch_schema(self, args: str):
        """Switch current schema."""
        dialect = self.cli.db_connector.dialect
        if not connector_registry.support_schema(dialect):
            print_error(self.cli.console, f"The {dialect} database does not support schema", prefix=False)
            return
        schema_name = args.strip()
        if not schema_name:
            print_warning(self.cli.console, "You need to give the name of the schema you want to switch to")
            return
        self.cli.db_connector.switch_context(
            catalog_name=self.cli.cli_context.current_catalog,
            database_name=self.cli.cli_context.current_db_name,
            schema_name=schema_name,
        )
        self.cli.cli_context.current_schema = schema_name
        print_success(self.cli.console, f"Schema switched to: {self.cli.cli_context.current_schema}")

    def cmd_table_schema(self, args: str):
        """Show schema information for tables."""
        if not self.cli.db_connector:
            print_error(self.cli.console, "No database connection.")
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

                # Display schema for the specific table via the shared helper.
                schema_table = build_row_table(
                    result,
                    title=f"Schema for {table_name}",
                    columns=[
                        ("cid", "Column Position"),
                        ("name", "Name"),
                        ("type", "Type"),
                        ("nullable", "Nullable"),
                        ("default_value", "Default"),
                        ("pk", "PK"),
                    ],
                )
                if schema_table is not None:
                    self.cli.console.print(schema_table)
                else:
                    print_empty_set(self.cli.console)
            else:
                # List all tables with basic schema info
                table_names = self.cli.db_connector.get_tables(
                    catalog_name=self.cli.cli_context.current_catalog,
                    database_name=self.cli.cli_context.current_db_name,
                    schema_name=self.cli.cli_context.current_schema,
                )
                self.cli.last_result = table_names

                # Display list of tables
                self.cli.console.print(f"[{TABLE_HEADER_STYLE}]Available tables:[/]")
                # Display table list
                for idx, table_name in enumerate(table_names):
                    self.cli.console.print(f"{idx + 1}. {table_name}")

                self.cli.console.print("\n[dim]Use /schemas [table_name] to view detailed schema.[/]")

        except Exception as e:
            logger.error(f"Schema listing error: {str(e)}")
            print_error(self.cli.console, str(e))
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
            print_error(self.cli.console, "Table name required")
            return

        if not self.cli.db_connector:
            print_error(self.cli.console, "No database connection.")
            return

        try:
            # For SQLite, query the sqlite_master table
            if self.cli.db_connector.get_type() == DBType.SQLITE:
                sql = f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'"
                result = self.cli.db_connector.execute_pandas(sql)

                if result is None or not result.success:
                    print_error(self.cli.console, "Query failed")
                    return

                indexes = result.sql_return.replace({np.nan: None}).to_dict(orient="records")
                index_table = build_row_table(
                    indexes,
                    title=f"Indexes for {table_name}",
                    columns=[("name", "Index Name")],
                )
                if index_table is not None:
                    self.cli.console.print(index_table)
                else:
                    print_warning(self.cli.console, f"Table {table_name} has no indexes")
            else:
                # For other database types, use information schema or equivalent
                # This is a placeholder for future database type support
                print_warning(
                    self.cli.console,
                    f"Index listing not yet supported for {self.cli.db_connector.get_type()}",
                )

        except Exception as e:
            logger.error(f"Index listing error: {str(e)}")
            print_error(self.cli.console, str(e))
