#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Manage command for Namespace.

This module provides an interactive CLI for setting up the namespace configuration
without requiring users to manually write conf/agent.yml files.
"""

import os
from getpass import getpass
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from datus.cli.init_util import detect_db_connectivity
from datus.configuration.agent_config import AgentConfig, DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
console = Console()


class NamespaceManager:
    @staticmethod
    def list(agent_config: AgentConfig):
        if not agent_config.namespaces:
            print("No namespace configured.")
            return 0

        console.print("[bold yellow]Configured namespaces:[/bold yellow]")
        for namespace_name, db_configs in agent_config.namespaces.items():
            print(f"\nNamespace: {namespace_name}")
            for db_name, db_config in db_configs.items():
                print(f"  Database: {db_name}")
                print(f"    Type: {db_config.type}")
                if db_config.host:
                    print(f"    Host: {db_config.host}:{db_config.port}")
                if db_config.uri:
                    print(f"    URI: {db_config.uri}")
                if db_config.database:
                    print(f"    Database: {db_config.database}")
                if db_config.schema:
                    print(f"    Schema: {db_config.schema}")
                if db_config.account:
                    print(f"    Account: {db_config.account}")
                if db_config.warehouse:
                    print(f"    Warehouse: {db_config.warehouse}")
                if db_config.catalog:
                    print(f"    Catalog: {db_config.catalog}")
                if db_config.username:
                    print(f"    Username: {db_config.username}")
                print()
        return 0

    @staticmethod
    def add(agent_config: AgentConfig):
        """Interactive method to add a new namespace configuration."""
        console.print("[bold yellow]Add New Namespace[/bold yellow]")

        # Namespace name
        namespace_name = Prompt.ask("- Namespace name")
        if not namespace_name.strip():
            console.print("❌ Namespace name cannot be empty")
            return False

        # Check if namespace already exists
        if namespace_name in agent_config.namespaces:
            console.print(f"❌ Namespace '{namespace_name}' already exists")
            return False

        # Database type selection
        db_types = ["sqlite", "duckdb", "snowflake", "mysql", "postgresql", "starrocks"]
        db_type = Prompt.ask("- Database type", choices=db_types, default="duckdb")

        # Connection configuration based on database type
        if db_type in ["starrocks", "mysql"]:
            # Host-based database configuration (StarRocks/MySQL)
            host = Prompt.ask("- Host", default="127.0.0.1")
            port = Prompt.ask("- Port", default="9030" if db_type == "starrocks" else "3306")
            username = Prompt.ask("- Username")
            password = getpass("- Password: ")
            database = Prompt.ask("- Database")

            # Store configuration
            config_data = {
                "type": db_type,
                "name": namespace_name,
                "host": host,
                "port": int(port),
                "username": username,
                "password": password,
                "database": database,
            }

            # Add StarRocks-specific catalog field
            if db_type == "starrocks":
                config_data["catalog"] = "default_catalog"

        elif db_type == "snowflake":
            # Snowflake specific configuration
            username = Prompt.ask("- Username")
            account = Prompt.ask("- Account")
            warehouse = Prompt.ask("- Warehouse")
            password = getpass("- Password: ")
            database = Prompt.ask("- Database", default="")
            schema = Prompt.ask("- Schema", default="")

            # Store Snowflake-specific configuration
            config_data = {
                "type": db_type,
                "name": namespace_name,
                "account": account,
                "username": username,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
            }
        else:
            # For other database types (sqlite, duckdb, postgresql), use connection string
            if db_type == "duckdb":
                default_conn_string = str(Path.home() / ".datus" / "sample" / "duckdb-demo.duckdb")
                conn_string = Prompt.ask("- Connection string", default=default_conn_string)
            else:
                conn_string = Prompt.ask("- Connection string")

            config_data = {
                "type": db_type,
                "name": namespace_name,
                "uri": conn_string,
            }

        # Test database connectivity
        console.print("→ Testing database connectivity...")

        success, error_msg = detect_db_connectivity(namespace_name, config_data)

        if success:
            console.print("✔ Database connection test successful\n")

            # Add to agent configuration
            if namespace_name not in agent_config.namespaces:
                agent_config.namespaces[namespace_name] = {}

            # Create DbConfig object and add to namespace
            from datus.configuration.agent_config import DbConfig

            db_config = DbConfig.filter_kwargs(DbConfig, config_data)
            agent_config.namespaces[namespace_name][config_data["name"]] = db_config

            # Save configuration
            if NamespaceManager._save_configuration(agent_config):
                console.print(f"✔ Namespace '{namespace_name}' added successfully")
                return True
            else:
                console.print("❌ Failed to save configuration")
                return False
        else:
            console.print(f"❌ Database connectivity test failed: {error_msg}\n")
            return False

    @staticmethod
    def delete(agent_config: AgentConfig):
        """Interactive method to delete a namespace configuration."""
        console.print("[bold yellow]Delete Namespace[/bold yellow]")

        # Check if there are any namespaces to delete
        if not agent_config.namespaces:
            console.print("❌ No namespaces configured to delete")
            return False

        # List available namespaces
        console.print("Available namespaces:")
        for namespace_name in agent_config.namespaces.keys():
            console.print(f"  - {namespace_name}")

        # Get namespace name to delete
        namespace_name = Prompt.ask("- Namespace name to delete")
        if not namespace_name.strip():
            console.print("❌ Namespace name cannot be empty")
            return False

        # Check if namespace exists
        if namespace_name not in agent_config.namespaces:
            console.print(f"❌ Namespace '{namespace_name}' does not exist")
            return False

        # Confirm deletion
        confirm = Confirm.ask(
            f"Are you sure you want to delete namespace '{namespace_name}'? This action cannot be undone."
        )
        if not confirm:
            console.print("❌ Namespace deletion cancelled")
            return False

        # Delete namespace from configuration
        del agent_config.namespaces[namespace_name]

        # Save configuration
        if NamespaceManager._save_configuration(agent_config):
            console.print(f"✔ Namespace '{namespace_name}' deleted successfully")
            return True
        else:
            console.print("❌ Failed to save configuration after deletion")
            return False

    @staticmethod
    def _save_configuration(agent_config: AgentConfig) -> bool:
        """Save configuration to agent.yml file."""
        try:
            from datus.configuration.agent_config_loader import configuration_manager

            configure_manager = configuration_manager()
            namespace_section = {}

            for ns_name, db_configs in agent_config.namespaces.items():
                namespace_dict = {}
                db_configs_list = list(db_configs.values())

                if len(db_configs_list) == 1:
                    db_config = db_configs_list[0]
                    if db_config.type in (DBType.SQLITE, DBType.DUCKDB):
                        if not db_config.uri.startswith(db_config.type):
                            uri = f"{db_config.type}:///{os.path.expanduser(db_config.uri)}"
                            namespace_dict["uri"] = uri
                        else:
                            namespace_dict["uri"] = db_config.uri
                        namespace_dict["type"] = db_config.type
                        namespace_dict["name"] = db_config.logic_name
                    else:
                        namespace_dict = db_config.to_dict()
                else:
                    namespace_dict["type"] = db_configs_list[0].type
                    namespace_dict["dbs"] = []
                    for db_config in db_configs_list:
                        _db_config = {}
                        _db_config["name"] = db_config.logic_name
                        if not db_config.uri.startswith(db_config.type):
                            uri = f"{db_config.type}:///{os.path.expanduser(db_config.uri)}"
                            _db_config["uri"] = uri
                        else:
                            _db_config["uri"] = db_config.uri
                        namespace_dict["dbs"].append(_db_config)

                namespace_section[ns_name] = namespace_dict

            configure_manager.update(updates={"namespace": namespace_section}, delete_old_key=True)
            console.print(f"Configuration saved to {configure_manager.config_path}")
            return True
        except Exception as e:
            console.print(f"❌ Failed to save configuration: {e}")
            logger.error(f"Failed to save configuration: {e}")
            return False
