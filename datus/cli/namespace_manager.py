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

import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

from datus.configuration.agent_config import AgentConfig
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
        success, error_msg = NamespaceManager._test_db_connectivity(config_data)
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
    def _test_db_connectivity(config_data: dict) -> tuple[bool, str]:
        """Test database connectivity."""
        try:
            from datus.configuration.agent_config import DbConfig
            from datus.tools.db_tools.db_manager import DBManager

            db_type = config_data["type"]
            namespace_name = config_data["name"]

            # Create DbConfig object with appropriate fields based on database type
            if db_type in ["starrocks", "mysql", "snowflake"]:
                db_config = DbConfig(
                    type=db_type,
                    host=config_data.get("host", ""),
                    port=config_data.get("port", 0),
                    username=config_data.get("username", ""),
                    password=config_data.get("password", ""),
                    database=config_data.get("database", ""),
                    catalog=config_data.get("catalog", "default_catalog"),
                    account=config_data.get("account", ""),
                    warehouse=config_data.get("warehouse", ""),
                    schema=config_data.get("schema", ""),
                )
            else:
                uri = config_data.get("uri", "")
                db_path = None
                if uri.startswith(f"{db_type}:///"):
                    db_path = uri[len(db_type) + 4 :]
                    db_path = os.path.expanduser(db_path)
                    uri = f"{db_type}:///{db_path}"
                else:
                    uri = os.path.expanduser(uri)
                    db_path = uri

                if db_type == "sqlite" and not Path(db_path).exists():
                    return False, f"SQLite database file does not exist: {db_path}"

                db_config = DbConfig(
                    type=db_type,
                    uri=uri,
                    database=config_data.get("name", namespace_name),
                )

            # Create DB manager and test connection
            namespaces = {namespace_name: {namespace_name: db_config}}
            db_manager = DBManager(namespaces)
            connector = db_manager.get_conn(namespace_name, namespace_name)
            test_result = connector.test_connection()

            if isinstance(test_result, bool):
                return (test_result, "") if test_result else (False, "Connection test failed")
            elif isinstance(test_result, dict):
                success = test_result.get("success", False)
                error_msg = test_result.get("error", "Connection test failed") if not success else ""
                return success, error_msg
            else:
                return False, "Unknown connection test result format"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Database connectivity test failed: {error_msg}")
            return False, error_msg

    @staticmethod
    def _save_configuration(agent_config: AgentConfig) -> bool:
        """Save configuration to agent.yml file."""
        try:
            from datus.utils.path_manager import get_path_manager

            path_manager = get_path_manager()
            conf_dir = path_manager.conf_dir
            conf_dir.mkdir(parents=True, exist_ok=True)

            config_path = conf_dir / "agent.yml"

            if config_path.exists():
                config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            else:
                config_dict = {}

            agent_section = config_dict.setdefault("agent", {})
            agent_section.setdefault("target", agent_config.target)
            agent_section.setdefault("models", {name: model.to_dict() for name, model in agent_config.models.items()})
            agent_section.setdefault(
                "storage", {"embedding_device_type": "cpu", "workspace_root": agent_config.workspace_root}
            )
            agent_section.setdefault(
                "nodes", {name: {"model": node.model} for name, node in agent_config.nodes.items()}
            )

            namespace_section = agent_section.setdefault("namespace", {})
            for ns_name, db_configs in agent_config.namespaces.items():
                namespace_section[ns_name] = {
                    logic_name: db_config.to_dict() for logic_name, db_config in db_configs.items()
                }

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            console.print(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            console.print(f"❌ Failed to save configuration: {e}")
            logger.error(f"Failed to save configuration: {e}")
            return False
