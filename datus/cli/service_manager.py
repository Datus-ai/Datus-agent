#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Manage command for Services (databases, semantic layer, BI tools, schedulers).

Replaces the legacy NamespaceManager. Works with the new services.databases
config structure where each database is an independent entry.
"""

from getpass import getpass

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from datus.cli.init_util import detect_db_connectivity
from datus.configuration.agent_config import DbConfig
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)
console = Console()


def _validate_db_name(name: str) -> tuple[bool, str]:
    """Validate database entry name format."""
    if not name.strip():
        return False, "Database name cannot be empty"
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " ", "\t", "\n"]
    for char in invalid_chars:
        if char in name:
            return False, f"Database name cannot contain '{char}'"
    return True, ""


def _validate_port(port_str: str) -> tuple[bool, str]:
    """Validate port number."""
    try:
        port = int(port_str)
        if port < 1 or port > 65535:
            return False, "Port must be between 1 and 65535"
        return True, ""
    except ValueError:
        return False, "Port must be a valid number"


class ServiceManager:
    """Manage services (databases, semantic layer, BI tools, schedulers) in agent.yml."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        try:
            self.agent_config = load_agent_config(config=config_path, action="service", reload=True)
        except DatusException as e:
            if e.code == ErrorCode.COMMON_FILE_NOT_FOUND:
                console.print("Configuration file not found.")
                console.print("Please run 'datus configure' first to create the configuration.")
                console.print("Or specify a config file with --config <path>")
            else:
                console.print(f"{e.message}")
            self.agent_config = None
        except Exception as e:
            console.print(f"Failed to load configuration: {e}")
            self.agent_config = None

    def run(self, command: str) -> int:
        if self.agent_config is None:
            return 1

        if command == "list":
            return self.list()
        elif command == "add":
            return self.add()
        elif command == "delete":
            return self.delete()
        else:
            console.print(f"Unknown command: {command}")
            return 1

    def list(self) -> int:
        databases = self.agent_config.services.databases
        if not databases:
            console.print("No databases configured.")
            return 0

        table = Table(title="Configured Databases", show_header=True, header_style="bold green")
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Connection")
        table.add_column("Default")

        default_db = self.agent_config.services.default_database
        for db_name, db_config in databases.items():
            connection = ""
            if db_config.uri:
                connection = db_config.uri
            elif db_config.host:
                connection = f"{db_config.host}:{db_config.port}"
            elif db_config.account:
                connection = f"account={db_config.account}"

            is_default = "*" if db_name == default_db else ""
            table.add_row(db_name, db_config.type, connection, is_default)

        console.print(table)

        semantic_layer = self.agent_config.services.semantic_layer
        if semantic_layer:
            console.print("\n[bold yellow]Semantic Layer:[/bold yellow]")
            for name, cfg in semantic_layer.items():
                console.print(f"  {name}: {cfg}")

        bi_tools = self.agent_config.services.bi_tools
        if bi_tools:
            console.print("\n[bold yellow]BI Tools:[/bold yellow]")
            for name, cfg in bi_tools.items():
                console.print(f"  {name}: {cfg}")

        schedulers = self.agent_config.services.schedulers
        if schedulers:
            console.print("\n[bold yellow]Schedulers:[/bold yellow]")
            for name, cfg in schedulers.items():
                console.print(f"  {name}: {cfg}")

        return 0

    def add(self) -> int:
        """Interactive method to add a new database configuration."""
        console.print("[bold yellow]Add New Database[/bold yellow]")

        db_name = Prompt.ask("- Database name")
        valid, error_msg = _validate_db_name(db_name)
        if not valid:
            console.print(f"{error_msg}")
            return 1

        if db_name in self.agent_config.services.databases:
            console.print(f"Database '{db_name}' already exists")
            return 1

        from datus.tools.db_tools import connector_registry

        available_adapters = connector_registry.list_available_adapters()
        if not available_adapters:
            console.print("No database adapters available.")
            return 1

        db_types = sorted(available_adapters.keys())
        default_type = "duckdb" if "duckdb" in db_types else db_types[0]
        db_type = Prompt.ask("- Database type", choices=db_types, default=default_type)

        adapter_metadata = available_adapters[db_type]
        config_fields = adapter_metadata.get_config_fields()

        config_data = {"type": db_type}

        if not config_fields:
            console.print(f"Adapter '{db_type}' does not have a configuration schema registered.")
            return 1

        for field_name, field_info in config_fields.items():
            if field_name == "type":
                continue

            label = f"- {field_name.replace('_', ' ').capitalize()}"
            required = field_info.get("required", False)
            default_value = field_info.get("default")
            input_type = field_info.get("input_type", "text")

            if input_type == "password" or field_name == "password":
                value = getpass(f"{label}: ")
            elif input_type == "file_path":
                sample_file = field_info.get("default_sample")
                if sample_file:
                    default_path = str(get_path_manager().sample_dir / sample_file)
                    value = Prompt.ask(label, default=default_path)
                else:
                    value = Prompt.ask(label, default=str(default_value) if default_value else "")
            elif field_info.get("type") == "int" or field_name == "port":
                value_str = Prompt.ask(label, default=str(default_value) if default_value else "")
                if value_str:
                    if field_name == "port":
                        valid, error_msg = _validate_port(value_str)
                        if not valid:
                            console.print(f"{error_msg}")
                            return 1
                    try:
                        value = int(value_str)
                    except ValueError:
                        console.print(f"Invalid integer value: '{value_str}'")
                        return 1
                else:
                    value = default_value
            elif not required and default_value is not None:
                value = Prompt.ask(label, default=str(default_value))
            elif not required:
                value = Prompt.ask(label, default="")
            else:
                value = Prompt.ask(label)

            if value != "" and value is not None:
                config_data[field_name] = value

        # Ask if this should be the default
        if not self.agent_config.services.databases:
            config_data["default"] = True
        elif Confirm.ask("- Set as default database?", default=False):
            config_data["default"] = True

        # Test connectivity
        console.print("Testing database connectivity...")
        success, error_msg = detect_db_connectivity(db_name, config_data)

        if success:
            console.print("Database connection test successful\n")

            db_config = DbConfig.filter_kwargs(DbConfig, config_data)
            db_config.logic_name = db_name
            db_config.default = config_data.get("default", False)
            self.agent_config.services.databases[db_name] = db_config

            if self._save_configuration():
                console.print(f"Database '{db_name}' added successfully")
                return 0
            else:
                console.print("Failed to save configuration")
                return 1
        else:
            console.print(f"Database connectivity test failed: {error_msg}\n")
            return 1

    def delete(self) -> int:
        """Interactive method to delete a database configuration."""
        console.print("[bold yellow]Delete Database[/bold yellow]")

        databases = self.agent_config.services.databases
        if not databases:
            console.print("No databases configured to delete")
            return 1

        console.print("Available databases:")
        for name in databases:
            console.print(f"  - {name}")

        db_name = Prompt.ask("- Database name to delete")
        if not db_name.strip():
            console.print("Database name cannot be empty")
            return 1

        if db_name not in databases:
            console.print(f"Database '{db_name}' does not exist")
            return 1

        confirm = Confirm.ask(
            f"Are you sure you want to delete database '{db_name}'? This action cannot be undone.",
            default=False,
        )
        if not confirm:
            console.print("Deletion cancelled")
            return 1

        del self.agent_config.services.databases[db_name]

        if self._save_configuration():
            console.print(f"Database '{db_name}' deleted successfully")
            return 0
        else:
            console.print("Failed to save configuration after deletion")
            return 1

    def _save_configuration(self) -> bool:
        """Save services configuration to the agent.yml file."""
        try:
            configure_manager = configuration_manager(config_path=self.config_path, reload=True)
            databases_section = {}

            for db_name, db_config in self.agent_config.services.databases.items():
                if db_config.type in (DBType.SQLITE, DBType.DUCKDB):
                    entry = {
                        "type": db_config.type,
                        "uri": db_config.uri,
                    }
                    if db_config.logic_name and db_config.logic_name != db_name:
                        entry["name"] = db_config.logic_name
                else:
                    entry = {k: v for k, v in db_config.to_dict().items() if v}
                    # Remove internal fields
                    for key in ("logic_name", "path_pattern", "extra", "default"):
                        entry.pop(key, None)

                if db_config.default:
                    entry["default"] = True

                databases_section[db_name] = entry

            services_section = {
                "databases": databases_section,
                "semantic_layer": dict(self.agent_config.services.semantic_layer),
                "bi_tools": dict(self.agent_config.services.bi_tools),
                "schedulers": dict(self.agent_config.services.schedulers),
            }

            configure_manager.update(updates={"services": services_section}, delete_old_key=True)
            # Remove legacy namespace key if it exists
            if "namespace" in configure_manager.data:
                del configure_manager.data["namespace"]
                configure_manager.save()

            console.print(f"Configuration saved to {configure_manager.config_path}")
            return True
        except Exception as e:
            console.print(f"Failed to save configuration: {e}")
            logger.error(f"Failed to save configuration: {e}")
            return False
