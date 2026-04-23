#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Manage command for Datasource.

This module provides an interactive CLI for setting up the datasource configuration
without requiring users to manually write conf/agent.yml files.
"""

from getpass import getpass
from urllib.parse import urlsplit, urlunsplit

from rich.console import Console
from rich.prompt import Confirm, Prompt

from datus.cli.init_util import detect_db_connectivity
from datus.configuration.agent_config import DbConfig, file_stem_from_uri
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)
console = Console()


def _redact_uri(uri: str) -> str:
    """Redact any password in a database URI, keeping scheme/host/path visible."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return uri
    if parts.password is None:
        return uri
    username = parts.username or ""
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{username}:***@{host}{port}" if username else f"***@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _validate_datasource_name(name: str) -> tuple[bool, str]:
    """Validate datasource name format."""
    if not name.strip():
        return False, "Datasource name cannot be empty"
    # Check for invalid characters that could cause issues in YAML/paths
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " ", "\t", "\n"]
    for char in invalid_chars:
        if char in name:
            return False, f"Datasource name cannot contain '{char}'"
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


def serialize_services_section(services_config) -> dict:
    """Convert a ServicesConfig into a dict suitable for ConfigurationManager.update().

    Shared by :class:`DatasourceManager` and :class:`DatasourceCommands`.
    """
    datasources_section = {}
    for db_name, db_config in services_config.datasources.items():
        if db_config.type in (DBType.SQLITE, DBType.DUCKDB):
            entry: dict = {"type": db_config.type}
            if db_config.path_pattern:
                entry["path_pattern"] = db_config.path_pattern
            elif db_config.uri:
                entry["uri"] = db_config.uri
            if db_config.logic_name and db_config.logic_name != db_name:
                entry["name"] = db_config.logic_name
        else:
            entry = {
                k: v
                for k, v in db_config.to_dict().items()
                if v and k not in ("logic_name", "path_pattern", "extra", "default")
            }
            if isinstance(db_config.extra, dict):
                for extra_key, extra_value in db_config.extra.items():
                    if extra_value in (None, ""):
                        continue
                    entry.setdefault(extra_key, extra_value)

        if db_config.default:
            entry["default"] = True

        datasources_section[db_name] = entry

    return {
        "datasources": datasources_section,
        "semantic_layer": dict(services_config.semantic_layer),
        "bi_platforms": dict(services_config.bi_platforms),
        "schedulers": dict(services_config.schedulers),
    }


class DatasourceManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        try:
            # Load without action-specific default-db enforcement because
            # datasource CRUD should work even when no default DB is selected.
            self.agent_config = load_agent_config(config=config_path, reload=True)
        except DatusException as e:
            if e.code == ErrorCode.COMMON_FILE_NOT_FOUND:
                console.print("\u274c Configuration file not found.")
                console.print("Please run 'datus-agent init' first to create the configuration.")
                console.print("Or specify a config file with --config <path>")
            else:
                console.print(f"\u274c {e.message}")
            self.agent_config = None
        except Exception as e:
            console.print(f"\u274c Failed to load configuration: {e}")
            self.agent_config = None

    def run(self, command: str) -> int:
        """Run the specified datasource command."""
        if self.agent_config is None:
            return 1

        if command == "list":
            return self.list()
        elif command == "add":
            return self.add()
        elif command == "delete":
            return self.delete()
        else:
            console.print(f"\u274c Unknown command: {command}")
            return 1

    def list(self) -> int:
        datasources = self.agent_config.services.datasources
        if not datasources:
            console.print("No datasource configured.")
            return 0

        console.print("[bold yellow]Configured datasources:[/bold yellow]")
        for datasource_name, db_config in datasources.items():
            console.print(f"\nDatasource: {datasource_name}")
            console.print(f"    Type: {db_config.type}")
            if db_config.host:
                console.print(f"    Host: {db_config.host}:{db_config.port}")
            if db_config.uri:
                console.print(f"    URI: {_redact_uri(db_config.uri)}")
            if db_config.database:
                console.print(f"    Database: {db_config.database}")
            if db_config.schema:
                console.print(f"    Schema: {db_config.schema}")
            if db_config.account:
                console.print(f"    Account: {db_config.account}")
            if db_config.warehouse:
                console.print(f"    Warehouse: {db_config.warehouse}")
            if db_config.catalog:
                console.print(f"    Catalog: {db_config.catalog}")
            if db_config.username:
                console.print(f"    Username: {db_config.username}")
            console.print()
        return 0

    def add(self) -> int:
        """Interactive method to add a new datasource configuration."""
        console.print("[bold yellow]Add New Datasource[/bold yellow]")

        # Datasource name
        datasource_name = Prompt.ask("- Datasource name")
        valid, error_msg = _validate_datasource_name(datasource_name)
        if not valid:
            console.print(f"\u274c {error_msg}")
            return 1

        # Check if datasource already exists
        if datasource_name in self.agent_config.services.datasources:
            console.print(f"\u274c Datasource '{datasource_name}' already exists")
            return 1

        # Get available adapters dynamically
        from datus.tools.db_tools import connector_registry

        available_adapters = connector_registry.list_available_adapters()
        if not available_adapters:
            console.print("\u274c No database adapters available. Please install at least one adapter.")
            return 1

        # Database type selection
        db_types = sorted(available_adapters.keys())
        default_type = "duckdb" if "duckdb" in db_types else db_types[0]
        db_type = Prompt.ask("- Database type", choices=db_types, default=default_type)

        # Get adapter metadata
        adapter_metadata = available_adapters[db_type]
        config_fields = adapter_metadata.get_config_fields()

        # Initialize config data
        config_data = {"type": db_type}
        logical_name = datasource_name  # Default logical name

        # If adapter provides config schema, use it to prompt for fields
        if not config_fields:
            console.print(f"\u274c Adapter '{db_type}' does not have a configuration schema registered.")
            return 1

        for field_name, field_info in config_fields.items():
            # Skip type field
            if field_name == "type":
                continue

            # Determine prompt label and default value
            label = f"- {field_name.replace('_', ' ').capitalize()}"
            required = field_info.get("required", False)
            default_value = field_info.get("default")
            input_type = field_info.get("input_type", "text")

            # Handle input based on input_type metadata
            if input_type == "password" or field_name == "password":
                value = getpass(f"{label}: ")
            elif input_type == "file_path":
                # Handle file path inputs
                sample_file = field_info.get("default_sample")
                if sample_file:
                    default_path = str(get_path_manager(agent_config=self.agent_config).sample_dir / sample_file)
                    value = Prompt.ask(label, default=default_path)
                else:
                    value = Prompt.ask(label, default=str(default_value) if default_value else "")
            elif field_info.get("type") == "int" or field_name == "port":
                # Handle integer inputs
                value_str = Prompt.ask(label, default=str(default_value) if default_value else "")
                if value_str:
                    if field_name == "port":
                        valid, error_msg = _validate_port(value_str)
                        if not valid:
                            console.print(f"\u274c {error_msg}")
                            return 1
                    try:
                        value = int(value_str)
                    except ValueError:
                        console.print(f"\u274c Invalid integer value: '{value_str}'. Please enter a valid number.")
                        return 1
                else:
                    value = default_value
            elif not required and default_value is not None:
                value = Prompt.ask(label, default=str(default_value))
            elif not required:
                value = Prompt.ask(label, default="")
            else:
                value = Prompt.ask(label)

            # Only add non-empty values
            if value != "" and value is not None:
                config_data[field_name] = value

                # Determine logical name from database or uri
                if field_name == "database" and value:
                    logical_name = value
                elif field_name == "uri" and value:
                    logical_name = file_stem_from_uri(value)

        # Add logical name to config
        config_data["name"] = logical_name

        # Test database connectivity
        console.print("\u2192 Testing database connectivity...")

        success, error_msg = detect_db_connectivity(datasource_name, config_data)

        if success:
            console.print("\u2714 Database connection test successful\n")

            db_config = DbConfig.filter_kwargs(DbConfig, config_data)
            self.agent_config.services.datasources[datasource_name] = db_config

            # Save configuration
            if self._save_configuration():
                console.print(f"\u2714 Datasource '{datasource_name}' added successfully")
                return 0
            else:
                console.print("\u274c Failed to save configuration")
                return 1
        else:
            console.print(f"\u274c Database connectivity test failed: {error_msg}\n")
            return 1

    def delete(self) -> int:
        """Interactive method to delete a datasource configuration."""
        console.print("[bold yellow]Delete Datasource[/bold yellow]")

        # Check if there are any datasources to delete
        datasources = self.agent_config.services.datasources
        if not datasources:
            console.print("\u274c No datasources configured to delete")
            return 1

        # List available datasources
        console.print("Available datasources:")
        for datasource_name in datasources.keys():
            console.print(f"  - {datasource_name}")

        # Get datasource name to delete
        datasource_name = Prompt.ask("- Datasource name to delete")
        if not datasource_name.strip():
            console.print("\u274c Datasource name cannot be empty")
            return 1

        # Check if datasource exists
        if datasource_name not in datasources:
            console.print(f"\u274c Datasource '{datasource_name}' does not exist")
            return 1

        # Confirm deletion
        confirm = Confirm.ask(
            f"Are you sure you want to delete datasource '{datasource_name}'? This action cannot be undone.",
            default=False,
        )
        if not confirm:
            console.print("\u274c Datasource deletion cancelled")
            return 1

        # Delete datasource from configuration
        del self.agent_config.services.datasources[datasource_name]

        # Save configuration
        if self._save_configuration():
            console.print(f"\u2714 Datasource '{datasource_name}' deleted successfully")
            return 0
        else:
            console.print("\u274c Failed to save configuration after deletion")
            return 1

    def _save_configuration(self) -> bool:
        """Save configuration to agent.yml file."""
        try:
            configure_manager = configuration_manager(config_path=self.config_path, reload=True)
            services_section = serialize_services_section(self.agent_config.services)
            configure_manager.update(updates={"services": services_section}, delete_old_key=True)
            console.print(f"Configuration saved to {configure_manager.config_path}")
            return True
        except Exception as e:
            console.print(f"\u274c Failed to save configuration: {e}")
            logger.error(f"Failed to save configuration: {e}")
            return False
