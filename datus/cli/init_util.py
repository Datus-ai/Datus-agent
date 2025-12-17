#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from pathlib import Path

from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)
console = Console()


def detect_db_connectivity(namespace_name, db_config_data) -> tuple[bool, str]:
    """Test database connectivity."""
    try:
        # Test database connectivity using connector's built-in method
        from datus.configuration.agent_config import DbConfig
        from datus.tools.db_tools.db_manager import DBManager

        # Get database configuration
        db_type = db_config_data["type"]

        # Create DbConfig object with appropriate fields based on database type
        if db_type in ["starrocks", "mysql"]:
            # For host-based connectors (StarRocks/MySQL)
            port = db_config_data.get("port")
            if not port:
                default_ports = {"starrocks": 9030, "mysql": 3306}
                port = default_ports.get(db_type, 0)

            config_params = {
                "type": db_type,
                "host": db_config_data.get("host", ""),
                "port": int(port) if port else 0,
                "username": db_config_data.get("username", ""),
                "password": db_config_data.get("password", ""),
                "database": db_config_data.get("database", ""),
            }

            # Add catalog field only for StarRocks
            if db_type == "starrocks":
                config_params["catalog"] = db_config_data.get("catalog", "default_catalog")

            db_config = DbConfig(**config_params)

        elif db_type == "snowflake":
            # For Snowflake connector
            db_config = DbConfig(
                type=db_type,
                account=db_config_data.get("account", ""),
                username=db_config_data.get("username", ""),
                password=db_config_data.get("password", ""),
                warehouse=db_config_data.get("warehouse", ""),
                database=db_config_data.get("database", ""),
                schema=db_config_data.get("schema", ""),
            )
        else:
            # For URI-based connectors (sqlite, duckdb, postgresql)
            uri = db_config_data.get("uri", "")

            # Handle ~ expansion and extract file path
            db_path = None
            if uri.startswith(f"{db_type}:///"):
                db_path = uri[len(db_type) + 4 :]  # Remove 'dbtype:///' prefix
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
                database=db_config_data.get("name", namespace_name),
            )

        # Create DB manager with minimal config
        namespaces = {namespace_name: {namespace_name: db_config}}
        db_manager = DBManager(namespaces)

        # Get connector and test connection using built-in method
        connector = db_manager.get_conn(namespace_name, namespace_name)
        test_result = connector.test_connection()

        # Handle different return types from different connectors
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
