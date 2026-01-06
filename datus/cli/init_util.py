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
    """Test database connectivity.

    Uses DbConfig.filter_kwargs to handle all database types uniformly.
    Adapter-specific fields are stored in the 'extra' field and expanded
    when creating the connector.
    """
    try:
        from datus.configuration.agent_config import DbConfig
        from datus.tools.db_tools.db_manager import DBManager

        db_type = db_config_data.get("type", "")
        if not db_type:
            return False, "Database type is required"

        # Handle ~ expansion for uri field
        config_data = db_config_data.copy()
        uri = config_data.get("uri", "")
        if uri:
            if uri.startswith(f"{db_type}:///"):
                db_path = uri[len(db_type) + 4 :]
                db_path = os.path.expanduser(db_path)
                config_data["uri"] = f"{db_type}:///{db_path}"

                if db_type == "sqlite" and not Path(db_path).exists():
                    return False, f"SQLite database file does not exist: {db_path}"
            else:
                config_data["uri"] = os.path.expanduser(uri)

        # Use filter_kwargs to create DbConfig
        # Unknown fields will be stored in 'extra' and expanded by DBManager
        db_config = DbConfig.filter_kwargs(DbConfig, config_data)

        # Create DB manager with minimal config
        namespaces = {namespace_name: {namespace_name: db_config}}
        db_manager = DBManager(namespaces)

        # Get connector and test connection
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
