# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from .base import BaseSqlConnector
from .registry import AdapterMetadata, ConnectorRegistry, connector_registry
from .sqlite_connector import SQLiteConnector

__all__ = [
    "BaseSqlConnector",
    "SQLiteConnector",
    "DuckdbConnector",
    "connector_registry",
    "ConnectorRegistry",
    "AdapterMetadata",
]


def _register_builtin_connectors():
    """Register built-in connectors (SQLite and DuckDB only)"""
    # SQLite (0 dependencies)
    try:
        from .builtin_configs import SQLiteConfig
        from .sqlite_connector import SQLiteConnector

        connector_registry.register("sqlite", SQLiteConnector, config_class=SQLiteConfig, display_name="SQLite")
    except ImportError:
        pass

    # DuckDB (small dependency)
    try:
        from .builtin_configs import DuckDBConfig
        from .duckdb_connector import DuckdbConnector

        connector_registry.register("duckdb", DuckdbConnector, config_class=DuckDBConfig, display_name="DuckDB")
        # Add to __all__ dynamically
        if "DuckdbConnector" not in __all__:
            __all__.append("DuckdbConnector")
        globals()["DuckdbConnector"] = DuckdbConnector
    except ImportError:
        pass


# Initialize built-in connectors and discover plugins
_register_builtin_connectors()
connector_registry.discover_adapters()
