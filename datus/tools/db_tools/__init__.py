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
    # SQLite (0 dependencies, no namespace support)
    try:
        from .builtin_configs import SQLiteConfig
        from .sqlite_connector import SQLiteConnector

        connector_registry.register(
            "sqlite",
            SQLiteConnector,
            config_class=SQLiteConfig,
            display_name="SQLite",
            capabilities=set(),
        )
    except ImportError:
        pass

    # DuckDB (small dependency, database + schema)
    try:
        from .builtin_configs import DuckDBConfig
        from .duckdb_connector import DuckdbConnector

        connector_registry.register(
            "duckdb",
            DuckdbConnector,
            config_class=DuckDBConfig,
            display_name="DuckDB",
            capabilities={"database", "schema"},
        )
        # Add to __all__ dynamically
        if "DuckdbConnector" not in __all__:
            __all__.append("DuckdbConnector")
        globals()["DuckdbConnector"] = DuckdbConnector
    except ImportError:
        pass


def _register_builtin_handlers():
    """Register URI builders and context resolvers for dialects without adapter packages."""
    from .builtin_handlers import (
        build_bigquery_uri,
        build_mssql_uri,
        build_oracle_uri,
        resolve_bigquery_context,
        resolve_mssql_context,
        resolve_oracle_context,
    )

    connector_registry.register_handlers(
        "bigquery",
        capabilities={"catalog", "database", "schema"},
        uri_builder=build_bigquery_uri,
        context_resolver=resolve_bigquery_context,
    )
    connector_registry.register_handlers(
        "mssql",
        capabilities={"database", "schema"},
        uri_builder=build_mssql_uri,
        context_resolver=resolve_mssql_context,
    )
    connector_registry.register_handlers(
        "oracle",
        capabilities={"database", "schema"},
        uri_builder=build_oracle_uri,
        context_resolver=resolve_oracle_context,
    )


# Initialize built-in connectors, handlers, and discover plugins
_register_builtin_connectors()
_register_builtin_handlers()
connector_registry.discover_adapters()
