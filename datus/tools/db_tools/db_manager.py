# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

from datus_db_core import BaseSqlConnector, ConnectionConfig, DatusDbException, connector_registry
from sqlalchemy.engine.url import URL, make_url

from datus.configuration.agent_config import DbConfig
from datus.tools.db_tools.config import DuckDBConfig, SQLiteConfig
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _normalize_dialect_name(db_type: Union[str, DBType, None]) -> str:
    """
    Normalize dialect names and collapse aliases so downstream checks work reliably.
    """
    if isinstance(db_type, DBType):
        value = db_type.value
    else:
        value = str(db_type or "").strip().lower()
    alias_map = {
        "postgres": "postgresql",
        "sqlserver": "mssql",
    }
    return alias_map.get(value, value)


def _clean_str(value: Optional[Union[str, int]]) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item:
                return str(item).strip()
        return ""
    return str(value).strip()


def _resolve_connection_context(db_config: DbConfig, uri: str) -> Tuple[str, str, str, str]:
    """
    Infer catalog, database, and schema information from a SQLAlchemy URL.
    Returns (dialect, catalog_name, database_name, schema_name).
    """
    normalized_type = _normalize_dialect_name(db_config.type)
    try:
        url = make_url(uri)
    except Exception as exc:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Invalid database uri `{uri}`: {exc}",
        ) from exc

    backend_normalized = _normalize_dialect_name(url.get_backend_name())
    dialect = backend_normalized or normalized_type
    if not dialect:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Unable to determine database type from uri `{uri}`",
        )

    # Delegate to a registered context resolver if available
    resolver = connector_registry.get_context_resolver(dialect)
    if resolver:
        try:
            return resolver(db_config, uri)
        except DatusDbException:
            raise
        except Exception as exc:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Context resolver failed for dialect '{dialect}': {exc}",
            ) from exc

    # Generic fallback
    catalog = _clean_str(db_config.catalog)
    database = _clean_str(url.database) or _clean_str(db_config.database)
    schema = _clean_str(db_config.schema)

    return dialect or "", catalog, database, schema


def gen_uri(db_config: DbConfig) -> str:
    if db_config.uri:
        return db_config.uri

    dialect = _normalize_dialect_name(db_config.type)

    # Delegate to a registered URI builder if available
    builder = connector_registry.get_uri_builder(dialect)
    if builder:
        try:
            return builder(db_config)
        except DatusDbException:
            raise
        except Exception as exc:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message=f"URI builder failed for dialect '{dialect}': {exc}",
            ) from exc

    # Generic fallback
    return str(
        URL.create(
            drivername=dialect,
            username=_value_or_none(db_config.username),
            password=_value_or_none(db_config.password),
            host=_value_or_none(db_config.host),
            port=_port_or_none(db_config.port),
            database=_value_or_none(db_config.database),
        )
    )


def _value_or_none(value: Optional[Union[str, int]]) -> Optional[str]:
    cleaned = _clean_str(value)
    return cleaned or None


def _port_or_none(port_value: Optional[Union[str, int]]) -> Optional[int]:
    cleaned = _clean_str(port_value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def get_connection(
    connections: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]], logic_name: str = ""
) -> BaseSqlConnector:
    if isinstance(connections, BaseSqlConnector):
        return connections
    if len(connections) == 1:
        return next(iter(connections.values()))

    if not logic_name:
        return list(connections.values())[0]
    if logic_name not in connections:
        raise DatusException(
            code=ErrorCode.DB_CONNECTION_FAILED,
            message_args={
                "error_message": f"Database {logic_name} not found in current namespace",
            },
        )
    return connections[logic_name]


class DBManager:
    def __init__(self, db_configs: Dict[str, Dict[str, DbConfig]]):
        self._conn_dict: Dict[str, Union[BaseSqlConnector, Dict[str, BaseSqlConnector]]] = defaultdict(dict)
        self._db_configs: Dict[str, Dict[str, DbConfig]] = db_configs

    def get_conn(self, namespace: str, logic_name: str = "") -> BaseSqlConnector:
        self._init_connections(namespace)
        connector_or_dict = self._conn_dict[namespace]
        return get_connection(connector_or_dict, logic_name)

    def get_connections(self, namespace: str = "") -> Union[BaseSqlConnector, Dict[str, BaseSqlConnector]]:
        self._init_connections(namespace)
        return self._conn_dict[namespace]

    def current_db_configs(self, namespace: str) -> Dict[str, DbConfig]:
        return self._db_configs[namespace]

    def _init_connections(self, namespace):
        if namespace in self._conn_dict:
            return
        if namespace not in self._db_configs:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR, message=f"Namespace {namespace} not found in config"
            )
        configs = self._db_configs[namespace]
        if len(configs) == 1:
            db_config = list(configs.values())[0]
            self._init_conn(namespace, db_config)
            return
        # Multiple database configuration
        for database_name, db_config in configs.items():
            self._init_conn(namespace, db_config, database_name=database_name)

        if namespace not in self._conn_dict:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"Database initialization under namespace {namespace} failed with the current configuration:"
                    f" {configs}"
                ),
            )

    def first_conn(self, namespace: str) -> BaseSqlConnector:
        self._init_connections(namespace)
        dbs: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]] = self._conn_dict[namespace]
        if isinstance(dbs, dict):
            return list(dbs.values())[0]
        return dbs

    def first_conn_with_name(self, namespace: str) -> Tuple[str, BaseSqlConnector]:
        self._init_connections(namespace)
        dbs: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]] = self._conn_dict[namespace]
        if isinstance(dbs, dict):
            name = list(dbs.keys())[0]
            conn = dbs[name]
            return name, conn

        config = list(self._db_configs[namespace].values())[0]
        return config.logic_name, dbs

    def get_db_uris(self, namespace: str) -> Dict[str, str]:
        dbs = self._db_configs.get(namespace, {})
        return {name: db.uri for name, db in dbs.items()}

    def _init_conn(self, namespace: str, db_config: DbConfig, database_name: Optional[str] = None) -> BaseSqlConnector:
        """Initialize connection using the registry

        Args:
            namespace: Namespace identifier
            db_config: Database configuration
            database_name: Optional database name for multi-database setup

        Returns:
            Initialized connector instance
        """
        # Convert DbConfig to ConnectionConfig
        connection_config = self._db_config_to_connection_config(db_config)

        # Use registry to create connector
        conn = connector_registry.create_connector(db_config.type, connection_config)

        # Store connection
        if database_name:
            self._conn_dict[namespace][database_name] = conn
        else:
            self._conn_dict[namespace] = conn

        return conn

    def _db_config_to_connection_config(self, db_config: DbConfig) -> Union[ConnectionConfig, dict]:
        """Convert DbConfig to appropriate ConnectionConfig subclass or dict.

        Args:
            db_config: Database configuration from agent config

        Returns:
            ConnectionConfig instance for built-in databases or dict for adapters
        """
        db_type = _normalize_dialect_name(db_config.type)
        timeout_seconds = 30  # Default timeout

        if db_type == DBType.SQLITE:
            # SQLite uses file path - prioritize uri over database field
            db_path = db_config.uri or db_config.database
            if db_path.startswith("sqlite:///"):
                db_path = db_path.replace("sqlite:///", "")
            return SQLiteConfig(
                db_path=db_path,
                timeout_seconds=timeout_seconds,
                database_name=None,  # Let connector extract from file path
            )

        elif db_type == DBType.DUCKDB:
            # DuckDB uses file path - prioritize uri over database field
            db_path = db_config.uri or db_config.database
            if db_path.startswith("duckdb:///"):
                db_path = db_path.replace("duckdb:///", "")
            return DuckDBConfig(
                db_path=db_path,
                timeout_seconds=timeout_seconds,
                database_name=None,  # Let connector extract from file path
            )

        else:
            # For adapters, convert DbConfig to dict and filter out empty values
            # This allows adapters to receive all configuration parameters they need
            config_dict = db_config.to_dict()

            # Add standard connection parameters
            config_dict["timeout_seconds"] = timeout_seconds

            # Remove None and empty string values, and internal fields
            # Keep False, 0, and empty containers to allow explicit configuration
            excluded_fields = ["type", "path_pattern", "logic_name", "extra"]

            filtered_config = {
                k: v
                for k, v in config_dict.items()
                if not (v is None or (isinstance(v, str) and v.strip() == "")) and k not in excluded_fields
            }

            # Expand extra field to include adapter-specific config
            if db_config.extra:
                filtered_config.update(db_config.extra)

            # Convert port to int if present
            if "port" in filtered_config:
                try:
                    filtered_config["port"] = int(filtered_config["port"])
                except (ValueError, TypeError):
                    pass

            return filtered_config

    def close(self):
        """Close all database connections."""
        for name, conn in list(self._conn_dict.items()):
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {name}: {str(e)}")
                finally:
                    self._conn_dict[name] = None

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()


def db_config_name(namespace: str, db_type: str, name: str = "") -> str:
    if db_type == DBType.SQLITE or db_type == DBType.DUCKDB:
        return f"{namespace}::{name}"
    # fix local snowflake
    return f"{namespace}::{namespace}"


_INSTANCE = None


def db_manager_instance(
    db_configs: Optional[Dict[str, Dict[str, DbConfig]]] = None,
) -> DBManager:
    global _INSTANCE
    if _INSTANCE is None or db_configs is not None:
        _INSTANCE = _db_manager(db_configs)
    return _INSTANCE


def _db_manager(
    db_configs: Optional[Dict[str, Dict[str, DbConfig]]] = None,
) -> DBManager:
    if db_configs is None:
        return DBManager({})
    manager = DBManager(db_configs)
    return manager
