# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import dataclasses
from typing import Callable, Dict, Optional, Tuple, Union

from datus_db_core import BaseSqlConnector, ConnectionConfig, DatusDbException, connector_registry
from sqlalchemy.engine.url import URL, make_url

from datus.configuration.agent_config import DbConfig
from datus.tools.db_tools.config import DuckDBConfig, SQLiteConfig
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_utils import get_files_from_glob_pattern

logger = get_logger(__name__)


def _auto_install_adapter(db_type: str) -> None:
    """Attempt to pip-install the adapter package for *db_type* and register it."""
    import importlib
    import shutil
    import subprocess
    import sys

    package = f"datus-{db_type}"
    logger.info("Adapter '%s' not found, attempting auto-install: %s", db_type, package)

    python = sys.executable
    uv_path = shutil.which("uv")
    cmd = (
        [uv_path, "pip", "install", "--python", python, package]
        if uv_path
        else [python, "-m", "pip", "install", package]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Auto-install of %s failed: %s", package, result.stderr.strip())
            return

        importlib.invalidate_caches()
        module_name = f"datus_{db_type}"
        module = importlib.import_module(module_name)
        if hasattr(module, "register"):
            module.register()
        logger.info("Auto-installed and loaded adapter: %s", db_type)
    except subprocess.TimeoutExpired:
        logger.warning("Auto-install of %s timed out", package)
    except Exception as e:
        logger.warning("Auto-install of %s failed: %s", package, e)


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
    connections: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]], name: str = ""
) -> BaseSqlConnector:
    if isinstance(connections, BaseSqlConnector):
        return connections
    if len(connections) == 1:
        return next(iter(connections.values()))

    if not name:
        return list(connections.values())[0]
    if name not in connections:
        raise DatusException(
            code=ErrorCode.DB_CONNECTION_FAILED,
            message_args={
                "error_message": f"Database {name} not found in current datasource",
            },
        )
    return connections[name]


class DBManager:
    def __init__(self, db_configs: Dict[str, DbConfig]):
        # {datasource: {database: connector}} — a datasource may serve multiple databases.
        self._conn_dict: Dict[str, Dict[str, BaseSqlConnector]] = {}
        self._db_configs: Dict[str, DbConfig] = db_configs

    def get_conn(self, datasource: str, database: str = "") -> BaseSqlConnector:
        """Connector for ``(datasource, database)``.

        ``database`` empty → the datasource's default database. For a glob (``path_pattern``)
        datasource each matched file is a database; for server DBs ``database`` overrides the
        configured one (cloned into the connection URI).
        """
        return self._init_connection(datasource, database)

    def get_connections(self, datasource: str = "") -> Dict[str, BaseSqlConnector]:
        """Connectors for every database served by ``datasource``, keyed by database name.

        A glob (``path_pattern``) datasource yields one connector per matched file; a server
        datasource yields its single configured database. Callers that probe health or list
        databases must iterate the map so multi-database datasources aren't reduced to one.
        """
        return {db_name: self._init_connection(datasource, db_name) for db_name in self.get_db_uris(datasource)}

    def first_conn(self, datasource: str) -> BaseSqlConnector:
        return self._init_connection(datasource, "")

    def first_conn_with_name(self, datasource: str) -> Tuple[str, BaseSqlConnector]:
        return datasource, self._init_connection(datasource, "")

    def get_db_uris(self, datasource: str) -> Dict[str, str]:
        cfg = self._db_configs.get(datasource)
        if cfg is None:
            return {}
        if cfg.path_pattern:
            return {f["name"]: f["uri"] for f in get_files_from_glob_pattern(cfg.path_pattern, cfg.type)}
        return {cfg.database or datasource: cfg.uri}

    def _resolve_db_config(self, datasource: str, database: str) -> Tuple[str, DbConfig]:
        """Resolve (datasource, database) → (db_name, DbConfig bound to that database)."""
        if datasource not in self._db_configs:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR, message=f"Datasource {datasource} not found in config"
            )
        cfg = self._db_configs[datasource]
        if cfg.path_pattern:
            files = get_files_from_glob_pattern(cfg.path_pattern, cfg.type)
            if not files:
                raise DatusException(
                    code=ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"No database files for datasource '{datasource}' (path_pattern: {cfg.path_pattern}).",
                )
            target = database or cfg.database or files[0]["name"]
            match = next((f for f in files if f["name"] == target), None)
            if match is None:
                raise DatusException(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    message=f"Database '{target}' is not in datasource '{datasource}'. "
                    f"Available: {', '.join(f['name'] for f in files)}.",
                )
            return match["name"], dataclasses.replace(cfg, uri=match["uri"], database=match["name"], path_pattern="")
        if database and database != cfg.database:
            return database, dataclasses.replace(cfg, database=database)
        return cfg.database or "", cfg

    def _init_connection(self, datasource: str, database: str) -> BaseSqlConnector:
        db_name, db_config = self._resolve_db_config(datasource, database)
        # Self-heal: a caller may have nulled this datasource's entry to mark its
        # connectors closed (e.g. an external pool on eviction). setdefault keeps
        # an existing None, so rebuild a fresh group rather than calling .get on it.
        group = self._conn_dict.get(datasource)
        if not isinstance(group, dict):
            group = {}
            self._conn_dict[datasource] = group
        cached = group.get(db_name)
        if cached is not None:
            return cached
        conn = self._build_conn(db_config)
        group[db_name] = conn
        return conn

    def _build_conn(self, db_config: DbConfig) -> BaseSqlConnector:
        """Build a connector from a (fully-resolved) DbConfig via the registry."""
        connection_config = self._db_config_to_connection_config(db_config)
        db_type = _normalize_dialect_name(db_config.type)
        if not connector_registry.is_registered(db_type):
            _auto_install_adapter(db_type)
        return connector_registry.create_connector(db_config.type, connection_config)

    def _db_config_to_connection_config(self, db_config: DbConfig) -> Union[ConnectionConfig, dict]:
        """Convert DbConfig to appropriate ConnectionConfig subclass or dict.

        Args:
            db_config: Database configuration from agent config

        Returns:
            ConnectionConfig instance for built-in databases or dict for adapters
        """
        db_type = _normalize_dialect_name(db_config.type)
        timeout_seconds = 30  # Default timeout

        def _bool_extra(value, default: bool = False) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        if db_type == DBType.SQLITE:
            # SQLite uses file path - prioritize uri over database field
            db_path = db_config.uri or db_config.database
            if db_path.startswith("sqlite:///"):
                db_path = db_path.replace("sqlite:///", "")
            extra = db_config.extra or {}
            return SQLiteConfig(
                db_path=db_path,
                timeout_seconds=timeout_seconds,
                database_name=None,  # Let connector extract from file path
                read_only=_bool_extra(extra.get("read_only"), False),
            )

        elif db_type == DBType.DUCKDB:
            # DuckDB uses file path - prioritize uri over database field
            db_path = db_config.uri or db_config.database
            if db_path.startswith("duckdb:///"):
                db_path = db_path.replace("duckdb:///", "")
            extra = db_config.extra or {}
            return DuckDBConfig(
                db_path=db_path,
                timeout_seconds=timeout_seconds,
                database_name=None,  # Let connector extract from file path
                read_only=_bool_extra(extra.get("read_only"), False),
                enable_external_access=_bool_extra(extra.get("enable_external_access"), True),
                memory_limit=extra.get("memory_limit"),
                iceberg=extra.get("iceberg"),
            )

        else:
            # For adapters, convert DbConfig to dict and filter out empty values
            # This allows adapters to receive all configuration parameters they need
            config_dict = db_config.to_dict()

            # Add standard connection parameters
            config_dict["timeout_seconds"] = timeout_seconds

            # Remove None and empty string values, and internal fields
            # Keep False, 0, and empty containers to allow explicit configuration
            excluded_fields = ["type", "path_pattern", "default", "extra"]

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
        for datasource, group in list(self._conn_dict.items()):
            if not isinstance(group, dict):
                continue
            for db_name, conn in list(group.items()):
                if conn is None:
                    continue
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {datasource}.{db_name}: {str(e)}")
                finally:
                    group.pop(db_name, None)

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()


def db_config_name(datasource: str, db_type: str, name: str = "") -> str:
    if db_type == DBType.SQLITE or db_type == DBType.DUCKDB:
        return f"{datasource}::{name}"
    # fix local snowflake
    return f"{datasource}::{datasource}"


# External factory for DBManager creation (used by SaaS backend for connection pooling)
_factory: Optional[Callable[[Dict[str, DbConfig]], DBManager]] = None
# CLI-mode cache: keyed by frozenset of datasource names to avoid creating
# duplicate DBManager instances (and leaking connections) for the same config.
_cli_cache: Dict[frozenset, DBManager] = {}


def set_db_manager_factory(factory: Optional[Callable[[Dict[str, DbConfig]], "DBManager"]] = None) -> None:
    """Set an external factory for DBManager creation.

    When set, ``db_manager_instance()`` delegates to this factory instead of
    creating a new ``DBManager`` directly.  This allows a SaaS backend to inject
    a pooled factory that manages connection lifecycle (reference counting,
    eviction, close_all).

    Pass ``None`` to reset to the default behaviour.

    Args:
        factory: Callable that accepts ``db_configs`` and returns a ``DBManager``.
    """
    global _factory
    _factory = factory
    # Clear CLI cache when switching modes
    _cli_cache.clear()


def db_manager_instance(
    db_configs: Optional[Dict[str, DbConfig]] = None,
) -> DBManager:
    """Create or obtain a DBManager instance.

    - With a factory set (SaaS mode): delegates to the factory every call,
      which typically returns a pooled/ref-counted instance.
    - Without a factory (CLI mode): caches by datasource keys. The per-database
      dimension lives inside DBManager (get_conn(datasource, database)), so the
      manager is shared across databases of the same datasource set.
    """
    if _factory is not None:
        return _factory(db_configs or {})
    configs = db_configs or {}
    cache_key = frozenset(configs.keys())
    cached = _cli_cache.get(cache_key)
    if cached is not None:
        return cached
    manager = DBManager(configs)
    _cli_cache[cache_key] = manager
    return manager
