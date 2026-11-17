# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import dataclasses
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from enum import StrEnum
from typing import Callable, Dict, Optional, Set, Tuple, Union
from weakref import finalize

from datus_db_core import BaseSqlConnector, ConnectionConfig, DatusDbException, connector_registry
from sqlalchemy.engine.url import URL, make_url

from datus.configuration.agent_config import DbConfig, file_stem_from_uri
from datus.schemas.database_context import DatabaseExecutionContext
from datus.tools.db_tools.config import DuckDBConfig, SQLiteConfig
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_utils import get_files_from_glob_pattern

logger = get_logger(__name__)


# Single-flight guard for adapter auto-installs: concurrent callers (e.g. the
# REPL connection init and the background schema sync) must not spawn duplicate
# ``pip install`` processes racing on the same environment, and a failed install
# must not be retried on every connection attempt within the process.
# Deliberately ONE process-wide lock rather than one per db_type: concurrent
# pip/uv processes mutate the same environment, so installs of *different*
# adapters must serialize too.
_adapter_install_lock = threading.Lock()
_adapter_install_attempted: Set[str] = set()
# db_types whose install subprocess is currently running. Read without the
# lock (the installer holds it for the whole install) — set membership is
# GIL-atomic and UI staleness of one check is harmless.
_adapter_install_active: Set[str] = set()


class AdapterInstallStatus(StrEnum):
    """Install state of an adapter's connector, for UI layers to message accurately.

    Self-contained: consults the connector registry first, so a concurrently
    completed install reports ``REGISTERED`` rather than leaking the
    "attempted" memo as a false failure — callers need no particular read
    ordering relative to ``missing_connector_type``.
    """

    REGISTERED = "registered"  # connector available; any earlier attempt is moot
    NOT_ATTEMPTED = "not_attempted"  # next connection attempt will auto-install
    INSTALLING = "installing"  # an install subprocess is running right now
    FAILED = "failed"  # attempted this process and the connector is still missing


def adapter_install_status(db_type: str) -> AdapterInstallStatus:
    """Return the process-wide auto-install state for *db_type*."""
    if connector_registry.is_registered(db_type):
        return AdapterInstallStatus.REGISTERED
    if db_type in _adapter_install_active:
        return AdapterInstallStatus.INSTALLING
    if db_type in _adapter_install_attempted:
        return AdapterInstallStatus.FAILED
    return AdapterInstallStatus.NOT_ATTEMPTED


def _auto_install_adapter(db_type: str) -> None:
    """Attempt to pip-install the adapter package for *db_type* and register it.

    Serialized process-wide: the first caller runs the install while holding the
    lock (so concurrent callers wait for its outcome instead of racing), and
    each *db_type* is attempted at most once per process.
    """
    with _adapter_install_lock:
        if db_type in _adapter_install_attempted:
            # The memo only guards the expensive pip subprocess. Retry the
            # cheap import + register so a manual ``pip install datus-<type>``
            # after a failed auto-install takes effect without a restart.
            if _try_register_adapter_module(db_type):
                logger.info("Adapter '%s' registered after manual install", db_type)
            else:
                logger.debug("Adapter '%s' auto-install already attempted; skipping", db_type)
            return
        # Active before attempted: lock-free status readers must never see
        # "attempted but not active" for an install that is about to run —
        # that combination means FAILED.
        _adapter_install_active.add(db_type)
        _adapter_install_attempted.add(db_type)
        try:
            _run_adapter_install(db_type)
        finally:
            _adapter_install_active.discard(db_type)


def _try_register_adapter_module(db_type: str) -> bool:
    """Import and register an already-installed ``datus_<type>`` package.

    Cheap (no subprocess): used after a fresh auto-install, and on every
    retry after a failed one so a manual install becomes usable in the
    running process.
    """
    import importlib

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(f"datus_{db_type}")
    except ImportError:
        return False
    except Exception as e:  # noqa: BLE001 - a broken adapter must not crash the caller
        logger.warning("Adapter module datus_%s failed to load: %s", db_type, e)
        return False
    register = getattr(module, "register", None)
    if callable(register):
        try:
            register()
        except Exception as e:  # noqa: BLE001 - a broken adapter must not crash the caller
            logger.warning("Adapter datus_%s register() failed: %s", db_type, e)
            return False
    return True


def _run_adapter_install(db_type: str) -> None:
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

        if _try_register_adapter_module(db_type):
            logger.info("Auto-installed and loaded adapter: %s", db_type)
        else:
            logger.warning("Auto-install of %s succeeded but the adapter module did not load", package)
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


class ConnectionLease:
    """A borrowed connector; releasing it never closes another borrower's work."""

    def __init__(self, manager: "DBManager", key: tuple, connector: BaseSqlConnector):
        self.connector = connector
        self._released = False
        self._lock = threading.Lock()
        self._finalizer = finalize(self, manager._release_connection, key)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._finalizer()

    def __enter__(self):
        return self.connector

    def __exit__(self, *args):
        self.release()


class DBManager:
    def __init__(self, db_configs: Dict[str, DbConfig], *, max_connections: int = 8, idle_ttl: Optional[float] = None):
        # {datasource: {database: connector}} — a datasource may serve multiple databases.
        self._conn_dict: Dict[str, Dict[Union[str, Tuple[str, str, str]], BaseSqlConnector]] = {}
        if max_connections < 0 or (idle_ttl is not None and idle_ttl < 0):
            raise ValueError("Connection capacity and idle TTL must be non-negative")
        self._db_configs: Dict[str, DbConfig] = db_configs
        self._condition = threading.Condition(threading.RLock())
        self._building: dict[tuple, Future] = {}
        self._leases: dict[tuple, int] = {}
        # Legacy callers receive naked connectors, so their lifetime is unknown.
        self._legacy_keys: set[tuple] = set()
        self._last_used = OrderedDict()
        self._max_connections = max_connections
        self._idle_ttl = idle_ttl
        self._closing = False

    def acquire_conn(
        self, datasource: str, database: str = "", *, context: Optional[DatabaseExecutionContext] = None
    ) -> ConnectionLease:
        """Borrow a connector until release(), including result iteration."""
        key, connector = self._get_connection(datasource, database, context=context, _leased=True)
        return ConnectionLease(self, key, connector)

    def get_conn(
        self, datasource: str, database: str = "", *, context: Optional[DatabaseExecutionContext] = None
    ) -> BaseSqlConnector:
        """Connector for ``(datasource, database)``.

        Without ``context``, ``database`` empty selects the datasource default.
        A resolved ``context`` is authoritative. An empty database is allowed for
        server profiles without a default target, but cannot restore a configured
        database or select a file database.
        For a glob (``path_pattern``)
        datasource each matched file is a database; for server DBs ``database`` overrides the
        configured one (cloned into the connection URI).
        """
        return self._init_connection(datasource, database, context=context)

    def get_connections(self, datasource: str = "") -> Dict[str, BaseSqlConnector]:
        """Connectors for every database served by ``datasource``, keyed by database name.

        A glob (``path_pattern``) datasource yields one connector per matched file; a server
        datasource yields its single configured database. Callers that probe health or list
        databases must iterate the map so multi-database datasources aren't reduced to one.
        """
        cfg = self._db_configs.get(datasource)
        # Only a glob datasource's keys are genuine per-file database names safe to pass back
        # as an override. A single datasource's key may be a display fallback (the datasource
        # name when ``database`` is unconfigured); feeding that back would mis-set the
        # connection's database to the datasource name, so resolve it via the default path.
        if cfg is not None and cfg.path_pattern:
            return {db_name: self._init_connection(datasource, db_name) for db_name in self.get_db_uris(datasource)}
        return {db_name: self._init_connection(datasource, "") for db_name in self.get_db_uris(datasource)}

    def first_conn(self, datasource: str) -> BaseSqlConnector:
        return self._init_connection(datasource, "")

    def first_conn_with_name(self, datasource: str) -> Tuple[str, BaseSqlConnector]:
        """Default connector for ``datasource`` plus its resolved database name.

        The name is the *database* name (a glob datasource's first file, or a server
        datasource's configured ``database``), not the datasource name — empty when no
        database is configured. Callers needing a display label must fall back to the
        datasource name themselves rather than relying on this slot.
        """
        connector = self._init_connection(datasource, "")
        resolved, _ = self._resolve_db_config(datasource, "")
        # Prefer the connector's own database name (e.g. SQLite/DuckDB derive it from the
        # file path, which ``cfg.database`` doesn't carry); fall back to the config-resolved
        # name. Both are real database names — never the datasource name.
        return (connector.database_name or resolved), connector

    def missing_connector_type(self, datasource: str) -> Optional[str]:
        """Return ``datasource``'s (normalized) db type when its connector is
        not registered yet, else ``None``.

        A metadata-only probe — never triggers the adapter auto-install — so
        UI layers can announce a pending install before the blocking
        connection attempt starts. Combine with :meth:`adapter_install_status`
        to phrase the announcement accurately (about to install / installing /
        already failed).
        """
        cfg = self._db_configs.get(datasource)
        if cfg is None:
            return None
        db_type = _normalize_dialect_name(cfg.type)
        if not db_type or connector_registry.is_registered(db_type):
            return None
        return db_type

    def adapter_install_status(self, db_type: str) -> AdapterInstallStatus:
        """Process-wide auto-install state for ``db_type``."""
        return adapter_install_status(db_type)

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
        db_type = _normalize_dialect_name(cfg.type)
        if cfg.uri and db_type in (DBType.SQLITE, DBType.DUCKDB):
            physical = file_stem_from_uri(cfg.uri)
            if db_type == DBType.SQLITE:
                if database and database != physical:
                    raise DatusException(
                        ErrorCode.COMMON_VALIDATION_FAILED,
                        message=f"Database {database!r} is not provided by this SQLite file; available: {physical!r}",
                    )
                return physical, dataclasses.replace(cfg, database=physical)
            # DuckDB may attach additional databases during connection setup.
            # Native USE validates this target before the connector is cached.
            target = database or cfg.database or physical
            if physical == ":memory:" and target == ":memory:":
                target = "memory"
            return target, dataclasses.replace(cfg, database=target)
        if database and database != cfg.database:
            return database, dataclasses.replace(cfg, database=database)
        return cfg.database or "", cfg

    def _init_connection(
        self, datasource: str, database: str, *, context: Optional[DatabaseExecutionContext] = None
    ) -> BaseSqlConnector:
        return self._get_connection(datasource, database, context=context)[1]

    def _get_connection(
        self,
        datasource: str,
        database: str,
        *,
        context: Optional[DatabaseExecutionContext] = None,
        _leased: bool = False,
    ) -> Tuple[tuple, BaseSqlConnector]:
        if context is not None:
            if context.datasource != datasource or (database and database != context.database):
                raise ValueError("Connection context does not match its datasource/database")
            if not context.database:
                configured = self._db_configs.get(datasource)
                # Server adapters may connect without selecting a database. Only
                # reject an empty binding when resolution could restore a target
                # from configuration, a URI, or a file datasource.
                requires_selection = bool(
                    configured is None
                    or configured.database
                    or configured.uri
                    or configured.path_pattern
                    or _normalize_dialect_name(configured.type) in (DBType.SQLITE, DBType.DUCKDB)
                )
                if configured is not None and not requires_selection:
                    # Adapter options in extra are merged last and can supply
                    # a target even when the top-level fields are empty.
                    effective_config = self._db_config_to_connection_config(configured)
                    requires_selection = not isinstance(effective_config, dict) or bool(
                        effective_config.get("database") or effective_config.get("uri")
                    )
                if requires_selection:
                    raise ValueError(
                        "Select a database explicitly; a resolved empty database cannot use a connection default"
                    )
            # Only the context-free legacy path interprets an empty argument
            # as a request for the configured default.
            database = context.database
        db_name, db_config = self._resolve_db_config(datasource, database)
        cache_key = db_name
        if context is not None:
            if context.datasource != datasource or (context.database and context.database != db_name):
                raise ValueError("Connection context does not match its datasource/database")
            db_config = dataclasses.replace(db_config, catalog=context.catalog, schema=context.schema)
            cache_key = (db_name, context.catalog, context.schema)
        key = (datasource, cache_key)
        with self._condition:
            while self._closing:
                self._condition.wait()
            group = self._conn_dict.get(datasource)
            if not isinstance(group, dict):
                group = self._conn_dict[datasource] = {}
            cached = group.get(cache_key)
            if cached is not None:
                return self._borrow_locked(key, cached, _leased)
            future = self._building.get(key)
            creator = future is None
            if creator:
                future = self._building[key] = Future()
        if not creator:
            future.result()
            return self._get_connection(datasource, database, context=context, _leased=_leased)

        conn = None
        try:
            conn = self._build_conn(db_config)
            if _normalize_dialect_name(db_config.type) == DBType.DUCKDB:
                conn.connect()
        except BaseException as exc:
            if conn is not None:
                self._close_connectors([conn])
            with self._condition:
                self._building.pop(key)
                self._condition.notify_all()
            future.set_exception(exc)
            raise
        with self._condition:
            self._conn_dict[datasource][cache_key] = conn
            result = self._borrow_locked(key, conn, _leased)
            self._building.pop(key)
            self._condition.notify_all()
        future.set_result(conn)
        return result

    def _borrow_locked(self, key, connector, leased):
        self._last_used[key] = time.monotonic()
        self._last_used.move_to_end(key)
        if leased:
            self._leases[key] = self._leases.get(key, 0) + 1
        else:
            self._legacy_keys.add(key)
        return key, connector

    def _release_connection(self, key):
        with self._condition:
            count = self._leases.get(key, 0)
            if count <= 1:
                self._leases.pop(key, None)
            else:
                self._leases[key] = count - 1
            self._last_used[key] = time.monotonic()
            self._last_used.move_to_end(key)
            evicted = self._evict_idle_locked()
            self._condition.notify_all()
        self._close_connectors(evicted)

    def _evict_idle_locked(self):
        evicted = []
        now = time.monotonic()
        total = sum(len(group) for group in self._conn_dict.values() if isinstance(group, dict))
        for key, last_used in list(self._last_used.items()):
            if key in self._legacy_keys or self._leases.get(key, 0):
                continue
            expired = self._idle_ttl is not None and now - last_used >= self._idle_ttl
            if not expired and (self._max_connections <= 0 or total <= self._max_connections):
                continue
            self._last_used.pop(key, None)
            group = self._conn_dict.get(key[0])
            connector = group.pop(key[1], None) if isinstance(group, dict) else None
            if connector is not None:
                evicted.append(connector)
                total -= 1
        return evicted

    def purge_idle_connections(self):
        """Apply the owner limit to idle managed connections, never legacy pins."""
        with self._condition:
            evicted = self._evict_idle_locked()
        self._close_connectors(evicted)

    @staticmethod
    def _close_connectors(connectors):
        for connector in {id(conn): conn for conn in connectors}.values():
            try:
                connector.close()
            except Exception:
                logger.warning("Error closing database connector", exc_info=True)

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
            if db_config.uri and db_config.database and db_config.database != file_stem_from_uri(db_path):
                raise DatusException(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    message=f"Database {db_config.database!r} is not provided by this SQLite file",
                )
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
                initial_database=(db_config.database or None) if db_config.uri else None,
                schema_name=db_config.schema or "",
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
        """Drain managed users and close cached connections; later use may rebuild."""
        with self._condition:
            # A running operation may need another connection before it can
            # return its first lease (e.g. a transfer). Only gate new borrows
            # during the actual close, after all current operations drain.
            while self._closing or self._building or self._leases:
                self._condition.wait()
            self._closing = True
            connectors = []
            for group in self._conn_dict.values():
                if isinstance(group, dict):
                    connectors.extend(conn for conn in group.values() if conn is not None)
                    group.clear()
            self._legacy_keys.clear()
            self._last_used.clear()
        try:
            self._close_connectors(connectors)
        finally:
            with self._condition:
                self._closing = False
                self._condition.notify_all()

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()


def db_config_name(datasource: str, db_type: str, name: str = "") -> str:
    # ``datasource::<database>`` for every dialect. The second component is the physical
    # database name (empty when unspecified); the datasource name is never duplicated into
    # it — a datasource is a connection profile, not a database.
    return f"{datasource}::{name}"


# External factory for DBManager creation (used by SaaS backend for connection pooling)
_factory: Optional[Callable[[Dict[str, DbConfig]], DBManager]] = None
# CLI-mode cache: keyed by frozenset of datasource names to avoid creating
# duplicate DBManager instances (and leaking connections) for the same config.
_cli_cache: Dict[frozenset, DBManager] = {}
_cli_cache_lock = threading.RLock()


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
    with _cli_cache_lock:
        _factory = factory
        # Existing tools may still borrow these managers; do not close them.
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
    with _cli_cache_lock:
        factory = _factory
        if factory is None:
            configs = db_configs or {}
            cache_key = frozenset(configs.keys())
            cached = _cli_cache.get(cache_key)
            if cached is None:
                cached = _cli_cache[cache_key] = DBManager(configs)
            return cached
    return factory(db_configs or {})
