# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Global backend singleton — manages RDB and vector backend instances."""

import threading
from typing import Optional

from datus_storage_base.backend_config import StorageBackendConfig
from datus_storage_base.rdb.base import BaseRdbBackend, RdbDatabase
from datus_storage_base.vector.base import VectorDatabase

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_config: Optional[StorageBackendConfig] = None
_data_dir: str = ""
_namespace: str = ""
_vector_backend = None
_vector_initialized: bool = False
_rdb_backend: Optional[BaseRdbBackend] = None
_rdb_initialized: bool = False
_rdb_lock = threading.Lock()
_vector_lock = threading.Lock()


def init_backends(
    config: Optional[StorageBackendConfig] = None,
    data_dir: str = "",
    namespace: str = "",
) -> None:
    """Initialize storage backends from configuration.

    Should be called once during application startup (from AgentConfig).
    If *config* is ``None``, defaults are used (sqlite + lance).

    Args:
        config: Storage backend configuration.
        data_dir: Root data directory (e.g. ``{home}/data``).
        namespace: Current namespace for data isolation.
    """
    global _config, _data_dir, _namespace, _vector_backend, _vector_initialized
    global _rdb_backend, _rdb_initialized
    _config = config or StorageBackendConfig()
    _data_dir = data_dir
    _namespace = namespace
    # Lazily initialize vector backend on first use
    _vector_backend = None
    _vector_initialized = False
    # Reset RDB backend for lazy re-initialization
    _rdb_backend = None
    _rdb_initialized = False
    logger.debug(f"Storage backends configured: rdb={_config.rdb.type}, vector={_config.vector.type}")


def set_namespace(namespace: str) -> None:
    """Switch namespace (called when AgentConfig.current_namespace changes)."""
    global _namespace
    _namespace = namespace


def _ensure_config() -> StorageBackendConfig:
    """Return the current config, defaulting to sqlite + lance if not initialized."""
    global _config
    if _config is None:
        _config = StorageBackendConfig()
    return _config


def _get_rdb_backend() -> BaseRdbBackend:
    """Return the global RDB backend instance (lazy-initialized singleton)."""
    global _rdb_backend, _rdb_initialized

    if not _rdb_initialized:
        with _rdb_lock:
            if not _rdb_initialized:
                from datus.storage.rdb import RdbRegistry

                cfg = _ensure_config()
                rdb_config = dict(cfg.rdb.params)
                rdb_config["data_dir"] = _data_dir
                rdb_config["isolation"] = _parse_isolation_type(cfg)
                _rdb_backend = RdbRegistry.create_backend(cfg.rdb.type, rdb_config)
                _rdb_initialized = True
                logger.debug(f"RDB backend initialized: {cfg.rdb.type}")

    return _rdb_backend


def get_vector_backend():
    """Return the global vector backend instance (lazy-initialized)."""
    global _vector_backend, _vector_initialized

    if not _vector_initialized:
        with _vector_lock:
            if not _vector_initialized:
                from datus.storage.vector import VectorRegistry

                cfg = _ensure_config()
                logger.debug(f"Initializing vector backend: type={cfg.vector.type}")
                vector_config = dict(cfg.vector.params)
                vector_config["data_dir"] = _data_dir
                vector_config["isolation"] = _parse_isolation_type(cfg)
                _vector_backend = VectorRegistry.create_backend(cfg.vector.type, vector_config)
                _vector_initialized = True
                logger.debug(f"Vector backend initialized: {cfg.vector.type}")

    return _vector_backend


def get_current_namespace() -> str:
    """Return the current global namespace."""
    return _namespace


def get_isolation_type() -> str:
    """Return the current isolation type as a string ('physical' or 'logical')."""
    cfg = _ensure_config()
    return _parse_isolation_type(cfg)


def _parse_isolation_type(cfg) -> str:
    isolation = getattr(cfg, "isolation", "physical")
    if hasattr(isolation, "value"):
        return isolation.value
    return str(isolation)


def create_rdb_for_store(store_db_name: str, namespace: str = "") -> RdbDatabase:
    """Create an RDB database handle for a specific store.

    The backend singleton is reused; ``connect()`` produces a per-store database.

    Args:
        store_db_name: Logical store name (e.g. ``"subject_tree"``).
        namespace: Namespace for path isolation.  Defaults to global ``_namespace``.
    """
    backend = _get_rdb_backend()
    ns = namespace or _namespace
    return backend.connect(ns, store_db_name)


def create_vector_connection(namespace: str = "") -> VectorDatabase:
    """Create a vector db connection.

    Main storage uses the unified ``datus_db`` directory (default).
    Pass an explicit *namespace* to create an isolated database for
    special stores (e.g. ``docstore__snowflake`` for document stores).

    When no *namespace* is given, the global ``_namespace`` is used so that:
    - PHYSICAL mode: connects to ``datus_db_{namespace}`` directory
    - LOGICAL mode: connects to shared ``datus_db`` with datasource_id filtering
    """
    backend = get_vector_backend()
    ns = namespace if namespace else _namespace
    return backend.connect(namespace=ns)


def reset_backends() -> None:
    """Reset all backend instances. Called by ``clear_cache()``."""
    global _config, _data_dir, _namespace, _vector_backend, _vector_initialized
    global _rdb_backend, _rdb_initialized
    # Close existing backends before resetting references
    if _rdb_backend is not None:
        try:
            _rdb_backend.close()
        except Exception as e:
            logger.debug(f"Error closing RDB backend: {e}")
    if _vector_backend is not None:
        try:
            _vector_backend.close()
        except Exception as e:
            logger.debug(f"Error closing vector backend: {e}")
    _config = None
    _data_dir = ""
    _namespace = ""
    _vector_backend = None
    _vector_initialized = False
    _rdb_backend = None
    _rdb_initialized = False
    logger.debug("Storage backends reset")
