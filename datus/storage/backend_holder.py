# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Global backend singleton — manages RDB and vector backend instances."""

import threading
from typing import Optional

from datus.storage.backend_config import StorageBackendConfig
from datus.storage.rdb.base import BaseRdbBackend, RdbDatabase
from datus.storage.vector.base import VectorDatabase
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
                from datus.storage.rdb.registry import RdbRegistry

                cfg = _ensure_config()
                rdb_config = dict(cfg.rdb.params)
                rdb_config["data_dir"] = _data_dir
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
                from datus.storage.vector.registry import VectorRegistry

                cfg = _ensure_config()
                logger.debug(f"Initializing vector backend: type={cfg.vector.type}")
                vector_config = dict(cfg.vector.params)
                vector_config["data_dir"] = _data_dir
                _vector_backend = VectorRegistry.create_backend(cfg.vector.type, vector_config)
                _vector_initialized = True
                logger.debug(f"Vector backend initialized: {cfg.vector.type}")

    return _vector_backend


def create_rdb_for_store(store_db_name: str) -> RdbDatabase:
    """Create an RDB database handle for a specific store.

    The storage path is resolved from the global ``_data_dir`` and ``_namespace``.
    The backend singleton is reused; ``connect()`` produces a per-store database.
    """
    backend = _get_rdb_backend()
    return backend.connect(_namespace, store_db_name)


def create_vector_connection() -> VectorDatabase:
    """Create a vector db connection."""
    backend = get_vector_backend()
    return backend.connect(namespace=_namespace)


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
