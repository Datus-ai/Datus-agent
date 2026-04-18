# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Global backend singleton — manages RDB and vector backend instances.

Backends are stateless with respect to project: ``initialize()`` only carries
backend-wide configuration (data_dir, isolation), while the active project
identifier is passed at ``connect()`` time by the ``create_*`` helpers. This
lets one backend instance serve many projects.
"""

import threading
from typing import Optional

from datus_storage_base.backend_config import StorageBackendConfig
from datus_storage_base.rdb.base import BaseRdbBackend, RdbDatabase
from datus_storage_base.vector.base import VectorDatabase

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_config: Optional[StorageBackendConfig] = None
_data_dir: str = ""
_vector_backend = None
_vector_initialized: bool = False
_rdb_backend: Optional[BaseRdbBackend] = None
_rdb_initialized: bool = False
_rdb_lock = threading.Lock()
_vector_lock = threading.Lock()


def init_backends(
    config: Optional[StorageBackendConfig] = None,
    data_dir: str = "",
) -> None:
    """Initialize storage backends from configuration.

    Should be called once during application startup. The per-call project
    identifier is not stored here — it is passed to ``create_rdb_for_store``
    / ``create_vector_connection`` at lookup time.

    Args:
        config: Storage backend configuration. Defaults to sqlite + lance.
        data_dir: Root data directory for file-based backends (e.g.
            ``~/.datus/data``).
    """
    global _config, _data_dir, _vector_backend, _vector_initialized
    global _rdb_backend, _rdb_initialized
    _config = config or StorageBackendConfig()
    _data_dir = data_dir
    _vector_backend = None
    _vector_initialized = False
    _rdb_backend = None
    _rdb_initialized = False
    logger.debug(f"Storage backends configured: rdb={_config.rdb.type}, vector={_config.vector.type}")


def _ensure_config() -> StorageBackendConfig:
    """Return the current config, defaulting to sqlite + lance if not initialized."""
    global _config
    if _config is None:
        _config = StorageBackendConfig()
    return _config


def _parse_isolation_type(cfg) -> str:
    """Normalize ``cfg.isolation`` (enum or string) to a plain string."""
    isolation = getattr(cfg, "isolation", "physical")
    if hasattr(isolation, "value"):
        return isolation.value
    return str(isolation)


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
                rdb_config.setdefault("isolation", _parse_isolation_type(cfg))
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
                vector_config.setdefault("isolation", _parse_isolation_type(cfg))
                _vector_backend = VectorRegistry.create_backend(cfg.vector.type, vector_config)
                _vector_initialized = True
                logger.debug(f"Vector backend initialized: {cfg.vector.type}")

    return _vector_backend


def get_isolation_type() -> str:
    """Return the current isolation type as a string ('physical' or 'logical')."""
    cfg = _ensure_config()
    return _parse_isolation_type(cfg)


def create_rdb_for_store(store_db_name: str, project: str) -> RdbDatabase:
    """Create an RDB database handle for *store_db_name* scoped to *project*.

    The backend singleton is reused; ``connect()`` produces a per-store,
    per-project database handle. ``project`` is a path component (PHYSICAL
    isolation) and must be non-empty.

    Args:
        store_db_name: Logical store name (e.g. ``"subject_tree"``).
        project: Project identifier; must be non-empty.

    Raises:
        DatusException: when ``project`` is empty.
    """
    if not project:
        raise DatusException(
            ErrorCode.STORAGE_FAILED,
            message=f"create_rdb_for_store requires a non-empty project (store_db_name={store_db_name!r}).",
        )
    backend = _get_rdb_backend()
    return backend.connect(project, store_db_name)


def create_vector_connection(project: str) -> VectorDatabase:
    """Create a vector db connection scoped to *project*.

    Args:
        project: Project identifier passed to the backend's ``connect()``
            first argument; the backend uses it as a path component for
            per-project isolation. Must be non-empty.

    Raises:
        DatusException: when ``project`` is empty.
    """
    if not project:
        raise DatusException(
            ErrorCode.STORAGE_FAILED,
            message="create_vector_connection requires a non-empty project.",
        )
    backend = get_vector_backend()
    return backend.connect(project)


def reset_backends() -> None:
    """Reset all backend instances. Called by ``clear_cache()``."""
    global _config, _data_dir, _vector_backend, _vector_initialized
    global _rdb_backend, _rdb_initialized
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
    _vector_backend = None
    _vector_initialized = False
    _rdb_backend = None
    _rdb_initialized = False
    logger.debug("Storage backends reset")
