# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Global backend singleton — manages RDB and vector backend instances."""

import threading
import warnings
from typing import Optional

from datus_storage_base.backend_config import StorageBackendConfig
from datus_storage_base.rdb.base import BaseRdbBackend, RdbDatabase
from datus_storage_base.vector.base import VectorDatabase

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_config: Optional[StorageBackendConfig] = None
_data_dir: str = ""
_project: str = ""
_vector_backend = None
_vector_initialized: bool = False
_rdb_backend: Optional[BaseRdbBackend] = None
_rdb_initialized: bool = False
_rdb_lock = threading.Lock()
_vector_lock = threading.Lock()


def init_backends(
    config: Optional[StorageBackendConfig] = None,
    data_dir: str = "",
    project: str = "",
) -> None:
    """Initialize storage backends from configuration.

    Should be called once during application startup (from AgentConfig).
    If *config* is ``None``, defaults are used (sqlite + lance).

    Args:
        config: Storage backend configuration.
        data_dir: Root data directory for file-based backends (e.g.
            ``~/.datus/data``).  This is the *parent* of any per-project
            sub-layout; each backend owns its project isolation strategy.
        project: Project identifier used by backends that isolate by project
            (e.g. sqlite / lance use a ``{project}/`` subdirectory).
    """
    global _config, _data_dir, _project, _vector_backend, _vector_initialized
    global _rdb_backend, _rdb_initialized
    _config = config or StorageBackendConfig()
    _data_dir = data_dir
    _project = project
    # Lazily initialize vector backend on first use
    _vector_backend = None
    _vector_initialized = False
    # Reset RDB backend for lazy re-initialization
    _rdb_backend = None
    _rdb_initialized = False
    logger.debug(f"Storage backends configured: rdb={_config.rdb.type}, vector={_config.vector.type}")


def set_project(project: str) -> None:
    """Switch the current project identifier.

    Triggered by :pymeth:`datus.configuration.agent_config.AgentConfig.project_name`
    setter; backends that cache a project-scoped path must be re-initialized
    via a follow-up :func:`init_backends` call.
    """
    global _project
    _project = project


def set_namespace(namespace: str) -> None:  # pragma: no cover - deprecated shim
    """Deprecated alias for :func:`set_project`.

    Kept so older external extensions keep importing; new code should call
    :func:`set_project` directly.
    """
    warnings.warn(
        "datus.storage.backend_holder.set_namespace is deprecated; use set_project().",
        DeprecationWarning,
        stacklevel=2,
    )
    set_project(namespace)


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
                # Project isolation is RDB-backend-specific: the built-in
                # sqlite backend builds a ``{data_dir}/{project}/datus_db/``
                # layout; other backends may ignore this or map it to a
                # schema/bucket name.
                rdb_config["project"] = _project
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
                # LOGICAL isolation is still meaningful for vector backends:
                # lance scopes rows by ``datasource_id``; physical isolation
                # for the backend-level layout is handled per-backend via the
                # ``project`` key.
                vector_config["isolation"] = _parse_isolation_type(cfg)
                vector_config["project"] = _project
                _vector_backend = VectorRegistry.create_backend(cfg.vector.type, vector_config)
                _vector_initialized = True
                logger.debug(f"Vector backend initialized: {cfg.vector.type}")

    return _vector_backend


def get_current_project() -> str:
    """Return the current global project identifier."""
    return _project


def get_current_namespace() -> str:  # pragma: no cover - deprecated shim
    """Deprecated alias for :func:`get_current_project`."""
    warnings.warn(
        "datus.storage.backend_holder.get_current_namespace is deprecated; use get_current_project().",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_current_project()


def get_isolation_type() -> str:
    """Return the current isolation type as a string ('physical' or 'logical')."""
    cfg = _ensure_config()
    return _parse_isolation_type(cfg)


def _parse_isolation_type(cfg) -> str:
    isolation = getattr(cfg, "isolation", "physical")
    if hasattr(isolation, "value"):
        return isolation.value
    return str(isolation)


def create_rdb_for_store(store_db_name: str, project: str = "") -> RdbDatabase:
    """Create an RDB database handle for a specific store.

    The backend singleton is reused; ``connect()`` produces a per-store database.

    Args:
        store_db_name: Logical store name (e.g. ``"subject_tree"``).
        project: Optional override for the current global project.  Most
            callers leave this empty so the global project from
            :func:`init_backends` is used.  The backend ultimately decides how
            project maps to storage (directory / schema / bucket); the base
            ``connect(namespace, store_db_name)`` signature is preserved, so
            this value is passed as the first argument.
    """
    backend = _get_rdb_backend()
    proj = project or _project
    return backend.connect(proj, store_db_name)


def create_vector_connection(datasource_id: str = "") -> VectorDatabase:
    """Create a vector db connection.

    Project isolation is handled by the vector backend itself (``lance`` uses
    a ``{data_dir}/{project}/datus_db`` directory).  This helper only carries
    the optional *datasource_id*, which the backend uses for LOGICAL row-
    scoping when the underlying config has ``isolation: logical``.

    Args:
        datasource_id: Optional logical filter key (e.g. the current database
            name, or a ``document__{platform}`` identifier for per-platform
            document stores).  Leave empty to skip LOGICAL filtering.
    """
    backend = get_vector_backend()
    return backend.connect(namespace=datasource_id)


def reset_backends() -> None:
    """Reset all backend instances. Called by ``clear_cache()``."""
    global _config, _data_dir, _project, _vector_backend, _vector_initialized
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
    _project = ""
    _vector_backend = None
    _vector_initialized = False
    _rdb_backend = None
    _rdb_initialized = False
    logger.debug("Storage backends reset")
