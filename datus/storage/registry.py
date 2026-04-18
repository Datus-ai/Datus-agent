# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Storage registry with per-project LRU cache.

Each (factory, project) pair gets its own storage instance with an isolated
VectorDatabase connection. Project isolation is PHYSICAL: each project gets
its own directory under ``{data_dir}/{project}/``.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from datus_storage_base.backend_config import StorageBackendConfig

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import get_embedding_model
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.storage.subject_tree.store import SubjectTreeStore

logger = get_logger(__name__)

_registry_lock = threading.Lock()

# Factory registry: maps factory name → factory callable for lru_cache lookup
_factory_registry: Dict[str, Callable[..., BaseEmbeddingStore]] = {}

# Deployment-level config injected once via configure_storage_defaults().
_storage_defaults: Dict[str, Any] = {}


def configure_storage_defaults(
    **kwargs: Any,
) -> None:
    """Set deployment-level defaults applied to every new storage instance.

    Call once at application startup (e.g. in SaaS backend lifespan).
    Subsequent calls overwrite previous defaults.

    Args:
        **kwargs: Forwarded to ``BaseEmbeddingStore.__init__``:
            ``table_prefix``, ``extra_fields``.

    Example::

        configure_storage_defaults(
            table_prefix="tb_",
        )
    """
    _storage_defaults.clear()
    _storage_defaults.update(kwargs)


def get_storage_defaults() -> Dict[str, Any]:
    """Return the current deployment-level defaults (read-only copy)."""
    return dict(_storage_defaults)


@lru_cache(maxsize=128)
def _get_storage_cached(factory_name: str, embedding_model_conf_name: str, project: str) -> BaseEmbeddingStore:
    """LRU-cached storage creation, keyed by (factory, embedding model, project)."""
    from datus.storage.backend_holder import create_vector_connection

    with _registry_lock:
        factory = _factory_registry[factory_name]
    kwargs = dict(_storage_defaults)
    kwargs["db"] = create_vector_connection(project)

    store = factory(get_embedding_model(embedding_model_conf_name), **kwargs)
    from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore

    if isinstance(store, BaseSubjectEmbeddingStore):
        store.subject_tree = _get_subject_tree_cached(project)
    return store


def get_storage(
    factory: Callable[..., BaseEmbeddingStore],
    embedding_model_conf_name: str,
    project: str,
) -> BaseEmbeddingStore:
    """Return a storage instance scoped to *project*.

    Project isolation is PHYSICAL: each project gets a per-project directory
    under ``{data_dir}/{project}/``. ``project`` must be non-empty and is
    forwarded to the backend ``connect()`` call.

    Uses an LRU cache (maxsize=128) so that inactive projects are evicted.
    Global defaults set via ``configure_storage_defaults()`` are
    automatically forwarded to the factory constructor.
    """
    with _registry_lock:
        _factory_registry[factory.__name__] = factory
    return _get_storage_cached(factory.__name__, embedding_model_conf_name, project)


@lru_cache(maxsize=128)
def _get_subject_tree_cached(project: str) -> "SubjectTreeStore":
    """LRU-cached SubjectTreeStore creation."""
    from datus.storage.subject_tree.store import SubjectTreeStore

    return SubjectTreeStore(project=project)


def get_subject_tree_store(project: str) -> "SubjectTreeStore":
    """Return a SubjectTreeStore instance (LRU-cached per project)."""
    return _get_subject_tree_cached(project)


def preload_all_storages(
    project: str,
    data_dir: str = "",
    config: Optional[StorageBackendConfig] = None,
    **defaults: Any,
) -> None:
    """One-stop initialization: backends + defaults + all storage singletons.

    Combines ``init_backends()``, ``configure_storage_defaults()``, and
    eager loading of every storage singleton into a single call.

    Args:
        project: Project identifier for per-project isolation, forwarded
            to every storage factory.  Must be non-empty.
        data_dir: Root data directory for file-based backends (e.g.
            ``~/.datus/data``).  Passed to ``init_backends()``.
        config: Storage backend configuration. Controls which RDB
            (sqlite/postgresql) and vector (lance) backends are used.
            Defaults to sqlite + lance if omitted.
        **defaults: Deployment-level defaults forwarded to
            ``configure_storage_defaults()`` and then to every
            storage constructor (e.g. ``table_prefix="tb_"``).

    Example (SaaS — PostgreSQL + LanceDB)::

        from datus_storage_base.backend_config import (
            StorageBackendConfig, RdbBackendConfig, VectorBackendConfig,
        )
        preload_all_storages(
            data_dir="/data/tenants/t1/workspaces/ws1/data",
            config=StorageBackendConfig(
                rdb=RdbBackendConfig(type="postgresql", params={...}),
                vector=VectorBackendConfig(type="lance"),
            ),
            project="ws1",
            table_prefix="tb_",
        )

    Example (CLI — default sqlite + lance)::

        preload_all_storages(data_dir="~/.datus/data", project="my_project")
    """
    from datus.storage.backend_holder import init_backends

    # 1. Initialize backends (vector DB + RDB connections)
    init_backends(config=config, data_dir=data_dir)

    # 2. Apply deployment-level defaults
    if defaults:
        configure_storage_defaults(**defaults)

    # 3. Eagerly create all storage singletons
    from datus.storage.ext_knowledge.store import ExtKnowledgeStore
    from datus.storage.metric.store import MetricStorage
    from datus.storage.reference_sql.store import ReferenceSqlStorage
    from datus.storage.schema_metadata.store import SchemaStorage, SchemaValueStorage
    from datus.storage.semantic_model.store import SemanticModelStorage

    get_storage(SchemaStorage, "database", project=project)
    get_storage(SchemaValueStorage, "database", project=project)
    get_storage(SemanticModelStorage, "semantic_model", project=project)
    get_storage(MetricStorage, "metric", project=project)
    get_storage(ReferenceSqlStorage, "reference_sql", project=project)
    get_storage(ExtKnowledgeStore, "ext_knowledge", project=project)
    get_subject_tree_store(project=project)
    logger.info("All storage singletons pre-loaded")


def clear_storage_registry() -> None:
    """Clear all cached storage instances and reset backends.

    Does NOT clear ``_storage_defaults``. The teardown runs under
    ``_registry_lock`` so concurrent ``get_storage()`` callers do not observe
    a half-initialized backend state.
    """
    from datus.storage.backend_holder import reset_backends

    with _registry_lock:
        _get_storage_cached.cache_clear()
        _factory_registry.clear()
        _get_subject_tree_cached.cache_clear()
        reset_backends()
