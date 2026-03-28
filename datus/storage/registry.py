# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Storage registry with per-namespace LRU cache.

Each (factory, namespace) pair gets its own storage instance with an isolated
VectorDatabase connection.  Multi-tenant isolation is handled at the backend
level via ``IsolationType.LOGICAL`` (datasource_id column) or ``PHYSICAL``
(separate directories).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from datus_storage_base.backend_config import StorageBackendConfig

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import get_embedding_model
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.storage.subject_tree.store import SubjectTreeStore

logger = get_logger(__name__)

_subject_tree_instance: Optional[Any] = None

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
def _get_storage_cached(factory_name: str, embedding_model_conf_name: str, namespace: str) -> BaseEmbeddingStore:
    """LRU-cached storage creation keyed by ``(factory_name, embedding_model_conf_name, namespace)``."""
    factory = _factory_registry[factory_name]
    kwargs = dict(_storage_defaults)
    if namespace:
        from datus.storage.backend_holder import create_vector_connection

        kwargs["db"] = create_vector_connection(namespace=namespace)

    return factory(get_embedding_model(embedding_model_conf_name), **kwargs)


def get_storage(
    factory: Callable[..., BaseEmbeddingStore],
    embedding_model_conf_name: str,
    namespace: str = "",
) -> BaseEmbeddingStore:
    """Return a storage instance keyed by ``(factory_name, namespace)``.

    Each *namespace* (i.e. datasource_id) gets its own storage instance with
    its own ``VectorDatabase`` connection so the backend can apply
    ``IsolationType.LOGICAL`` scoping automatically.

    Uses an LRU cache (maxsize=128) so that inactive namespaces are evicted.
    Global defaults set via ``configure_storage_defaults()`` are automatically
    forwarded to the factory constructor.
    """
    _factory_registry[factory.__name__] = factory
    return _get_storage_cached(factory.__name__, embedding_model_conf_name, namespace)


def get_subject_tree_store() -> "SubjectTreeStore":
    """Return the global singleton SubjectTreeStore.

    SubjectTreeStore is RDB-backed (not embedding-based), so it has its own
    cache separate from the vector storage registry.
    """
    global _subject_tree_instance
    if _subject_tree_instance is not None:
        return _subject_tree_instance

    from datus.storage.subject_tree.store import SubjectTreeStore

    _subject_tree_instance = SubjectTreeStore()
    return _subject_tree_instance


def preload_all_storages(
    data_dir: str = "",
    config: Optional[StorageBackendConfig] = None,
    namespace: str = "",
    **defaults: Any,
) -> None:
    """One-stop initialization: backends + defaults + all storage singletons.

    Combines ``init_backends()``, ``configure_storage_defaults()``, and
    eager loading of every storage singleton into a single call.

    Args:
        data_dir: Root data directory (e.g. ``{home}/data``).
            Passed to ``init_backends()``.
        config: Storage backend configuration.
            Controls which RDB (sqlite/postgresql) and vector (lance)
            backends are used.  Defaults to sqlite + lance if omitted.
        namespace: Namespace (datasource_id) for per-tenant storage instances.
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
            namespace="ds_001",
            table_prefix="tb_",
        )

    Example (CLI — default sqlite + lance)::

        preload_all_storages(data_dir="~/.datus/data")
    """
    from datus.storage.backend_holder import init_backends

    # 1. Initialize backends (vector DB + RDB connections)
    init_backends(config=config, data_dir=data_dir, namespace=namespace)

    # 2. Apply deployment-level defaults
    if defaults:
        configure_storage_defaults(**defaults)

    # 3. Eagerly create all storage singletons
    from datus.storage.ext_knowledge.store import ExtKnowledgeStore
    from datus.storage.metric.store import MetricStorage
    from datus.storage.reference_sql.store import ReferenceSqlStorage
    from datus.storage.schema_metadata.store import SchemaStorage, SchemaValueStorage
    from datus.storage.semantic_model.store import SemanticModelStorage

    get_storage(SchemaStorage, "database", namespace=namespace)
    get_storage(SchemaValueStorage, "database", namespace=namespace)
    get_storage(SemanticModelStorage, "semantic_model", namespace=namespace)
    get_storage(MetricStorage, "metric", namespace=namespace)
    get_storage(ReferenceSqlStorage, "reference_sql", namespace=namespace)
    get_storage(ExtKnowledgeStore, "ext_knowledge", namespace=namespace)
    get_subject_tree_store()
    logger.info("All storage singletons pre-loaded")


def clear_storage_registry() -> None:
    """Clear all cached storage instances and reset backends.

    Does NOT clear ``_storage_defaults``.
    """
    global _subject_tree_instance
    _get_storage_cached.cache_clear()
    _factory_registry.clear()
    _subject_tree_instance = None

    from datus.storage.backend_holder import reset_backends

    reset_backends()
