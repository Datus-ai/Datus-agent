# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Storage singleton registry.

Stores are true singletons keyed by factory name only.
Multi-tenant isolation (datasource_id filtering) is handled at the RAG layer,
not at the storage layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from datus_storage_base.backend_config import StorageBackendConfig

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import get_embedding_model
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.storage.subject_tree.store import SubjectTreeStore

logger = get_logger(__name__)

_storage_instances: Dict[str, BaseEmbeddingStore] = {}
_subject_tree_instance: Optional[Any] = None

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


def get_storage(
    factory: Callable[..., BaseEmbeddingStore],
    embedding_model_conf_name: str,
) -> BaseEmbeddingStore:
    """Return a singleton storage instance keyed by factory name.

    Global defaults set via ``configure_storage_defaults()`` are automatically
    forwarded to the factory constructor.
    """
    key = factory.__name__
    cached = _storage_instances.get(key)
    if cached is not None:
        return cached

    storage = factory(get_embedding_model(embedding_model_conf_name), **_storage_defaults)
    _storage_instances[key] = storage
    return storage


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
            table_prefix="tb_",
        )

    Example (CLI — default sqlite + lance)::

        preload_all_storages(data_dir="~/.datus/data")
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

    get_storage(SchemaStorage, "database")
    get_storage(SchemaValueStorage, "database")
    get_storage(SemanticModelStorage, "semantic_model")
    get_storage(MetricStorage, "metric")
    get_storage(ReferenceSqlStorage, "reference_sql")
    get_storage(ExtKnowledgeStore, "ext_knowledge")
    get_subject_tree_store()
    logger.info("All storage singletons pre-loaded")


def clear_storage_registry() -> None:
    """Clear all cached storage instances and reset backends.

    Does NOT clear ``_storage_defaults``.
    """
    global _subject_tree_instance
    _storage_instances.clear()
    _subject_tree_instance = None

    from datus.storage.backend_holder import reset_backends

    reset_backends()
