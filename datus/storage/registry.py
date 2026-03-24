# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Storage singleton registry.

Two-level configuration:

1. **Deployment-level** (call once at startup via ``configure_storage_defaults``):
   ``table_prefix``, ``extra_fields`` — schema structure.
   ``scope_fields`` — which ``request_context`` fields are used as read filters.

2. **Request-level** (per-request, via ``agent_config.request_context``):
   ``get_rag_storage(...)`` builds a scoped view that:
   - Reads: auto-applies WHERE for ``scope_fields`` (e.g. workspace_id)
   - Writes: auto-fills ALL ``request_context`` fields (workspace_id + creator_id + …)
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from datus_storage_base.conditions import Node, and_, eq

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import get_embedding_model
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig
    from datus.storage.subject_tree.store import SubjectTreeStore

logger = get_logger(__name__)

_storage_instances: Dict[Tuple[str, str], BaseEmbeddingStore] = {}
_subject_tree_instances: Dict[str, Any] = {}  # keyed by namespace

# Deployment-level config injected once via configure_storage_defaults().
_storage_defaults: Dict[str, Any] = {}
# Which request_context fields are used as WHERE filters on reads.
_scope_fields: List[str] = []


def configure_storage_defaults(
    scope_fields: Optional[List[str]] = None,
    **kwargs: Any,
) -> None:
    """Set deployment-level defaults applied to every new storage instance.

    Call once at application startup (e.g. in SaaS backend lifespan).
    Subsequent calls overwrite previous defaults.

    Args:
        scope_fields: ``request_context`` field names that become WHERE
            filters on reads (e.g. ``["workspace_id"]``).  Other
            ``request_context`` fields are write-only.
        **kwargs: Forwarded to ``BaseEmbeddingStore.__init__``:
            ``table_prefix``, ``extra_fields``.

    Example::

        configure_storage_defaults(
            table_prefix="tb_",
            extra_fields=[
                pa.field("workspace_id", pa.string()),
                pa.field("creator_id", pa.string()),
                pa.field("updator_id", pa.string()),
            ],
            scope_fields=["workspace_id"],
        )
    """
    _storage_defaults.clear()
    _storage_defaults.update(kwargs)
    # scope_fields are also used as scalar indices on the storage tables
    if scope_fields:
        _storage_defaults["scope_indices"] = list(scope_fields)
    _scope_fields.clear()
    if scope_fields:
        _scope_fields.extend(scope_fields)


def get_storage_defaults() -> Dict[str, Any]:
    """Return the current deployment-level defaults (read-only copy)."""
    return dict(_storage_defaults)


def get_storage(
    factory: Callable[..., BaseEmbeddingStore],
    embedding_model_conf_name: str,
    namespace: str = "",
) -> BaseEmbeddingStore:
    """Return a singleton storage instance keyed by (factory name, namespace).

    Global defaults set via ``configure_storage_defaults()`` are automatically
    forwarded to the factory constructor.
    """
    key = (factory.__name__, namespace)
    cached = _storage_instances.get(key)
    if cached is not None:
        return cached

    storage = factory(get_embedding_model(embedding_model_conf_name), **_storage_defaults)
    _storage_instances[key] = storage
    return storage


def get_rag_storage(
    factory: Callable[..., BaseEmbeddingStore],
    embedding_model_conf_name: str,
    agent_config: "AgentConfig",
    sub_agent_name: Optional[str],
    check_scope_attr: str,
) -> BaseEmbeddingStore:
    """One-stop helper for RAG class constructors.

    Combines singleton lookup + scope filtering + write defaults:

    - **Read**: sub-agent scope filter + ``scope_fields`` WHERE clause
    - **Write**: all ``request_context`` fields auto-filled

    Args:
        factory: Storage class constructor.
        embedding_model_conf_name: Embedding model config key.
        agent_config: Current request's agent configuration.
        sub_agent_name: Sub-agent name, or None for global access.
        check_scope_attr: Scoped context attribute
            (e.g. "tables", "metrics", "sqls", "ext_knowledge").
    """
    from datus.storage.rag_scope import build_rag_scope

    storage = get_storage(factory, embedding_model_conf_name, agent_config.current_namespace)
    scope_filter = build_rag_scope(agent_config, sub_agent_name, storage, check_scope_attr)
    write_defaults = _build_write_defaults(agent_config)
    return create_scoped_view(storage, scope_filter, write_defaults)


def create_scoped_view(
    storage: BaseEmbeddingStore,
    scope_filter: Optional[Node],
    write_defaults: Optional[Dict[str, Any]] = None,
) -> BaseEmbeddingStore:
    """Create a shallow copy of *storage* with independent scope filter and write defaults.

    The copy shares the underlying db connection, table reference, and locks
    so that writes to the singleton are visible through the scoped view.

    Args:
        storage: The singleton storage instance.
        scope_filter: WHERE filter to apply on reads, or None.
        write_defaults: Per-request values to auto-fill on writes.
    """
    if scope_filter is None and not write_defaults:
        return storage

    # Shallow copy shares db/table/locks/_init_state with the singleton.
    # _init_state is a shared mutable list — once the singleton initializes
    # the table, all scoped views see it immediately without re-entering
    # the lock. No eager _ensure_table_ready() call needed here.
    view = copy.copy(storage)
    if scope_filter is not None:
        view._scope_filter = scope_filter
    if write_defaults:
        # New dict so we don't mutate the singleton's _default_values
        view._default_values = {**storage._default_values, **write_defaults}
    return view


def _build_write_defaults(agent_config: "AgentConfig") -> Optional[Dict[str, Any]]:
    """Build per-request write defaults from ``agent_config.request_context``.

    ALL request_context fields are written; only ``_scope_fields`` are read-filtered.
    Returns None in CLI mode (no request_context).
    """
    request_context = getattr(agent_config, "request_context", None)
    if not request_context:
        return None
    return dict(request_context)


def build_scope_filter_from_context(agent_config: "AgentConfig") -> Optional[Node]:
    """Build read-time WHERE filter from ``request_context`` + ``scope_fields``.

    Only fields listed in ``_scope_fields`` (set via ``configure_storage_defaults``)
    become equality conditions.  Returns None if no scope_fields configured
    or no matching values in request_context.
    """
    if not _scope_fields:
        return None
    request_context = getattr(agent_config, "request_context", None)
    if not request_context:
        return None

    conditions: list[Node] = []
    for field_name in _scope_fields:
        value = request_context.get(field_name)
        if value is not None:
            conditions.append(eq(field_name, value))

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else and_(*conditions)


def get_subject_tree_store(namespace: str = "") -> "SubjectTreeStore":
    """Return a singleton SubjectTreeStore per namespace.

    SubjectTreeStore is RDB-backed (not embedding-based), so it has its own
    cache separate from the vector storage registry.
    """
    cached = _subject_tree_instances.get(namespace)
    if cached is not None:
        return cached

    from datus.storage.subject_tree.store import SubjectTreeStore

    instance = SubjectTreeStore()
    _subject_tree_instances[namespace] = instance
    return instance


def clear_storage_registry() -> None:
    """Clear all cached storage instances and reset backends.

    Does NOT clear ``_storage_defaults`` or ``_scope_fields``.
    """
    _storage_instances.clear()
    _subject_tree_instances.clear()

    from datus.storage.backend_holder import reset_backends

    reset_backends()
