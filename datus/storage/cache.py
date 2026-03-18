# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable, Optional, TypeVar

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel, get_embedding_model
from datus.storage.scoped_filter import ScopedFilterBuilder
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=12)
def _cached_storage[T: BaseEmbeddingStore](
    factory: Callable[[EmbeddingModel], T], storage_path: str, model_name: str
) -> T:
    return factory(get_embedding_model(model_name))


# Module-level cache for scoped storage instances so that clear_cache() can
# invalidate them regardless of which caller created them.
_scoped_storage_cache: dict[tuple[str, ...], BaseEmbeddingStore] = {}


def _build_scope_filter(
    agent_config: AgentConfig,
    sub_agent_config: SubAgentConfig,
    check_scope_attr: str,
    storage: BaseEmbeddingStore,
):
    """Build a scope filter Node from the sub-agent's scoped context."""
    scope_value = getattr(sub_agent_config.scoped_context, check_scope_attr, None)
    if not scope_value:
        return None

    if check_scope_attr == "tables":
        dialect = getattr(agent_config, "db_type", "")
        return ScopedFilterBuilder.build_table_filter(scope_value, dialect)
    elif check_scope_attr in ("metrics", "sqls", "ext_knowledge"):
        # Subject-based stores need the subject_tree to resolve paths to node IDs
        subject_tree = getattr(storage, "subject_tree", None)
        if subject_tree is None:
            logger.warning(f"Storage for '{check_scope_attr}' has no subject_tree; cannot build subject filter")
            return None
        return ScopedFilterBuilder.build_subject_filter(scope_value, subject_tree)
    return None


def create_storage_with_scope(
    storage_factory: Callable[[EmbeddingModel], T],
    agent_config: AgentConfig,
    embedding_model_conf_name: str,
    check_scope_attr: str,
    sub_agent_name: Optional[str] = None,
) -> T:
    """Create a storage instance, optionally scoped to a sub-agent's context.

    If sub_agent_name is provided and the sub-agent has a scoped_context
    with a non-empty value for check_scope_attr, a WHERE filter is applied
    to restrict all queries to that scope.
    """
    if sub_agent_name and (config := agent_config.sub_agent_config(sub_agent_name)):
        sub_agent_config = SubAgentConfig.model_validate(config)
        scope_value = None
        if sub_agent_config.has_scoped_context_by(check_scope_attr):
            scope_value = getattr(sub_agent_config.scoped_context, check_scope_attr, None)
        if scope_value:
            storage_path = agent_config.rag_storage_path()
            scope_hash = hashlib.md5(str(scope_value).encode()).hexdigest()[:8]
            factory_name = storage_factory.__name__
            cache_key = (factory_name, storage_path, sub_agent_name, check_scope_attr, scope_hash)
            cached = _scoped_storage_cache.get(cache_key)
            if cached is not None:
                return cached  # type: ignore[return-value]
            # Use global storage with a scope filter instead of a separate sub-agent copy
            storage = storage_factory(
                get_embedding_model(embedding_model_conf_name),
            )
            scope_filter = _build_scope_filter(agent_config, sub_agent_config, check_scope_attr, storage)
            if scope_filter is None:
                logger.warning(
                    "Failed to build scope filter for sub-agent '%s' (attr='%s'); "
                    "denying access to prevent unscoped reads.",
                    sub_agent_name,
                    check_scope_attr,
                )
                raise DatusException(
                    code=ErrorCode.COMMON_VALIDATION_FAILED,
                    message=(
                        f"Cannot build scope filter for sub-agent '{sub_agent_name}' "
                        f"(scope attr='{check_scope_attr}'). "
                        "Refusing to return unscoped storage."
                    ),
                )
            storage._scope_filter = scope_filter
            _scoped_storage_cache[cache_key] = storage
            logger.debug(
                f"Sub agent {sub_agent_name} has scoped context, "
                f"using global storage with WHERE filter for '{check_scope_attr}'"
            )
            return storage
    return _cached_storage(storage_factory, agent_config.rag_storage_path(), embedding_model_conf_name)


def clear_cache():
    _cached_storage.cache_clear()
    _scoped_storage_cache.clear()

    from datus.storage.backend_holder import reset_backends

    reset_backends()
