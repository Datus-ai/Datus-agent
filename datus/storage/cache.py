# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel, get_embedding_model
from datus.storage.ext_knowledge import ExtKnowledgeStore
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import SchemaValueStorage
from datus.storage.scoped_filter import ScopedFilterBuilder
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=12)
def _cached_storage[
    T: BaseEmbeddingStore
](factory: Callable[[EmbeddingModel], T], storage_path: str, model_name: str) -> T:
    return factory(get_embedding_model(model_name))


# Module-level cache for scoped storage instances so that clear_cache() can
# invalidate them regardless of which StorageCache instance created them.
_scoped_storage_cache: dict[tuple[str, ...], BaseEmbeddingStore] = {}


class StorageCacheHolder[T: BaseEmbeddingStore]:
    def __init__(
        self,
        storage_factory: Callable[[EmbeddingModel], T],
        agent_config: AgentConfig,
        embedding_model_conf_name: str,
        check_scope_attr: str,
    ):
        self.storage_factory = storage_factory
        self.embedding_model_conf_name = embedding_model_conf_name
        self._agent_config = agent_config
        self.check_scope_attr = check_scope_attr

    def storage_instance(self, sub_agent_name: Optional[str] = None) -> T:
        if sub_agent_name and (config := self._agent_config.sub_agent_config(sub_agent_name)):
            sub_agent_config = SubAgentConfig.model_validate(config)
            scope_value = None
            if sub_agent_config.has_scoped_context_by(self.check_scope_attr):
                scope_value = getattr(sub_agent_config.scoped_context, self.check_scope_attr, None)
            if scope_value:
                storage_path = self._agent_config.rag_storage_path()
                scope_hash = hashlib.md5(str(scope_value).encode()).hexdigest()[:8]
                factory_name = self.storage_factory.__name__
                cache_key = (factory_name, storage_path, sub_agent_name, self.check_scope_attr, scope_hash)
                cached = _scoped_storage_cache.get(cache_key)
                if cached is not None:
                    return cached  # type: ignore[return-value]
                # Use global storage with a scope filter instead of a separate sub-agent copy
                storage = self.storage_factory(
                    get_embedding_model(self.embedding_model_conf_name),
                )
                scope_filter = self._build_scope_filter(sub_agent_config, storage)
                if scope_filter is None:
                    logger.warning(
                        "Failed to build scope filter for sub-agent '%s' (attr='%s'); "
                        "denying access to prevent unscoped reads.",
                        sub_agent_name,
                        self.check_scope_attr,
                    )
                    raise DatusException(
                        code=ErrorCode.COMMON_VALIDATION_FAILED,
                        message=(
                            f"Cannot build scope filter for sub-agent '{sub_agent_name}' "
                            f"(scope attr='{self.check_scope_attr}'). "
                            "Refusing to return unscoped storage."
                        ),
                    )
                storage._scope_filter = scope_filter
                _scoped_storage_cache[cache_key] = storage
                logger.debug(
                    f"Sub agent {sub_agent_name} has scoped context, "
                    f"using global storage with WHERE filter for '{self.check_scope_attr}'"
                )
                return storage
        return _cached_storage(
            self.storage_factory, self._agent_config.rag_storage_path(), self.embedding_model_conf_name
        )

    def _build_scope_filter(self, sub_agent_config: SubAgentConfig, storage: T):
        """Build a scope filter Node from the sub-agent's scoped context."""
        scope_value = getattr(sub_agent_config.scoped_context, self.check_scope_attr, None)
        if not scope_value:
            return None

        if self.check_scope_attr == "tables":
            dialect = getattr(self._agent_config, "db_type", "")
            return ScopedFilterBuilder.build_table_filter(scope_value, dialect)
        elif self.check_scope_attr in ("metrics", "sqls", "ext_knowledge"):
            # Subject-based stores need the subject_tree to resolve paths to node IDs
            subject_tree = getattr(storage, "subject_tree", None)
            if subject_tree is None:
                logger.warning(
                    f"Storage for '{self.check_scope_attr}' has no subject_tree; cannot build subject filter"
                )
                return None
            return ScopedFilterBuilder.build_subject_filter(scope_value, subject_tree)
        return None


class StorageCache:
    """Cache access to global and sub-agent storage instances.

    Each storage accessor accepts an optional ``sub_agent_name``. When omitted, the accessor
    returns the shared/global storage instance. When provided, a scoped storage rooted at the
    sub-agent's knowledge base path is returned (and cached for subsequent calls).
    """

    def __init__(
        self,
        agent_config: AgentConfig,
    ):
        self._agent_config = agent_config
        self._schema_holder = StorageCacheHolder(SchemaStorage, agent_config, "database", "tables")
        self._sample_data_holder = StorageCacheHolder(SchemaValueStorage, agent_config, "database", "tables")
        self._metric_holder = StorageCacheHolder(MetricStorage, agent_config, "metric", "metrics")
        self._semantic_holder = StorageCacheHolder(SemanticModelStorage, agent_config, "semantic_model", "tables")
        self._reference_sql_holder = StorageCacheHolder(ReferenceSqlStorage, agent_config, "reference_sql", "sqls")
        self._ext_knowledge_holder = StorageCacheHolder(
            ExtKnowledgeStore, agent_config, "ext_knowledge", "ext_knowledge"
        )
        self._subject_tree_store = None

    def schema_storage(self, sub_agent_name: Optional[str] = None) -> SchemaStorage:
        return self._schema_holder.storage_instance(sub_agent_name)

    def schema_value_storage(self, sub_agent_name: Optional[str] = None) -> SchemaValueStorage:
        return self._sample_data_holder.storage_instance(sub_agent_name)

    def metric_storage(self, sub_agent_name: Optional[str] = None) -> MetricStorage:
        """Access dedicated MetricStorage for metrics only."""
        return self._metric_holder.storage_instance(sub_agent_name)

    def semantic_storage(self, sub_agent_name: Optional[str] = None) -> SemanticModelStorage:
        return self._semantic_holder.storage_instance(sub_agent_name)

    def reference_sql_storage(self, sub_agent_name: Optional[str] = None) -> ReferenceSqlStorage:
        return self._reference_sql_holder.storage_instance(sub_agent_name)

    def ext_knowledge_storage(self, sub_agent_name: Optional[str] = None) -> ExtKnowledgeStore:
        return self._ext_knowledge_holder.storage_instance(sub_agent_name)


_CACHE_INSTANCE = None


def get_storage_cache_instance(agent_config: AgentConfig) -> StorageCache:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = StorageCache(agent_config)
    return _CACHE_INSTANCE


def clear_cache():
    _cached_storage.cache_clear()
    _scoped_storage_cache.clear()
    global _CACHE_INSTANCE
    _CACHE_INSTANCE = None

    from datus.storage.backend_holder import reset_backends

    reset_backends()
