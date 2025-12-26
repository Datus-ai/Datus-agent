# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage import BaseEmbeddingStore
from datus.storage.document import DocumentStore
from datus.storage.embedding_models import EmbeddingModel, get_embedding_model
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.schema_metadata.store import SchemaValueStorage
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=12)
def _cached_storage[
    T: BaseEmbeddingStore
](factory: Callable[[str, EmbeddingModel], T], path: str, model_name: str) -> T:
    return factory(path, get_embedding_model(model_name))


class StorageCacheHolder[T: BaseEmbeddingStore]:
    def __init__(
        self,
        storage_factory: Callable[[str, EmbeddingModel], T],
        agent_config: AgentConfig,
        storage_name: str,
        check_scope_attr: str,
    ):
        self.storage_factory = storage_factory
        self.storage_name = storage_name
        self._agent_config = agent_config
        self.check_scope_attr = check_scope_attr

    def storage_instance(self, sub_agent_name: Optional[str] = None) -> T:
        if sub_agent_name and (config := self._agent_config.sub_agent_config(sub_agent_name)):
            sub_agent_config = SubAgentConfig.model_validate(config)
            if sub_agent_config.has_scoped_context_by(self.check_scope_attr) and getattr(
                sub_agent_config.scoped_context, self.check_scope_attr
            ):
                logger.debug(
                    (
                        f"Sub agent {sub_agent_name} has a scope context,"
                        f"so use {self._agent_config.sub_agent_storage_path(sub_agent_name)} for LanceDB"
                    )
                )
                return self.storage_factory(
                    self._agent_config.sub_agent_storage_path(sub_agent_name), get_embedding_model(self.storage_name)
                )
        return _cached_storage(self.storage_factory, self._agent_config.rag_storage_path(), self.storage_name)


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
        self._semantic_holder = StorageCacheHolder(SemanticModelStorage, agent_config, "metric", "semantic_models")
        self._reference_sql_holder = StorageCacheHolder(ReferenceSqlStorage, agent_config, "metric", "sqls")
        self._document_holder = StorageCacheHolder(DocumentStore, agent_config, "document", "")
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

    def document_storage(self, sub_agent_name: Optional[str] = None) -> DocumentStore:
        return self._document_holder.storage_instance(sub_agent_name)


_CACHE_INSTANCE = None


def get_storage_cache_instance(agent_config: AgentConfig) -> StorageCache:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = StorageCache(agent_config)
    return _CACHE_INSTANCE


def clear_cache():
    _cached_storage.cache_clear()
    global _CACHE_INSTANCE
    _CACHE_INSTANCE = None
