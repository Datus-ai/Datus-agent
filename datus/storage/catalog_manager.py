# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Catalog
"""
from typing import Any, Dict

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.cache import get_storage_cache_instance
from datus.storage.lancedb_conditions import And, eq
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CatalogUpdater:
    """
    Used to update all catalog data, including vector databases specific to Sub-Agents.
    """

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self.semantic_model_storage = get_storage_cache_instance(agent_config).semantic_storage()

    def _sub_agent_storage(self, sub_agent_config: SubAgentConfig) -> SemanticModelStorage | None:
        name = sub_agent_config.system_prompt
        return get_storage_cache_instance(self._agent_config).semantic_storage(name)

    def update_semantic_model(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        where = And(
            [
                eq("catalog_name", old_values["catalog_name"]),
                eq("database_name", old_values["database_name"]),
                eq("schema_name", old_values["schema_name"]),
                eq("table_name", old_values["table_name"]),
                eq("semantic_model_name", old_values["semantic_model_name"]),
            ]
        )

        self.semantic_model_storage.update(where, update_values, unique_filter=None)
        logger.debug("Updated the semantic model in the main space successfully")
        for name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if (
                sub_agent_config.is_in_namespace(self._agent_config.current_namespace)
                and sub_agent_config.has_scoped_context()
            ):
                self._sub_agent_storage(sub_agent_config).update(where, update_values, unique_filter=None)
                logger.debug(f"Updated the semantic model in the sub_agent `{name}` successfully")
