# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Subject
"""
from typing import Any, Dict, List

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.cache import get_storage_cache_instance
from datus.storage.metric import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubjectUpdater:
    """Used to update all subject data, including vector databases specific to Sub-Agents."""

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self.storage_cache = get_storage_cache_instance(self._agent_config)
        self.metrics_storage: MetricStorage = self.storage_cache.metric_storage()
        self.reference_sql_storage: ReferenceSqlStorage = self.storage_cache.reference_sql_storage()

    def _sub_agent_storage_metrics(self, sub_agent_config: SubAgentConfig) -> MetricStorage:
        name = sub_agent_config.system_prompt

        return self.storage_cache.metric_storage(name)

    def _sub_agent_storage_sql(self, sub_agent_config: SubAgentConfig) -> ReferenceSqlStorage:
        name = sub_agent_config.system_prompt
        return self.storage_cache.reference_sql_storage(name)

    def update_metrics_detail(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        """Update metrics detail fields using subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metrics entry
            update_values: Dictionary of fields to update (excluding subject_node_id and name)
        """
        if not update_values:
            return

        # Update in main storage
        self.metrics_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the metrics details in the main space successfully")

        # Update in sub-agent storages
        for sub_agent_name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_metrics(sub_agent_config).update_entry(subject_path, name, update_values)
                    logger.debug(f"Updated the metrics details in the sub_agent `{sub_agent_name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to update the metrics details in the sub_agent `{sub_agent_name}`: {e}")

    def update_historical_sql(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        """Update reference SQL detail fields using subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the SQL entry
            update_values: Dictionary of fields to update (excluding subject_node_id and name)
        """
        if not update_values:
            return

        # Update in main storage
        self.reference_sql_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the reference SQL details in the main space successfully")

        # Update in sub-agent storages
        for sub_agent_name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_sql(sub_agent_config).update_entry(subject_path, name, update_values)
                    logger.debug(f"Updated the reference SQL details in the sub_agent `{sub_agent_name}` successfully")
                except Exception as e:
                    logger.warning(
                        f"Failed to update the reference SQL details in the sub_agent `{sub_agent_name}`: {e}"
                    )
