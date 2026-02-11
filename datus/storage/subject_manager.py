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
from datus.storage.ext_knowledge import ExtKnowledgeStore
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
        self.ext_knowledge_storage: ExtKnowledgeStore = self.storage_cache.ext_knowledge_storage()

    def _sub_agent_storage_metrics(self, sub_agent_config: SubAgentConfig) -> MetricStorage:
        name = sub_agent_config.system_prompt

        return self.storage_cache.metric_storage(name)

    def _sub_agent_storage_sql(self, sub_agent_config: SubAgentConfig) -> ReferenceSqlStorage:
        name = sub_agent_config.system_prompt
        return self.storage_cache.reference_sql_storage(name)

    def _sub_agent_storage_ext_knowledge(self, sub_agent_config: SubAgentConfig) -> ExtKnowledgeStore:
        name = sub_agent_config.system_prompt
        return self.storage_cache.ext_knowledge_storage(name)

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

    def update_ext_knowledge(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        """Update external knowledge detail fields using subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the ext_knowledge entry
            update_values: Dictionary of fields to update (excluding subject_node_id and name)
        """
        if not update_values:
            return

        # Update in main storage
        self.ext_knowledge_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the ext_knowledge details in the main space successfully")

        # Update in sub-agent storages
        for sub_agent_name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_ext_knowledge(sub_agent_config).update_entry(
                        subject_path, name, update_values
                    )
                    logger.debug(f"Updated the ext_knowledge details in the sub_agent `{sub_agent_name}` successfully")
                except Exception as e:
                    logger.warning(
                        f"Failed to update the ext_knowledge details in the sub_agent `{sub_agent_name}`: {e}"
                    )

    def delete_metric(self, subject_path: List[str], name: str) -> Dict[str, Any]:
        """Delete metric by subject_path and name from main storage and all sub-agent storages.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metric to delete

        Returns:
            Dict with 'success', 'message', and optional 'yaml_updated' fields from main storage
        """
        # Delete from main storage (handles both lancedb and yaml cleanup)
        result = self.metrics_storage.delete_metric(subject_path, name)

        # Delete from sub-agent storages
        for sub_agent_name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_metrics(sub_agent_config).delete_entry(subject_path, name)
                    logger.debug(f"Deleted the metric in the sub_agent `{sub_agent_name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to delete the metric in the sub_agent `{sub_agent_name}`: {e}")

        return result

    def delete_reference_sql(self, subject_path: List[str], name: str) -> bool:
        """Delete reference SQL by subject_path and name from main storage and all sub-agent storages.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Analytics', 'Reports'])
            name: Name of the reference SQL to delete

        Returns:
            True if deleted successfully from main storage, False if entry not found
        """
        # Delete from main storage
        deleted = self.reference_sql_storage.delete_reference_sql(subject_path, name)

        # Delete from sub-agent storages
        for sub_agent_name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_sql(sub_agent_config).delete_reference_sql(subject_path, name)
                    logger.debug(f"Deleted the reference SQL in the sub_agent `{sub_agent_name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to delete the reference SQL in the sub_agent `{sub_agent_name}`: {e}")

        return deleted

    def delete_ext_knowledge(self, subject_path: List[str], name: str) -> bool:
        """Delete ext_knowledge by subject_path and name from main storage and all sub-agent storages.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Business', 'Terms'])
            name: Name of the knowledge entry to delete

        Returns:
            True if deleted successfully from main storage, False if entry not found
        """
        # Delete from main storage
        deleted = self.ext_knowledge_storage.delete_knowledge(subject_path, name)

        # Delete from sub-agent storages
        for sub_agent_name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_ext_knowledge(sub_agent_config).delete_knowledge(subject_path, name)
                    logger.debug(f"Deleted the ext_knowledge in the sub_agent `{sub_agent_name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to delete the ext_knowledge in the sub_agent `{sub_agent_name}`: {e}")

        return deleted
