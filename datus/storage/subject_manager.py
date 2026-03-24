# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Subject.

Sub-agents with scoped context now query the main (global) storage directly
via WHERE filters, so updates to the main storage are automatically visible
to all sub-agents.  No per-sub-agent propagation is needed.
"""

from typing import Any, Dict, List

from datus.configuration.agent_config import AgentConfig
from datus.storage.ext_knowledge import ExtKnowledgeStore
from datus.storage.metric import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.registry import get_storage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubjectUpdater:
    """Used to update all subject data.

    Since sub-agents now use the shared global storage with scope filters,
    any mutation applied to the main storage is automatically reflected in
    sub-agent queries.
    """

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self.metrics_storage: MetricStorage = get_storage(MetricStorage, "metric", agent_config.current_namespace)
        self.reference_sql_storage: ReferenceSqlStorage = get_storage(
            ReferenceSqlStorage, "reference_sql", agent_config.current_namespace
        )
        self.ext_knowledge_storage: ExtKnowledgeStore = get_storage(
            ExtKnowledgeStore, "ext_knowledge", agent_config.current_namespace
        )

    def update_metrics_detail(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        """Update metrics detail fields using subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metrics entry
            update_values: Dictionary of fields to update (excluding subject_node_id and name)
        """
        if not update_values:
            return
        self.metrics_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the metrics details in the main space successfully")

    def update_historical_sql(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        """Update reference SQL detail fields using subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the SQL entry
            update_values: Dictionary of fields to update (excluding subject_node_id and name)
        """
        if not update_values:
            return
        self.reference_sql_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the reference SQL details in the main space successfully")

    def update_ext_knowledge(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        """Update external knowledge detail fields using subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the ext_knowledge entry
            update_values: Dictionary of fields to update (excluding subject_node_id and name)
        """
        if not update_values:
            return
        self.ext_knowledge_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the ext_knowledge details in the main space successfully")

    def delete_metric(self, subject_path: List[str], name: str) -> Dict[str, Any]:
        """Delete metric by subject_path and name from main storage.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metric to delete

        Returns:
            Dict with 'success', 'message', and optional 'yaml_updated' fields from main storage
        """
        return self.metrics_storage.delete_metric(subject_path, name)

    def delete_reference_sql(self, subject_path: List[str], name: str) -> bool:
        """Delete reference SQL by subject_path and name from main storage.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Analytics', 'Reports'])
            name: Name of the reference SQL to delete

        Returns:
            True if deleted successfully from main storage, False if entry not found
        """
        return self.reference_sql_storage.delete_reference_sql(subject_path, name)

    def delete_ext_knowledge(self, subject_path: List[str], name: str) -> bool:
        """Delete ext_knowledge by subject_path and name from main storage.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Business', 'Terms'])
            name: Name of the knowledge entry to delete

        Returns:
            True if deleted successfully from main storage, False if entry not found
        """
        return self.ext_knowledge_storage.delete_knowledge(subject_path, name)
