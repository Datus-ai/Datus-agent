# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Subject.

All mutations are scoped to the active datasource_id to prevent
cross-datasource interference in multi-tenant setups.
"""

from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.storage.ext_knowledge import ExtKnowledgeStore
from datus.storage.metric import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.registry import get_storage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubjectUpdater:
    """Used to update all subject data, scoped by datasource_id."""

    def __init__(self, agent_config: AgentConfig, datasource_id: Optional[str] = None):
        self._agent_config = agent_config
        self.datasource_id = datasource_id or agent_config.current_database or ""
        self.metrics_storage: MetricStorage = get_storage(MetricStorage, "metric", project=agent_config.project_name)
        self.reference_sql_storage: ReferenceSqlStorage = get_storage(
            ReferenceSqlStorage, "reference_sql", project=agent_config.project_name
        )
        self.ext_knowledge_storage: ExtKnowledgeStore = get_storage(
            ExtKnowledgeStore, "ext_knowledge", project=agent_config.project_name
        )

    def update_metrics_detail(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        if not update_values:
            return
        self.metrics_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the metrics details in the main space successfully")

    def update_historical_sql(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        if not update_values:
            return
        self.reference_sql_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the reference SQL details in the main space successfully")

    def update_ext_knowledge(self, subject_path: List[str], name: str, update_values: Dict[str, Any]):
        if not update_values:
            return
        self.ext_knowledge_storage.update_entry(subject_path, name, update_values)
        logger.debug("Updated the ext_knowledge details in the main space successfully")

    def delete_metric(self, subject_path: List[str], name: str) -> Dict[str, Any]:
        return self.metrics_storage.delete_metric(subject_path, name)

    def delete_reference_sql(self, subject_path: List[str], name: str) -> bool:
        return self.reference_sql_storage.delete_reference_sql(subject_path, name)

    def delete_ext_knowledge(self, subject_path: List[str], name: str) -> bool:
        return self.ext_knowledge_storage.delete_knowledge(subject_path, name)
