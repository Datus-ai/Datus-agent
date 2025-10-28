# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Subject
"""
from typing import Any, Dict, Optional, Set

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.cache import get_storage_cache_instance
from datus.storage.lancedb_conditions import And, WhereExpr, eq
from datus.storage.metric import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubjectUpdater:
    """Used to update all subject data, including vector databases specific to Sub-Agents."""

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self.storage_cache = get_storage_cache_instance(self._agent_config)
        self.metrics_storage: MetricStorage = self.storage_cache.metrics_storage()
        self.reference_sql_storage: ReferenceSqlStorage = self.storage_cache.reference_sql_storage()

    def _sub_agent_storage_metrics(self, sub_agent_config: SubAgentConfig) -> MetricStorage:
        name = sub_agent_config.system_prompt

        return self.storage_cache.metrics_storage(name)

    def _sub_agent_storage_sql(self, sub_agent_config: SubAgentConfig) -> ReferenceSqlStorage:
        name = sub_agent_config.system_prompt
        return self.storage_cache.reference_sql_storage(name)

    def update_domain_layers(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        """
        Only update the fields of the domain layer, including domain, layer1, layer2, name.
        This method updates both metrics and reference SQL.
        """
        if "name" in update_values:
            unique_filter = And(
                [
                    eq("domain", update_values.get("domain", old_values.get("domain"))),
                    eq("layer1", update_values.get("layer1", old_values.get("layer1"))),
                    eq("layer2", update_values.get("layer2", old_values.get("layer2"))),
                    eq("name", update_values["name"]),
                ]
            )
        else:
            unique_filter = None
        must_check_keys = set()
        for k in ("domain", "layer1", "layer2", "name"):
            if k in update_values:
                must_check_keys.add(k)
        where = self._build_where(old_values, must_check_keys)
        if not where:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={
                    "operation": "update",
                    "table_name": self.metrics_storage.table_name,
                    "error_message": "Missing WHERE for metrics update",
                },
            )
        self.metrics_storage.update(where, update_values, unique_filter=unique_filter)
        logger.debug("Updated the domain layers of matrics in the main space successfully")
        self.reference_sql_storage.update(where, update_values, unique_filter=unique_filter)
        logger.debug("Updated the domain layers of reference SQL in the main space successfully")
        for name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_metrics(sub_agent_config).update(
                        where, update_values, unique_filter=unique_filter
                    )
                    logger.debug(f"Updated the domain layers of metrics in the sub_agent `{name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to update the domain layers of metrics in the sub_agent `{name}`: {e}")
                try:
                    self._sub_agent_storage_sql(sub_agent_config).update(
                        where, update_values, unique_filter=unique_filter
                    )
                except Exception as e:
                    logger.warning(f"Failed to update the domain layers of sql in the sub_agent `{name}`: {e}")

    def _build_where(self, old_values: Dict[str, Any], must_check_key: Optional[Set[str]] = None) -> WhereExpr:
        where_conditions = []
        for k in ("domain", "layer1", "layer2", "name"):
            v = old_values.get(k)
            if v:
                where_conditions.append(eq(k, v))
            elif must_check_key and k in must_check_key:
                raise DatusException(ErrorCode.COMMON_VALIDATION_FAILED, f"Field {k} is required")
        if not where_conditions:
            return None
        return And(where_conditions)

    def _clean_domain_layers(self, update_values: Dict[str, Any]) -> Dict[str, Any]:
        if not update_values:
            return update_values
        for k in ["domain", "layer1", "layer2", "name"]:
            if k in update_values:
                update_values.pop(k)
        return update_values

    def update_metrics_detail(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        """Only the details field is updated, domain, layer 1, layer 2, and name are not updated."""
        if not update_values:
            return
        cleaned_update_payload = self._clean_domain_layers(update_values)
        where = self._build_where(old_values, {"domain", "layer1", "layer2", "name"})
        if not where:
            return
        self.metrics_storage.update(where, cleaned_update_payload, unique_filter=None)
        logger.debug("Updated the metrics details in the main space successfully")

        for name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_metrics(sub_agent_config).update(where, update_values, unique_filter=None)
                    logger.debug(f"Updated the metrics details in the sub_agent `{name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to update the metrics details in the sub_agent `{name}`: {e}")

    def update_historical_sql(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        """Only the details field is updated, domain, layer 1, layer 2, and name are not updated."""
        if not update_values:
            return
        cleaned_update_payload = self._clean_domain_layers(update_values)
        where = self._build_where(old_values, {"domain", "layer1", "layer2", "name"})
        if not where:
            return
        self.reference_sql_storage.update(where, cleaned_update_payload)
        logger.debug("Updated the reference SQL details in the main space successfully")
        for name, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if sub_agent_config.is_in_namespace(self._agent_config.current_namespace):
                try:
                    self._sub_agent_storage_sql(sub_agent_config).update(where, update_values, unique_filter=None)
                    logger.debug(f"Updated the reference SQL details in the sub_agent `{name}` successfully")
                except Exception as e:
                    logger.warning(f"Failed to update the reference SQL details in the sub_agent `{name}`: {e}")
