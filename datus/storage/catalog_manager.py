# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Catalog
"""
import json
from typing import Any, Dict, List, Optional

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

    def _get_all_storages(self) -> List[SemanticModelStorage]:
        """Get main storage and all sub-agent storages."""
        storages = [self.semantic_model_storage]
        for _, value in self._agent_config.agentic_nodes.items():
            sub_agent_config = SubAgentConfig.model_validate(value)
            if (
                sub_agent_config.is_in_namespace(self._agent_config.current_namespace)
                and sub_agent_config.has_scoped_context()
            ):
                sub_storage = self._sub_agent_storage(sub_agent_config)
                if sub_storage:
                    storages.append(sub_storage)
        return storages

    def _parse_json_field(self, value: Any) -> Optional[List[Dict[str, Any]]]:
        """Parse JSON string or return list directly."""
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else None
            except json.JSONDecodeError:
                return None
        return None

    def update_semantic_model(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        catalog_name = old_values.get("catalog_name", "")
        database_name = old_values.get("database_name", "")
        schema_name = old_values.get("schema_name", "")
        table_name = old_values.get("table_name", "")
        semantic_model_name = old_values.get("semantic_model_name", "")

        storages = self._get_all_storages()

        # 1. Update table-level record (description)
        if "description" in update_values:
            table_where = And(
                [
                    eq("kind", "table"),
                    eq("catalog_name", catalog_name),
                    eq("database_name", database_name),
                    eq("schema_name", schema_name),
                    eq("table_name", table_name),
                    eq("name", semantic_model_name),
                ]
            )
            table_update = {"description": update_values["description"]}
            for storage in storages:
                storage.update(table_where, table_update, unique_filter=None)
            logger.debug("Updated table-level semantic model description")

        # 2. Update column-level records (dimensions, measures, identifiers)
        self._update_columns(
            storages,
            catalog_name,
            database_name,
            schema_name,
            table_name,
            old_values.get("dimensions"),
            update_values.get("dimensions"),
            "is_dimension",
            {"description", "expr", "column_type", "is_partition", "time_granularity"},
        )
        self._update_columns(
            storages,
            catalog_name,
            database_name,
            schema_name,
            table_name,
            old_values.get("measures"),
            update_values.get("measures"),
            "is_measure",
            {"description", "expr", "agg", "create_metric", "agg_time_dimension"},
        )
        self._update_columns(
            storages,
            catalog_name,
            database_name,
            schema_name,
            table_name,
            old_values.get("identifiers"),
            update_values.get("identifiers"),
            "is_entity_key",
            {"description", "expr", "column_type", "entity"},
        )

    def _update_columns(
        self,
        storages: List[SemanticModelStorage],
        catalog_name: str,
        database_name: str,
        schema_name: str,
        table_name: str,
        old_columns: Any,
        new_columns: Any,
        kind_field: str,
        allowed_fields: set,
    ):
        """Update column-level records by matching old and new values."""
        old_list = self._parse_json_field(old_columns) or []
        new_list = self._parse_json_field(new_columns) or []

        # Build lookup by name for old values
        old_by_name = {item.get("name"): item for item in old_list if item.get("name")}

        for new_item in new_list:
            col_name = new_item.get("name")
            if not col_name:
                continue

            old_item = old_by_name.get(col_name, {})

            # Compute changed fields (only allowed fields)
            changed = {}
            for field in allowed_fields:
                # Map 'type' field to 'column_type' in storage
                new_key = field
                old_key = "type" if field == "column_type" else field
                new_val = new_item.get("type" if field == "column_type" else field)
                old_val = old_item.get(old_key)
                if new_val != old_val:
                    changed[new_key] = new_val

            if not changed:
                continue

            # Build where clause for this column
            col_where = And(
                [
                    eq("kind", "column"),
                    eq(kind_field, True),
                    eq("catalog_name", catalog_name),
                    eq("database_name", database_name),
                    eq("schema_name", schema_name),
                    eq("table_name", table_name),
                    eq("name", col_name),
                ]
            )

            for storage in storages:
                storage.update(col_where, changed, unique_filter=None)
            logger.debug(f"Updated column '{col_name}' ({kind_field}): {list(changed.keys())}")
