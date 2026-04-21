# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Used to manage editing operations related to Catalog
"""

import json
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.storage.registry import get_storage
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CatalogUpdater:
    """
    Used to update all catalog data, including vector databases specific to Sub-Agents.
    """

    def __init__(self, agent_config: AgentConfig, datasource_id: Optional[str] = None):
        self._agent_config = agent_config
        self.datasource_id = datasource_id or agent_config.current_datasource or ""
        self.semantic_model_storage = get_storage(
            SemanticModelStorage, "semantic_model", project=agent_config.project_name
        )

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
        table_name = old_values.get("table_name", "")
        semantic_model_name = old_values.get("semantic_model_name", "")

        # 1. Update table-level record (description)
        if "description" in update_values:
            entry_id = f"table:{table_name}"
            try:
                self.semantic_model_storage.update_entry(entry_id, {"description": update_values["description"]})
            except DatusException as e:
                if e.code == ErrorCode.STORAGE_ENTRY_NOT_FOUND:
                    logger.warning(f"Table entry not found: {entry_id}")
                else:
                    raise
            else:
                logger.debug("Updated table-level semantic model description")

        # 2. Update column-level records (dimensions, measures, identifiers)
        self._update_columns(
            table_name,
            semantic_model_name,
            old_values.get("dimensions"),
            update_values.get("dimensions"),
            "is_dimension",
            {"description", "expr", "column_type", "is_partition", "time_granularity"},
        )
        self._update_columns(
            table_name,
            semantic_model_name,
            old_values.get("measures"),
            update_values.get("measures"),
            "is_measure",
            {"description", "expr", "agg", "create_metric", "agg_time_dimension"},
        )
        self._update_columns(
            table_name,
            semantic_model_name,
            old_values.get("identifiers"),
            update_values.get("identifiers"),
            "is_entity_key",
            {"description", "expr", "column_type", "entity"},
        )

    def _update_columns(
        self,
        table_name: str,
        semantic_model_name: str,
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
                old_key = "type" if field == "column_type" else field
                new_val = new_item.get("type" if field == "column_type" else field)
                old_val = old_item.get(old_key)
                if new_val != old_val:
                    changed[field] = new_val

            if not changed:
                continue

            entry_id = f"column:{table_name}.{col_name}"
            try:
                self.semantic_model_storage.update_entry(entry_id, changed)
            except DatusException as e:
                if e.code == ErrorCode.STORAGE_ENTRY_NOT_FOUND:
                    logger.warning(f"Column entry not found: {entry_id}")
                else:
                    raise
            else:
                logger.debug(f"Updated column '{col_name}' ({kind_field}): {list(changed.keys())}")
