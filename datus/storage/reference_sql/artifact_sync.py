# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Sync a reference SQL YAML artifact file into the Knowledge Base.

The YAML file is the source of truth: re-syncing an artifact with an existing
business ID updates the stored row instead of skipping it. Incremental /
overwrite bootstrap semantics belong to the bootstrap orchestration layer, not
here.
"""

import yaml

from datus.configuration.agent_config import AgentConfig
from datus.storage.reference_sql.init_utils import gen_reference_sql_id
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def sync_reference_sql_artifact_to_kb(file_path: str, agent_config: AgentConfig) -> dict:
    """
    Sync a reference SQL YAML file to the Knowledge Base.

    Returns a dict with ``success`` and either ``message`` or ``error``.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        if isinstance(doc, dict) and "sql" in doc:
            # Direct format without reference_sql wrapper
            reference_sql_data = doc
        else:
            return {"success": False, "error": "No reference_sql data found in YAML file"}

        # Generate ID if not present or if it's a placeholder
        sql_query = reference_sql_data.get("sql", "")
        comment = reference_sql_data.get("comment", "")
        item_id = reference_sql_data.get("id", "")

        if not item_id or item_id == "auto_generated":
            item_id = gen_reference_sql_id(sql_query)
            reference_sql_data["id"] = item_id

        # Parse subject_tree format: "path/component1/component2/..."
        subject_path = []
        subject_tree_str = reference_sql_data.get("subject_tree", "")
        if subject_tree_str:
            parts = subject_tree_str.split("/")
            subject_path = [part.strip() for part in parts if part.strip()]

        reference_sql_dict = {
            "id": item_id,
            "name": reference_sql_data.get("name", ""),
            "sql": sql_query,
            "comment": comment,
            "summary": reference_sql_data.get("summary", ""),
            "search_text": reference_sql_data.get("search_text", ""),
            "filepath": file_path,
            "subject_path": subject_path,
            "tags": reference_sql_data.get("tags", ""),
        }

        storage = ReferenceSqlRAG(agent_config)
        storage.upsert_batch([reference_sql_dict])

        logger.info(f"Successfully synced reference SQL {item_id} to Knowledge Base")
        return {"success": True, "message": f"Synced reference SQL: {reference_sql_dict['name']}"}

    except Exception as e:
        logger.error(f"Error syncing reference SQL to DB: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
