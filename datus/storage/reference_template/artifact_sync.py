# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Sync a reference template YAML artifact file into the Knowledge Base.

The YAML file is the source of truth: re-syncing an artifact with an existing
business ID updates the stored row instead of skipping it. Incremental /
overwrite bootstrap semantics belong to the bootstrap orchestration layer, not
here.
"""

import json

import yaml

from datus.configuration.agent_config import AgentConfig
from datus.storage.reference_template.init_utils import gen_reference_template_id
from datus.storage.reference_template.store import ReferenceTemplateRAG
from datus.storage.reference_template.template_file_processor import extract_template_parameters
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def sync_reference_template_artifact_to_kb(file_path: str, agent_config: AgentConfig) -> dict:
    """
    Sync a reference template YAML file to the Knowledge Base.

    Returns a dict with ``success`` and either ``message`` or ``error``.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        if isinstance(doc, dict) and "sql" in doc:
            reference_template_data = doc
        else:
            return {"success": False, "error": "No reference_template data found in YAML file"}

        # The template content is stored in the "sql" field by SqlSummaryAgenticNode
        template_content = reference_template_data.get("sql", "")
        if not isinstance(template_content, str) or not template_content.strip():
            return {"success": False, "error": "Reference template 'sql' must be a non-empty string"}
        comment = reference_template_data.get("comment", "")
        item_id = reference_template_data.get("id", "")

        if not item_id or item_id == "auto_generated":
            item_id = gen_reference_template_id(template_content)
            reference_template_data["id"] = item_id

        subject_path = []
        subject_tree_str = reference_template_data.get("subject_tree", "")
        if subject_tree_str:
            parts = subject_tree_str.split("/")
            subject_path = [part.strip() for part in parts if part.strip()]

        parameters = extract_template_parameters(template_content)

        reference_template_dict = {
            "id": item_id,
            "name": reference_template_data.get("name", ""),
            "template": template_content,
            "parameters": json.dumps(parameters),
            "comment": comment,
            "summary": reference_template_data.get("summary", ""),
            "search_text": reference_template_data.get("search_text", ""),
            "filepath": file_path,
            "subject_path": subject_path,
            "tags": reference_template_data.get("tags", ""),
        }

        storage = ReferenceTemplateRAG(agent_config)
        storage.upsert_batch([reference_template_dict])

        logger.info(f"Successfully synced reference template {item_id} to Knowledge Base")
        return {"success": True, "message": f"Synced reference template: {reference_template_dict['name']}"}

    except Exception as e:
        logger.error(f"Error syncing reference template to DB: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
