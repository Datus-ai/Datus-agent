# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Set

from datus.storage.ext_knowledge.store import ExtKnowledgeStore


def exists_ext_knowledge(storage: ExtKnowledgeStore, build_mode: str = "overwrite") -> Set[str]:
    """Get existing external knowledge IDs based on build mode.

    Args:
        storage: ExtKnowledgeStore instance
        build_mode: "overwrite" to ignore existing data, "incremental" to check existing

    Returns:
        Set of existing knowledge IDs
    """
    existing_knowledge = set()
    if build_mode == "overwrite":
        return existing_knowledge

    if build_mode == "incremental":
        # Get all existing knowledge entries to avoid duplicates
        knowledge_list = storage.search_all_knowledge()

        for item in knowledge_list:
            subject_path = item.get("subject_path", [])
            search_text = item.get("search_text", "")
            knowledge_id = gen_ext_knowledge_id(subject_path, search_text)
            existing_knowledge.add(knowledge_id)

    return existing_knowledge


def gen_ext_knowledge_id(subject_path: list, search_text: str) -> str:
    """Generate unique ID for external knowledge entry.

    Args:
        subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
        search_text: Business search_text/concept

    Returns:
        Unique knowledge ID
    """
    # Clean inputs to avoid issues with special characters
    clean_path_parts = [part.replace("/", "_") for part in subject_path]
    clean_search_text = search_text.replace("/", "_")

    path_str = "/".join(clean_path_parts) if clean_path_parts else ""
    return f"{path_str}/{clean_search_text}" if path_str else clean_search_text
