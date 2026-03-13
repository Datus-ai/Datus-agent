#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Prepare dynamic context for SQL summary generation.

Queries vector store for existing subject trees and similar reference SQLs.
Configuration is read from environment variables injected by SkillBashTool.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _skill_common import build_agent_config  # noqa: E402


def get_existing_subject_trees(agent_config):
    """Query existing subject_path values from reference SQL storage."""
    try:
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        rag = ReferenceSqlRAG(agent_config)
        return sorted(rag.reference_sql_storage.get_subject_tree_flat())
    except Exception as e:
        print(f"Warning: Failed to get subject trees: {e}", file=sys.stderr)
        return []


def get_similar_sqls(agent_config, query_text, top_n=5):
    """Find similar reference SQLs based on query text."""
    try:
        if not query_text:
            return []

        from datus.storage.reference_sql.store import ReferenceSqlRAG

        rag = ReferenceSqlRAG(agent_config)
        similar_items = rag.search_reference_sql(query_text=query_text, top_n=top_n)

        results = []
        for item in similar_items:
            subject_path = item.get("subject_path", [])
            subject_tree = "/".join(subject_path) if subject_path else ""
            results.append(
                {
                    "name": item.get("name", ""),
                    "subject_tree": subject_tree,
                    "tags": item.get("tags", ""),
                    "comment": item.get("comment", ""),
                    "summary": item.get("summary", ""),
                }
            )
        return results
    except Exception as e:
        print(f"Warning: Failed to get similar SQLs: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare context for SQL summary generation")
    parser.add_argument("--sql-query", type=str, default="", help="SQL query (first 200 chars for similarity search)")
    parser.add_argument("--subject-tree", type=str, default="", help="Predefined subject tree categories (JSON list)")
    args = parser.parse_args()

    agent_config = build_agent_config()

    # Resolve sql_summary_dir
    from datus.utils.path_manager import get_path_manager

    path_manager = get_path_manager(agent_config.home)
    sql_summary_dir = str(path_manager.sql_summary_path(agent_config.current_namespace))

    # Handle subject_tree
    has_subject_tree = False
    subject_tree = []
    if args.subject_tree:
        try:
            subject_tree = json.loads(args.subject_tree)
            has_subject_tree = bool(subject_tree)
        except json.JSONDecodeError:
            pass

    # Query existing subject trees (only if no predefined tree)
    existing_subject_trees = []
    if not has_subject_tree:
        existing_subject_trees = get_existing_subject_trees(agent_config)

    # Query similar SQLs
    query_text = args.sql_query[:200] if args.sql_query else ""
    similar_items = get_similar_sqls(agent_config, query_text)

    # Output context as JSON
    context = {
        "sql_summary_dir": sql_summary_dir,
        "has_subject_tree": has_subject_tree,
        "subject_tree": subject_tree,
        "existing_subject_trees": existing_subject_trees,
        "similar_items": similar_items,
    }
    print(json.dumps(context, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
