#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Prepare dynamic context for external knowledge generation.

Queries subject trees and resolves the ext knowledge directory.
Configuration is read from environment variables injected by SkillBashTool.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _skill_common import build_agent_config  # noqa: E402


def get_existing_subject_trees(agent_config):
    """Query existing subject_tree values from ext knowledge storage."""
    try:
        from datus.storage.ext_knowledge.store import ExtKnowledgeRAG

        rag = ExtKnowledgeRAG(agent_config)
        return sorted(rag.store.get_subject_tree_flat())
    except Exception as e:
        print(f"Warning: Failed to get subject trees: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare context for ext knowledge generation")
    parser.add_argument("--subject-tree", type=str, default="", help="Predefined subject tree categories (JSON list)")
    args = parser.parse_args()

    agent_config = build_agent_config()

    from datus.utils.path_manager import get_path_manager

    path_manager = get_path_manager(agent_config.home)
    ext_knowledge_dir = str(path_manager.ext_knowledge_path(agent_config.current_namespace))

    has_subject_tree = False
    subject_tree = []
    if args.subject_tree:
        try:
            subject_tree = json.loads(args.subject_tree)
            has_subject_tree = bool(subject_tree)
        except json.JSONDecodeError:
            pass

    existing_subject_trees = []
    if not has_subject_tree:
        existing_subject_trees = get_existing_subject_trees(agent_config)

    context = {
        "ext_knowledge_dir": ext_knowledge_dir,
        "has_subject_tree": has_subject_tree,
        "subject_tree": subject_tree,
        "existing_subject_trees": existing_subject_trees,
    }
    print(json.dumps(context, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
