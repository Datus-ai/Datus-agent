#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Prepare dynamic context for metrics generation.

Queries subject trees and resolves the semantic model directory.
Configuration is read from environment variables injected by SkillBashTool.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _skill_common import build_agent_config  # noqa: E402


def get_subject_tree(agent_config, subject_tree_json):
    """Get predefined subject tree if provided."""
    if subject_tree_json:
        try:
            tree = json.loads(subject_tree_json)
            if tree:
                return True, tree
        except json.JSONDecodeError:
            pass
    return False, []


def get_existing_subject_trees(agent_config):
    """Query existing subject_tree values from metric storage."""
    try:
        from datus.storage.metric.store import MetricRAG

        rag = MetricRAG(agent_config)
        return sorted(rag.storage.get_subject_tree_flat())
    except Exception as e:
        print(f"Warning: Failed to get subject trees: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare context for metrics generation")
    parser.add_argument("--subject-tree", type=str, default="", help="Predefined subject tree categories (JSON list)")
    args = parser.parse_args()

    agent_config = build_agent_config()

    from datus.utils.path_manager import get_path_manager

    path_manager = get_path_manager(agent_config.home)
    semantic_model_dir = str(path_manager.semantic_model_path(agent_config.current_namespace))

    has_subject_tree, subject_tree = get_subject_tree(agent_config, args.subject_tree)

    existing_subject_trees = []
    if not has_subject_tree:
        existing_subject_trees = get_existing_subject_trees(agent_config)

    context = {
        "semantic_model_dir": semantic_model_dir,
        "has_subject_tree": has_subject_tree,
        "subject_tree": subject_tree,
        "existing_subject_trees": existing_subject_trees,
    }
    print(json.dumps(context, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
