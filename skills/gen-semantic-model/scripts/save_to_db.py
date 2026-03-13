#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Save generated semantic model YAML file to the knowledge base.

Configuration is read from environment variables injected by SkillBashTool.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _skill_common import build_agent_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Save semantic model to knowledge base")
    parser.add_argument("--file-path", type=str, required=True, help="Semantic model YAML file name")
    parser.add_argument("--metric-sqls-json", type=str, default="{}", help="Metric SQLs as JSON string")
    args = parser.parse_args()

    agent_config = build_agent_config()

    from datus.utils.path_manager import get_path_manager

    path_manager = get_path_manager(agent_config.home)
    semantic_model_dir = path_manager.semantic_model_path(agent_config.current_namespace)
    full_path = os.path.realpath(str(semantic_model_dir / args.file_path))

    if not full_path.startswith(os.path.realpath(str(semantic_model_dir)) + os.sep):
        print(json.dumps({"success": False, "error": "Invalid file path: path traversal not allowed"}))
        sys.exit(1)

    if not os.path.isfile(full_path):
        print(json.dumps({"success": False, "error": f"File not found: {full_path}"}))
        sys.exit(1)

    try:
        metric_sqls = json.loads(args.metric_sqls_json) if args.metric_sqls_json else {}

        from datus.cli.generation_hooks import GenerationHooks

        result = GenerationHooks._sync_semantic_to_db(full_path, agent_config, metric_sqls=metric_sqls)

        if result.get("success"):
            print(json.dumps({"success": True, "message": result.get("message", "Saved successfully")}))
        else:
            print(json.dumps({"success": False, "error": result.get("error", "Unknown error")}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
