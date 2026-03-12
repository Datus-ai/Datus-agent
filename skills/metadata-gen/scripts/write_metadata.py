#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Write schema metadata and sample values to vector store via SchemaWithValueRAG.

Uses the abstract vector backend (not hardcoded to LanceDB).
Input: JSON file with schemas and values arrays.
Output: JSON with success status and counts.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _skill_common import build_agent_config  # noqa: E402


def output_json(data):
    """Print JSON result to stdout."""
    print(json.dumps(data, ensure_ascii=False, default=str))


def output_error(message):
    """Print error JSON and exit with code 1."""
    print(json.dumps({"success": False, "error": str(message)}, ensure_ascii=False))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Write schema metadata to vector store")
    parser.add_argument("--input_file", required=True, help="JSON file with schemas and values")
    parser.add_argument("--mode", default="overwrite", choices=["overwrite", "incremental"], help="Write mode")
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input_file)
    if not os.path.isfile(input_path):
        output_error(f"Input file not found: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        output_error(f"Failed to read input file: {e}")

    schemas = data.get("schemas", [])
    values = data.get("values", [])

    if not schemas and not values:
        output_error("Input file must contain 'schemas' and/or 'values' arrays")

    try:
        agent_config = build_agent_config()

        if args.mode == "overwrite":
            agent_config.save_storage_config("database")
        else:
            agent_config.check_init_storage_config("database")

        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        rag = SchemaWithValueRAG(agent_config)

        if args.mode == "overwrite":
            rag.truncate()

        rag.store_batch(schemas, values)
        rag.after_init()

        output_json(
            {
                "success": True,
                "schema_metadata_count": rag.get_schema_size(),
                "schema_value_count": rag.get_value_size(),
                "mode": args.mode,
            }
        )

    except Exception as e:
        output_error(str(e))


if __name__ == "__main__":
    main()
