#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Get sample rows from database tables using datus connector API."""

import argparse
import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _skill_common import build_agent_config  # noqa: E402


def output_json(data):
    print(json.dumps(data, ensure_ascii=False, default=str))


def output_error(message):
    print(json.dumps({"success": False, "error": str(message)}, ensure_ascii=False))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Get sample rows from database tables")
    parser.add_argument("--tables", required=True, help="Comma-separated table names")
    parser.add_argument("--limit", type=int, default=5, help="Number of sample rows (default: 5)")
    args = parser.parse_args()

    agent_config = build_agent_config()

    try:
        from datus.tools.db_tools.db_manager import db_manager_instance

        db_manager = db_manager_instance(agent_config.namespaces)
        connector = db_manager.get_conn(agent_config.current_namespace)
        connector.connect()

        try:
            table_names = [t.strip() for t in args.tables.split(",") if t.strip()]
            results = {}

            try:
                sample_rows = connector.get_sample_rows(tables=table_names, top_n=args.limit)
                for item in sample_rows:
                    name = item.get("table_name", "")
                    csv_data = item.get("sample_rows", "")
                    if name and csv_data:
                        # Parse CSV properly to handle quoted fields and embedded commas
                        reader = csv.reader(io.StringIO(csv_data.strip()))
                        parsed = list(reader)
                        if parsed:
                            columns = parsed[0]
                            rows = parsed[1:]
                            results[name] = {"columns": columns, "rows": rows, "count": len(rows)}
                        else:
                            results[name] = {"columns": [], "rows": [], "count": 0}
                    elif name:
                        results[name] = {"columns": [], "rows": [], "count": 0}
            except (NotImplementedError, AttributeError):
                pass

            # Report missing tables
            for t in table_names:
                if t not in results:
                    results[t] = {"error": f"Could not retrieve sample rows for: {t}"}

            output_json({"success": True, "tables": results})
        finally:
            connector.close()
    except Exception as e:
        output_error(str(e))


if __name__ == "__main__":
    main()
