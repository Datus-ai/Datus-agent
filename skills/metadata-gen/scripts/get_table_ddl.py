#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Get table DDL from database using datus connector API."""

import argparse
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
    parser = argparse.ArgumentParser(description="Get table DDL from database")
    parser.add_argument("--tables", required=True, help="Comma-separated list of table names")
    args = parser.parse_args()

    agent_config = build_agent_config()

    try:
        from datus.tools.db_tools.db_manager import db_manager_instance

        db_manager = db_manager_instance(agent_config.namespaces)
        connector = db_manager.get_conn(agent_config.current_namespace)
        connector.connect()

        try:
            table_names = [t.strip() for t in args.tables.split(",") if t.strip()]
            if not table_names:
                output_error("No valid table names provided in --tables")
            results = {}

            # Get tables with DDL
            try:
                tables_ddl = connector.get_tables_with_ddl(tables=table_names)
                for item in tables_ddl:
                    name = item.get("table_name", "")
                    results[name] = {
                        "definition": item.get("definition", ""),
                        "table_type": item.get("table_type", "table"),
                        "identifier": item.get("identifier", ""),
                    }
            except (NotImplementedError, AttributeError):
                pass

            # Get views with DDL for any remaining tables
            remaining = [t for t in table_names if t not in results]
            if remaining:
                try:
                    views_ddl = connector.get_views_with_ddl()
                    for item in views_ddl:
                        name = item.get("table_name", "")
                        if name in remaining:
                            results[name] = {
                                "definition": item.get("definition", ""),
                                "table_type": item.get("table_type", "view"),
                                "identifier": item.get("identifier", ""),
                            }
                except (NotImplementedError, AttributeError):
                    pass

            # Report missing tables
            for t in table_names:
                if t not in results:
                    results[t] = {"error": f"Table or view not found: {t}"}

            output_json({"success": True, "tables": results})
        finally:
            connector.close()
    except Exception as e:
        output_error(str(e))


if __name__ == "__main__":
    main()
