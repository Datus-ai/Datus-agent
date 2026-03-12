#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""List all tables/views in a database namespace using datus connector API."""

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
    agent_config = build_agent_config()

    try:
        from datus.tools.db_tools.db_manager import db_manager_instance

        db_manager = db_manager_instance(agent_config.namespaces)
        connector = db_manager.get_conn(agent_config.current_namespace)
        connector.connect()

        try:
            tables = []
            for t in connector.get_tables():
                tables.append({"name": t, "type": "table", "comment": ""})

            try:
                for v in connector.get_views():
                    tables.append({"name": v, "type": "view", "comment": ""})
            except (NotImplementedError, AttributeError):
                pass

            output_json(
                {
                    "success": True,
                    "namespace": agent_config.current_namespace,
                    "count": len(tables),
                    "tables": tables,
                }
            )
        finally:
            connector.close()
    except Exception as e:
        output_error(str(e))


if __name__ == "__main__":
    main()
