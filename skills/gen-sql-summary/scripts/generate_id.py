#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Generate a unique SQL summary ID from SQL query content.

Uses the existing gen_reference_sql_id utility function.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate unique SQL summary ID")
    parser.add_argument("--sql-query", type=str, required=True, help="SQL query to generate ID for")
    args = parser.parse_args()

    if not args.sql_query.strip():
        print("Error: Empty SQL query provided", file=sys.stderr)
        sys.exit(1)

    from datus.storage.reference_sql.init_utils import gen_reference_sql_id

    sql_id = gen_reference_sql_id(args.sql_query)
    print(sql_id)


if __name__ == "__main__":
    main()
