#!/usr/bin/env python3
"""Render deterministic SQL checks from a structured table contract spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_not_null(table: str, check: dict) -> str:
    columns = check["columns"]
    name = check.get("name", "not_null_check")
    parts = [f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END)" for column in columns]
    observed = " + ".join(parts)
    return f"""SELECT
  {sql_string(name)} AS check_name,
  'not_null' AS check_type,
  {observed} AS observed_value,
  0 AS expected_threshold,
  CASE WHEN {observed} = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM {table}"""


def render_duplicate_check(table: str, check: dict, check_type: str) -> str:
    columns = check["columns"]
    cols = ", ".join(columns)
    name = check.get("name", f"{check_type}_check")
    return f"""SELECT
  {sql_string(name)} AS check_name,
  {sql_string(check_type)} AS check_type,
  COUNT(*) AS observed_value,
  0 AS expected_threshold,
  CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM (
  SELECT {cols}
  FROM {table}
  GROUP BY {cols}
  HAVING COUNT(*) > 1
) duplicate_rows"""


def render_check(table: str, check: dict) -> str:
    check_type = check["type"]
    if check_type == "not_null":
        return render_not_null(table, check)
    if check_type == "unique_key":
        return render_duplicate_check(table, check, "unique_key")
    if check_type == "grain":
        return render_duplicate_check(table, check, "grain")
    raise ValueError(f"unsupported check type: {check_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to a table contract validation JSON file")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    table = spec["table"]
    checks = spec.get("checks", [])
    if not checks:
        raise ValueError("spec must contain at least one check")
    print("\nUNION ALL\n\n".join(render_check(table, check) for check in checks) + ";")


if __name__ == "__main__":
    main()
