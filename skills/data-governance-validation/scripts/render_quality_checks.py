#!/usr/bin/env python3
"""Render deterministic SQL data-quality checks from a structured contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def regex_expr(db_type: str, column: str, pattern: str) -> str:
    quoted_pattern = sql_string(pattern)
    if db_type in {"duckdb", "postgres", "postgresql"}:
        return f"regexp_matches(CAST({column} AS VARCHAR), {quoted_pattern})"
    if db_type == "sqlite":
        raise ValueError("sqlite regex checks are not supported by this renderer")
    raise ValueError(f"unsupported db_type for regex checks: {db_type}")


def render_null_ratio(table: str, check: dict) -> str:
    column = check["column"]
    max_ratio = check["max_ratio"]
    name = check.get("name", f"{column}_null_ratio")
    return f"""SELECT
  {sql_string(name)} AS check_name,
  'null_ratio' AS check_type,
  SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS observed_value,
  {max_ratio} AS expected_threshold,
  CASE
    WHEN SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) <= {max_ratio} THEN 'PASS'
    ELSE 'FAIL'
  END AS status
FROM {table}"""


def render_range(table: str, check: dict) -> str:
    column = check["column"]
    name = check.get("name", f"{column}_range")
    conditions: list[str] = []
    if "min_value" in check:
        conditions.append(f"MIN({column}) >= {check['min_value']}")
    if "max_value" in check:
        conditions.append(f"MAX({column}) <= {check['max_value']}")
    if not conditions:
        raise ValueError("range check requires min_value and/or max_value")
    details = []
    if "min_value" in check:
        details.append(f"min={check['min_value']}")
    if "max_value" in check:
        details.append(f"max={check['max_value']}")
    return f"""SELECT
  {sql_string(name)} AS check_name,
  'range' AS check_type,
  CONCAT('min=', MIN({column}), ', max=', MAX({column})) AS observed_value,
  {sql_string(", ".join(details))} AS expected_threshold,
  CASE
    WHEN {" AND ".join(conditions)} THEN 'PASS'
    ELSE 'FAIL'
  END AS status
FROM {table}"""


def render_accepted_values(table: str, check: dict) -> str:
    column = check["column"]
    name = check.get("name", f"{column}_accepted_values")
    values = ", ".join(sql_string(value) for value in check["values"])
    max_fail_ratio = check.get("max_fail_ratio", 0.0)
    return f"""SELECT
  {sql_string(name)} AS check_name,
  'accepted_values' AS check_type,
  SUM(CASE WHEN {column} IS NOT NULL AND {column} NOT IN ({values}) THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS observed_value,
  {max_fail_ratio} AS expected_threshold,
  CASE
    WHEN SUM(CASE WHEN {column} IS NOT NULL AND {column} NOT IN ({values}) THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) <= {max_fail_ratio} THEN 'PASS'
    ELSE 'FAIL'
  END AS status
FROM {table}"""


def render_regex(table: str, db_type: str, check: dict) -> str:
    column = check["column"]
    name = check.get("name", f"{column}_regex")
    max_fail_ratio = check.get("max_fail_ratio", 0.0)
    predicate = regex_expr(db_type, column, check["pattern"])
    return f"""SELECT
  {sql_string(name)} AS check_name,
  'regex' AS check_type,
  SUM(CASE WHEN {column} IS NOT NULL AND NOT {predicate} THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS observed_value,
  {max_fail_ratio} AS expected_threshold,
  CASE
    WHEN SUM(CASE WHEN {column} IS NOT NULL AND NOT {predicate} THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) <= {max_fail_ratio} THEN 'PASS'
    ELSE 'FAIL'
  END AS status
FROM {table}"""


def render_uniqueness(table: str, check: dict) -> str:
    columns = check["columns"]
    name = check.get("name", "_".join(columns) + "_unique")
    max_duplicate_rows = check.get("max_duplicate_rows", 0)
    cols = ", ".join(columns)
    return f"""SELECT
  {sql_string(name)} AS check_name,
  'uniqueness' AS check_type,
  COUNT(*) AS observed_value,
  {max_duplicate_rows} AS expected_threshold,
  CASE
    WHEN COUNT(*) <= {max_duplicate_rows} THEN 'PASS'
    ELSE 'FAIL'
  END AS status
FROM (
  SELECT {cols}
  FROM {table}
  GROUP BY {cols}
  HAVING COUNT(*) > 1
) duplicate_rows"""


def render_check(table: str, db_type: str, check: dict) -> str:
    check_type = check["type"]
    if check_type == "null_ratio":
        return render_null_ratio(table, check)
    if check_type == "range":
        return render_range(table, check)
    if check_type == "accepted_values":
        return render_accepted_values(table, check)
    if check_type == "regex":
        return render_regex(table, db_type, check)
    if check_type == "uniqueness":
        return render_uniqueness(table, check)
    raise ValueError(f"unsupported check type: {check_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to a table quality contract JSON file")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    table = spec["table"]
    db_type = spec.get("db_type", "duckdb")
    checks = spec.get("checks", [])
    if not checks:
        raise ValueError("spec must contain at least one check")

    rendered = [render_check(table, db_type, check) for check in checks]
    print("\nUNION ALL\n\n".join(rendered) + ";")


if __name__ == "__main__":
    main()
