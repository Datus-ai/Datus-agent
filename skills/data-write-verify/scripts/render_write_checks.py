#!/usr/bin/env python3
import argparse
import json
import sys


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_check(table: str, check: dict) -> str:
    kind = check["kind"]
    column = check.get("column")
    if kind == "null_ratio":
        return (
            f"SELECT {sql_string('null_ratio:' + column)} AS check_name, "
            f"CAST(SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS VARCHAR) AS observed_value, "
            f"{sql_string(str(check['max_ratio']))} AS expected_value "
            f"FROM {table}"
        )
    if kind == "numeric_range":
        return (
            f"SELECT {sql_string('numeric_range:' + column)} AS check_name, "
            f"CAST(MIN({column}) AS VARCHAR) || ' .. ' || CAST(MAX({column}) AS VARCHAR) AS observed_value, "
            f"{sql_string(str(check['min']) + ' .. ' + str(check['max']))} AS expected_value "
            f"FROM {table}"
        )
    if kind == "accepted_values":
        values = ", ".join(sql_string(v) for v in check["values"])
        return (
            f"SELECT {sql_string('accepted_values:' + column)} AS check_name, "
            f"CAST(COUNT(*) FILTER (WHERE {column} IS NOT NULL AND {column} NOT IN ({values})) AS VARCHAR) AS observed_value, "
            f"{sql_string('0 invalid rows')} AS expected_value "
            f"FROM {table}"
        )
    if kind == "regex":
        pattern = check["pattern"]
        return (
            f"SELECT {sql_string('regex:' + column)} AS check_name, "
            f"CAST(COUNT(*) FILTER (WHERE {column} IS NOT NULL AND NOT regexp_matches(CAST({column} AS VARCHAR), {sql_string(pattern)})) AS VARCHAR) AS observed_value, "
            f"{sql_string('0 invalid rows')} AS expected_value "
            f"FROM {table}"
        )
    if kind == "uniqueness":
        keys = check["columns"]
        key_expr = ", ".join(keys)
        return (
            f"SELECT {sql_string('uniqueness:' + '_'.join(keys))} AS check_name, "
            f"CAST(COUNT(*) AS VARCHAR) AS observed_value, "
            f"{sql_string('0 duplicate groups')} AS expected_value "
            f"FROM (SELECT {key_expr}, COUNT(*) AS n FROM {table} GROUP BY {key_expr} HAVING COUNT(*) > 1) d"
        )
    raise ValueError(f"Unsupported check kind: {kind}")


def render(spec: dict) -> str:
    table = spec["table"]
    parts = []
    row_range = spec.get("row_count_range")
    if row_range:
        parts.append(
            f"SELECT {sql_string('row_count_range')} AS check_name, "
            f"CAST(COUNT(*) AS VARCHAR) AS observed_value, "
            f"{sql_string(str(row_range['min']) + ' .. ' + str(row_range['max']))} AS expected_value "
            f"FROM {table}"
        )
    for check in spec.get("checks", []):
        parts.append(render_check(table, check))
    return "\nUNION ALL\n".join(parts) + "\nORDER BY check_name;"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to write validation JSON spec")
    args = parser.parse_args()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    sys.stdout.write(render(spec))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
