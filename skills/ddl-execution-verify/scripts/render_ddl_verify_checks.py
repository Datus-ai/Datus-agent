#!/usr/bin/env python3
import argparse
import json
import sys


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalize_nullable(value):
    if value is None:
        return "NULL"
    return "TRUE" if bool(value) else "FALSE"


def render(spec: dict) -> str:
    table = spec["table"]
    if "." in table:
        schema_name, table_name = table.split(".", 1)
    else:
        schema_name, table_name = "main", table

    rows = []
    for column in spec.get("expected_columns", []):
        rows.append(
            "("
            + ", ".join(
                [
                    sql_string(column["name"]),
                    sql_string(column["type"].upper()),
                    normalize_nullable(column.get("nullable")),
                ]
            )
            + ")"
        )
    expected_values = (
        ",\n    ".join(rows) or f"({sql_string('__no_expected_columns__')}, {sql_string('VARCHAR')}, NULL)"
    )

    return f"""WITH expected(column_name, expected_type, expected_nullable) AS (
    VALUES
    {expected_values}
),
actual AS (
    SELECT
        column_name,
        UPPER(data_type) AS actual_type,
        CASE
            WHEN is_nullable = 'YES' THEN TRUE
            WHEN is_nullable = 'NO' THEN FALSE
            ELSE NULL
        END AS actual_nullable
    FROM information_schema.columns
    WHERE table_schema = {sql_string(schema_name)}
      AND table_name = {sql_string(table_name)}
),
object_exists AS (
    SELECT COUNT(*) AS object_exists
    FROM information_schema.tables
    WHERE table_schema = {sql_string(schema_name)}
      AND table_name = {sql_string(table_name)}
)
SELECT 'object_exists' AS check_name, CAST(object_exists AS VARCHAR) AS observed_value, '1' AS expected_value
FROM object_exists
UNION ALL
SELECT 'missing_column:' || e.column_name, 'missing', 'present'
FROM expected e
LEFT JOIN actual a ON a.column_name = e.column_name
WHERE a.column_name IS NULL
UNION ALL
SELECT 'extra_column:' || a.column_name, 'present', 'absent'
FROM actual a
LEFT JOIN expected e ON e.column_name = a.column_name
WHERE e.column_name IS NULL
UNION ALL
SELECT 'type_mismatch:' || e.column_name, COALESCE(a.actual_type, 'NULL'), e.expected_type
FROM expected e
JOIN actual a ON a.column_name = e.column_name
WHERE a.actual_type <> e.expected_type
UNION ALL
SELECT 'nullability_mismatch:' || e.column_name, COALESCE(CAST(a.actual_nullable AS VARCHAR), 'NULL'), COALESCE(CAST(e.expected_nullable AS VARCHAR), 'NULL')
FROM expected e
JOIN actual a ON a.column_name = e.column_name
WHERE e.expected_nullable IS NOT NULL
  AND a.actual_nullable IS DISTINCT FROM e.expected_nullable
ORDER BY check_name;"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to schema verification JSON spec")
    args = parser.parse_args()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    sys.stdout.write(render(spec))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
