#!/usr/bin/env python3
"""Analyze SQL query for optimization opportunities."""

import argparse
import json


def analyze(sql: str, db_type: str = "sqlite") -> dict:
    """Simple query analysis."""
    issues = []
    suggestions = []

    if "SELECT *" in sql.upper():
        issues.append("Using SELECT * - specify columns explicitly")
        suggestions.append("Replace SELECT * with specific column names")

    if "WHERE" not in sql.upper():
        issues.append("No WHERE clause - full table scan")
        suggestions.append("Add filtering conditions")

    return {
        "sql": sql,
        "db_type": db_type,
        "issues": issues,
        "suggestions": suggestions,
        "score": max(0, 100 - len(issues) * 25),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SQL query")
    parser.add_argument("--sql", required=True, help="SQL query to analyze")
    parser.add_argument("--db-type", default="sqlite", help="Database type")
    args = parser.parse_args()

    result = analyze(args.sql, args.db_type)
    print(json.dumps(result, indent=2))
