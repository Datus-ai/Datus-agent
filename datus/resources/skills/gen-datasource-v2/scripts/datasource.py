"""Schema-preserving, industry-independent SQL generation and validation runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import duckdb


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def catalog(con):
    """Use DuckDB's parser/catalog, including complete composite constraint tuples."""
    tables = {}
    for schema, name in con.execute(
        "SELECT schema_name, table_name FROM duckdb_tables() WHERE NOT temporary ORDER BY 1,2"
    ).fetchall():
        if schema != "main":
            raise ValueError("This runner currently requires unqualified/main-schema DDL; normalize explicitly.")
        cols = con.execute(
            "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema=? AND table_name=? ORDER BY ordinal_position",
            [schema, name],
        ).fetchall()
        constraints = con.execute(
            "SELECT constraint_type, constraint_column_names, referenced_table, referenced_column_names, "
            "expression FROM duckdb_constraints() WHERE schema_name=? AND table_name=? ORDER BY constraint_index",
            [schema, name],
        ).fetchall()
        tables[name] = {
            "columns": [
                {"name": n, "type": t, "nullable": nullable == "YES", "default": default}
                for n, t, nullable, default in cols
            ],
            "constraints": [
                {"type": t, "columns": cs, "parent": p, "parent_columns": pcs, "expression": expr}
                for t, cs, p, pcs, expr in constraints
            ],
        }
    return tables


def apply_ddl(con, ddl):
    statements = con.extract_statements(ddl)
    if not statements or any(s.type.name not in {"CREATE", "ALTER"} for s in statements):
        raise ValueError("schema.sql must contain schema DDL, not data mutations or queries")
    remaining = [s.query for s in statements]
    while remaining:
        retry, errors = [], []
        for sql in remaining:
            try:
                con.execute(sql)
            except duckdb.Error as exc:
                retry.append(sql)
                errors.append(str(exc))
        if len(retry) == len(remaining):
            raise ValueError("DDL cannot be created (syntax, missing dependency or cycle): " + " | ".join(errors))
        remaining = retry
    if not catalog(con):
        raise ValueError("DDL contains no persistent tables")


def read_schema(ddl):
    with duckdb.connect(":memory:") as con:
        apply_ddl(con, ddl)
        return catalog(con)


def dependency_order(schema):
    pending, ordered = set(schema), []
    while pending:
        ready = sorted(
            t
            for t in pending
            if not {
                c["parent"]
                for c in schema[t]["constraints"]
                if c["type"] == "FOREIGN KEY" and c["parent"] != t and c["parent"] in pending
            }
        )
        if not ready:
            raise ValueError("Cyclic table dependencies require an explicit generation strategy: " + ", ".join(pending))
        ordered.extend(ready)
        pending.difference_update(ready)
    return ordered


def constraint_signature(constraint):
    return json.dumps(constraint, sort_keys=True)


def validate(con, expected, assertions, min_rows, max_rows, require_assertions=True):
    checks, counts = [], {}

    def add(name, ok, actual, **extra):
        checks.append({"name": name, "ok": bool(ok), "actual": actual, **extra})

    actual = catalog(con)
    add("schema.tables", set(actual) == set(expected), sorted(actual), expected=sorted(expected))
    for table, spec in expected.items():
        if table not in actual:
            continue
        q = quote(table)
        counts[table] = con.execute(f"SELECT count(*) FROM {q}").fetchone()[0]
        add(f"{table}.populated", counts[table] > 0, counts[table])
        add(f"{table}.columns", actual[table]["columns"] == spec["columns"], actual[table]["columns"])
        expected_constraints = Counter(map(constraint_signature, spec["constraints"]))
        actual_constraints = Counter(map(constraint_signature, actual[table]["constraints"]))
        add(f"{table}.constraints", actual_constraints == expected_constraints, actual[table]["constraints"])
        available = {c["name"] for c in actual[table]["columns"]}
        for c in spec["constraints"]:
            columns = c["columns"]
            if not set(columns) <= available:
                continue
            if c["type"] == "NOT NULL":
                n = con.execute(f"SELECT count(*) FROM {q} WHERE {quote(columns[0])} IS NULL").fetchone()[0]
            elif c["type"] in {"PRIMARY KEY", "UNIQUE"}:
                keys = ", ".join(map(quote, columns))
                where = " AND ".join(f"{quote(k)} IS NOT NULL" for k in columns)
                n = con.execute(
                    f"SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM {q} WHERE {where} "
                    f"GROUP BY {keys} HAVING count(*)>1)"
                ).fetchone()[0]
            elif c["type"] == "FOREIGN KEY" and c["parent"] in actual:
                pairs = list(zip(columns, c["parent_columns"], strict=True))
                where = " AND ".join(f"child.{quote(a)} IS NOT NULL" for a, _ in pairs)
                match = " AND ".join(f"child.{quote(a)}=parent.{quote(b)}" for a, b in pairs)
                n = con.execute(
                    f"SELECT count(*) FROM {q} child WHERE {where} AND NOT EXISTS "
                    f"(SELECT 1 FROM {quote(c['parent'])} parent WHERE {match})"
                ).fetchone()[0]
            elif c["type"] == "CHECK":
                n = con.execute(f"SELECT count(*) FROM {q} WHERE NOT ({c['expression']})").fetchone()[0]
            else:
                continue
            add(f"{table}.{c['type']}({','.join(columns)})", n == 0, n)
    total = sum(counts.values())
    add("database.row_budget", min_rows <= total <= max_rows, total, expected=[min_rows, max_rows])
    if require_assertions:
        add("business.assertions_present", bool(assertions), len(assertions))
    for assertion in assertions:
        name = "business." + assertion.get("name", "unnamed")
        try:
            sql = assertion["sql"]
            statements = con.extract_statements(sql)
            if len(statements) != 1 or statements[0].type.name != "SELECT":
                raise ValueError("An assertion must be one SELECT returning one numeric scalar")
            rows = con.execute(sql).fetchmany(2)
            if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is None:
                raise ValueError("Expected exactly one non-NULL scalar")
            value = float(rows[0][0])
            if not math.isfinite(value):
                raise ValueError("Non-finite scalar")
            expect = assertion["expect"]
            if expect == "zero":
                ok = value == 0
            elif expect == "nonzero":
                ok = value != 0
            elif isinstance(expect, dict) and expect and set(expect) <= {"min", "max"}:
                ok = float(expect.get("min", -math.inf)) <= value <= float(expect.get("max", math.inf))
            else:
                raise ValueError("expect must be zero, nonzero, or a min/max interval")
            add(name, ok, value, expected=expect)
        except (duckdb.Error, ValueError, TypeError, KeyError) as exc:
            add(name, False, None, error=str(exc))
    return {"ok": all(c["ok"] for c in checks), "rows": total, "tables": counts, "checks": checks}


def run(directory):
    directory = Path(directory).resolve()
    build = directory / "_build"
    build.mkdir(exist_ok=True)
    next_path = build / "next.duckdb"
    next_path.unlink(missing_ok=True)
    report, started = {"ok": False}, time.perf_counter()
    # Invalidate the previous success before reading anything that might be missing/malformed.
    write_json(directory / "quality.json", {"ok": False, "status": "running"})
    try:
        settings = json.loads((directory / "settings.json").read_text())
        expected = read_schema((directory / "schema.sql").read_text())
        assertions = json.loads((directory / "checks.json").read_text()).get("assertions", [])
        with duckdb.connect(str(next_path)) as con:
            con.execute("SET threads=1")
            apply_ddl(con, (directory / "schema.sql").read_text())
            seed = int(settings["seed"])
            start = date.fromisoformat(settings["start_date"])
            end = date.fromisoformat(settings["end_date"])
            con.execute(f"CREATE TEMP MACRO u01(k, s) AS ((hash(k,s,{seed}) % 1000003)::DOUBLE/1000003.0)")
            con.execute(f"CREATE TEMP MACRO data_start() AS DATE '{start}'")
            con.execute(f"CREATE TEMP MACRO data_end() AS DATE '{end}'")
            statements = con.extract_statements((directory / "generate.sql").read_text())
            for number, statement in enumerate(statements, 1):
                try:
                    con.execute(statement.query)
                except duckdb.Error as exc:
                    raise ValueError(f"generate.sql statement {number}: {exc}") from exc
            report = validate(con, expected, assertions, settings["min_rows"], settings["max_rows"])
        os.replace(next_path, build / "datasource.duckdb")
    except (OSError, duckdb.Error, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["duckdb_version"] = duckdb.__version__
        write_json(directory / "quality.json", report)
    failures = [c for c in report.get("checks", []) if not c["ok"]]
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, default=str))
    if failures:
        print(json.dumps({"failures": failures}, default=str))
    return 0 if report["ok"] else 1


GENERATOR = '''#!/usr/bin/env python3
"""Reproduce the database from schema.sql, generate.sql, checks.json and settings.json."""
import pathlib
import sys
import datus

scripts = pathlib.Path(datus.__file__).resolve().parent / "resources/skills/gen-datasource-v2/scripts"
sys.path.insert(0, str(scripts))
from datasource import run

if __name__ == "__main__":
    raise SystemExit(run(pathlib.Path(__file__).resolve().parent))
'''


def initialize(args):
    directory = Path(args.directory).resolve()
    ddl = Path(args.ddl).read_text()
    schema = read_schema(ddl)
    order = dependency_order(schema)
    if args.months < 1 or not 0 < args.min_rows <= args.rows <= args.max_rows:
        raise ValueError("Require months >= 1 and 0 < min_rows <= rows <= max_rows")
    end = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)
    month_index = end.year * 12 + end.month - args.months
    year, month = divmod(month_index, 12)
    start = date(year, month + 1, 1)
    directory.mkdir(parents=True, exist_ok=True)
    settings_path = directory / "settings.json"
    if settings_path.exists():
        raise ValueError("settings.json already exists; edit the saved settings instead of resetting the run")
    if Path(args.ddl).resolve() != directory / "schema.sql":
        shutil.copyfile(args.ddl, directory / "schema.sql")
    write_json(directory / "schema.json", {"tables": schema, "dependency_order": order})
    write_json(
        settings_path,
        {
            "rows": args.rows,
            "min_rows": args.min_rows,
            "max_rows": args.max_rows,
            "seed": args.seed,
            "start_date": str(start),
            "end_date": str(end),
        },
    )
    (directory / "gen.py").write_text(GENERATOR)
    print(
        json.dumps(
            {
                "dependency_order": order,
                "settings": json.loads(settings_path.read_text()),
                "tables": {
                    t: {"columns": len(s["columns"]), "constraints": s["constraints"]} for t, s in schema.items()
                },
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--ddl", required=True)
    init.add_argument("--directory", default="data")
    init.add_argument("--rows", type=int, default=80000)
    init.add_argument("--min-rows", type=int, default=50000)
    init.add_argument("--max-rows", type=int, default=100000)
    init.add_argument("--months", type=int, default=17)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--end-date", default="")
    execute = sub.add_parser("run")
    execute.add_argument("--directory", default="data")
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args)
            return 0
        return run(args.directory)
    except (OSError, ValueError, duckdb.Error) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
