# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Independent checks for explicitly declared sample-data semantics.

No industry inference and no agent-authored SQL predicates: identifiers and aggregation
choices form a typed contract, while the implementation owns checks and thresholds.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, timedelta
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Table(ContractModel):
    role: Literal["dimension", "fact", "event", "summary"]
    grain: str = Field(min_length=1)
    min_rows: int = Field(default=1, ge=1)
    logical_key: list[str] = Field(default_factory=list)
    keyless_reason: str | None = None
    dates: dict[str, Literal["actual", "planned", "audit"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def key_or_reason(self):
        if bool(self.logical_key) == bool(self.keyless_reason and self.keyless_reason.strip()):
            raise ValueError("Declare a logical_key or a nonempty keyless_reason, exclusively")
        return self


class Relationship(ContractModel):
    table: str
    columns: list[str] = Field(min_length=1)
    parent: str
    parent_columns: list[str] = Field(min_length=1)
    nullable: bool = False

    @model_validator(mode="after")
    def same_arity(self):
        if len(self.columns) != len(self.parent_columns):
            raise ValueError("Relationship tuples must have the same arity")
        return self


class Measure(ContractModel):
    aggregate: Literal["count", "sum", "avg"]
    column: str | None = None
    unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def column_required(self):
        if (self.aggregate == "count") != (self.column is None):
            raise ValueError("count means row count; sum/avg require a numeric column")
        return self


class Series(ContractModel):
    name: str = Field(min_length=1)
    table: str
    date_column: str
    measure: Measure
    scope: dict[str, list[str | int]] = Field(default_factory=dict)
    monthly: Literal["stable", "growth", "decline", "seasonal"]
    weekly: Literal["flat", "weekday_heavy", "weekend_heavy"]
    reason: str = Field(min_length=1)


class Distribution(ContractModel):
    name: str = Field(min_length=1)
    table: str
    entity: list[str] = Field(min_length=1)
    measure: Measure
    shape: Literal["long_tail", "balanced"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def additive_only(self):
        if self.measure.aggregate == "avg":
            raise ValueError("Concentration requires an additive measure (count or sum)")
        return self


class Sequence(ContractModel):
    table: str
    partition_by: list[str] = Field(min_length=1)
    order_by: list[str] = Field(min_length=1)
    timestamp: str
    end_timestamp: str | None = None


class Anomaly(ContractModel):
    name: str = Field(min_length=1)
    series: str
    start: date
    end: date
    direction: Literal["up", "down"]
    explanation: str = Field(min_length=1)


class Semantics(ContractModel):
    quality_contract_version: Literal[1]
    tables: dict[str, Table] = Field(min_length=1)
    relationships: list[Relationship] = Field(default_factory=list)
    series: list[Series] = Field(default_factory=list)
    distributions: list[Distribution] = Field(default_factory=list)
    sequences: list[Sequence] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    not_applicable: dict[Literal["relationships", "series", "distributions", "sequences", "anomalies"], str] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def explicit_coverage(self):
        for category in ("relationships", "series", "distributions", "sequences", "anomalies"):
            reason = self.not_applicable.get(category, "").strip()
            if bool(getattr(self, category)) == bool(reason):
                raise ValueError(f"{category}: supply rules or one nonempty not_applicable reason, exclusively")
        for category in ("series", "distributions", "anomalies"):
            names = [item.name for item in getattr(self, category)]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate {category} names")
        return self


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def catalog(con):
    """Canonical structural snapshot shared by the builder and post-import gate."""
    tables = {}
    for schema, name in con.execute(
        "SELECT schema_name,table_name FROM duckdb_tables() "
        "WHERE database_name=current_database() AND NOT temporary ORDER BY 1,2"
    ).fetchall():
        if schema != "main":
            raise ValueError("This runner requires unqualified/main-schema DDL; normalize explicitly.")
        cols = con.execute(
            "SELECT column_name,data_type,is_nullable,column_default FROM information_schema.columns "
            "WHERE table_catalog=current_database() AND table_schema=? AND table_name=? ORDER BY ordinal_position",
            [schema, name],
        ).fetchall()
        constraints = con.execute(
            "SELECT constraint_type,constraint_column_names,referenced_table,referenced_column_names,expression "
            "FROM duckdb_constraints() WHERE database_name=current_database() AND schema_name=? AND table_name=? "
            "ORDER BY constraint_index",
            [schema, name],
        ).fetchall()
        tables[name] = {
            "columns": [
                {"name": n, "type": t, "nullable": null == "YES", "default": default} for n, t, null, default in cols
            ],
            "constraints": [
                {"type": t, "columns": cs, "parent": p, "parent_columns": pcs, "expression": expr}
                for t, cs, p, pcs, expr in constraints
            ],
        }
    return tables


def check_semantic_metadata(con, metadata):
    """Validate the delivered schema and run semantics from the accepted build metadata."""
    try:
        if metadata.get("quality_contract_version") != 1:
            raise ValueError("Unsupported quality_contract_version")
        actual, expected = catalog(con), metadata["schema"]
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Missing original schema snapshot")
        checks = [
            {
                "check": "schema.tables",
                "status": "PASS" if set(actual) == set(expected) else "FAIL",
                "detail": f"actual={sorted(actual)}, expected={sorted(expected)}",
            }
        ]
        for table in expected:
            if table not in actual:
                continue

            def signature(c):
                return json.dumps(c, sort_keys=True)

            equal = actual[table]["columns"] == expected[table]["columns"] and (
                Counter(map(signature, actual[table]["constraints"]))
                == Counter(map(signature, expected[table]["constraints"]))
            )
            checks.append(
                {
                    "check": f"{table}.schema",
                    "status": "PASS" if equal else "FAIL",
                    "detail": "Original columns and declared constraints must survive import",
                }
            )
        total = sum(con.execute(f"SELECT count(*) FROM {quote(t)}").fetchone()[0] for t in actual)
        low, high = metadata["min_rows"], metadata["max_rows"]
        checks.append(
            {
                "check": "database.row_budget",
                "status": "PASS" if 0 < low <= total <= high else "FAIL",
                "detail": f"{total} rows; expected [{low}, {high}]",
            }
        )
        checks.extend(check_semantics(con, metadata["semantics"], metadata["start_date"], metadata["end_date"]))
        if metadata.get("contract_changes"):
            checks.append(
                {
                    "check": "semantic contract corrections",
                    "status": "WARN",
                    "detail": json.dumps(metadata["contract_changes"], default=str),
                }
            )
        return checks
    except (ValueError, TypeError, KeyError, duckdb.Error) as exc:
        return [{"check": "semantic metadata", "status": "FAIL", "detail": str(exc)}]


def check_semantics(con, payload, start_date, end_date):
    """Return PASS/FAIL/WARN records; malformed contracts and failed queries fail closed."""
    checks = []

    def add(name, ok, detail, warning=False):
        checks.append({"check": name, "status": "PASS" if ok else "WARN" if warning else "FAIL", "detail": str(detail)})

    try:
        contract = Semantics.model_validate(payload)
        start, end = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
        if start > end or (end - start).days > 36600:
            raise ValueError("Invalid observation window (maximum 100 years)")
        columns = {}
        for table, column, dtype in con.execute(
            "SELECT table_name,column_name,data_type FROM information_schema.columns "
            "WHERE table_catalog=current_database() AND table_schema='main'"
        ).fetchall():
            columns.setdefault(table, {})[column] = dtype
        actual_tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE database_name=current_database() AND NOT temporary"
            ).fetchall()
        }
        if actual_tables != set(contract.tables):
            raise ValueError(
                f"Semantic table coverage mismatch: actual={sorted(actual_tables)}, declared={sorted(contract.tables)}"
            )

        def refs(table, names):
            if table not in contract.tables or not names or len(names) != len(set(names)):
                raise ValueError(f"Invalid table or column tuple: {table}.{names}")
            for name in names:
                if name not in columns.get(table, {}):
                    raise ValueError(f"Unknown column: {table}.{name}")
            return ", ".join(map(quote, names))

        def number(sql, params=None):
            value = con.execute(sql, params or []).fetchone()[0]
            if value is None or not math.isfinite(float(value)):
                raise ValueError("Quality query returned NULL/non-finite value")
            return float(value)

        def measure(table, metric):
            if metric.aggregate == "count":
                return "count(*)"
            refs(table, [metric.column])
            col = quote(metric.column)
            dtype = columns[table][metric.column]
            if not dtype.startswith(
                (
                    "DECIMAL",
                    "DOUBLE",
                    "FLOAT",
                    "REAL",
                    "BIGINT",
                    "INTEGER",
                    "SMALLINT",
                    "TINYINT",
                    "HUGEINT",
                    "UBIGINT",
                    "UINTEGER",
                )
            ):
                raise ValueError(f"Measure {table}.{metric.column} must be numeric")
            bad = number(f"SELECT count(*) FROM {quote(table)} WHERE {col} IS NULL OR NOT isfinite({col}::DOUBLE)")
            if bad:
                raise ValueError(f"Measure {table}.{metric.column} has {bad:g} NULL/non-finite values")
            return f"{metric.aggregate}({col})"

        def unique(table, keys, name, nullable=False):
            group = refs(table, keys)
            any_null = " OR ".join(f"{quote(k)} IS NULL" for k in keys)
            n = number(f"SELECT count(*) FROM {quote(table)} WHERE {any_null}")
            duplicates = number(
                f"SELECT count(*) FROM (SELECT {group} FROM {quote(table)} GROUP BY {group} HAVING count(*)>1)"
            )
            add(name, duplicates == 0 and (nullable or n == 0), f"duplicate tuples={duplicates:g}, NULL tuples={n:g}")

        for category, reason in contract.not_applicable.items():
            add(f"coverage.{category}", False, f"Not assessed: {reason}", warning=True)
        for table, spec in contract.tables.items():
            qt = quote(table)
            count = number(f"SELECT count(*) FROM {qt}")
            add(f"{table}.populated", count >= spec.min_rows, f"{count:g} rows, minimum={spec.min_rows}; {spec.grain}")
            table_comment = con.execute(
                "SELECT comment FROM duckdb_tables() WHERE database_name=current_database() "
                "AND schema_name='main' AND table_name=? AND NOT temporary",
                [table],
            ).fetchone()[0]
            missing_comments = con.execute(
                "SELECT column_name FROM duckdb_columns() WHERE database_name=current_database() "
                "AND schema_name='main' AND table_name=? AND (comment IS NULL OR trim(comment)='')",
                [table],
            ).fetchall()
            add(
                f"{table}.comments",
                bool(table_comment and table_comment.strip()) and not missing_comments,
                f"table comment={bool(table_comment)}, columns missing comments={[x[0] for x in missing_comments]}",
            )
            if spec.logical_key:
                unique(table, spec.logical_key, f"{table}.logical_key")
            else:
                add(f"{table}.logical_key", False, f"Not assessed: {spec.keyless_reason}", warning=True)
            date_columns = {c for c, typ in columns[table].items() if typ.startswith(("DATE", "TIMESTAMP"))}
            if set(spec.dates) != date_columns:
                raise ValueError(
                    f"{table}: classify every date/timestamp as actual, planned or audit: {sorted(date_columns)}"
                )
            for column, role in spec.dates.items():
                if role == "actual":
                    n = number(f"SELECT count(*) FROM {qt} WHERE CAST({quote(column)} AS DATE)>?", [end])
                    add(f"{table}.{column}.actual_date", n == 0, f"{n:g} actual events after {end}")

        for rel in contract.relationships:
            refs(rel.table, rel.columns)
            refs(rel.parent, rel.parent_columns)
            unique(rel.parent, rel.parent_columns, f"{rel.table}->{rel.parent}.parent_key")
            match = " AND ".join(
                f"c.{quote(a)}=p.{quote(b)}" for a, b in zip(rel.columns, rel.parent_columns, strict=True)
            )
            present = " AND ".join(f"c.{quote(k)} IS NOT NULL" for k in rel.columns)
            bad = number(
                f"SELECT count(*) FROM {quote(rel.table)} c WHERE ({present}) AND NOT EXISTS "
                f"(SELECT 1 FROM {quote(rel.parent)} p WHERE {match})"
            )
            nulls = number(f"SELECT count(*) FROM {quote(rel.table)} c WHERE NOT ({present})")
            add(
                f"{rel.table}.relationship({','.join(rel.columns)})",
                bad == 0 and (rel.nullable or nulls == 0),
                f"orphans={bad:g}, NULL tuples={nulls:g}, nullable={rel.nullable}",
            )

        for seq in contract.sequences:
            partition = refs(seq.table, seq.partition_by)
            order = refs(seq.table, seq.order_by)
            refs(seq.table, [seq.timestamp])
            end_col = seq.end_timestamp or seq.timestamp
            refs(seq.table, [end_col])
            if any(not columns[seq.table][c].startswith(("DATE", "TIMESTAMP")) for c in (seq.timestamp, end_col)):
                raise ValueError("Sequence endpoints must be temporal columns")
            unique(seq.table, seq.partition_by + seq.order_by, f"{seq.table}.sequence_grain")
            n = number(
                f"SELECT count(*) FROM (SELECT {quote(seq.timestamp)} t, {quote(end_col)} e, "
                f"lag({quote(end_col)}) OVER (PARTITION BY {partition} ORDER BY {order}) prev "
                f"FROM {quote(seq.table)}) WHERE t IS NULL OR e IS NULL OR e<t OR t<prev"
            )
            add(f"{seq.table}.sequence", n == 0, f"{n:g} reversed, overlapping or missing event times")

        daily = {}
        for series in contract.series:
            refs(series.table, [series.date_column])
            if contract.tables[series.table].dates.get(series.date_column) != "actual":
                raise ValueError(f"{series.name}: time series must use an actual-event date")
            agg = measure(series.table, series.measure)
            scope_sql, scope_args = "", []
            for column, allowed in series.scope.items():
                refs(series.table, [column])
                if not allowed:
                    raise ValueError(f"{series.name}: scope values cannot be empty")
                scope_sql += f" AND {quote(column)} IN ({','.join('?' for _ in allowed)})"
                scope_args.extend(allowed)
            # Include inactive dates for additive measures; AVG excludes days without observations.
            raw = dict(
                con.execute(
                    f"SELECT CAST({quote(series.date_column)} AS DATE), {agg} FROM {quote(series.table)} "
                    f"WHERE CAST({quote(series.date_column)} AS DATE) BETWEEN ? AND ? {scope_sql} GROUP BY 1",
                    [start, end, *scope_args],
                ).fetchall()
            )
            if not raw:
                raise ValueError(f"{series.name}: empty observed series")
            values = {}
            for offset in range((end - start).days + 1):
                day = start + timedelta(days=offset)
                value = raw.get(day, None if series.measure.aggregate == "avg" else 0)
                if value is not None:
                    values[day] = float(value)
            daily[series.name] = values
            months = {}
            for day, value in values.items():
                months.setdefault(day.replace(day=1), []).append(value)
            complete = []
            for month, vals in sorted(months.items()):
                last = (month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                if month >= start and last <= end:
                    complete.append(sum(vals) / len(vals) if series.measure.aggregate == "avg" else sum(vals))
            if len(complete) < 3 or min(complete) <= 0 or not all(math.isfinite(v) for v in complete):
                raise ValueError(f"{series.name}: need at least three complete months with positive finite aggregates")
            ratio, change = max(complete) / min(complete), complete[-1] / complete[0]
            verdict = {
                "stable": ratio <= 1.3,
                "growth": change >= 1.15,
                "decline": change <= 0.85,
                "seasonal": ratio >= 1.3,
            }[series.monthly]
            add(
                f"{series.name}.monthly",
                verdict,
                f"{series.monthly}: max/min={ratio:.3f}, last/first={change:.3f}; {series.reason}",
            )
            weekdays = [v for d, v in values.items() if d.weekday() < 5]
            weekends = [v for d, v in values.items() if d.weekday() >= 5]
            if not weekdays or not weekends or sum(weekdays) <= 0:
                raise ValueError(f"{series.name}: weekly comparison has no positive weekday baseline")
            ratio = (sum(weekends) / len(weekends)) / (sum(weekdays) / len(weekdays))
            verdict = {"flat": 0.88 < ratio < 1.12, "weekday_heavy": ratio <= 0.88, "weekend_heavy": ratio >= 1.12}[
                series.weekly
            ]
            add(f"{series.name}.weekly", verdict, f"{series.weekly}: weekend/weekday={ratio:.3f}")

        for distribution in contract.distributions:
            group = refs(distribution.table, distribution.entity)
            agg = measure(distribution.table, distribution.measure)
            values = sorted(
                float(r[0])
                for r in con.execute(f"SELECT {agg} FROM {quote(distribution.table)} GROUP BY {group}").fetchall()
            )
            if len(values) < 2 or min(values) < 0 or sum(values) <= 0 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{distribution.name}: need >=2 nonnegative finite entity totals and a positive total")
            count, total = len(values), sum(values)
            top_share = sum(values[-max(1, math.ceil(count * 0.1)) :]) / total
            head = values[-1] / total
            if distribution.shape == "long_tail":
                floor = 0.3 if count < 60 else 0.4 if count < 200 else 0.5
                ok = count >= 10 and floor <= top_share <= 0.92 and head <= 0.5
            else:
                ok = min(values) > 0 and head <= min(1.0, 2.0 / count)
            add(
                f"{distribution.name}.distribution",
                ok,
                f"{distribution.shape}: {count} entities, top10%={top_share:.3f}, head={head:.3f}; {distribution.reason}",
            )

        for anomaly in contract.anomalies:
            if anomaly.series not in daily or not start <= anomaly.start <= anomaly.end <= end:
                raise ValueError(f"{anomaly.name}: invalid series or anomaly interval")
            values = daily[anomaly.series]
            baseline_start = anomaly.start - timedelta(days=(anomaly.end - anomaly.start).days + 1)
            if baseline_start < start:
                raise ValueError(f"{anomaly.name}: comparison window is outside observed dates")
            effect = [v for d, v in values.items() if anomaly.start <= d <= anomaly.end]
            baseline = [v for d, v in values.items() if baseline_start <= d < anomaly.start]
            if not effect or not baseline or sum(baseline) <= 0:
                raise ValueError(f"{anomaly.name}: missing effect or positive baseline")
            ratio = (sum(effect) / len(effect)) / (sum(baseline) / len(baseline))
            ok = ratio >= 1.2 if anomaly.direction == "up" else ratio <= 0.8
            add(
                f"{anomaly.name}.anomaly",
                ok,
                f"{anomaly.direction}: window/preceding-window={ratio:.3f}; {anomaly.explanation}",
            )
    except (ValueError, TypeError, KeyError, IndexError, duckdb.Error) as exc:
        add("semantic contract executable", False, str(exc))
    return checks
