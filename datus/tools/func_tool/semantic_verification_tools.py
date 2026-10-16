# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Query business outputs once and compare them against trusted source results."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from time import perf_counter
from typing import Any, Callable, Literal

import sqlglot
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import expressions as exp
from sqlglot.optimizer.scope import traverse_scope

from datus.tools.func_tool.base import FuncToolResult, tool_schema, trans_to_function_tool


class MetricQueryCheck(BaseModel):
    """Query arguments for full-result comparison, with narrower scope alignment than query_metrics."""

    model_config = ConfigDict(extra="forbid")
    dimensions: list[str] = Field(default_factory=list, description="Exact queryable dimension names from get_metric")
    time_start: str | None = Field(default=None, description="Inclusive business-time lower bound")
    time_end: str | None = Field(default=None, description="Exclusive business-time upper bound")
    time_granularity: str | None = Field(default=None, description="Requested time grain, e.g. month or day")
    where: str | None = Field(
        default=None,
        description=(
            "SQL condition without WHERE. Numerical comparison supports only result-row filters on mapped "
            "grouping dimensions and the primary metric; input filters, subqueries, aggregates and windows "
            "cannot be aligned automatically. Query execution is recorded even if comparison is unverified."
        ),
    )
    context_filter: str | None = Field(
        default=None,
        description=(
            "Input-population filter passed to query_metrics. Any nonempty value currently makes numerical "
            "comparison unverified (input_population_unconfirmed); query execution is recorded separately. "
            "Preserve required population filters rather than removing them to obtain a numerical pass."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="One scalar value per parameter; no binding arrays"
    )


class SemanticOutputCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Requested business output, not an extra KPI discovered in SQL")
    metric: str = Field(description="One final business metric; its helpers are queried through its definition")
    companion_metrics: list[str] = Field(
        default_factory=list,
        description="Additional metrics for a necessary combined query, after checking the primary output independently",
    )
    query: MetricQueryCheck = Field(default_factory=MetricQueryCheck)
    source_output_id: str = Field(
        min_length=1,
        description=(
            "Copy the exact id from inspect_semantic_sources result.calculation_evidence.outputs[].id. "
            "Select the trusted original SQL's final output by source statement, column name and expression. "
            "Pass the returned output ID unchanged."
        ),
    )
    reference: Literal["sql", "stored"] = Field(
        default="sql", description="Original SELECT, or its confirmed stored columns"
    )
    dimension_mapping: dict[str, str] = Field(
        default_factory=dict, description="Reference column -> query dimension; unique exact-name matches are inferred"
    )
    source_parameters: dict[str, Any] = Field(default_factory=dict, description="Scalar bindings for ${name} in SQL")
    reference_where: str | None = Field(default=None, description="Explicit filter on the reference result rows")
    absolute_tolerance: str = Field(
        default="0.000001", description="Nonnegative finite absolute tolerance; do not widen to obtain a pass"
    )
    relative_tolerance: str = Field(
        default="0", description="Nonnegative finite relative tolerance; default exact relative comparison"
    )


class _VerificationProblem(ValueError):
    """A comparison precondition, independent of the generated metric's correctness."""

    def __init__(self, category: str, code: str, reason: str, action: str, **details):
        super().__init__(reason)
        self.result = {
            "status": "unverified",
            "query_status": "not_run",
            "comparison_status": "not_run",
            "category": category,
            "reason_code": code,
            "reason": reason,
            "next_action": action,
            **details,
        }


def _rows(data: Any, columns: list[str] | None = None) -> list[dict]:
    if hasattr(data, "to_pylist"):
        data = data.to_pylist()
    elif hasattr(data, "to_dict"):
        data = data.to_dict(orient="records")
    if not isinstance(data, list):
        raise ValueError("Complete result rows are unavailable")
    if data and not isinstance(data[0], dict):
        if not columns:
            raise ValueError("Result column names are unavailable")
        return [dict(zip(columns, row)) for row in data]
    return data


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
        return parsed
    except (ValueError, InvalidOperation):
        return None


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = value.isoformat() if isinstance(value, date) else str(value)
    if len(text) == 4:
        text += "-01-01"
    elif len(text) == 7:
        text += "-01"
    return datetime.fromisoformat(text)


def compare_output_rows(
    expected: list[dict],
    actual: list[dict],
    reference_value: str,
    metric_value: str,
    dimensions: dict[str, str],
    *,
    absolute_tolerance: Decimal = Decimal("0.000001"),
    relative_tolerance: Decimal = Decimal(0),
    time_columns: set[str] | None = None,
    time_granularity: str | None = None,
    temporal_value: bool = False,
) -> dict:
    """Compare group sets and values without reaggregating or dropping duplicates."""

    def key_value(value, column):
        if column in (time_columns or set()) and value is not None:
            try:
                parsed = _as_datetime(value)
            except ValueError:
                raise ValueError(f"Cannot normalize time key {value!r} at {time_granularity} grain") from None
            if time_granularity == "month":
                return parsed.strftime("%Y-%m")
            if time_granularity == "year":
                return parsed.strftime("%Y")
            if time_granularity == "day":
                return parsed.strftime("%Y-%m-%d")
            return parsed.isoformat()
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value

    def index(rows, keys, value_column, source):
        indexed = {}
        for row in rows:
            missing = [k for k in [*keys, value_column] if k not in row]
            if missing:
                raise ValueError(f"{source} result is missing columns: {', '.join(missing)}")
            key = tuple(key_value(row[c], c) for c in keys)
            if key in indexed:
                raise ValueError(f"{source} has duplicate rows at the comparison grain: {key!r}")
            indexed[key] = row[value_column]
        return indexed

    try:
        left = index(expected, list(dimensions), reference_value, "Reference")
        right = index(actual, list(dimensions.values()), metric_value, "Metric")
    except (ValueError, TypeError) as exc:
        return {"status": "unverified", "reason": str(exc)}
    if not left and not right:
        return {"status": "unverified", "reason": "Empty results cannot prove business equivalence"}
    missing, extra = left.keys() - right.keys(), right.keys() - left.keys()
    differences = []
    difference_count = 0
    non_null = False
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if a is None or b is None:
            equal = a is None and b is None
        elif temporal_value or isinstance(a, (date, datetime)) or isinstance(b, (date, datetime)):
            non_null = True
            try:
                equal = _as_datetime(a) == _as_datetime(b)
            except (ValueError, TypeError):
                return {"status": "unverified", "reason": "Cannot normalize a declared temporal result value"}
        else:
            non_null = True
            da, db = _decimal(a), _decimal(b)
            if any(value is not None and not value.is_finite() for value in (da, db)):
                return {"status": "unverified", "reason": "Non-finite values cannot prove business equivalence"}
            if da is not None and db is not None:
                equal = (
                    da == db
                    if isinstance(a, int)
                    else abs(da - db) <= max(absolute_tolerance, relative_tolerance * max(abs(da), abs(db)))
                )
            else:
                equal = a == b
        if not equal:
            difference_count += 1
            if len(differences) < 5:
                differences.append({"group": list(key), "reference": a, "metric": b})
    status = "failed" if missing or extra or difference_count else "passed" if non_null else "unverified"
    return {
        "status": status,
        "reason": (
            f"Missing {len(missing)} groups, extra {len(extra)} groups, {difference_count} value differences"
            if status == "failed"
            else "All groups and values agree"
            if status == "passed"
            else "All compared values are NULL; business equivalence is unverified"
        ),
        "reference_groups": len(left),
        "metric_groups": len(right),
        "missing_groups": len(missing),
        "extra_groups": len(extra),
        "value_differences": difference_count,
        "examples": {
            "missing": [list(k) for k in sorted(missing, key=repr)[:5]],
            "extra": [list(k) for k in sorted(extra, key=repr)[:5]],
            "different": differences,
        },
    }


class SemanticVerificationTools:
    permission_category = "semantic_tools"

    def __init__(self, db_tool, semantic_tools, discovery_tools, generation_evidence, artifact_provider: Callable):
        self.db_tool = db_tool
        self.semantic_tools = semantic_tools
        self.discovery = discovery_tools
        self.evidence = generation_evidence
        self.artifact_provider = artifact_provider
        self._references: dict[tuple, list[dict]] = {}
        self._blocked_checks: dict[tuple, dict] = {}

    def reset_request_cache(self) -> None:
        self._references.clear()
        self._blocked_checks.clear()

    def available_tools(self):
        return [trans_to_function_tool(self.verify_semantic_outputs)]

    @tool_schema(strict_mode=False)
    def verify_semantic_outputs(self, checks: list[SemanticOutputCheck]) -> FuncToolResult:
        """Compare business outputs to trusted original SQL or confirmed stored values.

        Use only when a trusted reference exists. For execution-only checks without
        a trusted baseline, call query_metrics directly instead. This tool runs
        each metric query once internally, so do not query it separately first.
        It then checks the reference grain, mapping, population and filters before comparing.
        Never add dimensions or change a metric's window to satisfy a reference
        that cannot be mapped. Results include category,
        reason_code and next_action: fix_check/inspect_query/inspect_difference concern
        different evidence; record_unverified means stop changing the model for this check.
        Repeating an unchanged blocked check returns its diagnosis without rerunning it.
        Supply final metric names and query dimensions from get_metric. The required
        source_output_id identifies a final output from trusted original SQL.
        Preserve the source calculation grain and
        bind the same population, time and parameters on both sides. A reference_where
        filters result rows, never a window's input population. `stored` reads the
        confirmed target columns directly without SUM. A missing original grouping key,
        duplicate grain, empty/all-NULL result, template or missing dependency is reported
        rather than approximated. After independent checks, companion_metrics exercises a
        necessary combined query against the same primary reference output. Derived metrics
        exercise their definition chains. No arbitrary preview limit is introduced.

        Args:
            checks: Requested business outputs, their final metrics, query arguments,
                required source IDs and reference-column to Dosi-dimension mappings.
        """
        try:
            validated = [
                c if isinstance(c, SemanticOutputCheck) else SemanticOutputCheck.model_validate(c) for c in checks
            ]
            if not validated:
                raise ValueError("Provide at least one requested business output")
            artifact = self.artifact_provider()
            if not artifact or self.evidence._semantic_artifact_state(artifact) is None:
                raise ValueError("Bind an existing semantic-model artifact before verifying its outputs")
            snapshot = self.discovery.source_snapshot()
            results = []
            run_id = self.evidence.run_id
            for check in validated:
                started = perf_counter()
                before = self.evidence._business_artifact_state(artifact)
                source_revision = snapshot["revision"]
                cache_key = (
                    run_id,
                    json.dumps(before, sort_keys=True),
                    source_revision,
                    json.dumps(check.model_dump(exclude={"name"}), sort_keys=True),
                )
                if cache_key in self._blocked_checks:
                    result = {
                        **self._blocked_checks[cache_key],
                        "unchanged_check": True,
                    }
                else:
                    try:
                        result = self._verify(check, snapshot)
                    except _VerificationProblem as exc:
                        result = exc.result
                        self._blocked_checks[cache_key] = dict(result)
                    except Exception as exc:
                        result = {
                            "query_status": "unverified",
                            "status": "unverified",
                            "comparison_status": "not_run",
                            "category": "verification_error",
                            "reason_code": "verification_error",
                            "reason": str(exc),
                            "next_action": "record_unverified",
                        }
                result.update(name=check.name, metric=check.metric, wall_clock_seconds=perf_counter() - started)
                result["query"] = check.query.model_dump(exclude_none=True)
                if check.companion_metrics:
                    result["companion_metrics"] = check.companion_metrics
                result.update(source_output_id=check.source_output_id, reference=check.reference)
                record = {
                    **result,
                    "check": check.model_dump(),
                    "source_revision": source_revision,
                    "artifact": before,
                    "run_id": run_id,
                }
                self.evidence.record_business_verification(artifact, record)
                results.append(result)
            return FuncToolResult(result={"checks": results})
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))

    def _verify(self, check: SemanticOutputCheck, snapshot: dict) -> dict:
        query = check.query.model_dump(exclude_none=True)
        if any(
            isinstance(value, (dict, list))
            for value in [*check.query.params.values(), *check.source_parameters.values()]
        ):
            raise _VerificationProblem(
                "invalid_check",
                "scalar_binding_required",
                "Use one scalar parameter binding per check, not binding arrays",
                "fix_check",
            )
        metrics = list(dict.fromkeys([check.metric, *check.companion_metrics]))
        # query_metrics already compiles before execution. A separate dry run
        # repeats that work and cannot establish numerical correctness.
        response = self.semantic_tools.query_metrics(metrics=metrics, **query)
        if not response.success:
            return self._query_failure(response, "query")
        try:
            return self._compare(check, snapshot, response)
        except _VerificationProblem as exc:
            exc.result["query_status"] = "passed"
            raise
        except (ValueError, KeyError, TypeError) as exc:
            problem = _VerificationProblem(
                "reference_unresolved", "comparison_unavailable", str(exc), "record_unverified"
            )
            problem.result["query_status"] = "passed"
            raise problem from exc

    def _compare(self, check: SemanticOutputCheck, snapshot: dict, response: FuncToolResult) -> dict:
        query = check.query.model_dump(exclude_none=True)
        requested = query.get("dimensions", [])
        # Infer only unique exact field-name correspondences; an explicit mapping
        # is still required for transforms, time aliases and ambiguous names.
        analysis = snapshot["analysis"]
        output = next((o for o in analysis.get("outputs", []) if o["id"] == check.source_output_id), None)
        statement = next((s for s in analysis.get("statements", []) if check.source_output_id in s["outputs"]), None)
        if output and statement:
            by_id = {o["id"]: o for o in analysis["outputs"]}
            queries = {q["id"]: q for q in analysis["queries"]}
            try:
                grain = self._grain(
                    statement, output, by_id, queries, set(check.dimension_mapping), dialect=snapshot["dialect"]
                )
            except ValueError:
                grain = []
            mapping = dict(check.dimension_mapping)
            for column in grain:
                candidates = [d for d in requested if d.rsplit(".", 1)[-1] == column]
                if column not in mapping and len(candidates) == 1 and candidates[0] not in mapping.values():
                    mapping[column] = candidates[0]
            check.dimension_mapping = mapping
        if (
            set(check.dimension_mapping.values()) != set(requested)
            or len(set(check.dimension_mapping.values())) != len(check.dimension_mapping)
            or len(set(requested)) != len(requested)
        ):
            raise _VerificationProblem(
                "invalid_check",
                "dimension_mapping_invalid",
                "dimension_mapping must map one-to-one onto the query's dimensions",
                "fix_check",
                query_dimensions=requested,
            )
        try:
            absolute, relative = Decimal(check.absolute_tolerance), Decimal(check.relative_tolerance)
        except InvalidOperation as exc:
            raise _VerificationProblem(
                "invalid_check",
                "invalid_tolerance",
                "Comparison tolerances must be finite and nonnegative",
                "fix_check",
            ) from exc
        if not all(v.is_finite() and v >= 0 for v in (absolute, relative)):
            raise _VerificationProblem(
                "invalid_check",
                "invalid_tolerance",
                "Comparison tolerances must be finite and nonnegative",
                "fix_check",
            )
        if check.reference_where:
            try:
                sqlglot.parse_one(check.reference_where, read=snapshot["dialect"], into=exp.Condition)
            except sqlglot.errors.ParseError as exc:
                raise _VerificationProblem("invalid_check", "invalid_reference_filter", str(exc), "fix_check") from exc
        reference = self._prepare_reference(check, snapshot)
        payload = response.result or {}
        cache = self.semantic_tools.get_cached_query_metrics_result(payload.get("result_id", ""))
        if cache is None:
            return {
                "query_status": "passed",
                "status": "unverified",
                "comparison_status": "not_run",
                "category": "result_unavailable",
                "reason_code": "metric_result_unavailable",
                "reason": "Complete metric result is unavailable",
                "next_action": "record_unverified",
            }
        actual = _rows(cache["data"], cache["columns"])
        query_ast, source_column, grain, time_scope = reference
        sql = query_ast.sql(dialect=snapshot["dialect"])
        key = (snapshot["revision"], sql)
        if key not in self._references:
            database = snapshot["context"].get("database")
            read = self.db_tool.read_query(sql, compress=False, **({"database": database} if database else {}))
            if not read.success:
                return {
                    "query_status": "passed",
                    "status": "unverified",
                    "comparison_status": "not_run",
                    "category": "reference_query_error",
                    "reason_code": "reference_query_failed",
                    "reason": f"Reference query could not execute: {read.error}",
                    "next_action": "record_unverified",
                }
            self._references[key] = _rows(read.result)
        expected = self._references[key]
        if time_scope:
            column, lower, upper = time_scope
            try:
                expected = [
                    row
                    for row in expected
                    if row[column] is not None
                    and (lower is None or _as_datetime(row[column]) >= lower)
                    and (upper is None or _as_datetime(row[column]) < upper)
                ]
            except (KeyError, ValueError, TypeError) as exc:
                return {
                    "query_status": "passed",
                    "status": "unverified",
                    "comparison_status": "not_run",
                    "category": "reference_unresolved",
                    "reason_code": "reference_time_unresolved",
                    "reason": f"Cannot apply the mapped time range to the reference result: {exc}",
                    "next_action": "record_unverified",
                }
        columns = cache["columns"]
        bindings = (payload.get("metadata") or {}).get("outputs") or []
        metric_columns = [
            b.get("column") or b.get("name") for b in bindings if b.get("metric", b.get("name")) == check.metric
        ]
        metric_column = metric_columns[0] if len(metric_columns) == 1 else check.metric
        if len(metric_columns) > 1:
            raise ValueError("One scalar parameter binding is required per business-output check")
        dimensions = {}
        time_columns = set()
        detail = self._metric_detail(check.metric)
        time_dimensions = {"metric_time", detail.get("time_dimension")}
        time_dimensions.update(
            d["name"]
            for d in detail.get("dimensions", [])
            if d.get("type") in {"time", "date", "datetime", "timestamp"} or d.get("is_time")
        )
        for source, dimension in check.dimension_mapping.items():
            candidates = [dimension, dimension.rsplit(".", 1)[-1]]
            if dimension == "metric_time" and check.query.time_granularity:
                candidates.append(f"metric_time__{check.query.time_granularity}")
            matches = list(dict.fromkeys(c for c in candidates if c in columns))
            if len(matches) != 1:
                raise ValueError(f"Cannot identify result column for dimension {dimension!r}")
            dimensions[source] = matches[0]
            if dimension in time_dimensions and check.query.time_granularity:
                time_columns.update((source, matches[0]))
        source_output = next(o for o in snapshot["analysis"]["outputs"] if o["id"] == check.source_output_id)
        types = snapshot.get("column_types", {})
        value_expression = (
            sqlglot.parse_one(source_output["expression"], read=snapshot["dialect"])
            if source_output.get("expression")
            else None
        )
        if isinstance(value_expression, (exp.Max, exp.Min)):
            value_expression = value_expression.this
        temporal_value = (
            isinstance(value_expression, (exp.Date, exp.Timestamp))
            or (
                isinstance(value_expression, exp.Column)
                and len(source_output["source_columns"]) == 1
                and all(
                    str(types.get(c["table"], {}).get(c["column"], "")).lower().startswith(("date", "time"))
                    for c in source_output["source_columns"]
                )
            )
            or (
                isinstance(value_expression, exp.Cast)
                and value_expression.args["to"].sql().lower().startswith(("date", "timestamp", "time"))
            )
        )
        result = compare_output_rows(
            expected,
            actual,
            source_column,
            metric_column,
            dimensions,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            time_columns=time_columns,
            time_granularity=check.query.time_granularity,
            temporal_value=temporal_value,
        )
        return {
            "query_status": "passed",
            **result,
            "comparison_grain": grain,
            "comparison_status": result["status"],
            "category": {"passed": "verified", "failed": "result_mismatch"}.get(result["status"], "result_unavailable"),
            "reason_code": {"passed": "values_match", "failed": "groups_or_values_differ"}.get(
                result["status"], "comparison_unavailable"
            ),
            "next_action": {"passed": "none", "failed": "inspect_difference"}.get(
                result["status"], "record_unverified"
            ),
        }

    def _metric_detail(self, metric: str) -> dict:
        getter = getattr(self.semantic_tools, "get_metric", None)
        if getter is None:
            return {}
        detail = getter(metric)
        return detail.result if detail.success and isinstance(detail.result, dict) else {}

    @staticmethod
    def _query_failure(response: FuncToolResult, stage: str) -> dict:
        return {
            "query_status": "failed",
            "status": "unverified",
            "comparison_status": "not_run",
            "category": "model_query_error",
            "reason_code": f"metric_{stage}_failed",
            "reason": response.error or "Metric query failed",
            "next_action": "inspect_query",
            "query_diagnostic": response.result,
        }

    def _prepare_reference(self, check: SemanticOutputCheck, snapshot: dict) -> tuple:
        """Resolve comparison prerequisites without running warehouse queries."""
        analysis = snapshot["analysis"]
        outputs = {o["id"]: o for o in analysis.get("outputs", [])}
        queries = {q["id"]: q for q in analysis.get("queries", [])}
        output = outputs.get(check.source_output_id)
        if output is None:
            raise _VerificationProblem(
                "invalid_check",
                "source_output_missing",
                "Unknown source_output_id; inspect the current SQL sources first",
                "fix_check",
            )
        statement = next((s for s in analysis["statements"] if check.source_output_id in s["outputs"]), None)
        if statement is None:
            raise _VerificationProblem(
                "invalid_check",
                "reference_not_final",
                "A reference must be a final query output, not a pure intermediate calculation",
                "fix_check",
            )
        if statement.get("input_kind") not in {"request", "original_file"}:
            raise _VerificationProblem(
                "no_reference",
                "diagnostic_reference",
                "Query executed; this SQL is diagnostic, not a caller-supplied original baseline. "
                "Agreement cannot prove reproduction of the original calculation.",
                "record_unverified",
                reference_origin=statement.get("input_kind", "unknown"),
            )
        if check.reference == "sql" and (statement.get("parameters") or statement.get("templated")):
            missing = sorted(set(statement.get("parameters", [])) - check.source_parameters.keys())
            if missing or not check.source_parameters:
                raise _VerificationProblem(
                    "invalid_check",
                    "reference_parameters_unbound",
                    "Source SQL parameters must be bound before result comparison",
                    "fix_check",
                    missing_parameters=missing,
                )
        try:
            grain = self._grain(
                statement, output, outputs, queries, set(check.dimension_mapping), dialect=snapshot["dialect"]
            )
        except ValueError as exc:
            raise _VerificationProblem(
                "reference_unresolved", "reference_grain_unresolved", str(exc), "record_unverified"
            ) from exc
        if set(check.dimension_mapping) != set(grain):
            raise _VerificationProblem(
                "invalid_check",
                "reference_grain_mapping_incomplete",
                f"Full reference grain is required: {grain!r}",
                "fix_check",
                reference_grain=grain,
                missing_reference_dimensions=sorted(set(grain) - set(check.dimension_mapping)),
                extra_reference_dimensions=sorted(set(check.dimension_mapping) - set(grain)),
            )
        try:
            required = [
                oid for oid in statement["outputs"] if oid == check.source_output_id or outputs[oid]["name"] in grain
            ]
            query = self._reference_query(check, statement, outputs, snapshot, required)
        except (ValueError, sqlglot.errors.ParseError) as exc:
            raise _VerificationProblem(
                "reference_unresolved", "reference_query_unresolved", str(exc), "record_unverified"
            ) from exc
        query = self._align_scope(check, query, output["name"], snapshot["dialect"])
        time_scope = self._time_scope(check)
        return query, output["name"], grain, time_scope

    def _time_scope(self, check: SemanticOutputCheck) -> tuple | None:
        """Select reference result buckets without changing the source window population."""
        if not (check.query.time_start or check.query.time_end):
            return None
        axes = {"metric_time"}
        if "metric_time" not in check.dimension_mapping.values():
            detail = self._metric_detail(check.metric)
            if detail:
                axes.add(detail.get("time_dimension"))
                axes.update(
                    d["name"] for d in detail.get("dimensions", []) if d.get("type") == "time" or d.get("is_time")
                )
        columns = [source for source, dimension in check.dimension_mapping.items() if dimension in axes]
        if len(columns) != 1 or not check.query.time_granularity:
            raise _VerificationProblem(
                "scope_unconfirmed",
                "time_axis_unconfirmed",
                "A bounded comparison needs one mapped business time axis and its query grain",
                "record_unverified",
            )
        try:
            lower = _as_datetime(check.query.time_start) if check.query.time_start else None
            upper = _as_datetime(check.query.time_end) if check.query.time_end else None
            if lower and upper and lower >= upper:
                raise ValueError("time_start must precede time_end")
            grain = check.query.time_granularity
            for bound in (lower, upper):
                if bound is None:
                    continue
                aligned = {
                    "second": bound.microsecond == 0,
                    "minute": bound.second == bound.microsecond == 0,
                    "hour": bound.minute == bound.second == bound.microsecond == 0,
                    "day": bound.time().isoformat() == "00:00:00",
                    "week": bound.weekday() == 0 and bound.time().isoformat() == "00:00:00",
                    "month": bound.day == 1 and bound.time().isoformat() == "00:00:00",
                    "quarter": bound.month in (1, 4, 7, 10)
                    and bound.day == 1
                    and bound.time().isoformat() == "00:00:00",
                    "year": bound.month == bound.day == 1 and bound.time().isoformat() == "00:00:00",
                }.get(grain, False)
                if not aligned:
                    raise ValueError(f"Time bounds must retain whole {grain} result buckets")
        except (ValueError, TypeError) as exc:
            raise _VerificationProblem(
                "scope_unconfirmed", "time_range_unconfirmed", str(exc), "record_unverified"
            ) from exc
        return columns[0], lower, upper

    @staticmethod
    def _align_scope(check: SemanticOutputCheck, reference: exp.Query, source_column: str, dialect: str) -> exp.Query:
        """Only translate proven result-row filters; never move input filters across a window."""
        if check.query.context_filter:
            raise _VerificationProblem(
                "scope_unconfirmed",
                "input_population_unconfirmed",
                "The metric has an input population filter whose equivalence to the source SQL is unconfirmed. "
                "Do not change the metric or reference to force a match; retain numerical verification as unverified. "
                "Query execution is recorded separately from numerical verification.",
                "record_unverified",
                comparison_scope={
                    "context_filter": check.query.context_filter,
                    "time_start": check.query.time_start,
                    "time_end": check.query.time_end,
                    "reference_where": check.reference_where,
                },
            )
        if not check.query.where and not check.reference_where:
            return reference
        if not check.query.where:
            raise _VerificationProblem(
                "invalid_check",
                "result_scope_mismatch",
                "reference_where filters result rows but the metric query has no matching where",
                "fix_check",
            )
        mapping = {dimension: source for source, dimension in check.dimension_mapping.items()}
        mapping[check.metric] = source_column
        for dimension, source in list(mapping.items()):
            short = dimension.rsplit(".", 1)[-1]
            if sum(d.rsplit(".", 1)[-1] == short for d in mapping) == 1:
                mapping[short] = source
        try:
            predicate = sqlglot.parse_one(check.query.where, read=dialect, into=exp.Condition)
        except sqlglot.errors.ParseError as exc:
            raise _VerificationProblem("invalid_check", "invalid_result_filter", str(exc), "fix_check") from exc
        if any(predicate.find_all(exp.AggFunc, exp.Window, exp.Subquery)):
            raise _VerificationProblem(
                "scope_unconfirmed",
                "result_filter_unresolved",
                "Only filters over mapped result dimensions and the primary metric can be aligned",
                "record_unverified",
            )
        for column in list(predicate.find_all(exp.Column)):
            name = ".".join(part.name for part in column.parts)
            if name not in mapping:
                raise _VerificationProblem(
                    "scope_unconfirmed",
                    "input_filter_unmapped",
                    f"Filter column {name!r} is not a mapped result dimension or primary metric; "
                    "its effect on the input population is unconfirmed",
                    "record_unverified",
                )
            # Generated filters bind to quoted output aliases; explicit source
            # filters retain their authored identifier spelling for comparison.
            column.replace(exp.column(mapping[name], quoted=not bool(check.reference_where)))
        if check.reference_where:
            other = sqlglot.parse_one(check.reference_where, read=dialect, into=exp.Condition)
            if predicate != other:
                raise _VerificationProblem(
                    "invalid_check",
                    "result_scope_mismatch",
                    "Could not confirm equivalent mapped filters for metric where and reference_where",
                    "fix_check",
                    mapped_reference_where=predicate.sql(dialect=dialect),
                )
            return reference
        return exp.select("*").from_(reference.subquery("reference_result")).where(predicate)

    @staticmethod
    def _grain(statement, output, outputs, queries, preferred_dimensions=None, *, dialect=None) -> list[str]:
        root = queries[statement["query_id"]]
        if root.get("operation") == "UNION":
            position = root["outputs"].index(output["id"])
            key_positions = []
            branch_outputs = []
            for branch_id in root["branches"]:
                branch = queries[branch_id]
                branch_outputs.append(branch["outputs"])
                branch_grain = SemanticVerificationTools._grain(
                    {"query_id": branch_id, "outputs": branch["outputs"]},
                    outputs[branch["outputs"][position]],
                    outputs,
                    queries,
                    preferred_dimensions,
                    dialect=dialect,
                )
                positions = {i for i, oid in enumerate(branch["outputs"]) if outputs[oid]["name"] in branch_grain}
                key_positions.append(positions)
            if not key_positions or any(positions != key_positions[0] for positions in key_positions):
                raise ValueError("UNION branches have different calculation grains")
            positions = set(key_positions[0])
            # Constants such as a brand label distinguish otherwise identical
            # branch grains. They are output dimensions, not aggregate inputs.
            for i in range(len(root["outputs"])):
                if i == position:
                    continue
                expressions = [outputs[oids[i]]["expression"] for oids in branch_outputs]
                if all(expression is not None for expression in expressions):
                    nodes = [sqlglot.parse_one(expression, read=dialect) for expression in expressions]
                    if all(isinstance(node, (exp.Literal, exp.Null, exp.Boolean)) for node in nodes):
                        if len(set(expressions)) > 1:
                            positions.add(i)
            return [outputs[oid]["name"] for i, oid in enumerate(root["outputs"]) if i in positions]
        if root.get("operation") != "SELECT":
            raise ValueError("Set-operation result grain requires confirmation before numerical comparison")
        # An outer aggregate establishes its own grain. Otherwise trace the
        # selected calculation's grouping outputs through final projections.
        outer_aggregate = root.get("group_by") or any(outputs[oid]["kind"] == "aggregate" for oid in root["outputs"])
        calculations = [root] if outer_aggregate else [queries[q] for q in output["calculation_queries"]]
        calculations = [
            q
            for q in calculations
            if q.get("group_by") or not any(outputs[oid]["kind"] == "window" for oid in q["outputs"])
        ]
        if not calculations and not root.get("inputs"):
            return []
        if not calculations:
            raise ValueError("Original calculation grain is unconfirmed")
        final = statement["outputs"]

        def depends(oid, target, active=None):
            active = set() if active is None else active
            if oid == target:
                return True
            if oid in active:
                return False
            active.add(oid)
            # A value-transform is not evidence that it preserves a grouping key.
            current = outputs[oid]
            if current["kind"] != "passthrough":
                return False
            return any(depends(dep, target, active) for dep in current["dependencies"])

        names = []
        for calculation in calculations:
            if calculation.get("operation") != "SELECT":
                raise ValueError("Original calculation grain is unconfirmed")
            if calculation.get("grouping"):
                raise ValueError("Grouping sets require explicit grain confirmation")
            groups = calculation.get("group_by_output_sets")
            if groups is None:
                groups = [[oid] for oid in calculation.get("group_by_outputs", [])]
            if len(calculation.get("group_by", [])) != len(groups) or any(not group for group in groups):
                raise ValueError("Some original GROUP BY keys are not projected")
            for equivalent_outputs in groups:
                candidates = list(
                    dict.fromkeys(
                        outputs[oid]["name"]
                        for oid in final
                        if any(depends(oid, group) for group in equivalent_outputs)
                    )
                )
                preferred = [name for name in candidates if name in (preferred_dimensions or set())]
                if len(preferred) == 1:
                    candidates = preferred
                if len(candidates) != 1:
                    raise ValueError("Original grouping key does not map uniquely to a final output")
                if candidates[0] not in names:
                    names.append(candidates[0])
        return names

    def _reference_query(self, check, statement, outputs, snapshot, required_outputs) -> exp.Query:
        dialect = snapshot["dialect"]
        projection_outputs = statement["outputs"]
        if check.reference == "stored":
            table = statement.get("target_table")
            projection_outputs = required_outputs
            if not table or any(not outputs[o].get("target_column") for o in projection_outputs):
                raise ValueError("Stored target column mapping is unconfirmed")
            query = exp.select(
                *[
                    exp.alias_(exp.column(outputs[o]["target_column"], quoted=True), outputs[o]["name"], quoted=True)
                    for o in projection_outputs
                ]
            ).from_(exp.to_table(table, dialect=dialect))
        else:
            sql = statement["query_sql"]
            from datus.utils.sql_lineage import preprocess_template

            normalized, _ = preprocess_template(sql)
            query = sqlglot.parse_one(normalized, read=dialect)
            for node in list(query.walk()):
                if not isinstance(node, (exp.Literal, exp.Identifier)):
                    continue
                text = str(node.this)
                for name, value in check.source_parameters.items():
                    marker = f"__tpl_{name}__"
                    if text == marker and isinstance(node, exp.Identifier):
                        if isinstance(node.parent, exp.Table):
                            raise ValueError("Dynamic table parameters require explicit source resolution")
                        if not isinstance(node.parent, exp.Column) or node.parent.table:
                            raise ValueError("Only scalar SQL parameters can be bound")
                        node.parent.replace(exp.convert(value))
                        break
                    if marker in text and isinstance(node, exp.Literal) and node.is_string:
                        text = text.replace(marker, str(value))
                        node.set("this", text)
            if "__tpl" in query.sql():
                raise ValueError("Source SQL contains unresolved template constructs")
        context = snapshot["context"]
        for scope in traverse_scope(query):
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    if not source.db and context.get("prefix"):
                        source.set("db", exp.to_identifier(context["prefix"]))
                    if not source.catalog and context.get("catalog"):
                        source.set("catalog", exp.to_identifier(context["catalog"]))
        if isinstance(query, exp.Select) and not any(p.is_star for p in query.selects):
            if len(query.selects) != len(projection_outputs):
                raise ValueError("Reference output columns are unresolved")
            query.set(
                "expressions",
                [
                    exp.alias_(projection.unalias(), outputs[oid]["name"], quoted=True)
                    for projection, oid in zip(query.selects, projection_outputs)
                ],
            )
        if check.reference_where:
            predicate = sqlglot.parse_one(check.reference_where, read=dialect, into=exp.Condition)
            query = exp.select("*").from_(query.subquery("reference_result")).where(predicate)
        return query
