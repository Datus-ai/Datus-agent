# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Execute one metric and one reference query, then compare their full results."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from datus.tools.func_tool.base import FuncToolResult, tool_schema, trans_to_function_tool


class MetricQueryCheck(BaseModel):
    """Arguments passed unchanged to query_metrics, without a preview limit."""

    model_config = ConfigDict(extra="forbid")
    dimensions: list[str] = Field(default_factory=list, description="Query dimensions from get_metric")
    time_start: str | None = Field(default=None, description="Inclusive business-time lower bound")
    time_end: str | None = Field(default=None, description="Exclusive business-time upper bound")
    time_granularity: str | None = Field(default=None, description="Query time grain, e.g. month or day")
    where: str | None = Field(default=None, description="Metric result filter; match its scope in reference_sql")
    context_filter: str | None = Field(
        default=None, description="Metric input-population filter; match its scope in reference_sql"
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Metric parameter bindings")


class SemanticOutputCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Requested business output")
    metric: str = Field(description="One final business metric, checked independently of other outputs")
    reference_sql: str = Field(
        min_length=1,
        description=(
            "Executable read-only SELECT/WITH for the matching population, time range and result grain. "
            "Use the source SELECT or query stored result columns directly; do not reaggregate stored values "
            "or rewrite a calculation merely to obtain a pass."
        ),
    )
    reference_column: str = Field(min_length=1, description="Value column returned by reference_sql")
    query: MetricQueryCheck = Field(default_factory=MetricQueryCheck)
    dimension_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Reference result key -> metric query dimension or returned key column. "
            "Use {} for scalar results. Only these keys are compared; no dimensions are added. "
            "Time dimensions normalize ISO date representations, including YYYY-MM and YYYY. "
            "Duplicate keys in either full result are reported as unverified."
        ),
    )
    absolute_tolerance: str = Field(default="0.000001", description="Finite nonnegative absolute tolerance")
    relative_tolerance: str = Field(default="0", description="Finite nonnegative relative tolerance")


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
    time_dimensions: dict[str, str | None] | None = None,
) -> dict:
    """Compare group sets and values without reaggregating or dropping duplicates."""

    def key_value(value, column):
        if column in (time_dimensions or {}) and value is not None:
            time_granularity = time_dimensions[column]
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
            missing = [k for k in [*keys.values(), value_column] if k not in row]
            if missing:
                raise ValueError(f"{source} result is missing columns: {', '.join(missing)}")
            key = tuple(key_value(row[c], reference_key) for reference_key, c in keys.items())
            if key in indexed:
                raise ValueError(f"{source} has duplicate rows at the comparison grain: {key!r}")
            indexed[key] = row[value_column]
        return indexed

    try:
        left = index(expected, {key: key for key in dimensions}, reference_value, "Reference")
        right = index(actual, dimensions, metric_value, "Metric")
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
        elif isinstance(a, (date, datetime)) or isinstance(b, (date, datetime)):
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
                equal = abs(da - db) <= max(absolute_tolerance, relative_tolerance * max(abs(da), abs(db)))
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

    def __init__(self, db_tool, semantic_tools, generation_evidence, artifact_provider: Callable):
        self.db_tool = db_tool
        self.semantic_tools = semantic_tools
        self.evidence = generation_evidence
        self.artifact_provider = artifact_provider

    def available_tools(self):
        return [trans_to_function_tool(self.verify_semantic_outputs)]

    @tool_schema(strict_mode=False)
    def verify_semantic_outputs(self, checks: list[SemanticOutputCheck]) -> FuncToolResult:
        """Compare each business output with an explicit reference SELECT.

        Each check queries exactly one metric and executes reference_sql once through
        the existing read-only database tool. It compares full cached metric rows,
        never the compressed preview. Supply the reference value column and result-key
        mapping explicitly; these keys determine which result groups are compared.
        The metric must belong to the selected model when one is selected. Recorded
        comparisons are bound to the metric's actual model file and version.

        Choose matching populations, time ranges, parameters and business grouping in
        both queries. Query arguments are passed unchanged; where filters result rows
        and context_filter scopes inputs according to query_metrics semantics. Write
        those conditions at the corresponding stages of reference_sql, especially for
        windows. Comparison checks the supplied results; it does not prove that two
        independently chosen SQL definitions have the same business meaning.

        Results are passed, failed or unverified with a reason, group counts and bounded
        difference examples. Missing/extra groups and unequal values fail. Query errors,
        missing result columns, duplicate keys, empty/all-NULL or unavailable full results
        are unverified. Each invocation executes again. After fixing the model or query,
        rerun that check. Check outputs individually before using query_metrics for a
        necessary combined query; another metric must not supply missing zero groups.

        Args:
            checks: Business outputs, each with metric, query, reference_sql,
                reference_column and dimension_mapping. For example, reference_sql
                returns market,n; metric revenue groups by orders.market; set
                reference_column="n" and dimension_mapping={"market":"orders.market"}.
        """
        try:
            validated = [
                c if isinstance(c, SemanticOutputCheck) else SemanticOutputCheck.model_validate(c) for c in checks
            ]
            if not validated:
                raise ValueError("Provide at least one business output")
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))

        results = []
        for check in validated:
            started = perf_counter()
            before = None
            run_id = self.evidence.run_id
            result = {"status": "unverified", "query_status": "not_run", "reference_status": "not_run"}
            try:
                binding = self.semantic_tools.runtime.artifact_metric_binding(check.metric)
                actual_path = str(Path(binding["model_path"]).resolve())
                selected = self.artifact_provider()
                if selected and actual_path != str(Path(selected).resolve()):
                    raise ValueError(
                        f"Metric {check.metric!r} belongs to {actual_path}, not the selected model {selected}"
                    )
                before = {"path": actual_path, "sha256": binding["model_revision"]}
                self._compare(check, result)
                if self.evidence._semantic_artifact_state(actual_path) != before:
                    result.update(status="unverified", reason="Model changed during this comparison; rerun the check")
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                result.update(status="unverified", reason=str(exc))
            except Exception as exc:
                result.update(status="unverified", reason=f"Comparison could not complete: {exc}")
            result.update(
                name=check.name,
                metric=check.metric,
                query=check.query.model_dump(exclude_none=True),
                reference_column=check.reference_column,
                dimension_mapping=check.dimension_mapping,
                wall_clock_seconds=perf_counter() - started,
            )
            if before is not None:
                self.evidence.record_business_verification(
                    before["path"], {**result, "artifact": before, "check": check.model_dump(), "run_id": run_id}
                )
            results.append(result)
        return FuncToolResult(result={"checks": results})

    def _compare(self, check: SemanticOutputCheck, result: dict) -> None:
        absolute, relative = Decimal(check.absolute_tolerance), Decimal(check.relative_tolerance)
        if not all(value.is_finite() and value >= 0 for value in (absolute, relative)):
            raise ValueError("Comparison tolerances must be finite and nonnegative")
        if len(set(check.dimension_mapping.values())) != len(check.dimension_mapping):
            raise ValueError("Each reference key must map to a different metric result key")

        response = self.semantic_tools.query_metrics(
            metrics=[check.metric], **check.query.model_dump(exclude_none=True)
        )
        if not response.success:
            result.update(query_status="failed", reason=f"Metric query failed: {response.error}")
            if response.result:
                result["query_diagnostic"] = response.result
            return
        result["query_status"] = "passed"
        payload = response.result or {}
        cache = self.semantic_tools.get_cached_query_metrics_result(payload.get("result_id", ""))
        if cache is None:
            raise ValueError("Complete metric result is unavailable")
        actual = _rows(cache["data"], cache["columns"])
        bindings = (payload.get("metadata") or {}).get("outputs") or []
        metric_columns = list(
            dict.fromkeys(
                binding.get("column") or binding.get("name")
                for binding in bindings
                if binding.get("metric", binding.get("name")) == check.metric
            )
        )
        if len(metric_columns) > 1:
            raise ValueError("The parameter bindings produce multiple value columns; compare one binding per check")
        metric_column = metric_columns[0] if metric_columns else check.metric
        time_bindings = {
            dimension: (self._result_column(dimension, cache["columns"], grain), grain)
            for dimension, grain in (payload.get("metadata") or {}).get("time_dimensions", {}).items()
        }
        dimensions = {
            source: time_bindings[dimension][0]
            if dimension in time_bindings
            else self._result_column(dimension, cache["columns"], None)
            for source, dimension in check.dimension_mapping.items()
        }
        time_dimensions = {
            source: grain
            for source, column in dimensions.items()
            for time_column, grain in time_bindings.values()
            if column == time_column
        }

        reference = self.db_tool.read_query(check.reference_sql, compress=False)
        if not reference.success:
            result.update(reference_status="failed", reason=f"Reference query failed: {reference.error}")
            return
        result["reference_status"] = "passed"
        result.update(
            compare_output_rows(
                _rows(reference.result),
                actual,
                check.reference_column,
                metric_column,
                dimensions,
                absolute_tolerance=absolute,
                relative_tolerance=relative,
                time_dimensions=time_dimensions,
            )
        )

    @staticmethod
    def _result_column(dimension: str, columns: list[str], grain: str | None) -> str:
        candidates = [dimension, dimension.rsplit(".", 1)[-1]]
        if grain:
            candidates = [f"{candidate}__{grain}" for candidate in candidates]
        matches = list(dict.fromkeys(candidate for candidate in candidates if candidate in columns))
        if len(matches) != 1:
            raise ValueError(f"Cannot identify metric result column for {dimension!r}; returned columns: {columns}")
        return matches[0]
