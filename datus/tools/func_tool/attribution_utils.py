# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Adapter-agnostic dimension attribution analysis."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.models import (
    AttributionComparisonMetadata,
    AttributionDimensionDetail,
    AttributionDimensionScore,
    AttributionDrillDown,
    AttributionReconciliation,
    AttributionRequest,
    AttributionResult,
    AttributionTotalChange,
    AttributionUnsupportedInfo,
    AttributionValueContribution,
    AttributionWarning,
    AttributionWindow,
    QueryResult,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

JsonScalar = Union[str, int, float, bool]
_MAX_DIMENSION_VALUES = 1000
_ADDITIVITY_TOLERANCE = 0.02
_ZERO_DELTA_EPSILON = 1e-6


class AdditivityCheck(BaseModel):
    """Whether grouped values reconcile to the corresponding period totals."""

    status: Literal["passed", "failed", "skipped", "unknown"] = "unknown"
    baseline_residual: Optional[float] = None
    current_residual: Optional[float] = None
    baseline_residual_pct: Optional[float] = None
    current_residual_pct: Optional[float] = None
    delta_residual_pct: Optional[float] = None


class DimensionAnalysisError(BaseModel):
    """A query or result-validation failure isolated to one dimension."""

    error_type: Literal["query_error", "validation_error"]
    code: str
    message: str
    period: Optional[Literal["baseline", "current"]] = None
    columns: List[str] = Field(default_factory=list)
    row_count: Optional[int] = None


class AttributionValidationErrorPayload(BaseModel):
    """Structured fatal validation failure returned by the public tool wrapper."""

    error_type: Literal["attribution_validation_error"] = "attribution_validation_error"
    code: str
    message: str
    period: Optional[str] = None
    dimension: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    row_count: Optional[int] = None


class AttributionValidationException(Exception):
    """Raised when an adapter result cannot be safely used for attribution."""

    def __init__(self, payload: AttributionValidationErrorPayload):
        self.payload = payload
        super().__init__(payload.message)


# ==================== Attribution Util ====================


class GenericAttributeAnalyzer:
    """Term-wise attribution fallback for adapters without a native implementation."""

    def __init__(self, adapter: BaseSemanticAdapter):
        self.adapter = adapter

    async def attribute(self, request: AttributionRequest) -> AttributionResult:
        """Rank candidate dimensions over OSI half-open time windows."""
        metric_name = request.metric
        candidate_dimensions = list(dict.fromkeys(request.dimensions))
        baseline_days = self._validate_time_window(
            period="baseline",
            start=request.baseline.start,
            end=request.baseline.end,
        )
        current_days = self._validate_time_window(
            period="current",
            start=request.current.start,
            end=request.current.end,
        )
        warnings: List[AttributionWarning] = []
        effective_time_dimension = None
        if request.time_dimension:
            try:
                available_dimensions = await self.adapter.get_dimensions(
                    metric_name,
                    path=request.path,
                )
            except Exception as error:
                logger.warning(
                    "Could not verify generic attribution time dimension '%s': %s",
                    request.time_dimension,
                    error,
                )
                available_dimensions = []
            primary_time_dimensions = [
                dimension.name for dimension in available_dimensions if dimension.is_primary_time
            ]
            if request.time_dimension not in primary_time_dimensions:
                return AttributionResult(
                    metric=metric_name,
                    implementation="generic",
                    strategy="unsupported",
                    unsupported_reason=AttributionUnsupportedInfo(
                        code="time_dimension_not_supported",
                        message=(
                            "Generic attribution cannot honor the requested time dimension "
                            f"'{request.time_dimension}' through query_metrics; use native attribution "
                            "or request the adapter's primary time dimension."
                        ),
                    ),
                    comparison_metadata=self._comparison_metadata(
                        request=request,
                        baseline_days=baseline_days,
                        current_days=current_days,
                        queries_executed=0,
                        time_dimension=None,
                    ),
                    warnings=warnings,
                )
            effective_time_dimension = request.time_dimension

        requested_max_dimension_values = (
            500 if request.max_values_per_dimension is None else request.max_values_per_dimension
        )
        effective_max_dimension_values = max(
            1,
            min(requested_max_dimension_values, _MAX_DIMENSION_VALUES),
        )
        top_n_values = 10 if request.top_n_values is None else max(1, request.top_n_values)
        top_n_dimensions = 3 if request.top_n_dimensions is None else max(1, request.top_n_dimensions)
        grouped_query_limit = effective_max_dimension_values + 1
        queries_executed = 2

        baseline_total_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=[],
            path=request.path,
            time_start=request.baseline.start,
            time_end=request.baseline.end,
            where=request.where_sql,
            params=request.params or None,
        )
        current_total_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=[],
            path=request.path,
            time_start=request.current.start,
            time_end=request.current.end,
            where=request.where_sql,
            params=request.params or None,
        )

        baseline_total = self._parse_total(
            baseline_total_result,
            metric_name=metric_name,
            period="baseline",
            warnings=warnings,
        )
        current_total = self._parse_total(
            current_total_result,
            metric_name=metric_name,
            period="current",
            warnings=warnings,
        )
        total_delta = current_total - baseline_total

        if baseline_days != current_days:
            warnings.append(
                AttributionWarning(
                    code="unequal_windows",
                    message=(
                        f"Baseline and current windows contain {baseline_days} and {current_days} days; "
                        "values were not normalized."
                    ),
                )
            )

        total_delta_is_zero = self._is_effectively_zero_delta(
            total_delta,
            baseline_total=baseline_total,
            current_total=current_total,
        )
        dimension_rankings: List[AttributionDimensionScore] = []
        all_contributions: Dict[str, List[AttributionValueContribution]] = {}
        per_dimension: Dict[str, AttributionDimensionDetail] = {}

        for dimension in candidate_dimensions:
            baseline_result, baseline_lookup, dimension_error = await self._query_grouped_period(
                metric_name=metric_name,
                dimension=dimension,
                period="baseline",
                time_start=request.baseline.start,
                time_end=request.baseline.end,
                path=request.path,
                where=request.where_sql,
                limit=grouped_query_limit,
                params=request.params,
            )
            queries_executed += 1
            if dimension_error is not None:
                self._record_dimension_failure(
                    dimension=dimension,
                    error=dimension_error,
                    warnings=warnings,
                )
                continue

            current_result, current_lookup, dimension_error = await self._query_grouped_period(
                metric_name=metric_name,
                dimension=dimension,
                period="current",
                time_start=request.current.start,
                time_end=request.current.end,
                path=request.path,
                where=request.where_sql,
                limit=grouped_query_limit,
                params=request.params,
            )
            queries_executed += 1
            if dimension_error is not None:
                self._record_dimension_failure(
                    dimension=dimension,
                    error=dimension_error,
                    warnings=warnings,
                )
                continue

            assert baseline_result is not None and baseline_lookup is not None
            assert current_result is not None and current_lookup is not None

            logger.debug(
                "Analyzing dimension '%s': baseline=%d rows, current=%d rows",
                dimension,
                len(baseline_result.data),
                len(current_result.data),
            )

            union_keys = list(dict.fromkeys([*baseline_lookup, *current_lookup]))
            truncated = (
                len(baseline_result.data) >= grouped_query_limit
                or len(current_result.data) >= grouped_query_limit
                or len(union_keys) > effective_max_dimension_values
            )
            if truncated:
                per_dimension[dimension] = AttributionDimensionDetail(
                    truncated=True,
                )
                warnings.append(
                    AttributionWarning(
                        code="high_cardinality_dimension",
                        dimension=dimension,
                        message=(
                            f"Dimension '{dimension}' exceeded the {effective_max_dimension_values}-value limit. "
                            "Attribute a coarser dimension first, then use where to narrow this dimension."
                        ),
                    )
                )
                continue

            contributions = self._build_contributions(
                dimension=dimension,
                union_keys=union_keys,
                baseline_lookup=baseline_lookup,
                current_lookup=current_lookup,
                total_delta=total_delta,
                total_delta_is_zero=total_delta_is_zero,
            )
            deltas = [contribution.delta for contribution in contributions]
            score = (
                max(abs(delta) for delta in deltas) / abs(total_delta) if deltas and not total_delta_is_zero else None
            )
            additivity_check = self._check_additivity(
                baseline_total=baseline_total,
                current_total=current_total,
                baseline_values=[item[1] for item in baseline_lookup.values()],
                current_values=[item[1] for item in current_lookup.values()],
                total_delta_is_zero=total_delta_is_zero,
            )
            if additivity_check.status == "failed":
                warnings.append(
                    AttributionWarning(
                        code="non_additive_dimension",
                        dimension=dimension,
                        message=(
                            f"Grouped values for '{dimension}' do not reconcile to the period totals; "
                            "do not interpret its contribution percentages as an additive decomposition."
                        ),
                    )
                )

            if total_delta_is_zero and self._has_material_component_change(
                deltas,
                baseline_total=baseline_total,
                current_total=current_total,
            ):
                warnings.append(
                    AttributionWarning(
                        code="zero_total_delta_with_component_changes",
                        dimension=dimension,
                        message=(
                            f"Dimension '{dimension}' has offsetting component changes while the total change is "
                            "effectively zero; interpret absolute deltas, not contribution percentages."
                        ),
                    )
                )

            non_additive = additivity_check.status == "failed"
            per_dimension[dimension] = AttributionDimensionDetail(
                values=sorted(
                    contributions,
                    key=lambda item: abs(item.delta),
                    reverse=True,
                )[:top_n_values],
                score=score,
                non_additive=non_additive,
                reconciliation=AttributionReconciliation(
                    baseline_residual=additivity_check.baseline_residual or 0.0,
                    current_residual=additivity_check.current_residual or 0.0,
                    passed=not non_additive,
                ),
            )
            if non_additive:
                continue
            dimension_rankings.append(
                AttributionDimensionScore(
                    dimension=dimension,
                    score=score,
                    non_additive=False,
                    truncated=False,
                )
            )
            all_contributions[dimension] = contributions

        dimension_rankings.sort(
            key=lambda ranking: ranking.score or 0.0,
            reverse=True,
        )
        selected_dimensions = [ranking.dimension for ranking in dimension_rankings[:top_n_dimensions]]
        selected_contributions = [
            contribution for dimension in selected_dimensions for contribution in all_contributions[dimension]
        ]
        selected_contributions.sort(
            key=lambda contribution: abs(contribution.delta),
            reverse=True,
        )
        unsupported_reason = None
        strategy: Literal["term_wise", "unsupported"] = "term_wise"
        if not candidate_dimensions:
            strategy = "unsupported"
            unsupported_reason = AttributionUnsupportedInfo(
                code="dimensions_required",
                message="Generic attribution requires at least one candidate dimension.",
            )
        elif not dimension_rankings:
            strategy = "unsupported"
            unsupported_reason = AttributionUnsupportedInfo(
                code="no_additive_dimensions",
                message="No candidate dimension produced a complete additive decomposition.",
            )

        return AttributionResult(
            metric=metric_name,
            implementation="generic",
            strategy=strategy,
            unsupported_reason=unsupported_reason,
            total_change=AttributionTotalChange(
                baseline_value=baseline_total,
                current_value=current_total,
                delta=total_delta,
                pct_change=(total_delta / abs(baseline_total) * 100 if baseline_total else None),
            ),
            dimension_ranking=dimension_rankings,
            selected_dimensions=selected_dimensions,
            top_dimension_values=selected_contributions[:top_n_values],
            per_dimension=per_dimension,
            comparison_metadata=self._comparison_metadata(
                request=request,
                baseline_days=baseline_days,
                current_days=current_days,
                queries_executed=queries_executed,
                time_dimension=effective_time_dimension,
            ),
            warnings=warnings,
        )

    @staticmethod
    def _comparison_metadata(
        *,
        request: AttributionRequest,
        baseline_days: int,
        current_days: int,
        queries_executed: int,
        time_dimension: Optional[str],
    ) -> AttributionComparisonMetadata:
        """Build metadata from the options the generic queries actually honored."""
        return AttributionComparisonMetadata(
            baseline=AttributionWindow(
                start=request.baseline.start,
                end=request.baseline.end,
            ),
            current=AttributionWindow(
                start=request.current.start,
                end=request.current.end,
            ),
            baseline_days=baseline_days,
            current_days=current_days,
            equal_length_windows=baseline_days == current_days,
            time_dimension=time_dimension,
            queries_executed=queries_executed,
            params=request.params,
        )

    async def _query_grouped_period(
        self,
        *,
        metric_name: str,
        dimension: str,
        period: Literal["baseline", "current"],
        time_start: str,
        time_end: str,
        path: Optional[List[str]],
        where: Optional[str],
        limit: int,
        params: Dict[str, Any],
    ) -> tuple[
        Optional[QueryResult],
        Optional[Dict[str, tuple[Optional[JsonScalar], float]]],
        Optional[DimensionAnalysisError],
    ]:
        """Query and validate one dimension period without aborting its peers."""
        try:
            result = await self.adapter.query_metrics(
                metrics=[metric_name],
                dimensions=[dimension],
                path=path,
                time_start=time_start,
                time_end=time_end,
                where=where,
                limit=limit,
                params=params or None,
            )
        except Exception as error:
            payload = getattr(error, "payload", None)
            code = getattr(payload, "code", None)
            message = getattr(payload, "message", None) or str(error)
            if not isinstance(code, str) or not code:
                code = "DIMENSION_QUERY_FAILED"
            if not isinstance(message, str) or not message:
                message = "Dimension query failed."
            logger.warning(
                "Attribution query failed for dimension '%s' during %s: %s",
                dimension,
                period,
                message,
            )
            return (
                None,
                None,
                DimensionAnalysisError(
                    error_type="query_error",
                    code=code,
                    message=message,
                    period=period,
                ),
            )

        try:
            lookup = self._parse_grouped_result(
                result,
                metric_name=metric_name,
                dimension=dimension,
                period=period,
            )
        except AttributionValidationException as error:
            payload = error.payload
            logger.warning(
                "Attribution result validation failed for dimension '%s' during %s: %s",
                dimension,
                period,
                payload.message,
            )
            return (
                result,
                None,
                DimensionAnalysisError(
                    error_type="validation_error",
                    code=payload.code,
                    message=payload.message,
                    period=period,
                    columns=payload.columns,
                    row_count=payload.row_count,
                ),
            )

        return result, lookup, None

    @staticmethod
    def _record_dimension_failure(
        *,
        dimension: str,
        error: DimensionAnalysisError,
        warnings: List[AttributionWarning],
    ) -> None:
        warnings.append(
            AttributionWarning(
                code="dimension_analysis_failed",
                dimension=dimension,
                message=(
                    f"Dimension '{dimension}' could not be analyzed during {error.period}: "
                    f"[{error.code}] {error.message} Other dimensions were analyzed independently."
                ),
            )
        )

    def _parse_total(
        self,
        result: QueryResult,
        *,
        metric_name: str,
        period: Literal["baseline", "current"],
        warnings: List[AttributionWarning],
    ) -> float:
        row_count = len(result.data)
        if row_count > 1:
            self._raise_validation_error(
                code="MULTI_ROW_TOTAL",
                message=f"{period.title()} total query returned {row_count} rows; expected at most one.",
                period=period,
                columns=result.columns,
                row_count=row_count,
            )
        if row_count == 0:
            warnings.append(
                AttributionWarning(
                    code=f"no_data_{period}",
                    message=f"The {period} total query returned no rows; the total is treated as 0.",
                )
            )
            return 0.0
        metric_column = self._resolve_column(
            requested=metric_name,
            columns=result.columns,
            missing_code="MISSING_METRIC_COLUMN",
            period=period,
            row_count=row_count,
        )

        row = result.data[0]
        if metric_column not in row:
            self._raise_validation_error(
                code="MISSING_METRIC_COLUMN",
                message=f"Resolved metric column '{metric_column}' is absent from the {period} result row.",
                period=period,
                columns=result.columns,
                row_count=row_count,
            )
        value = row[metric_column]
        if value is None:
            warnings.append(
                AttributionWarning(
                    code=f"null_total_{period}",
                    message=f"The {period} total is NULL and is treated as 0; confirm data coverage.",
                )
            )
            return 0.0

        numeric_value = self._coerce_finite_metric(
            value,
            period=period,
            columns=result.columns,
            row_count=row_count,
        )
        if numeric_value == 0:
            warnings.append(
                AttributionWarning(
                    code=f"zero_or_no_data_{period}",
                    message=(
                        f"The {period} total is 0, which cannot distinguish a real zero from empty aggregate "
                        "input; confirm coverage with query_metrics at an appropriate time grain."
                    ),
                )
            )
        return numeric_value

    def _parse_grouped_result(
        self,
        result: QueryResult,
        *,
        metric_name: str,
        dimension: str,
        period: Literal["baseline", "current"],
    ) -> Dict[str, tuple[Optional[JsonScalar], float]]:
        row_count = len(result.data)
        if row_count == 0:
            return {}
        metric_column = self._resolve_column(
            requested=metric_name,
            columns=result.columns,
            missing_code="MISSING_METRIC_COLUMN",
            period=period,
            dimension=dimension,
            row_count=row_count,
        )
        dimension_column = self._resolve_column(
            requested=dimension,
            columns=result.columns,
            missing_code="MISSING_DIMENSION_COLUMN",
            period=period,
            dimension=dimension,
            row_count=row_count,
        )

        values: Dict[str, tuple[Optional[JsonScalar], float]] = {}
        for row in result.data:
            if metric_column not in row:
                self._raise_validation_error(
                    code="MISSING_METRIC_COLUMN",
                    message=f"Resolved metric column '{metric_column}' is absent from a {period} grouped row.",
                    period=period,
                    dimension=dimension,
                    columns=result.columns,
                    row_count=row_count,
                )
            if dimension_column not in row:
                self._raise_validation_error(
                    code="MISSING_DIMENSION_COLUMN",
                    message=f"Resolved dimension column '{dimension_column}' is absent from a {period} grouped row.",
                    period=period,
                    dimension=dimension,
                    columns=result.columns,
                    row_count=row_count,
                )

            raw_dimension_value = row[dimension_column]
            normalized_dimension_value = self._normalize_dimension_value(raw_dimension_value)
            key = self._dimension_key(normalized_dimension_value)
            if key in values:
                self._raise_validation_error(
                    code="DUPLICATE_DIMENSION_KEY",
                    message=(
                        f"Dimension '{dimension}' returned duplicate value "
                        f"'{self._display_dimension_value(normalized_dimension_value)}' in the {period} period; "
                        "check for an implicit time grain or extra grouping columns."
                    ),
                    period=period,
                    dimension=dimension,
                    columns=result.columns,
                    row_count=row_count,
                )
            metric_value = self._coerce_finite_metric(
                row[metric_column],
                period=period,
                dimension=dimension,
                columns=result.columns,
                row_count=row_count,
            )
            values[key] = (normalized_dimension_value, metric_value)
        return values

    def _build_contributions(
        self,
        *,
        dimension: str,
        union_keys: List[str],
        baseline_lookup: Dict[str, tuple[Optional[JsonScalar], float]],
        current_lookup: Dict[str, tuple[Optional[JsonScalar], float]],
        total_delta: float,
        total_delta_is_zero: bool,
    ) -> List[AttributionValueContribution]:
        contributions: List[AttributionValueContribution] = []
        for key in union_keys:
            dimension_value = (current_lookup.get(key) or baseline_lookup[key])[0]
            baseline_value = baseline_lookup.get(key, (dimension_value, 0.0))[1]
            current_value = current_lookup.get(key, (dimension_value, 0.0))[1]
            delta = current_value - baseline_value
            contribution_pct = None if total_delta_is_zero else delta / total_delta * 100
            if key not in baseline_lookup:
                segment_kind = "entered"
            elif key not in current_lookup:
                segment_kind = "exited"
            else:
                segment_kind = "normal"
            contributions.append(
                AttributionValueContribution(
                    dimension=dimension,
                    value=self._display_dimension_value(dimension_value),
                    baseline_value=baseline_value,
                    current_value=current_value,
                    delta=delta,
                    contribution_pct=contribution_pct,
                    segment_kind=segment_kind,
                    drill_down=AttributionDrillDown(
                        where_sql=self._drill_down_sql(
                            dimension,
                            dimension_value,
                        )
                    ),
                )
            )
        return contributions

    @staticmethod
    def _drill_down_sql(dimension: str, value: Optional[JsonScalar]) -> str:
        if value is None:
            return f"{dimension} IS NULL"
        if isinstance(value, bool):
            literal = "TRUE" if value else "FALSE"
        elif isinstance(value, (int, float)):
            literal = str(value)
        else:
            literal = "'" + str(value).replace("'", "''") + "'"
        return f"{dimension} = {literal}"

    @staticmethod
    def _check_additivity(
        *,
        baseline_total: float,
        current_total: float,
        baseline_values: List[float],
        current_values: List[float],
        total_delta_is_zero: bool,
    ) -> AdditivityCheck:
        baseline_sum = sum(baseline_values)
        current_sum = sum(current_values)
        baseline_residual = baseline_sum - baseline_total
        current_residual = current_sum - current_total
        baseline_tolerance = _ADDITIVITY_TOLERANCE * max(
            abs(baseline_total),
            sum(abs(value) for value in baseline_values),
        )
        current_tolerance = _ADDITIVITY_TOLERANCE * max(
            abs(current_total),
            sum(abs(value) for value in current_values),
        )
        status = (
            "passed"
            if abs(baseline_residual) <= baseline_tolerance and abs(current_residual) <= current_tolerance
            else "failed"
        )
        total_delta = current_total - baseline_total
        grouped_delta = current_sum - baseline_sum
        return AdditivityCheck(
            status=status,
            baseline_residual=baseline_residual,
            current_residual=current_residual,
            baseline_residual_pct=(baseline_residual / abs(baseline_total) * 100 if baseline_total != 0 else None),
            current_residual_pct=(current_residual / abs(current_total) * 100 if current_total != 0 else None),
            delta_residual_pct=(
                (grouped_delta - total_delta) / abs(total_delta) * 100 if not total_delta_is_zero else None
            ),
        )

    @staticmethod
    def _is_effectively_zero_delta(
        total_delta: float,
        *,
        baseline_total: float,
        current_total: float,
    ) -> bool:
        scale = max(abs(baseline_total), abs(current_total))
        return abs(total_delta) <= _ZERO_DELTA_EPSILON * scale if scale else total_delta == 0

    @staticmethod
    def _has_material_component_change(
        deltas: List[float],
        *,
        baseline_total: float,
        current_total: float,
    ) -> bool:
        threshold = _ADDITIVITY_TOLERANCE * max(abs(baseline_total), abs(current_total))
        return any(abs(delta) > threshold for delta in deltas)

    def _coerce_finite_metric(
        self,
        value: Any,
        *,
        period: Literal["baseline", "current"],
        columns: List[str],
        row_count: int,
        dimension: Optional[str] = None,
    ) -> float:
        if value is None:
            self._raise_validation_error(
                code="NON_NUMERIC_METRIC_VALUE",
                message=f"Metric value is NULL in a {period} grouped row.",
                period=period,
                dimension=dimension,
                columns=columns,
                row_count=row_count,
            )
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric_value = math.nan
        if not math.isfinite(numeric_value):
            self._raise_validation_error(
                code="NON_NUMERIC_METRIC_VALUE",
                message=f"Metric value {value!r} is not a finite number in the {period} result.",
                period=period,
                dimension=dimension,
                columns=columns,
                row_count=row_count,
            )
        return numeric_value

    def _resolve_column(
        self,
        *,
        requested: str,
        columns: List[str],
        missing_code: str,
        period: Literal["baseline", "current"],
        row_count: int,
        dimension: Optional[str] = None,
    ) -> str:
        exact_matches = [column for column in columns if column.casefold() == requested.casefold()]
        if len(exact_matches) == 1:
            return exact_matches[0]

        requested_leaf = self._column_leaf(requested)
        leaf_matches = [
            column for column in columns if self._column_leaf(column).casefold() == requested_leaf.casefold()
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0]

        detail = "not found" if not leaf_matches else f"ambiguous ({', '.join(leaf_matches)})"
        kind = "metric" if missing_code == "MISSING_METRIC_COLUMN" else "dimension"
        self._raise_validation_error(
            code=missing_code,
            message=f"Requested {kind} column '{requested}' was {detail} in the {period} result.",
            period=period,
            dimension=dimension,
            columns=columns,
            row_count=row_count,
        )

    @staticmethod
    def _column_leaf(column: str) -> str:
        return column.rsplit(".", 1)[-1].rsplit("__", 1)[-1]

    @staticmethod
    def _normalize_dimension_value(value: Any) -> Optional[JsonScalar]:
        if value is None:
            return None
        try:
            if not isinstance(value, (str, bytes)) and math.isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, Decimal):
            if value == value.to_integral_value():
                return int(value)
            normalized = float(value)
            return normalized if math.isfinite(normalized) else str(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _dimension_key(value: Optional[JsonScalar]) -> str:
        return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False, sort_keys=True)}"

    @staticmethod
    def _display_dimension_value(value: Optional[JsonScalar]) -> str:
        return "(null)" if value is None else str(value)

    def _validate_time_window(
        self,
        *,
        period: Literal["baseline", "current"],
        start: str,
        end: str,
    ) -> int:
        """Validate a concrete OSI half-open window and return its day count."""
        try:
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        except (TypeError, ValueError):
            self._raise_validation_error(
                code="INVALID_TIME_WINDOW",
                message=(
                    f"{period.title()} window must use ISO dates in an OSI "
                    f"half-open range [start, end); received [{start}, {end})."
                ),
                period=period,
            )
        if days <= 0:
            self._raise_validation_error(
                code="INVALID_TIME_WINDOW",
                message=(
                    f"{period.title()} window must be a non-empty OSI half-open range [start, end); "
                    f"received [{start}, {end})."
                ),
                period=period,
            )
        return days

    @staticmethod
    def _raise_validation_error(
        *,
        code: str,
        message: str,
        period: Optional[str] = None,
        dimension: Optional[str] = None,
        columns: Optional[List[str]] = None,
        row_count: Optional[int] = None,
    ) -> None:
        raise AttributionValidationException(
            AttributionValidationErrorPayload(
                code=code,
                message=message,
                period=period,
                dimension=dimension,
                columns=columns or [],
                row_count=row_count,
            )
        )
