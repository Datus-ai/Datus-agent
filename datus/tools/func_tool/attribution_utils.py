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
from datus.tools.semantic_tools.models import QueryResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

JsonScalar = Union[str, int, float, bool]
_MAX_DIMENSION_VALUES = 1000
_ADDITIVITY_TOLERANCE = 0.02
_ZERO_DELTA_EPSILON = 1e-6


# ==================== Data Models ====================


class DimensionRanking(BaseModel):
    """Ranking score for a dimension's change concentration."""

    dimension: str = Field(..., description="Dimension name")
    score: float = Field(
        ..., description="Max contribution ratio (max_abs_delta / total_delta), can exceed 1 when deltas offset"
    )


class FilterHint(BaseModel):
    """Typed filter information for a follow-up attribution query."""

    dimension: str
    operator: Literal["eq", "is_null"]
    value: Optional[JsonScalar] = None


class DimensionValueContribution(BaseModel):
    """Delta contribution of a dimension value."""

    dimension_values: Dict[str, str] = Field(..., description="Legacy human-readable dimension value(s)")
    baseline: float = Field(..., description="Baseline period metric value")
    current: float = Field(..., description="Current period metric value")
    delta: float = Field(..., description="Absolute change (current - baseline)")
    contribution_pct_of_total_delta: float = Field(..., description="Percentage contribution to total delta")
    filter_hint: Optional[FilterHint] = None


class AdditivityCheck(BaseModel):
    """Whether grouped values reconcile to the corresponding period totals."""

    status: Literal["passed", "failed", "skipped", "unknown"] = "unknown"
    baseline_residual: Optional[float] = None
    current_residual: Optional[float] = None
    baseline_residual_pct: Optional[float] = None
    current_residual_pct: Optional[float] = None
    delta_residual_pct: Optional[float] = None


class DimensionAttribution(BaseModel):
    """Attribution details and guardrail state for one dimension."""

    dimension: str
    score: Optional[float] = None
    additivity_check: AdditivityCheck = Field(default_factory=AdditivityCheck)
    truncated: bool = False
    contributions: List[DimensionValueContribution] = Field(default_factory=list)


class AttributionWarning(BaseModel):
    """A non-fatal limitation that must be disclosed when interpreting results."""

    code: str
    dimension: Optional[str] = None
    message: str


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


class AttributionAnalysisResult(BaseModel):
    """Result of unified attribution analysis."""

    metric_name: str = Field(..., description="Metric being analyzed")
    candidate_dimensions: List[str] = Field(..., description="Input candidate dimensions")
    dimension_ranking: List[DimensionRanking] = Field(..., description="Dimensions ranked by change concentration")
    selected_dimensions: List[str] = Field(..., description="Dimensions selected for analysis")
    top_dimension_values: List[DimensionValueContribution] = Field(
        ..., description="Legacy cross-dimension list of top contributors"
    )
    anomaly_context: Optional[Dict] = Field(None, description="Anomaly detection context")
    comparison_metadata: Dict = Field(..., description="Comparison period metadata")
    per_dimension: Dict[str, DimensionAttribution] = Field(default_factory=dict)
    warnings: List[AttributionWarning] = Field(default_factory=list)


# ==================== Attribution Util ====================


class DimensionAttributionUtil:
    """Dimension attribution utility that only depends on BaseSemanticAdapter."""

    def __init__(self, adapter: BaseSemanticAdapter):
        self.adapter = adapter

    async def attribution_analyze(
        self,
        metric_name: str,
        candidate_dimensions: List[str],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        path: Optional[List[str]] = None,
        anomaly_context: Optional[Dict] = None,
        max_selected_dimensions: int = 3,
        top_n_values: int = 10,
        where: Optional[str] = None,
        max_dimension_values: int = 500,
    ) -> AttributionAnalysisResult:
        """Rank candidate dimensions and calculate guarded delta contributions."""
        requested_max_dimension_values = max_dimension_values
        effective_max_dimension_values = max(1, min(max_dimension_values, _MAX_DIMENSION_VALUES))
        grouped_query_limit = effective_max_dimension_values + 1
        warnings: List[AttributionWarning] = []

        baseline_total_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=[],
            path=path,
            time_start=baseline_start,
            time_end=baseline_end,
            where=where,
        )
        current_total_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=[],
            path=path,
            time_start=current_start,
            time_end=current_end,
            where=where,
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

        baseline_days = self._window_days(baseline_start, baseline_end)
        current_days = self._window_days(current_start, current_end)
        if baseline_days is not None and current_days is not None and baseline_days != current_days:
            warnings.append(
                AttributionWarning(
                    code="UNEQUAL_WINDOWS",
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
        dimension_rankings: List[DimensionRanking] = []
        all_contributions: Dict[str, List[DimensionValueContribution]] = {}
        per_dimension: Dict[str, DimensionAttribution] = {}

        for dimension in candidate_dimensions:
            baseline_result = await self.adapter.query_metrics(
                metrics=[metric_name],
                dimensions=[dimension],
                path=path,
                time_start=baseline_start,
                time_end=baseline_end,
                where=where,
                limit=grouped_query_limit,
            )
            current_result = await self.adapter.query_metrics(
                metrics=[metric_name],
                dimensions=[dimension],
                path=path,
                time_start=current_start,
                time_end=current_end,
                where=where,
                limit=grouped_query_limit,
            )

            logger.debug(
                "Analyzing dimension '%s': baseline=%d rows, current=%d rows",
                dimension,
                len(baseline_result.data),
                len(current_result.data),
            )

            baseline_lookup = self._parse_grouped_result(
                baseline_result,
                metric_name=metric_name,
                dimension=dimension,
                period="baseline",
            )
            current_lookup = self._parse_grouped_result(
                current_result,
                metric_name=metric_name,
                dimension=dimension,
                period="current",
            )
            union_keys = list(dict.fromkeys([*baseline_lookup, *current_lookup]))
            truncated = (
                len(baseline_result.data) >= grouped_query_limit
                or len(current_result.data) >= grouped_query_limit
                or len(union_keys) > effective_max_dimension_values
            )
            if truncated:
                per_dimension[dimension] = DimensionAttribution(
                    dimension=dimension,
                    truncated=True,
                    additivity_check=AdditivityCheck(status="skipped"),
                )
                warnings.append(
                    AttributionWarning(
                        code="HIGH_CARDINALITY_DIMENSION",
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
                max(abs(delta) for delta in deltas) / abs(total_delta) if deltas and not total_delta_is_zero else 0.0
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
                        code="NON_ADDITIVE_DIMENSION",
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
                        code="ZERO_TOTAL_DELTA_WITH_COMPONENT_CHANGES",
                        dimension=dimension,
                        message=(
                            f"Dimension '{dimension}' has offsetting component changes while the total change is "
                            "effectively zero; interpret absolute deltas, not contribution percentages."
                        ),
                    )
                )

            dimension_rankings.append(DimensionRanking(dimension=dimension, score=score))
            all_contributions[dimension] = contributions
            per_dimension[dimension] = DimensionAttribution(
                dimension=dimension,
                score=score,
                additivity_check=additivity_check,
                contributions=sorted(contributions, key=lambda item: abs(item.delta), reverse=True)[
                    : max(0, top_n_values)
                ],
            )

        dimension_rankings.sort(key=lambda ranking: ranking.score, reverse=True)
        selected_dimensions = [ranking.dimension for ranking in dimension_rankings[: max(0, max_selected_dimensions)]]
        selected_contributions = [
            contribution for dimension in selected_dimensions for contribution in all_contributions[dimension]
        ]
        selected_contributions.sort(
            key=lambda contribution: contribution.contribution_pct_of_total_delta,
            reverse=True,
        )

        return AttributionAnalysisResult(
            metric_name=metric_name,
            candidate_dimensions=candidate_dimensions,
            dimension_ranking=dimension_rankings,
            selected_dimensions=selected_dimensions,
            top_dimension_values=selected_contributions[: max(0, top_n_values)],
            anomaly_context=anomaly_context,
            comparison_metadata={
                "baseline": {
                    "start": baseline_start,
                    "end": baseline_end,
                    "days": baseline_days,
                    "total": baseline_total,
                },
                "current": {
                    "start": current_start,
                    "end": current_end,
                    "days": current_days,
                    "total": current_total,
                },
                "total_delta": total_delta,
                "where": where,
                "path": path,
                "requested_max_dimension_values": requested_max_dimension_values,
                "effective_max_dimension_values": effective_max_dimension_values,
            },
            per_dimension=per_dimension,
            warnings=warnings,
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
                    code=f"NO_DATA_{period.upper()}",
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
                    code=f"NULL_TOTAL_{period.upper()}",
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
                    code=f"ZERO_OR_NO_DATA_{period.upper()}",
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
    ) -> List[DimensionValueContribution]:
        contributions: List[DimensionValueContribution] = []
        for key in union_keys:
            dimension_value = (current_lookup.get(key) or baseline_lookup[key])[0]
            baseline_value = baseline_lookup.get(key, (dimension_value, 0.0))[1]
            current_value = current_lookup.get(key, (dimension_value, 0.0))[1]
            delta = current_value - baseline_value
            contribution_pct = 0.0 if total_delta_is_zero else delta / total_delta * 100
            is_null = dimension_value is None
            contributions.append(
                DimensionValueContribution(
                    dimension_values={dimension: self._display_dimension_value(dimension_value)},
                    baseline=baseline_value,
                    current=current_value,
                    delta=delta,
                    contribution_pct_of_total_delta=contribution_pct,
                    filter_hint=FilterHint(
                        dimension=dimension,
                        operator="is_null" if is_null else "eq",
                        value=None if is_null else dimension_value,
                    ),
                )
            )
        return contributions

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

    @staticmethod
    def _window_days(start: str, end: str) -> Optional[int]:
        try:
            return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        except (TypeError, ValueError):
            return None

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
