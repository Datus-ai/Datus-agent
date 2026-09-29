# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DimensionInfo(BaseModel):
    """Information about a dimension."""

    name: str = Field(..., description="Dimension name")
    description: Optional[str] = Field(None, description="Dimension description")
    type: Optional[str] = Field(
        None,
        description="Dimension type (platform-native value, e.g. 'string', 'number', 'time', 'boolean', 'categorical')",
    )
    is_primary_key: Optional[bool] = Field(None, description="Whether this dimension is a primary key")
    is_primary_time: Optional[bool] = Field(
        None,
        description="Whether this is the metric's canonical time dimension",
    )
    time_granularities: List[str] = Field(
        default_factory=list,
        description=("Queryable time granularities for the canonical time dimension, ordered from finest to coarsest"),
    )
    recommended: Optional[bool] = Field(
        None,
        description=(
            "Whether this queryable dimension is recommended for analytical "
            "grouping; false does not prevent an explicit group-by"
        ),
    )
    recommendation_source: Optional[str] = Field(
        None,
        description="How the grouping recommendation was declared or inferred",
    )


class SemanticModelInfo(BaseModel):
    """Typed semantic model metadata (thin model + extra dict for platform-specific data)."""

    name: str = Field(..., description="Model name (cube name, explore name, semantic model name)")
    description: Optional[str] = Field(None, description="Model description")
    table_name: Optional[str] = Field(None, description="Physical table name backing this semantic model")
    catalog_name: Optional[str] = Field(None, description="Physical catalog name for the backing table")
    database_name: Optional[str] = Field(None, description="Physical database name for the backing table")
    schema_name: Optional[str] = Field(None, description="Physical schema name for the backing table")
    platform_type: Optional[str] = Field(
        None,
        description="Platform-native type (e.g. 'cube', 'view', 'explore', 'semantic_model')",
    )
    dimensions: List[DimensionInfo] = Field(default_factory=list, description="Dimensions in this model")
    measures: List[str] = Field(default_factory=list, description="Measure/metric names in this model")
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific metadata (joins, segments, etc.)",
    )


class MetricDefinition(BaseModel):
    """Metadata about a specific metric."""

    name: str = Field(..., description="Metric name")
    description: Optional[str] = Field(None, description="Metric description")
    type: Optional[str] = Field(None, description="Metric type (simple, ratio, derived, etc.)")
    dimensions: List[str] = Field(default_factory=list, description="Available dimensions for this metric")
    measures: List[str] = Field(default_factory=list, description="Underlying measures used")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'USD', 'count', 'percent')")
    format: Optional[str] = Field(None, description="Display format (e.g., ',.2f', '0.00%')")
    path: Optional[List[str]] = Field(
        None,
        description="Subject tree hierarchy path (e.g., ['domain', 'layer1', 'layer2'])",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResult(BaseModel):
    """Standardized query response for both actual execution and dry-run."""

    columns: List[str] = Field(default_factory=list, description="Column names")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (execution_time, warnings, etc.)",
    )


AttributionStrategy = Literal[
    "term_wise",
    "mix_shift",
    "factor_shapley",
    "unsupported",
]


class AttributionWindow(BaseModel):
    """One half-open ``[start, end)`` attribution window."""

    start: str
    end: str


class AttributionRequest(BaseModel):
    """Backend-neutral request for explaining a metric change."""

    metric: str
    dimensions: List[str] = Field(default_factory=list)
    baseline: AttributionWindow
    current: AttributionWindow
    where_sql: Optional[str] = None
    time_dimension: Optional[str] = None
    max_values_per_dimension: Optional[int] = Field(None, ge=0)
    top_n_dimensions: Optional[int] = Field(None, ge=0)
    top_n_values: Optional[int] = Field(None, ge=0)
    params: Dict[str, Any] = Field(default_factory=dict)


class AttributionUnsupportedInfo(BaseModel):
    """Stable reason why a metric cannot be attributed."""

    code: str
    message: str
    measure: Optional[str] = None
    agg: Optional[str] = None
    distinct: Optional[bool] = None
    side: Optional[str] = None
    count: Optional[int] = None
    max: Optional[int] = None


class AttributionTotalChange(BaseModel):
    baseline_value: float
    current_value: float
    delta: float
    pct_change: Optional[float] = None


class AttributionDrillDown(BaseModel):
    where_sql: str


class AttributionValueContribution(BaseModel):
    dimension: str
    value: str
    baseline_value: float
    current_value: float
    delta: float
    contribution_pct: Optional[float] = None
    segment_kind: Literal["normal", "entered", "exited", "fallback"] = "normal"
    mix_effect: Optional[float] = None
    rate_effect: Optional[float] = None
    baseline_rate: Optional[float] = None
    current_rate: Optional[float] = None
    baseline_share: Optional[float] = None
    current_share: Optional[float] = None
    drill_down: AttributionDrillDown


class AttributionReconciliation(BaseModel):
    baseline_residual: float
    current_residual: float
    passed: bool


class AttributionDimensionDetail(BaseModel):
    values: List[AttributionValueContribution] = Field(default_factory=list)
    score: Optional[float] = None
    non_additive: bool = False
    truncated: bool = False
    reconciliation: Optional[AttributionReconciliation] = None


class AttributionDimensionScore(BaseModel):
    dimension: str
    score: Optional[float] = None
    non_additive: bool = False
    truncated: bool = False


class AttributionFactorTotals(BaseModel):
    mix_effect: float
    rate_effect: float
    entered: float
    exited: float
    fallback: float
    residual: float


class AttributionFactorEffect(BaseModel):
    factor: str
    baseline_value: float
    current_value: float
    delta: float
    effect: float


class AttributionMemberDelta(BaseModel):
    metric: str
    coefficient: float
    baseline_value: float
    current_value: float
    delta: float
    weighted_delta: float


class AttributionWindowMapping(BaseModel):
    family: str
    calc: str
    note: str


class AttributionBreakdownSeries(BaseModel):
    baseline_value: float
    current_value: float
    delta: float


class AttributionFilterBreakdown(BaseModel):
    base_metric: str
    base: AttributionBreakdownSeries
    filtered: AttributionBreakdownSeries
    complement: AttributionBreakdownSeries


class AttributionComparisonMetadata(BaseModel):
    baseline: AttributionWindow
    current: AttributionWindow
    baseline_days: int
    current_days: int
    equal_length_windows: bool
    time_dimension: Optional[str] = None
    queries_executed: int
    params: Dict[str, Any] = Field(default_factory=dict)


class AttributionWarning(BaseModel):
    code: str
    message: str
    dimension: Optional[str] = None


class AttributionResult(BaseModel):
    """Unified attribution result returned by native and generic implementations."""

    metric: str
    implementation: Literal["dosi", "generic"]
    strategy: AttributionStrategy
    unsupported_reason: Optional[AttributionUnsupportedInfo] = None
    total_change: Optional[AttributionTotalChange] = None
    factor_totals: Optional[AttributionFactorTotals] = None
    dimension_ranking: List[AttributionDimensionScore] = Field(default_factory=list)
    selected_dimensions: List[str] = Field(default_factory=list)
    top_dimension_values: List[AttributionValueContribution] = Field(default_factory=list)
    per_dimension: Dict[str, AttributionDimensionDetail] = Field(default_factory=dict)
    member_breakdown: Optional[List[AttributionMemberDelta]] = None
    factors: Optional[List[AttributionFactorEffect]] = None
    affine_constant: Optional[float] = None
    window_mapping: Optional[AttributionWindowMapping] = None
    filter_breakdown: Optional[AttributionFilterBreakdown] = None
    comparison_metadata: AttributionComparisonMetadata
    warnings: List[AttributionWarning] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """A single validation issue."""

    severity: Literal["error", "warning", "info"] = Field(..., description="Severity level: error, warning, info")
    message: str = Field(..., description="Issue description")
    location: Optional[str] = Field(None, description="Location in config where issue was found")


class ValidationResult(BaseModel):
    """Result of a semantic configuration validation check."""

    valid: bool = Field(..., description="Whether the configuration is valid")
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of validation issues")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dosi validation evidence and diagnostics",
    )


class SemanticValidationError(BaseModel):
    """Structured Dosi query validation failure.

    Callers can revise query arguments from stable fields instead of parsing
    raw exception text. Unknown rejections use ``code='validation_error'``
    and a human-readable ``message``.
    """

    error_type: Literal["semantic_validation_error"] = "semantic_validation_error"
    code: str = Field(
        "validation_error",
        description="Validation category, e.g. 'cumulative_requires_metric_time', 'time_grain_required'",
    )
    metrics: List[str] = Field(default_factory=list, description="Affected metric names")
    required_dimensions: List[str] = Field(
        default_factory=list,
        description="Dimensions the caller must add, e.g. ['metric_time']",
    )
    required_time_granularity: Optional[str] = Field(
        None, description="Required or corrected time granularity, e.g. 'day'"
    )
    unsupported_dimensions: List[str] = Field(
        default_factory=list,
        description="Requested dimensions not supported by all selected metrics",
    )
    suggested_retry: Optional[Dict[str, Any]] = Field(
        None, description="Concrete query_metrics kwargs the caller can retry with"
    )
    message: str = Field("", description="Human-readable explanation")


class AnomalyContext(BaseModel):
    """Context information for anomaly detection in attribution analysis."""

    model_config = {"extra": "forbid"}

    rule: Optional[str] = Field(None, description="Anomaly detection rule name (e.g., 'wow_growth_gt_20pct')")
    observed_change_pct: Optional[float] = Field(None, description="Observed percentage change")
