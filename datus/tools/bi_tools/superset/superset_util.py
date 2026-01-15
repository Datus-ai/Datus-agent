# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Python implementation of buildQueryContext from superset-frontend.

This module converts form_data to QueryContext for the chart data API.
Implementation based on superset-frontend/packages/superset-ui-core/src/query/
and superset-frontend/plugins/*/buildQuery.ts files.

Supports chart types including:
- ECharts Timeseries (area, bar, line, scatter, smooth, step)
- BigNumber (total and with trendline)
- Table, Pivot Table
- BoxPlot, Heatmap, Histogram
- Pie, Funnel, Gauge, Radar
- Bubble, Waterfall, Word Cloud
- Sunburst, Treemap, Sankey
- Graph, Tree
- Mixed Timeseries
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, TypedDict, cast

# =============================================================================
# Enums and Constants
# =============================================================================


class QueryMode(str, Enum):
    AGGREGATE = "aggregate"
    RAW = "raw"


class ExpressionType(str, Enum):
    SIMPLE = "SIMPLE"
    SQL = "SQL"


class RollingType(str, Enum):
    CUMSUM = "cumsum"
    SUM = "sum"
    MEAN = "mean"
    STD = "std"


class ComparisonType(str, Enum):
    VALUES = "values"
    DIFFERENCE = "difference"
    PERCENTAGE = "percentage"
    RATIO = "ratio"


class ContributionMode(str, Enum):
    ROW = "row"
    COLUMN = "column"


class DatasourceType(str, Enum):
    TABLE = "table"
    QUERY = "query"
    SAVED_QUERY = "saved_query"
    VIEW = "view"


# Reserved filter columns for time-related fields
RESERVED_FILTER_COLUMNS = {
    "__time_range",
    "__time_col",
    "__time_grain",
    "__granularity",
    "__time_compare",
}

# Default field aliases mapping (from extractQueryFields.ts)
DEFAULT_FIELD_ALIASES: dict[str, str] = {
    "metric": "metrics",
    "metric_2": "metrics",
    "secondary_metric": "metrics",
    "x": "metrics",
    "y": "metrics",
    "size": "metrics",
    "all_columns": "columns",
    "series": "columns",
    "order_by_cols": "orderby",
}


# =============================================================================
# Type Definitions
# =============================================================================


class QueryObjectFilterClause(TypedDict, total=False):
    col: str | dict[str, Any]
    op: str
    val: Any
    grain: str | None
    isExtra: bool
    formattedVal: str | None


class AdhocFilter(TypedDict, total=False):
    expressionType: str
    clause: str
    subject: str
    operator: str
    comparator: Any
    sqlExpression: str
    filterOptionName: str
    isExtra: bool


class QueryObjectExtras(TypedDict, total=False):
    having: str | None
    where: str | None
    time_grain_sqla: str | None
    relative_start: str | None
    relative_end: str | None
    instant_time_comparison_range: str | None


class PostProcessingRule(TypedDict, total=False):
    operation: str
    options: dict[str, Any]


@dataclass
class QueryObject:
    """Query object compatible with chart data API."""

    # Time fields
    time_range: str | None = None
    since: str | None = None
    until: str | None = None
    granularity: str | None = None

    # Query fields
    columns: list[Any] = field(default_factory=list)
    metrics: list[Any] = field(default_factory=list)
    orderby: list[tuple[Any, bool]] = field(default_factory=list)

    # Filters
    filters: list[QueryObjectFilterClause] = field(default_factory=list)
    extras: QueryObjectExtras = field(default_factory=dict)

    # Pagination
    row_limit: int | None = None
    row_offset: int | None = None

    # Series configuration
    series_columns: list[Any] | None = None
    series_limit: int = 0
    series_limit_metric: Any | None = None

    # Time comparison
    time_offsets: list[str] = field(default_factory=list)

    # Timeseries flag
    is_timeseries: bool | None = None

    # Other fields
    annotation_layers: list[Any] = field(default_factory=list)
    applied_time_extras: dict[str, Any] = field(default_factory=dict)
    order_desc: bool = True
    url_params: dict[str, str] = field(default_factory=dict)
    custom_params: dict[str, Any] = field(default_factory=dict)
    custom_form_data: dict[str, Any] = field(default_factory=dict)
    post_processing: list[PostProcessingRule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, filtering out None and empty values."""
        result = {}
        for key, value in self.__dict__.items():
            # Include False for is_timeseries explicitly
            if value is None:
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            if isinstance(value, dict) and len(value) == 0:
                continue
            result[key] = value
        return result


@dataclass
class QueryContext:
    """Query context for chart data API."""

    datasource: dict[str, Any]
    force: bool
    queries: list[QueryObject]
    form_data: dict[str, Any]
    result_format: str = "json"
    result_type: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasource": self.datasource,
            "force": self.force,
            "queries": [q.to_dict() for q in self.queries],
            "form_data": self.form_data,
            "result_format": self.result_format,
            "result_type": self.result_type,
        }


@dataclass
class DatasourceKey:
    """Parse and represent datasource key like '1__table'."""

    id: int
    type: DatasourceType

    @classmethod
    def from_string(cls, datasource: str) -> DatasourceKey:
        """Parse datasource string like '1__table'."""
        if isinstance(datasource, dict):
            return cls(id=datasource.get("id", 0), type=DatasourceType(datasource.get("type", "table")))
        parts = datasource.split("__")
        if len(parts) != 2:
            raise ValueError(f"Invalid datasource format: {datasource}")
        return cls(id=int(parts[0]), type=DatasourceType(parts[1]))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type.value}


# =============================================================================
# Helper Functions
# =============================================================================


def ensure_list(value: Any) -> list[Any]:
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_column_label(column: Any) -> str:
    """Get label from column (string or adhoc column dict)."""
    if isinstance(column, str):
        return column
    if isinstance(column, dict):
        return column.get("label") or column.get("sqlExpression") or column.get("column_name", "")
    return str(column)


def get_metric_label(metric: Any) -> str:
    """Get label from metric (string or metric dict)."""
    if isinstance(metric, str):
        return metric
    if isinstance(metric, dict):
        label = metric.get("label")
        if label:
            return label
        # Build label from aggregate and column
        aggregate = metric.get("aggregate")
        column = metric.get("column", {})
        column_name = column.get("column_name", "") if isinstance(column, dict) else ""
        if aggregate and column_name:
            return f"{aggregate}({column_name})"
        return metric.get("sqlExpression", "")
    return str(metric)


def is_physical_column(column: Any) -> bool:
    """Check if column is a physical column (string) vs adhoc column."""
    return isinstance(column, str)


# Datetime alias used when is_timeseries=True
DTTM_ALIAS = "__timestamp"


def is_x_axis_set(form_data: dict[str, Any]) -> bool:
    """Check if x_axis is set in form_data."""
    x_axis = form_data.get("x_axis")
    # x_axis must be a physical column (string) or adhoc column (dict with sqlExpression)
    if isinstance(x_axis, str):
        return True
    if isinstance(x_axis, dict) and x_axis.get("sqlExpression"):
        return True
    return False


def get_x_axis_column(form_data: dict[str, Any]) -> Any:
    """
    Get x_axis column.
    Based on migrate_viz/query_functions.py get_x_axis_column.

    Returns:
        - The x_axis value if explicitly set
        - granularity_sqla if x_axis not set but granularity_sqla is set
        - None if neither is set

    Note: For modern ECharts charts, the frontend sets x_axis from granularity_sqla
    as a default. When x_axis is not set, we fall back to granularity_sqla directly
    (not DTTM_ALIAS) so that the column label matches the actual column name.
    """
    # If x_axis is explicitly set, return it
    if is_x_axis_set(form_data):
        return form_data.get("x_axis")

    # Fall back to granularity_sqla for charts that use it as x_axis
    # This matches frontend behavior where x_axis defaults to granularity_sqla
    granularity_sqla = form_data.get("granularity_sqla")
    if granularity_sqla:
        return granularity_sqla

    return None


def get_x_axis_label(form_data: dict[str, Any]) -> str | None:
    """
    Get x-axis label from form_data.
    Based on migrate_viz/query_functions.py get_x_axis_label.
    """
    col = get_x_axis_column(form_data)
    if col:
        return get_column_label(col)
    return None


def get_x_axis_column_with_time_grain(form_data: dict[str, Any]) -> Any:
    """
    Get x_axis column with time grain transformation for query columns.

    Falls back to granularity_sqla if x_axis is not explicitly set.
    This matches frontend behavior where x_axis defaults to granularity_sqla.
    """
    x_axis = form_data.get("x_axis")
    time_grain = form_data.get("time_grain_sqla")

    # Fall back to granularity_sqla if x_axis is not set
    if not x_axis:
        x_axis = form_data.get("granularity_sqla")

    if not x_axis:
        return None

    if is_physical_column(x_axis) and time_grain:
        return {
            "sqlExpression": x_axis,
            "label": x_axis,
            "expressionType": "SQL",
            "columnType": "BASE_AXIS",
            "timeGrain": time_grain,
        }
    elif isinstance(x_axis, dict):
        result = {**x_axis, "columnType": "BASE_AXIS"}
        if time_grain:
            result["timeGrain"] = time_grain
        return result

    return x_axis


def is_adhoc_column(column: Any) -> bool:
    """Check if column is an adhoc column (dict with SQL expression)."""
    if not isinstance(column, dict):
        return False
    return (
        column.get("sqlExpression") is not None
        and column.get("label") is not None
        and column.get("expressionType") == "SQL"
    )


def normalize_time_column(form_data: dict[str, Any], query_object: QueryObject) -> QueryObject:
    """
    Transform x_axis column to BASE_AXIS format with timeGrain.
    Based on migrate_viz/query_functions.py normalize_time_column.

    This function is called AFTER building the query to transform the x_axis
    column in the columns list to include columnType and timeGrain.

    Note: Some build functions may already use get_x_axis_column_with_time_grain()
    which pre-transforms the column. This function handles both cases:
    - If column is already transformed (has columnType), ensure timeGrain is set
    - If column is not transformed, convert to BASE_AXIS format

    Args:
        form_data: Chart form data
        query_object: The query object to transform

    Returns:
        QueryObject with transformed x_axis column
    """
    # Use get_x_axis_column which falls back to granularity_sqla
    x_axis = get_x_axis_column(form_data)
    if not x_axis:
        return query_object

    columns = list(query_object.columns or [])
    extras = query_object.extras or {}
    time_grain = extras.get("time_grain_sqla") or form_data.get("time_grain_sqla")

    # Find the index of x_axis in columns
    axis_idx = None
    for idx, col in enumerate(columns):
        # Match physical column (string)
        if is_physical_column(col) and is_physical_column(x_axis) and col == x_axis:
            axis_idx = idx
            break
        # Match adhoc column by sqlExpression
        if isinstance(col, dict) and isinstance(x_axis, str):
            # Column might already be transformed - check sqlExpression matches x_axis
            if col.get("sqlExpression") == x_axis or col.get("label") == x_axis:
                axis_idx = idx
                break
        if is_adhoc_column(col) and is_adhoc_column(x_axis) and col.get("sqlExpression") == x_axis.get("sqlExpression"):
            axis_idx = idx
            break

    if axis_idx is not None and x_axis:
        col = columns[axis_idx]

        if isinstance(col, dict):
            # Column is already a dict (adhoc or pre-transformed)
            # Ensure it has BASE_AXIS columnType and timeGrain
            updated = dict(col)
            if updated.get("columnType") != "BASE_AXIS":
                updated["columnType"] = "BASE_AXIS"
            if time_grain and not updated.get("timeGrain"):
                updated["timeGrain"] = time_grain
            columns[axis_idx] = updated
        else:
            # Transform physical column to BASE_AXIS format
            columns[axis_idx] = {
                "columnType": "BASE_AXIS",
                "sqlExpression": x_axis,
                "label": x_axis,
                "expressionType": "SQL",
            }
            if time_grain:
                columns[axis_idx]["timeGrain"] = time_grain

        # Update query object
        query_object.columns = columns
        # Remove is_timeseries when x_axis is explicitly set
        query_object.is_timeseries = None

    return query_object


def is_time_comparison(form_data: dict[str, Any], _query_object: QueryObject) -> bool:
    """Check if time comparison is enabled."""
    time_compare = form_data.get("time_compare")
    return bool(time_compare and len(ensure_list(time_compare)) > 0)


def normalize_orderby(query_object: QueryObject) -> list[tuple[Any, bool]]:
    """
    Normalize orderby format.
    Based on migrate_viz/query_functions.py normalize_order_by.

    If orderby already exists and is valid, normalize its format.
    Otherwise, generate orderby from series_limit_metric or first metric.
    """
    orderby = query_object.orderby or []

    # Check if valid orderby already exists
    if orderby and len(orderby) > 0:
        first_item = orderby[0]
        if (
            isinstance(first_item, (list, tuple))
            and len(first_item) == 2
            and first_item[0]
            and isinstance(first_item[1], bool)
        ):
            # Valid orderby exists, just normalize format
            result = []
            for item in orderby:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    result.append((item[0], bool(item[1])))
                elif isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, list) and len(parsed) == 2:
                            result.append((parsed[0], bool(parsed[1])))
                    except json.JSONDecodeError:
                        pass
            return result

    # No valid orderby, generate one
    # is_asc = not order_desc (order_desc=true means descending, so is_asc=false)
    is_asc = not query_object.order_desc

    # 1. Use series_limit_metric if available
    if query_object.series_limit_metric:
        return [(query_object.series_limit_metric, is_asc)]

    # 2. Use first metric if available
    metrics = query_object.metrics or []
    if metrics and len(metrics) > 0:
        return [(metrics[0], is_asc)]

    return []


# =============================================================================
# Post-Processing Operators
# =============================================================================


def pivot_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """
    Build pivot post-processing rule.
    Based on pivotOperator.ts and migrate_viz/query_functions.py
    """
    metrics = ensure_list(query_object.metrics)
    metric_labels = [get_metric_label(m) for m in metrics]
    columns = query_object.series_columns or query_object.columns or []
    # get_x_axis_label returns __timestamp when x_axis not set but granularity_sqla is
    x_axis_label = get_x_axis_label(form_data)

    if x_axis_label and metric_labels:
        return cast(
            PostProcessingRule,
            {
                "operation": "pivot",
                "options": {
                    "index": [x_axis_label],
                    "columns": [get_column_label(c) for c in ensure_list(columns)],
                    "aggregates": {m: {"operator": "mean"} for m in metric_labels},
                    "drop_missing_columns": not form_data.get("show_empty_columns", False),
                },
            },
        )
    return None


def time_compare_pivot_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """Build time comparison pivot operator."""
    metrics = ensure_list(query_object.metrics)
    metric_labels = [get_metric_label(m) for m in metrics]
    time_offsets = ensure_list(form_data.get("time_compare", []))
    columns = query_object.series_columns or query_object.columns or []
    x_axis_label = get_x_axis_label(form_data)

    if not x_axis_label or not metric_labels:
        return None

    # Add offset metrics
    all_metric_labels = list(metric_labels)
    for offset in time_offsets:
        for metric in metric_labels:
            all_metric_labels.append(f"{metric}__{offset}")

    return cast(
        PostProcessingRule,
        {
            "operation": "pivot",
            "options": {
                "index": [x_axis_label],
                "columns": [get_column_label(c) for c in ensure_list(columns)],
                "aggregates": {m: {"operator": "mean"} for m in all_metric_labels},
                "drop_missing_columns": not form_data.get("show_empty_columns", False),
            },
        },
    )


def flatten_operator(_form_data: dict[str, Any], _query_object: QueryObject) -> PostProcessingRule:
    """Build flatten post-processing rule. Based on flattenOperator.ts"""
    return cast(PostProcessingRule, {"operation": "flatten"})


def rolling_window_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """Build rolling window post-processing rule. Based on rollingWindowOperator.ts"""
    rolling_type = form_data.get("rolling_type")

    if not rolling_type or rolling_type.lower() == "none":
        return None

    metrics = ensure_list(query_object.metrics)
    columns = [get_metric_label(m) for m in metrics]
    columns_map = {c: c for c in columns}

    if rolling_type == RollingType.CUMSUM.value:
        return cast(
            PostProcessingRule,
            {
                "operation": "cum",
                "options": {
                    "operator": "sum",
                    "columns": columns_map,
                },
            },
        )

    if rolling_type in [RollingType.SUM.value, RollingType.MEAN.value, RollingType.STD.value]:
        return cast(
            PostProcessingRule,
            {
                "operation": "rolling",
                "options": {
                    "rolling_type": rolling_type,
                    "window": int(form_data.get("rolling_periods", 1)),
                    "min_periods": int(form_data.get("min_periods", 0)),
                    "columns": columns_map,
                },
            },
        )

    return None


def resample_operator(form_data: dict[str, Any], _query_object: QueryObject) -> PostProcessingRule | None:
    """Build resample post-processing rule. Based on resampleOperator.ts"""
    resample_rule = form_data.get("resample_rule")
    resample_method = form_data.get("resample_method")

    if resample_rule and resample_method:
        return cast(
            PostProcessingRule,
            {
                "operation": "resample",
                "options": {
                    "rule": resample_rule,
                    "method": resample_method,
                    "fill_value": form_data.get("resample_fill_value"),
                },
            },
        )
    return None


def rename_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """Build rename post-processing rule. Based on renameOperator.ts"""
    metrics = ensure_list(query_object.metrics)
    truncate_metric = form_data.get("truncate_metric")
    x_axis_label = get_x_axis_label(form_data)

    if not metrics or not x_axis_label:
        return None

    # Only rename when truncate_metric is enabled and single metric
    if truncate_metric and len(metrics) == 1:
        metric_label = get_metric_label(metrics[0])
        return cast(
            PostProcessingRule,
            {
                "operation": "rename",
                "options": {
                    "columns": {metric_label: None},
                    "level": 0,
                    "inplace": True,
                },
            },
        )

    return None


def sort_operator(form_data: dict[str, Any], _query_object: QueryObject) -> PostProcessingRule | None:
    """Build sort post-processing rule. Based on sortOperator.ts"""
    x_axis_sort = form_data.get("x_axis_sort")
    x_axis_sort_asc = form_data.get("x_axis_sort_asc")
    groupby = form_data.get("groupby", [])

    if x_axis_sort is None or x_axis_sort_asc is None:
        return None

    # Sort operator doesn't support sort-by multiple series
    if groupby and len(ensure_list(groupby)) > 0:
        return None

    x_axis_label = get_x_axis_label(form_data)

    if x_axis_sort == x_axis_label:
        return cast(
            PostProcessingRule,
            {
                "operation": "sort",
                "options": {
                    "is_sort_index": True,
                    "ascending": x_axis_sort_asc,
                },
            },
        )

    return cast(
        PostProcessingRule,
        {
            "operation": "sort",
            "options": {
                "by": x_axis_sort,
                "ascending": x_axis_sort_asc,
            },
        },
    )


def contribution_operator(
    form_data: dict[str, Any], _query_object: QueryObject, time_offsets: list[str] | None = None
) -> PostProcessingRule | None:
    """Build contribution post-processing rule. Based on contributionOperator.ts"""
    contribution_mode = form_data.get("contributionMode")

    if contribution_mode:
        return cast(
            PostProcessingRule,
            {
                "operation": "contribution",
                "options": {
                    "orientation": contribution_mode,
                    "time_shifts": time_offsets or [],
                },
            },
        )
    return None


def time_compare_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """Build time comparison post-processing rule. Based on timeCompareOperator.ts"""
    if not is_time_comparison(form_data, query_object):
        return None

    comparison_type = form_data.get("comparison_type", ComparisonType.VALUES.value)

    return cast(
        PostProcessingRule,
        {
            "operation": "compare",
            "options": {
                "compare_type": comparison_type,
            },
        },
    )


def boxplot_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """Build boxplot post-processing rule. Based on boxplotOperator.ts"""
    whisker_options = form_data.get("whiskerOptions")
    groupby = form_data.get("groupby", [])
    metrics = ensure_list(query_object.metrics)

    if not whisker_options:
        return None

    # Parse whisker type
    whisker_type = "tukey"
    percentiles = None

    if whisker_options == "Tukey":
        whisker_type = "tukey"
    elif whisker_options == "Min/max (no outliers)":
        whisker_type = "min/max"
    else:
        # Check for percentile pattern like "10/90 percentiles"
        match = re.match(r"(\d+)/(\d+) percentiles", str(whisker_options))
        if match:
            whisker_type = "percentile"
            percentiles = [int(match.group(1)), int(match.group(2))]

    result: PostProcessingRule = {
        "operation": "boxplot",
        "options": {
            "whisker_type": whisker_type,
            "groupby": [get_column_label(c) for c in ensure_list(groupby)],
            "metrics": [get_metric_label(m) for m in metrics],
        },
    }

    if percentiles:
        result["options"]["percentiles"] = percentiles

    return result


def rank_operator(form_data: dict[str, Any], query_object: QueryObject) -> PostProcessingRule | None:
    """Build rank post-processing rule. For heatmap normalization."""
    metrics = ensure_list(query_object.metrics)
    normalize_across = form_data.get("normalize_across")

    if not metrics or not normalize_across:
        return None

    metric_label = get_metric_label(metrics[0])
    group_by = None

    if normalize_across == "x":
        x_axis = get_x_axis_label(form_data)
        group_by = x_axis
    elif normalize_across == "y":
        groupby = form_data.get("groupby", [])
        if groupby:
            group_by = get_column_label(ensure_list(groupby)[0])

    return cast(
        PostProcessingRule,
        {
            "operation": "rank",
            "options": {
                "metric": metric_label,
                "group_by": group_by,
            },
        },
    )


def histogram_operator(form_data: dict[str, Any], _query_object: QueryObject) -> PostProcessingRule | None:
    """Build histogram post-processing rule."""
    column = form_data.get("column") or form_data.get("all_columns", [None])[0]
    bins = form_data.get("bins", 10)
    groupby = form_data.get("groupby", [])
    cumulative = form_data.get("cumulative", False)
    normalize = form_data.get("normalize", False)

    if not column:
        return None

    return cast(
        PostProcessingRule,
        {
            "operation": "histogram",
            "options": {
                "column": get_column_label(column),
                "groupby": [get_column_label(c) for c in ensure_list(groupby)],
                "bins": int(bins),
                "cumulative": cumulative,
                "normalize": normalize,
            },
        },
    )


def prophet_operator(form_data: dict[str, Any], _query_object: QueryObject) -> PostProcessingRule | None:
    """
    Build prophet forecasting post-processing rule.
    Based on migrate_viz/query_functions.py prophet_operator.

    IMPORTANT: Only triggered when forecastEnabled is explicitly set to True.
    The forecastPeriods field alone is NOT sufficient to enable prophet.
    """
    x_axis_label = get_x_axis_label(form_data)
    forecast_enabled = form_data.get("forecastEnabled")

    # Prophet requires BOTH forecastEnabled=True AND x_axis_label
    # forecastPeriods alone does NOT enable prophet
    if not (forecast_enabled and x_axis_label):
        return None

    forecast_periods = form_data.get("forecastPeriods", 0)

    try:
        periods = int(forecast_periods) if forecast_periods else 0
    except (TypeError, ValueError):
        periods = 0

    try:
        confidence_interval = float(form_data.get("forecastInterval", 0.8))
    except (TypeError, ValueError):
        confidence_interval = 0.8

    return cast(
        PostProcessingRule,
        {
            "operation": "prophet",
            "options": {
                "time_grain": form_data.get("time_grain_sqla"),
                "periods": periods,
                "confidence_interval": confidence_interval,
                "yearly_seasonality": form_data.get("forecastSeasonalityYearly", "auto"),
                "weekly_seasonality": form_data.get("forecastSeasonalityWeekly", "auto"),
                "daily_seasonality": form_data.get("forecastSeasonalityDaily", "auto"),
                "index": x_axis_label,
            },
        },
    )


# =============================================================================
# Core Query Building Functions
# =============================================================================


def extract_query_fields(
    form_data: dict[str, Any], query_field_aliases: dict[str, str] | None = None
) -> dict[str, list[Any]]:
    """
    Extract query fields (columns, metrics, orderby) from form_data.
    Based on extractQueryFields.ts
    """
    aliases = {**DEFAULT_FIELD_ALIASES, **(query_field_aliases or {})}

    columns: list[Any] = []
    metrics: list[Any] = []
    orderby: list[tuple[Any, bool]] = []

    seen_columns: set[str] = set()
    seen_metrics: set[str] = set()

    query_mode = form_data.get("query_mode", QueryMode.AGGREGATE.value)

    for field_name, value in form_data.items():
        if value is None:
            continue

        target_field = aliases.get(field_name, field_name)

        # Handle columns/groupby
        if target_field in ("columns", "groupby"):
            if query_mode == QueryMode.RAW.value and field_name == "groupby":
                continue
            if query_mode == QueryMode.AGGREGATE.value and field_name == "columns":
                continue

            for col in ensure_list(value):
                label = get_column_label(col)
                if label and label not in seen_columns:
                    seen_columns.add(label)
                    columns.append(col)

        # Handle metrics
        elif target_field == "metrics":
            if query_mode == QueryMode.RAW.value:
                continue

            for metric in ensure_list(value):
                if metric is None:
                    continue
                label = get_metric_label(metric)
                if label and label not in seen_metrics:
                    seen_metrics.add(label)
                    metrics.append(metric)

        # Handle orderby
        elif target_field == "orderby":
            for item in ensure_list(value):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    orderby.append((item[0], bool(item[1])))
                elif isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, list) and len(parsed) == 2:
                            orderby.append((parsed[0], bool(parsed[1])))
                    except json.JSONDecodeError:
                        pass

    return {
        "columns": columns,
        "metrics": metrics,
        "orderby": orderby,
    }


def extract_extras(form_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract time-related and special filter fields from form_data.
    Based on extractExtras.ts
    """
    extra_filters = form_data.get("extra_filters", [])

    filters: list[QueryObjectFilterClause] = []
    extras: QueryObjectExtras = {}
    applied_time_extras: dict[str, Any] = {}

    result = {
        "filters": filters,
        "extras": extras,
        "applied_time_extras": applied_time_extras,
        "time_range": form_data.get("time_range"),
        "granularity": form_data.get("granularity"),
        "granularity_sqla": form_data.get("granularity_sqla"),
    }

    for filter_item in extra_filters:
        col = filter_item.get("col", "")
        val = filter_item.get("val")

        if col in RESERVED_FILTER_COLUMNS:
            if col == "__time_range":
                result["time_range"] = val
                applied_time_extras["__time_range"] = val
            elif col == "__time_col":
                result["granularity_sqla"] = val
                applied_time_extras["__time_col"] = val
            elif col == "__time_grain":
                extras["time_grain_sqla"] = val
                applied_time_extras["__time_grain"] = val
            elif col == "__granularity":
                result["granularity"] = val
                applied_time_extras["__granularity"] = val
            elif col == "__time_compare":
                result["time_compare"] = val
                applied_time_extras["__time_compare"] = val
        else:
            filters.append(
                {
                    "col": col,
                    "op": filter_item.get("op", "=="),
                    "val": val,
                    "isExtra": True,
                }
            )

    if not extras.get("time_grain_sqla") and form_data.get("time_grain_sqla"):
        extras["time_grain_sqla"] = form_data.get("time_grain_sqla")

    result["extras"] = extras
    result["applied_time_extras"] = applied_time_extras

    return result


def process_filters(form_data: dict[str, Any]) -> dict[str, Any]:
    """
    Process adhoc_filters and split into WHERE and HAVING clauses.
    Based on processFilters.ts
    """
    adhoc_filters = form_data.get("adhoc_filters", [])
    existing_filters = form_data.get("filters", [])
    existing_extras = form_data.get("extras", {})
    existing_where = form_data.get("where", "")

    simple_where_filters: list[QueryObjectFilterClause] = list(existing_filters)
    freeform_where_clauses: list[str] = []
    freeform_having_clauses: list[str] = []

    if existing_where:
        freeform_where_clauses.append(f"({existing_where})")

    for adhoc_filter in adhoc_filters:
        expression_type = adhoc_filter.get("expressionType", "")
        clause_type = adhoc_filter.get("clause", "WHERE")
        is_extra = adhoc_filter.get("isExtra", False)

        if expression_type == ExpressionType.SIMPLE.value:
            subject = adhoc_filter.get("subject", "")
            operator = adhoc_filter.get("operator", "")
            comparator = adhoc_filter.get("comparator")

            if clause_type == "WHERE":
                simple_where_filters.append(
                    {
                        "col": subject,
                        "op": operator,
                        "val": comparator,
                        "isExtra": is_extra,
                    }
                )
            # HAVING clause with SIMPLE type goes to freeform

        elif expression_type == ExpressionType.SQL.value:
            sql_expression = adhoc_filter.get("sqlExpression", "")
            if sql_expression:
                sanitized = f"({sql_expression})"
                if clause_type == "WHERE":
                    freeform_where_clauses.append(sanitized)
                else:
                    freeform_having_clauses.append(sanitized)

    extras: QueryObjectExtras = cast(QueryObjectExtras, dict(existing_extras))

    if freeform_where_clauses:
        extras["where"] = " AND ".join(freeform_where_clauses)

    if freeform_having_clauses:
        extras["having"] = " AND ".join(freeform_having_clauses)

    return {
        "filters": simple_where_filters,
        "extras": extras,
    }


def build_query_object(form_data: dict[str, Any], query_field_aliases: dict[str, str] | None = None) -> QueryObject:
    """
    Build a QueryObject from form_data.
    Based on buildQueryObject.ts
    """
    query_fields = extract_query_fields(form_data, query_field_aliases)
    extracted_extras = extract_extras(form_data)

    extra_form_data = form_data.get("extra_form_data", {})
    append_adhoc_filters = extra_form_data.get("adhoc_filters", [])
    append_filters = extra_form_data.get("filters", [])

    combined_filters = extracted_extras["filters"] + append_filters
    combined_adhoc_filters = form_data.get("adhoc_filters", []) + append_adhoc_filters

    filter_data = {
        "filters": combined_filters,
        "adhoc_filters": combined_adhoc_filters,
        "extras": extracted_extras["extras"],
        "where": form_data.get("where", ""),
    }
    processed_filters = process_filters(filter_data)

    # Parse numeric limits
    row_limit = form_data.get("row_limit")
    row_offset = form_data.get("row_offset")
    series_limit = form_data.get("series_limit") or form_data.get("limit") or 0

    try:
        row_limit = int(row_limit) if row_limit is not None else None
    except (ValueError, TypeError):
        row_limit = None

    try:
        row_offset = int(row_offset) if row_offset is not None else None
    except (ValueError, TypeError):
        row_offset = None

    try:
        series_limit = int(series_limit)
    except (ValueError, TypeError):
        series_limit = 0

    series_limit_metric = form_data.get("series_limit_metric") or form_data.get("timeseries_limit_metric")

    query_object = QueryObject(
        time_range=extracted_extras.get("time_range") or form_data.get("time_range"),
        since=form_data.get("since"),
        until=form_data.get("until"),
        granularity=extracted_extras.get("granularity")
        or form_data.get("granularity")
        or form_data.get("granularity_sqla"),
        columns=query_fields["columns"],
        metrics=query_fields["metrics"],
        orderby=query_fields["orderby"],
        filters=processed_filters["filters"],
        extras=processed_filters["extras"],
        row_limit=row_limit,
        row_offset=row_offset,
        series_columns=form_data.get("series_columns"),
        series_limit=series_limit,
        series_limit_metric=series_limit_metric,
        annotation_layers=form_data.get("annotation_layers", []),
        applied_time_extras=extracted_extras.get("applied_time_extras", {}),
        order_desc=form_data.get("order_desc", True),
        url_params=form_data.get("url_params", {}),
        custom_params=form_data.get("custom_params", {}),
    )

    # Apply extra_form_data overrides
    overrides = {k: v for k, v in extra_form_data.items() if k not in ("adhoc_filters", "filters", "custom_form_data")}
    for key, value in overrides.items():
        if hasattr(query_object, key) and value is not None:
            setattr(query_object, key, value)

    query_object.custom_form_data = extra_form_data.get("custom_form_data", {})

    return query_object


# =============================================================================
# Chart-Specific Build Query Functions
# =============================================================================

BuildQueryFunc = Callable[[QueryObject, dict[str, Any]], list[QueryObject]]


def build_default_query(base_query: QueryObject, _form_data: dict[str, Any]) -> list[QueryObject]:
    """Default buildQuery - just wraps in array."""
    return [base_query]


def build_timeseries_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for timeseries charts (ECharts Area, Bar, Line, etc.)
    Based on Timeseries/buildQuery.ts
    """
    groupby = ensure_list(form_data.get("groupby", []))

    # Build columns: x_axis (with time grain) + groupby
    # When x_axis is set, use get_x_axis_column_with_time_grain for BASE_AXIS transformation
    # When x_axis is not set, columns only contain groupby (is_timeseries=True handles time)
    columns = []
    if is_x_axis_set(form_data):
        x_col = get_x_axis_column_with_time_grain(form_data)
        if x_col:
            columns.append(x_col)
    columns.extend(groupby)

    # Determine time_offsets for time comparison
    time_offsets = []
    if is_time_comparison(form_data, base_query):
        time_offsets = ensure_list(form_data.get("time_compare", []))

    # Build post_processing pipeline
    post_processing = []

    # 1. Pivot operator
    if is_time_comparison(form_data, base_query):
        pivot_op = time_compare_pivot_operator(form_data, base_query)
    else:
        pivot_op = pivot_operator(form_data, base_query)
    if pivot_op:
        post_processing.append(pivot_op)

    # 2. Rolling window
    rolling_op = rolling_window_operator(form_data, base_query)
    if rolling_op:
        post_processing.append(rolling_op)

    # 3. Time compare
    time_cmp_op = time_compare_operator(form_data, base_query)
    if time_cmp_op:
        post_processing.append(time_cmp_op)

    # 4. Resample
    resample_op = resample_operator(form_data, base_query)
    if resample_op:
        post_processing.append(resample_op)

    # 5. Rename
    rename_op = rename_operator(form_data, base_query)
    if rename_op:
        post_processing.append(rename_op)

    # 6. Contribution
    contribution_op = contribution_operator(form_data, base_query, time_offsets)
    if contribution_op:
        post_processing.append(contribution_op)

    # 7. Sort
    sort_op = sort_operator(form_data, base_query)
    if sort_op:
        post_processing.append(sort_op)

    # 8. Flatten
    post_processing.append(flatten_operator(form_data, base_query))

    # 9. Prophet
    prophet_op = prophet_operator(form_data, base_query)
    if prophet_op:
        post_processing.append(prophet_op)

    # Update query object
    base_query.columns = columns
    base_query.series_columns = groupby if groupby else None
    base_query.orderby = normalize_orderby(base_query)
    base_query.time_offsets = time_offsets
    base_query.post_processing = [p for p in post_processing if p]

    # Set is_timeseries if x_axis is not set
    if not is_x_axis_set(form_data):
        base_query.is_timeseries = True

    return [base_query]


def build_boxplot_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for BoxPlot chart.
    Based on BoxPlot/buildQuery.ts
    """
    groupby = ensure_list(form_data.get("groupby", []))
    form_columns = ensure_list(form_data.get("columns", []))
    time_grain = form_data.get("time_grain_sqla")
    granularity = form_data.get("granularity_sqla")
    temporal_lookup = form_data.get("temporal_columns_lookup", {})

    # Build columns
    columns = []

    # If no columns specified, use granularity_sqla for backwards compatibility
    if not form_columns and granularity:
        form_columns = [granularity]

    for col in form_columns:
        if is_physical_column(col) and time_grain and temporal_lookup.get(col):
            columns.append(
                {
                    "timeGrain": time_grain,
                    "columnType": "BASE_AXIS",
                    "sqlExpression": col,
                    "label": col,
                    "expressionType": "SQL",
                }
            )
        else:
            columns.append(col)

    columns.extend(groupby)

    base_query.columns = columns
    base_query.series_columns = groupby if groupby else None

    # Add boxplot post_processing
    boxplot_op = boxplot_operator(form_data, base_query)
    if boxplot_op:
        base_query.post_processing = [boxplot_op]

    return [base_query]


def build_table_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Table chart.
    Based on plugin-chart-table/src/buildQuery.ts
    """
    query_mode = form_data.get("query_mode", QueryMode.AGGREGATE.value)
    all_columns = form_data.get("all_columns", [])

    if all_columns:
        query_mode = QueryMode.RAW.value

    percent_metrics = form_data.get("percent_metrics", [])
    order_desc = form_data.get("order_desc", False)
    sort_by_metric = ensure_list(form_data.get("timeseries_limit_metric", []))
    time_grain = form_data.get("time_grain_sqla")
    temporal_lookup = form_data.get("temporal_columns_lookup", {})

    post_processing: list[PostProcessingRule] = []
    time_offsets: list[str] = []

    # Handle time comparison
    time_compare = ensure_list(form_data.get("time_compare", []))
    non_custom_shifts = [s for s in time_compare if s not in ("custom", "inherit")]
    custom_shifts = [s for s in time_compare if s in ("custom", "inherit")]

    if is_time_comparison(form_data, base_query) and non_custom_shifts:
        time_offsets = non_custom_shifts

    if is_time_comparison(form_data, base_query) and custom_shifts:
        if "custom" in custom_shifts:
            offset = form_data.get("start_date_offset")
            if offset:
                time_offsets.append(offset)
        if "inherit" in custom_shifts:
            time_offsets.append("inherit")

    columns = list(base_query.columns)
    metrics = list(base_query.metrics)
    orderby = list(base_query.orderby)

    if query_mode == QueryMode.AGGREGATE.value:
        # Override orderby with timeseries metric
        if sort_by_metric:
            orderby = [(sort_by_metric[0], not order_desc)]
        elif metrics:
            orderby = [(metrics[0], False)]

        # Add percent metrics post_processing
        if percent_metrics:
            percent_labels = [get_metric_label(m) for m in percent_metrics]
            post_processing.append(
                {
                    "operation": "contribution",
                    "options": {
                        "columns": percent_labels,
                        "rename_columns": [f"%{m}" for m in percent_labels],
                    },
                }
            )
            # Add percent metrics to metrics list
            for pm in percent_metrics:
                if pm not in metrics:
                    metrics.append(pm)

        # Add time compare operator
        if time_offsets:
            time_cmp_op = time_compare_operator(form_data, base_query)
            if time_cmp_op:
                post_processing.append(time_cmp_op)

        # Transform temporal columns
        new_columns = []
        temporal_added = False
        for col in columns:
            if is_physical_column(col) and time_grain and temporal_lookup.get(col):
                if not temporal_added:
                    new_columns.insert(
                        0,
                        {
                            "timeGrain": time_grain,
                            "columnType": "BASE_AXIS",
                            "sqlExpression": col,
                            "label": col,
                            "expressionType": "SQL",
                        },
                    )
                    temporal_added = True
            else:
                new_columns.append(col)
        columns = new_columns

    base_query.columns = columns
    base_query.metrics = metrics
    base_query.orderby = orderby
    base_query.post_processing = post_processing
    base_query.time_offsets = time_offsets

    return [base_query]


def build_big_number_query(base_query: QueryObject, _form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for BigNumber Total chart.
    Based on BigNumber/BigNumberTotal/buildQuery.ts
    """
    return [base_query]


def build_big_number_trendline_query(base_query: QueryObject, _form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for BigNumber with Trendline chart.
    Based on BigNumber/BigNumberWithTrendline/buildQuery.ts
    """
    metrics = ensure_list(base_query.metrics)
    metric_labels = [get_metric_label(m) for m in metrics]

    # Add pivot + flatten post_processing
    post_processing = []

    if metric_labels:
        post_processing.append(
            {
                "operation": "pivot",
                "options": {
                    "index": ["__timestamp"],
                    "columns": [],
                    "aggregates": {m: {"operator": "mean"} for m in metric_labels},
                    "drop_missing_columns": True,
                },
            }
        )
        post_processing.append({"operation": "flatten"})

    base_query.post_processing = post_processing
    base_query.is_timeseries = True

    return [base_query]


def build_pie_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Pie chart.
    Based on Pie/buildQuery.ts

    Pie chart ALWAYS adds contribution operator for percentage calculation.
    """
    sort_by_metric = form_data.get("sort_by_metric", False)
    metric = form_data.get("metric")
    metrics = ensure_list(base_query.metrics)

    if sort_by_metric and metric:
        base_query.orderby = [(metric, False)]

    # Pie chart always adds contribution for percentage display
    if metrics:
        metric_label = get_metric_label(metrics[0])
        base_query.post_processing = [
            cast(
                PostProcessingRule,
                {
                    "operation": "contribution",
                    "options": {
                        "columns": [metric_label],
                        "rename_columns": [f"% {metric_label}"],  # getContributionLabel
                    },
                },
            )
        ]

    return [base_query]


def build_funnel_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Funnel chart.
    Based on Funnel/buildQuery.ts
    """
    sort_by_metric = form_data.get("sort_by_metric", False)
    metric = form_data.get("metric")

    if sort_by_metric and metric:
        base_query.orderby = [(metric, False)]

    return [base_query]


def build_gauge_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Gauge chart.
    Based on Gauge/buildQuery.ts
    """
    sort_by_metric = form_data.get("sort_by_metric", False)
    metrics = ensure_list(base_query.metrics)

    if sort_by_metric and metrics:
        base_query.orderby = [(metrics[0], False)]

    return [base_query]


def build_heatmap_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Heatmap chart.
    Based on Heatmap/buildQuery.ts

    sort_x_axis/sort_y_axis values: "value_asc", "value_desc", "alpha_asc", "alpha_desc"
    - "value_*" sorts by metric
    - "alpha_*" sorts by column
    - "*_asc" means ascending
    """
    x_axis = form_data.get("x_axis")
    groupby = ensure_list(form_data.get("groupby", []))
    metrics = ensure_list(base_query.metrics)
    metric = get_metric_label(metrics[0]) if metrics else None

    # Build columns with proper x_axis transformation (BASE_AXIS with timeGrain)
    columns = []
    x_axis_col = get_x_axis_column_with_time_grain(form_data)
    if x_axis_col:
        columns.append(x_axis_col)
    columns.extend(groupby)

    base_query.columns = columns

    # Add orderby based on sort options (use original column name for orderby)
    orderby = []
    sort_x = form_data.get("sort_x_axis")
    sort_y = form_data.get("sort_y_axis")

    if sort_x:
        # If "value" in sort option, sort by metric; otherwise by column
        sort_by = metric if "value" in sort_x else (x_axis if x_axis else None)
        ascending = "asc" in sort_x
        if sort_by:
            orderby.append((sort_by, ascending))

    if sort_y:
        # For y-axis, sort by metric or first groupby column
        sort_by = metric if "value" in sort_y else (groupby[0] if groupby else None)
        ascending = "asc" in sort_y
        if sort_by:
            orderby.append((sort_by, ascending))

    if orderby:
        base_query.orderby = orderby

    # Add rank operator for normalization
    normalize_across = form_data.get("normalize_across")
    if normalize_across and metric:
        group_by = None
        if normalize_across == "x":
            group_by = get_column_label(x_axis) if x_axis else None
        elif normalize_across == "y" and groupby:
            group_by = get_column_label(groupby[0])

        base_query.post_processing = [
            cast(
                PostProcessingRule,
                {
                    "operation": "rank",
                    "options": {
                        "metric": metric,
                        "group_by": group_by,
                    },
                },
            )
        ]

    return [base_query]


def build_histogram_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Histogram chart.
    Based on Histogram/buildQuery.ts
    """
    column = form_data.get("column")
    groupby = ensure_list(form_data.get("groupby", []))

    columns = []
    if column:
        columns.append(column)
    columns.extend(groupby)

    base_query.columns = columns
    base_query.metrics = []  # Histogram uses raw data

    # Add histogram post_processing
    histogram_op = histogram_operator(form_data, base_query)
    if histogram_op:
        base_query.post_processing = [histogram_op]

    return [base_query]


def build_bubble_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Bubble chart.
    Based on Bubble/buildQuery.ts
    """
    entity = form_data.get("entity")
    series = form_data.get("series")

    columns = []
    if entity:
        columns.append(entity)
    if series:
        columns.append(series)

    base_query.columns = columns

    # Invert orderby direction
    orderby = base_query.orderby
    if orderby:
        base_query.orderby = [(o[0], not o[1]) for o in orderby]

    return [base_query]


def build_waterfall_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Waterfall chart.
    Based on Waterfall/buildQuery.ts
    """
    groupby = ensure_list(form_data.get("groupby", []))

    # Build columns with proper x_axis transformation (BASE_AXIS with timeGrain)
    columns = []
    x_axis_col = get_x_axis_column_with_time_grain(form_data)
    if x_axis_col:
        columns.append(x_axis_col)
    columns.extend(groupby)

    base_query.columns = columns

    # Order by x_axis (use the original column name for orderby, not the dict)
    x_axis = form_data.get("x_axis")
    orderby_columns = []
    if x_axis:
        orderby_columns.append(x_axis)
    orderby_columns.extend(groupby)

    if orderby_columns:
        base_query.orderby = [(col, True) for col in orderby_columns]

    return [base_query]


def build_sankey_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Sankey chart.
    Based on Sankey/buildQuery.ts
    """
    source = form_data.get("source")
    target = form_data.get("target")

    columns = []
    if source:
        columns.append(source)
    if target:
        columns.append(target)

    base_query.columns = columns

    # Optional orderby by metric
    sort_by_metric = form_data.get("sort_by_metric", False)
    metrics = ensure_list(base_query.metrics)

    if sort_by_metric and metrics:
        base_query.orderby = [(metrics[0], False)]

    return [base_query]


def build_sunburst_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Sunburst chart.
    Based on Sunburst/buildQuery.ts
    """
    sort_by_metric = form_data.get("sort_by_metric", False)
    metrics = ensure_list(base_query.metrics)

    if sort_by_metric and metrics:
        base_query.orderby = [(metrics[0], False)]

    return [base_query]


def build_treemap_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Treemap chart.
    Based on Treemap/buildQuery.ts
    """
    sort_by_metric = form_data.get("sort_by_metric", False)
    metrics = ensure_list(base_query.metrics)

    if sort_by_metric and metrics:
        base_query.orderby = [(metrics[0], False)]

    return [base_query]


def build_word_cloud_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Word Cloud chart.
    Based on plugin-chart-word-cloud/src/plugin/buildQuery.ts
    """
    sort_by_metric = form_data.get("sort_by_metric", False)
    metrics = ensure_list(base_query.metrics)

    if sort_by_metric and metrics:
        base_query.orderby = [(metrics[0], False)]

    return [base_query]


def build_graph_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Graph/Network chart.
    Based on Graph/buildQuery.ts
    """
    source = form_data.get("source")
    target = form_data.get("target")
    source_category = form_data.get("source_category")
    target_category = form_data.get("target_category")

    columns = []
    for col in [source, target, source_category, target_category]:
        if col:
            columns.append(col)

    base_query.columns = columns

    return [base_query]


def build_tree_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Tree chart.
    Based on Tree/buildQuery.ts
    """
    id_col = form_data.get("id")
    parent_col = form_data.get("parent")
    name_col = form_data.get("name")

    columns = []
    for col in [id_col, parent_col, name_col]:
        if col:
            columns.append(col)

    base_query.columns = columns

    return [base_query]


def build_radar_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Radar chart.
    Based on Radar/buildQuery.ts
    """
    x_axis = form_data.get("x_axis") or form_data.get("groupby", [])
    groupby = ensure_list(form_data.get("groupby", []))

    columns = []
    if x_axis:
        columns.extend(ensure_list(x_axis))
    for g in groupby:
        if g not in columns:
            columns.append(g)

    base_query.columns = columns

    # Add rank operator
    rank_op = rank_operator(form_data, base_query)
    if rank_op:
        base_query.post_processing = [rank_op]

    return [base_query]


def build_pivot_table_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Pivot Table v2 chart.
    Based on plugin-chart-pivot-table/src/plugin/buildQuery.ts
    """
    groupby_columns = ensure_list(form_data.get("groupbyColumns", []))
    groupby_rows = ensure_list(form_data.get("groupbyRows", []))
    time_grain = form_data.get("time_grain_sqla")
    temporal_lookup = form_data.get("temporal_columns_lookup", {})

    columns = []

    # Process groupby columns with temporal wrapping
    for col in groupby_columns + groupby_rows:
        if is_physical_column(col) and time_grain and temporal_lookup.get(col):
            columns.append(
                {
                    "timeGrain": time_grain,
                    "columnType": "BASE_AXIS",
                    "sqlExpression": col,
                    "label": col,
                    "expressionType": "SQL",
                }
            )
        else:
            columns.append(col)

    base_query.columns = columns

    # Add orderby
    metrics = ensure_list(base_query.metrics)
    series_limit_metric = form_data.get("series_limit_metric")
    order_desc = form_data.get("order_desc", True)

    if series_limit_metric:
        base_query.orderby = [(series_limit_metric, not order_desc)]
    elif metrics:
        base_query.orderby = [(metrics[0], not order_desc)]

    return [base_query]


def build_mixed_timeseries_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Mixed Timeseries chart (multiple query objects).
    Based on MixedTimeseries/buildQuery.ts
    """
    queries = []

    # First query (Query A)
    query_a = build_timeseries_query(deepcopy(base_query), form_data)[0]
    queries.append(query_a)

    # Second query (Query B) - uses _b suffix fields
    metrics_b = form_data.get("metrics_b", [])
    groupby_b = form_data.get("groupby_b", [])

    if metrics_b:
        query_b = deepcopy(base_query)
        query_b.metrics = metrics_b

        # Build columns for query B (with BASE_AXIS transformation)
        columns_b = []
        if is_x_axis_set(form_data):
            x_col = get_x_axis_column_with_time_grain(form_data)
            if x_col:
                columns_b.append(x_col)
        columns_b.extend(ensure_list(groupby_b))

        query_b.columns = columns_b
        query_b.series_columns = groupby_b if groupby_b else None

        # Build post_processing for query B
        form_data_b = {**form_data, "groupby": groupby_b}
        query_b = build_timeseries_query(query_b, form_data_b)[0]
        queries.append(query_b)

    return queries


# =============================================================================
# Additional Modern Chart Build Query Functions
# =============================================================================


def build_gantt_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Gantt chart.
    Based on Gantt/buildQuery.ts
    """
    start_time = form_data.get("start_time")
    end_time = form_data.get("end_time")
    y_axis = form_data.get("y_axis")
    series = form_data.get("series")
    tooltip_columns = ensure_list(form_data.get("tooltip_columns", []))
    tooltip_metrics = ensure_list(form_data.get("tooltip_metrics", []))
    order_by_cols = ensure_list(form_data.get("order_by_cols", []))

    groupby = ensure_list(series)

    # Parse orderby from JSON strings
    orderby: list[tuple[Any, bool]] = []
    for expr in order_by_cols:
        try:
            parsed = json.loads(expr)
            if isinstance(parsed, list) and len(parsed) == 2:
                orderby.append((parsed[0], bool(parsed[1])))
        except (json.JSONDecodeError, TypeError):
            pass

    # Collect all columns
    columns_set: set[str] = set()
    for col in [start_time, end_time, y_axis]:
        if col:
            columns_set.add(col)
    for col in groupby:
        if col:
            columns_set.add(col)
    for col in tooltip_columns:
        if col:
            columns_set.add(col)
    for ob in orderby:
        if ob[0]:
            columns_set.add(str(ob[0]))

    base_query.columns = list(columns_set)
    base_query.metrics = tooltip_metrics
    base_query.orderby = orderby
    base_query.series_columns = groupby

    return [base_query]


def build_pop_kpi_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Period Over Period KPI (pop_kpi) chart.
    Based on BigNumber/BigNumberPeriodOverPeriod/buildQuery.ts
    """
    cols = ensure_list(form_data.get("cols", []))

    # Time comparison logic
    non_custom_shifts = [s for s in ensure_list(form_data.get("time_compare", [])) if s not in ("custom", "inherit")]
    custom_or_inherit_shifts = [s for s in ensure_list(form_data.get("time_compare", [])) if s in ("custom", "inherit")]

    time_offsets: list[str] = list(non_custom_shifts)

    if custom_or_inherit_shifts:
        if "custom" in custom_or_inherit_shifts:
            start_offset = form_data.get("start_date_offset")
            if start_offset:
                time_offsets.append(start_offset)
        if "inherit" in custom_or_inherit_shifts:
            time_offsets.append("inherit")

    # Add time compare post-processing
    post_processing: list[PostProcessingRule] = []
    if time_offsets and is_time_comparison(form_data, base_query):
        compare_op = time_compare_operator(form_data, base_query)
        if compare_op:
            post_processing.append(compare_op)

    base_query.columns = cols
    base_query.post_processing = post_processing
    base_query.time_offsets = time_offsets if is_time_comparison(form_data, base_query) else []

    return [base_query]


# =============================================================================
# Deck.GL Chart Build Query Functions
# =============================================================================


def _get_spatial_columns(spatial: dict[str, Any] | None) -> list[str]:
    """Extract column names from spatial configuration."""
    if not spatial:
        return []

    spatial_type = spatial.get("type", "latlong")
    columns = []

    if spatial_type == "latlong":
        lat_col = spatial.get("latCol")
        lon_col = spatial.get("lonCol")
        if lat_col:
            columns.append(lat_col)
        if lon_col:
            columns.append(lon_col)
    elif spatial_type == "delimited":
        delimited_col = spatial.get("lonlatCol")
        if delimited_col:
            columns.append(delimited_col)
    elif spatial_type == "geohash":
        geohash_col = spatial.get("geohashCol")
        if geohash_col:
            columns.append(geohash_col)

    return columns


def _add_spatial_null_filters(
    spatial: dict[str, Any] | None, filters: list[QueryObjectFilterClause]
) -> list[QueryObjectFilterClause]:
    """Add IS NOT NULL filters for spatial columns."""
    if not spatial:
        return filters

    result = list(filters)
    for col in _get_spatial_columns(spatial):
        result.append(
            {
                "col": col,
                "op": "IS NOT NULL",
                "val": None,
            }
        )
    return result


def build_deck_arc_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Arc chart (viz_type: deck_arc).
    Based on legacy-preset-chart-deckgl/src/layers/Arc/buildQuery.ts
    """
    start_spatial = form_data.get("start_spatial")
    end_spatial = form_data.get("end_spatial")
    dimension = form_data.get("dimension")
    js_columns = ensure_list(form_data.get("js_columns", []))
    tooltip_contents = ensure_list(form_data.get("tooltip_contents", []))

    columns = list(base_query.columns or [])
    columns.extend(_get_spatial_columns(start_spatial))
    columns.extend(_get_spatial_columns(end_spatial))

    if dimension:
        columns.append(dimension)

    for col in js_columns:
        if col and col not in columns:
            columns.append(col)

    # Add tooltip columns
    for item in tooltip_contents:
        if isinstance(item, dict) and item.get("column"):
            col = item["column"]
            if col not in columns:
                columns.append(col)

    filters = _add_spatial_null_filters(start_spatial, list(base_query.filters or []))
    filters = _add_spatial_null_filters(end_spatial, filters)

    base_query.columns = columns
    base_query.filters = filters
    base_query.is_timeseries = bool(form_data.get("time_grain_sqla"))

    return [base_query]


def build_deck_scatter_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Scatter chart (viz_type: deck_scatter).
    Based on legacy-preset-chart-deckgl/src/layers/Scatter/buildQuery.ts
    """
    spatial = form_data.get("spatial")
    point_radius_fixed = form_data.get("point_radius_fixed", {})
    dimension = form_data.get("dimension")
    js_columns = ensure_list(form_data.get("js_columns", []))
    tooltip_contents = ensure_list(form_data.get("tooltip_contents", []))

    columns = list(base_query.columns or [])
    columns.extend(_get_spatial_columns(spatial))

    if dimension:
        columns.append(dimension)

    for col in js_columns:
        if col and col not in columns:
            columns.append(col)

    for item in tooltip_contents:
        if isinstance(item, dict) and item.get("column"):
            col = item["column"]
            if col not in columns:
                columns.append(col)

    # Metrics from point_radius_fixed
    metrics = []
    radius_value = point_radius_fixed.get("value") if isinstance(point_radius_fixed, dict) else None
    if radius_value:
        metrics.append(radius_value)

    filters = _add_spatial_null_filters(spatial, list(base_query.filters or []))

    base_query.columns = columns
    base_query.metrics = metrics
    base_query.filters = filters
    base_query.is_timeseries = False

    if radius_value:
        base_query.orderby = [(radius_value, False)]

    return [base_query]


def build_deck_grid_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Grid chart (viz_type: deck_grid).
    """
    spatial = form_data.get("spatial")
    js_columns = ensure_list(form_data.get("js_columns", []))

    columns = list(base_query.columns or [])
    columns.extend(_get_spatial_columns(spatial))

    for col in js_columns:
        if col and col not in columns:
            columns.append(col)

    filters = _add_spatial_null_filters(spatial, list(base_query.filters or []))

    base_query.columns = columns
    base_query.filters = filters
    base_query.is_timeseries = bool(form_data.get("time_grain_sqla"))

    return [base_query]


def build_deck_hex_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Hex chart (viz_type: deck_hex).
    """
    return build_deck_grid_query(base_query, form_data)


def build_deck_heatmap_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Heatmap chart (viz_type: deck_heatmap).
    Note: Different from ECharts heatmap.
    """
    return build_deck_grid_query(base_query, form_data)


def build_deck_contour_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Contour chart (viz_type: deck_contour).
    """
    return build_deck_grid_query(base_query, form_data)


def build_deck_screengrid_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Screen Grid chart (viz_type: deck_screengrid).
    """
    return build_deck_grid_query(base_query, form_data)


def build_deck_path_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Path chart (viz_type: deck_path).
    """
    line_column = form_data.get("line_column")
    js_columns = ensure_list(form_data.get("js_columns", []))

    columns = list(base_query.columns or [])

    if line_column:
        columns.append(line_column)

    for col in js_columns:
        if col and col not in columns:
            columns.append(col)

    base_query.columns = columns
    base_query.is_timeseries = bool(form_data.get("time_grain_sqla"))

    return [base_query]


def build_deck_polygon_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for Deck.gl Polygon chart (viz_type: deck_polygon).
    """
    line_column = form_data.get("line_column")
    js_columns = ensure_list(form_data.get("js_columns", []))

    columns = list(base_query.columns or [])

    if line_column:
        columns.append(line_column)

    for col in js_columns:
        if col and col not in columns:
            columns.append(col)

    base_query.columns = columns
    base_query.is_timeseries = bool(form_data.get("time_grain_sqla"))

    return [base_query]


# =============================================================================
# Legacy Chart Build Query Functions (for explore_json API)
# =============================================================================
# These functions implement query_obj() logic from superset/viz.py
# for legacy charts that use the explore_json endpoint instead of chart data API


def build_legacy_bubble_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy NVD3 Bubble chart (viz_type: bubble).
    Based on superset/viz.py BubbleViz.query_obj()

    Note: Different from ECharts bubble which uses x_axis/y_axis columns.
    Legacy bubble uses entity/series for groupby and x/y/size for metrics.
    """
    entity = form_data.get("entity")
    series = form_data.get("series")
    x_metric = form_data.get("x")
    y_metric = form_data.get("y")
    size_metric = form_data.get("size")
    limit = form_data.get("limit")

    # groupby = [entity, series?] - deduped
    groupby = []
    if entity:
        groupby.append(entity)
    if series and series != entity:
        groupby.append(series)

    base_query.columns = groupby

    # metrics = [size, x, y] in this specific order
    metrics = []
    for m in [size_metric, x_metric, y_metric]:
        if m:
            metrics.append(m)
    base_query.metrics = metrics

    # Use "limit" field for row_limit
    if limit:
        base_query.row_limit = int(limit)

    # Legacy bubble is not timeseries
    base_query.is_timeseries = False

    return [base_query]


def build_legacy_bullet_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy NVD3 Bullet chart (viz_type: bullet).
    Based on superset/viz.py BulletViz.query_obj()
    """
    metric = form_data.get("metric")

    if metric:
        base_query.metrics = [metric]

    base_query.is_timeseries = False

    return [base_query]


def build_legacy_timeseries_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy NVD3 timeseries charts.
    Based on superset/viz.py NVD3TimeSeriesViz.query_obj()

    Used by: compare, horizon, rose
    """
    timeseries_limit_metric = form_data.get("timeseries_limit_metric")
    order_desc = form_data.get("order_desc", True)
    metrics = ensure_list(base_query.metrics or [])

    # Add timeseries_limit_metric to metrics if not present
    sort_by = timeseries_limit_metric or (metrics[0] if metrics else None)

    if sort_by:
        sort_by_label = get_metric_label(sort_by)
        if sort_by_label not in [get_metric_label(m) for m in metrics]:
            metrics.append(sort_by)
            base_query.metrics = metrics

        # is_asc = not order_desc
        base_query.orderby = [(sort_by, not order_desc)]

    base_query.is_timeseries = True

    return [base_query]


def build_legacy_time_pivot_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy NVD3 Time Pivot chart (viz_type: time_pivot).
    Based on superset/viz.py NVD3TimePivotViz.query_obj()
    """
    # First apply base timeseries query logic
    queries = build_legacy_timeseries_query(base_query, form_data)
    query = queries[0]

    # Override metrics to single metric
    metric = form_data.get("metric")
    if metric:
        query.metrics = [metric]

    return [query]


def build_legacy_chord_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Chord diagram (viz_type: chord).
    Based on superset/viz.py ChordViz.query_obj()
    """
    groupby_col = form_data.get("groupby")
    columns_col = form_data.get("columns")
    metric = form_data.get("metric")
    sort_by_metric = form_data.get("sort_by_metric", False)

    # groupby = [groupby, columns]
    columns = []
    if groupby_col:
        columns.append(groupby_col)
    if columns_col:
        columns.append(columns_col)
    base_query.columns = columns

    # metrics = [metric]
    if metric:
        base_query.metrics = [metric]

    # Optional orderby
    if sort_by_metric and base_query.metrics:
        base_query.orderby = [(base_query.metrics[0], False)]

    base_query.is_timeseries = False

    return [base_query]


def build_legacy_country_map_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Country Map (viz_type: country_map).
    Based on superset/viz.py CountryMapViz.query_obj()
    """
    entity = form_data.get("entity")
    metric = form_data.get("metric")

    # groupby = [entity]
    if entity:
        base_query.columns = [entity]

    # metrics = [metric]
    if metric:
        base_query.metrics = [metric]

    base_query.is_timeseries = False

    return [base_query]


def build_legacy_world_map_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy World Map (viz_type: world_map).
    Based on superset/viz.py WorldMapViz.query_obj()
    """
    entity = form_data.get("entity")
    sort_by_metric = form_data.get("sort_by_metric", False)

    # groupby = [entity]
    if entity:
        base_query.columns = [entity]

    # Optional orderby
    metrics = ensure_list(base_query.metrics or [])
    if sort_by_metric and metrics:
        base_query.orderby = [(metrics[0], False)]

    base_query.is_timeseries = False

    return [base_query]


def build_legacy_parallel_coordinates_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Parallel Coordinates (viz_type: para).
    Based on superset/viz.py ParallelCoordinatesViz.query_obj()
    """
    series = form_data.get("series")
    timeseries_limit_metric = form_data.get("timeseries_limit_metric")
    order_desc = form_data.get("order_desc", True)
    metrics = ensure_list(base_query.metrics or [])

    # groupby = [series]
    if series:
        base_query.columns = [series]

    # Add timeseries_limit_metric to metrics if present and not already included
    if timeseries_limit_metric:
        sort_by_label = get_metric_label(timeseries_limit_metric)
        if sort_by_label not in [get_metric_label(m) for m in metrics]:
            metrics.append(timeseries_limit_metric)
            base_query.metrics = metrics

        if order_desc:
            base_query.orderby = [(timeseries_limit_metric, not order_desc)]

    base_query.is_timeseries = False

    return [base_query]


def build_legacy_mapbox_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Mapbox chart (viz_type: mapbox).
    Based on superset/viz.py MapboxViz.query_obj()
    """
    groupby = form_data.get("groupby")
    all_columns_x = form_data.get("all_columns_x")  # longitude
    all_columns_y = form_data.get("all_columns_y")  # latitude
    mapbox_label = form_data.get("mapbox_label", [])
    point_radius = form_data.get("point_radius")

    if not groupby:
        # Raw mode: use columns
        columns = []

        if all_columns_x:
            columns.append(all_columns_x)
        if all_columns_y:
            columns.append(all_columns_y)

        # Add label column if not "count"
        if mapbox_label and len(mapbox_label) >= 1:
            if mapbox_label[0] != "count":
                columns.append(mapbox_label[0])

        # Add point_radius column if not "Auto"
        if point_radius and point_radius != "Auto":
            columns.append(point_radius)

        # Sort and dedupe columns
        base_query.columns = sorted(set(columns))
        base_query.metrics = []

    # If groupby is set, validation is done on form_data side
    # The groupby must contain lat/lon columns

    base_query.is_timeseries = False

    return [base_query]


def build_legacy_cal_heatmap_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Calendar Heatmap (viz_type: cal_heatmap).
    Based on superset/viz.py CalHeatmapViz.query_obj()
    """
    metrics = form_data.get("metrics", [])
    subdomain_granularity = form_data.get("subdomain_granularity", "min")

    # Mapping for time_grain_sqla
    time_grain_mapping = {
        "min": "PT1M",
        "hour": "PT1H",
        "day": "P1D",
        "week": "P1W",
        "month": "P1M",
        "year": "P1Y",
    }

    base_query.metrics = metrics

    # Set time_grain_sqla in extras
    if not base_query.extras:
        base_query.extras = {}
    base_query.extras["time_grain_sqla"] = time_grain_mapping.get(subdomain_granularity, "PT1M")

    base_query.is_timeseries = True

    return [base_query]


def build_legacy_partition_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Partition diagram (viz_type: partition).
    Based on superset/viz.py PartitionViz.query_obj()
    """
    # First apply base timeseries query logic
    queries = build_legacy_timeseries_query(base_query, form_data)
    query = queries[0]

    # Return time series data if the user specifies so
    time_series_option = form_data.get("time_series_option", "not_time")
    query.is_timeseries = time_series_option != "not_time"

    return [query]


def build_legacy_paired_ttest_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Paired T-Test chart (viz_type: paired_ttest).
    Based on superset/viz.py PairedTTestViz.query_obj()
    """
    timeseries_limit_metric = form_data.get("timeseries_limit_metric")
    order_desc = form_data.get("order_desc", True)
    metrics = ensure_list(base_query.metrics or [])

    # Add timeseries_limit_metric to metrics if present and not already included
    if timeseries_limit_metric:
        sort_by_label = get_metric_label(timeseries_limit_metric)
        if sort_by_label not in [get_metric_label(m) for m in metrics]:
            metrics.append(timeseries_limit_metric)
            base_query.metrics = metrics

        if order_desc:
            base_query.orderby = [(timeseries_limit_metric, not order_desc)]

    base_query.is_timeseries = True

    return [base_query]


def build_legacy_time_table_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Time Table (viz_type: time_table).
    Based on superset/viz.py TimeTableViz.query_obj()
    """
    metrics = ensure_list(base_query.metrics or [])
    order_desc = form_data.get("order_desc", True)

    # orderby by first metric
    if metrics:
        sort_by = get_metric_label(metrics[0])
        base_query.orderby = [(sort_by, not order_desc)]

    base_query.is_timeseries = True

    return [base_query]


def build_legacy_deck_multi_query(_base_query: QueryObject, _form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Deck.gl Multiple Layers (viz_type: deck_multi).
    Based on superset/viz.py DeckGLMultiLayer.query_obj()

    This chart returns an empty query object because it aggregates
    data from other chart slices.
    """
    # Return empty query - this chart loads data from child slices
    empty_query = QueryObject(
        columns=[],
        metrics=[],
        filters=[],
        orderby=[],
        extras=cast(QueryObjectExtras, {}),
        is_timeseries=False,
    )
    return [empty_query]


def build_legacy_deck_geojson_query(base_query: QueryObject, form_data: dict[str, Any]) -> list[QueryObject]:
    """
    Build query for legacy Deck.gl GeoJSON (viz_type: deck_geojson).
    Based on superset/viz.py DeckGeoJson.query_obj()
    """
    geojson_col = form_data.get("geojson")

    # columns += [geojson]
    columns = list(base_query.columns or [])
    if geojson_col:
        columns.append(geojson_col)

    base_query.columns = columns
    base_query.metrics = []
    # Note: original sets groupby=[], but columns already handles raw mode

    base_query.is_timeseries = False

    return [base_query]


# =============================================================================
# Chart Build Query Registry
# =============================================================================


class ChartBuildQueryRegistry:
    """Registry for chart-type specific buildQuery functions."""

    def __init__(self):
        self._registry: dict[str, BuildQueryFunc] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default buildQuery functions for known chart types."""

        # Timeseries charts
        timeseries_types = [
            "echarts_timeseries",
            "echarts_timeseries_bar",
            "echarts_timeseries_line",
            "echarts_timeseries_scatter",
            "echarts_timeseries_smooth",
            "echarts_timeseries_step",
            "echarts_area",
        ]
        for chart_type in timeseries_types:
            self.register(chart_type, build_timeseries_query)

        # Mixed timeseries
        self.register("mixed_timeseries", build_mixed_timeseries_query)

        # BigNumber
        self.register("big_number_total", build_big_number_query)
        self.register("big_number", build_big_number_trendline_query)

        # Table
        self.register("table", build_table_query)

        # Pivot Table
        self.register("pivot_table_v2", build_pivot_table_query)

        # BoxPlot
        self.register("box_plot", build_boxplot_query)

        # Pie/Donut
        self.register("pie", build_pie_query)

        # Funnel
        self.register("funnel", build_funnel_query)

        # Gauge
        self.register("gauge_chart", build_gauge_query)

        # Heatmap
        self.register("heatmap", build_heatmap_query)
        self.register("heatmap_v2", build_heatmap_query)

        # Histogram
        self.register("histogram", build_histogram_query)

        # Bubble (ECharts version - v2 only, legacy "bubble" is registered below)
        self.register("bubble_v2", build_bubble_query)

        # Waterfall
        self.register("waterfall", build_waterfall_query)

        # Sankey
        self.register("sankey", build_sankey_query)
        self.register("sankey_v2", build_sankey_query)

        # Sunburst
        self.register("sunburst", build_sunburst_query)
        self.register("sunburst_v2", build_sunburst_query)

        # Treemap
        self.register("treemap", build_treemap_query)
        self.register("treemap_v2", build_treemap_query)

        # Word Cloud
        self.register("word_cloud", build_word_cloud_query)

        # Graph
        self.register("graph_chart", build_graph_query)

        # Tree
        self.register("tree", build_tree_query)
        self.register("tree_chart", build_tree_query)

        # Radar
        self.register("radar", build_radar_query)

        # Gantt
        self.register("gantt_chart", build_gantt_query)

        # Period Over Period KPI
        self.register("pop_kpi", build_pop_kpi_query)

        # Histogram v2 (same as histogram)
        self.register("histogram_v2", build_histogram_query)

        # AG Grid Table (same logic as table)
        self.register("ag-grid-table", build_table_query)

        # Simple charts with default behavior
        simple_types = [
            "handlebars",
            "cartodiagram",  # Delegates to inner chart's buildQuery
        ]
        for chart_type in simple_types:
            self.register(chart_type, build_default_query)

        # =================================================================
        # Legacy Charts (useLegacyApi: true in frontend)
        # These charts use /explore_json endpoint in frontend, but we
        # provide build_query functions for form_data → query_obj conversion
        # =================================================================

        # Legacy NVD3 charts
        self.register("bubble", build_legacy_bubble_query)  # Override ECharts version
        self.register("bullet", build_legacy_bullet_query)
        self.register("compare", build_legacy_timeseries_query)
        self.register("time_pivot", build_legacy_time_pivot_query)

        # Legacy map charts
        self.register("world_map", build_legacy_world_map_query)
        self.register("country_map", build_legacy_country_map_query)
        self.register("mapbox", build_legacy_mapbox_query)

        # Legacy visualization charts
        self.register("chord", build_legacy_chord_query)
        self.register("cal_heatmap", build_legacy_cal_heatmap_query)
        self.register("horizon", build_legacy_timeseries_query)  # Same as compare
        self.register("para", build_legacy_parallel_coordinates_query)
        self.register("partition", build_legacy_partition_query)
        self.register("rose", build_legacy_timeseries_query)  # Same as compare
        self.register("paired_ttest", build_legacy_paired_ttest_query)
        self.register("time_table", build_legacy_time_table_query)

        # Legacy Deck.gl charts
        self.register("deck_multi", build_legacy_deck_multi_query)
        self.register("deck_geojson", build_legacy_deck_geojson_query)

        # Deck.GL spatial charts
        self.register("deck_arc", build_deck_arc_query)
        self.register("deck_scatter", build_deck_scatter_query)
        self.register("deck_grid", build_deck_grid_query)
        self.register("deck_hex", build_deck_hex_query)
        self.register("deck_heatmap", build_deck_heatmap_query)
        self.register("deck_contour", build_deck_contour_query)
        self.register("deck_screengrid", build_deck_screengrid_query)
        self.register("deck_path", build_deck_path_query)
        self.register("deck_polygon", build_deck_polygon_query)

        # Note: "bubble_v2" uses ECharts build_bubble_query (non-legacy)
        # Note: "markup" and "iframe" are legacy but have no query logic

    def register(self, chart_type: str, build_query_func: BuildQueryFunc):
        """Register a buildQuery function for a chart type."""
        self._registry[chart_type] = build_query_func

    def get(self, chart_type: str) -> BuildQueryFunc:
        """Get buildQuery function for a chart type, default to wrap_in_array."""
        return self._registry.get(chart_type, build_default_query)


# Global registry instance
_chart_build_query_registry = ChartBuildQueryRegistry()


def get_chart_build_query_registry() -> ChartBuildQueryRegistry:
    """Get the global chart buildQuery registry."""
    return _chart_build_query_registry


def register_chart_build_query(chart_type: str, build_query_func: BuildQueryFunc):
    """Register a custom buildQuery function for a chart type."""
    registry = get_chart_build_query_registry()
    registry.register(chart_type, build_query_func)


# =============================================================================
# Main Function
# =============================================================================


def build_query_context(
    form_data: dict[str, Any],
    build_query: BuildQueryFunc | None = None,
    query_field_aliases: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Build QueryContext from form_data for ANY chart type.

    This is a Python implementation of the frontend's buildQueryContext.ts,
    designed to generate QueryContext for the /api/v1/chart/data endpoint.

    IMPORTANT: This function supports ALL chart types, including legacy charts
    (bubble, chord, world_map, etc.) that have non-standard field mappings.
    The chart-type specific build_query functions handle the field mapping
    differences (e.g., "entity" → "columns", "x/y/size" → "metrics").

    Data Flow:
        form_data → build_query_context() → QueryContext → /api/v1/chart/data → SQL

    Args:
        form_data: Chart form data containing all configuration.
                  Supports both modern (echarts) and legacy (nvd3) field formats.
        build_query: Optional custom function to transform base query.
                    If not provided, uses chart-type specific function from registry.
        query_field_aliases: Optional field name mappings for custom charts.

    Returns:
        QueryContext ready to be sent to /api/v1/chart/data endpoint.

    Supported Chart Types:
        - Modern ECharts: echarts_timeseries, pie, table, big_number, etc.
        - Legacy NVD3: bubble, bullet, compare, time_pivot
        - Legacy Maps: world_map, country_map, mapbox
        - Legacy Others: chord, partition, rose, horizon, cal_heatmap, etc.
        - Deck.gl: deck_geojson, deck_multi

    Example (Modern Chart):
        >>> my_form_data = {
        ...     "datasource": "1__table",
        ...     "viz_type": "echarts_timeseries",
        ...     "metrics": ["count"],
        ...     "groupby": ["category"],
        ...     "x_axis": "date",
        ...     "time_range": "Last 7 days",
        ... }
        >>> result = build_query_context(my_form_data)

    Example (Legacy Chart):
        >>> legacy_form_data = {
        ...     "datasource": "1__table",
        ...     "viz_type": "bubble",
        ...     "entity": "country",      # Legacy field → columns
        ...     "x": "avg_price",         # Legacy field → metrics
        ...     "y": "total_sales",       # Legacy field → metrics
        ...     "size": "count",          # Legacy field → metrics
        ... }
        >>> result = build_query_context(legacy_form_data)
    """
    # Step 1: Build base query object
    base_query_object = build_query_object(form_data, query_field_aliases)

    # Step 2: Determine buildQuery function
    if build_query is None:
        viz_type = form_data.get("viz_type", "")
        registry = get_chart_build_query_registry()
        chart_build_query = registry.get(viz_type)
        queries = chart_build_query(base_query_object, form_data)
    else:
        queries = build_query(base_query_object, form_data)

    # Step 3: Filter out None post_processing items
    for query in queries:
        if query.post_processing:
            query.post_processing = [p for p in query.post_processing if p]

    # Step 4: Normalize x_axis column to BASE_AXIS format (if x_axis or granularity_sqla is set)
    # This transforms the x_axis column in columns list to include columnType and timeGrain
    # Based on migrate_viz/query_functions.py build_query_context logic
    if get_x_axis_column(form_data):
        queries = [normalize_time_column(form_data, query) for query in queries]

    # Step 5: Build and return QueryContext
    datasource_str = form_data.get("datasource", "")
    try:
        datasource = DatasourceKey.from_string(datasource_str).to_dict()
    except (ValueError, IndexError):
        datasource = {"id": 0, "type": "table"}

    return QueryContext(
        datasource=datasource,
        force=form_data.get("force", False),
        queries=queries,
        form_data=form_data,
        result_format=form_data.get("result_format", "json"),
        result_type=form_data.get("result_type", "query"),
    ).to_dict()
