"""
Superset Query Context Builder

This module builds query_context from form_data, simulating the frontend's
buildQueryContext logic. It handles various chart types including:
- ECharts timeseries (area, bar, line, etc.)
- Big Number (with trendline and total)
- Table, Pivot Table
- Hierarchical (Sunburst, Treemap, Sankey)
- Bubble, Funnel, Gauge, Radar
- Box Plot, Waterfall, Word Cloud
- Graph, Tree, Heatmap

Based on analysis of:
1. Superset frontend source code (buildQuery.ts files)
2. Sample query_context data from actual Superset instances
3. @superset-ui/core buildQueryContext function behavior
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

# ============================================================================
# Chart Type Classifications
# ============================================================================

# Legacy chart types that ONLY support explore_json API
LEGACY_VIZ_TYPES = {
    # True legacy plugins (not migrated)
    "horizon",
    "calendar",
    "chord",
    "country_map",
    "event_flow",
    "force_directed",
    "heatmap",
    "histogram",
    "map_box",
    "paired_t_test",
    "parallel_coordinates",
    "partition",
    "pivot_table",
    "rose",
    "sankey",
    "sunburst",
    "treemap",
    "word_cloud",
    "world_map",
    "markup",
    "iframe",
    "filter_box",
    # Deck.GL charts
    "deck_arc",
    "deck_geojson",
    "deck_grid",
    "deck_hex",
    "deck_multi",
    "deck_path",
    "deck_polygon",
    "deck_scatter",
    "deck_screengrid",
    # Legacy NVD3
    "bubble",
    "bullet",
    "compare",
    "dual_line",
    "line_multi",
    "time_pivot",
}


def is_legacy_chart(viz_type: str) -> bool:
    """Check if chart type uses legacy API (explore_json)"""
    return viz_type in LEGACY_VIZ_TYPES


class QueryContextBuilder:
    """
    Build query_context from form_data.
    Simulates the frontend buildQueryContext logic.
    """

    # Chart type configurations
    CHART_CONFIGS = {
        # Charts that use x_axis as BASE_AXIS column
        "x_axis_base_axis_charts": {
            "heatmap_v2",
            "waterfall",
            "echarts_timeseries",
            "echarts_timeseries_bar",
            "echarts_timeseries_line",
            "echarts_timeseries_scatter",
            "echarts_timeseries_smooth",
            "echarts_timeseries_step",
            "echarts_area",
            "mixed_timeseries",
        },
        # Timeseries charts - use time_range/granularity in query, pivot post_processing
        "timeseries_charts": {
            "echarts_timeseries",
            "echarts_timeseries_bar",
            "echarts_timeseries_line",
            "echarts_timeseries_scatter",
            "echarts_timeseries_smooth",
            "echarts_timeseries_step",
            "echarts_area",
            "mixed_timeseries",
            "big_number",  # big_number with trendline is timeseries!
        },
        # Charts that always include time_range/granularity (even if not timeseries)
        "time_aware_charts": {
            "table",
            "sunburst_v2",
            "treemap_v2",
            "box_plot",
        },
        # Legacy charts that need time_range/granularity but use explore_json API
        "legacy_time_aware_charts": {
            "bubble",
            "world_map",
            "filter_box",
            "dist_bar",
            "line",
            "area",
            "bar",
            "pivot_table",
            "sunburst",
            "treemap",
            "country_map",
            "histogram",
        },
        # Charts that need pivot + rename + flatten post_processing
        "pivot_post_processing_charts": {
            "echarts_timeseries",
            "echarts_timeseries_bar",
            "echarts_timeseries_line",
            "echarts_timeseries_scatter",
            "echarts_timeseries_smooth",
            "echarts_timeseries_step",
            "echarts_area",
            "mixed_timeseries",
        },
        # Charts that need pivot + flatten only (big_number style)
        "simple_pivot_charts": {
            "big_number",
        },
        # Charts that need boxplot post_processing
        "boxplot_charts": {
            "box_plot",
        },
        # Charts that need rank post_processing
        "rank_charts": {
            "heatmap_v2",
        },
        # Metric-only charts (no groupby, typically no orderby)
        # NOTE: gauge_chart still uses groupby for breakdown!
        "metric_only_charts": {
            "big_number_total",
        },
        # Charts that use 'columns' field for hierarchy
        "hierarchy_charts": {
            "sunburst_v2",
            "treemap_v2",
        },
        # Charts that use special field mappings
        "bubble_charts": {"bubble_v2"},
        "graph_charts": {"graph_chart"},
        "tree_charts": {"tree_chart"},
        "word_cloud_charts": {"word_cloud"},
        "sankey_charts": {"sankey_v2"},
        "pivot_table_charts": {"pivot_table_v2"},
        # Legacy bubble chart (different from bubble_v2)
        "legacy_bubble_charts": {"bubble"},
        # World map chart
        "world_map_charts": {"world_map"},
        # Charts that have series_columns
        "series_columns_charts": {
            "echarts_area",
            "echarts_timeseries",
            "echarts_timeseries_bar",
            "echarts_timeseries_line",
            "box_plot",
        },
    }

    def __init__(self, form_data: Dict[str, Any]):
        self.form_data = deepcopy(form_data)
        self.viz_type = form_data.get("viz_type", "")

    def build(self) -> Dict[str, Any]:
        """Build complete query_context"""
        datasource = self._parse_datasource()

        query_context = {
            "datasource": datasource,
            "force": self.form_data.get("force", False),
            "queries": [self._build_query_object()],
            "form_data": self._enrich_form_data(),
        }

        return query_context

    def _parse_datasource(self) -> Dict[str, Any]:
        """Parse datasource field"""
        ds = self.form_data.get("datasource", "")
        if isinstance(ds, str) and "__" in ds:
            ds_id, ds_type = ds.split("__")
            return {"id": int(ds_id), "type": ds_type}
        return {"id": ds, "type": "table"}

    def _is_timeseries(self) -> bool:
        """Check if this is a timeseries chart"""
        return self.viz_type in self.CHART_CONFIGS["timeseries_charts"]

    def _uses_x_axis_base_axis(self) -> bool:
        """Check if chart uses x_axis as BASE_AXIS"""
        return self.viz_type in self.CHART_CONFIGS["x_axis_base_axis_charts"]

    def _build_query_object(self) -> Dict[str, Any]:
        """Build query object - core logic"""
        is_timeseries = self._is_timeseries()
        is_time_aware = self.viz_type in self.CHART_CONFIGS["time_aware_charts"]
        is_legacy_time_aware = self.viz_type in self.CHART_CONFIGS.get("legacy_time_aware_charts", set())

        # Build base query object
        query_obj = {
            "filters": self._build_filters(),
            "extras": self._build_extras(),
            "applied_time_extras": self.form_data.get("applied_time_extras", {}),
            "columns": self._build_columns(),
            "metrics": self._build_metrics(),
            "annotation_layers": self.form_data.get("annotation_layers", []),
            "row_limit": self._get_row_limit(),
            "series_limit": self._get_series_limit(),
            "order_desc": self.form_data.get("order_desc", True),
            "url_params": self.form_data.get("url_params", {}),
            "custom_params": {},
            "custom_form_data": {},
        }

        # Add time_range and granularity for timeseries charts
        if is_timeseries:
            time_range = self.form_data.get("time_range")
            granularity = self.form_data.get("granularity_sqla") or self.form_data.get("granularity")

            if time_range:
                query_obj["time_range"] = time_range
            if granularity:
                query_obj["granularity"] = granularity
            query_obj["is_timeseries"] = True

        # Time-aware charts also include time_range/granularity but are not timeseries
        elif is_time_aware or is_legacy_time_aware:
            time_range = self.form_data.get("time_range")
            granularity = self.form_data.get("granularity_sqla") or self.form_data.get("granularity")

            if time_range:
                query_obj["time_range"] = time_range
            if granularity:
                query_obj["granularity"] = granularity

        # Add series_columns for applicable charts
        if self.viz_type in self.CHART_CONFIGS["series_columns_charts"]:
            groupby = self.form_data.get("groupby", [])
            if groupby:
                if isinstance(groupby, str):
                    query_obj["series_columns"] = [groupby]
                else:
                    query_obj["series_columns"] = list(groupby)

        # Special handling for sankey_v2: uses groupby instead of columns
        if self.viz_type in self.CHART_CONFIGS["sankey_charts"]:
            source = self.form_data.get("source")
            target = self.form_data.get("target")
            if source and target:
                query_obj["groupby"] = [source, target]
                query_obj["columns"] = []

        # Add orderby
        orderby = self._build_orderby()
        if orderby:
            query_obj["orderby"] = orderby

        # Add post_processing
        post_processing = self._build_post_processing()
        if post_processing:
            query_obj["post_processing"] = post_processing
        elif self.viz_type == "table":
            # Table chart needs empty post_processing and time_offsets
            query_obj["post_processing"] = []
            query_obj["time_offsets"] = []

        # Add time_offsets for timeseries
        if is_timeseries or self.viz_type in self.CHART_CONFIGS["pivot_post_processing_charts"]:
            query_obj["time_offsets"] = self.form_data.get("time_offsets", [])

        return query_obj

    def _get_row_limit(self) -> Optional[int]:
        """Get row limit"""
        return self.form_data.get("row_limit")

    def _get_series_limit(self) -> int:
        """Get series limit"""
        # Check 'limit' field (used by many charts)
        limit = self.form_data.get("limit")
        if limit:
            try:
                return int(limit)
            except (ValueError, TypeError):
                pass

        return self.form_data.get("series_limit", 0)

    def _build_columns(self) -> List[Any]:
        """Build columns list based on chart type"""
        columns = []

        # Special handling for different chart types

        # 1. Metric-only charts have empty columns
        if self.viz_type in self.CHART_CONFIGS["metric_only_charts"]:
            return []

        # 2. big_number with trendline - empty columns (uses is_timeseries)
        if self.viz_type == "big_number":
            return []

        # 3. Legacy bubble chart: columns = [series]
        if self.viz_type in self.CHART_CONFIGS.get("legacy_bubble_charts", set()):
            series = self.form_data.get("series")
            if series:
                return [series]
            return []

        # 4. world_map: empty columns
        if self.viz_type in self.CHART_CONFIGS.get("world_map_charts", set()):
            return []

        # 5. pivot_table_v2 - uses groupbyColumns + groupbyRows
        if self.viz_type in self.CHART_CONFIGS["pivot_table_charts"]:
            groupby_columns = self.form_data.get("groupbyColumns", [])
            groupby_rows = self.form_data.get("groupbyRows", [])
            columns.extend(groupby_columns)
            columns.extend(groupby_rows)
            return columns

        # 6. box_plot - uses granularity column + groupby
        if self.viz_type == "box_plot":
            granularity = self.form_data.get("granularity_sqla") or self.form_data.get("granularity")
            if granularity:
                columns.append(granularity)
            groupby = self.form_data.get("groupby", [])
            if isinstance(groupby, str):
                columns.append(groupby)
            elif isinstance(groupby, list):
                columns.extend(groupby)
            return columns

        # 7. Timeseries charts with x_axis - use BASE_AXIS format
        if self._uses_x_axis_base_axis():
            x_axis = self.form_data.get("x_axis")
            if x_axis:
                columns.append(self._make_base_axis_column(x_axis))

        # 8. Hierarchy charts (sunburst, treemap) - use 'columns' field directly
        if self.viz_type in self.CHART_CONFIGS["hierarchy_charts"]:
            form_columns = self.form_data.get("columns", [])
            if form_columns:
                columns.extend(form_columns)
            # Also add groupby if present
            groupby = self.form_data.get("groupby", [])
            if isinstance(groupby, str) and groupby and groupby not in columns:
                columns.append(groupby)
            elif isinstance(groupby, list):
                for g in groupby:
                    if g not in columns:
                        columns.append(g)
            return columns

        # 9. Sankey chart - handled separately (uses groupby field in query)
        if self.viz_type in self.CHART_CONFIGS["sankey_charts"]:
            return []  # sankey uses groupby, not columns

        # 10. Bubble chart (v2) - entity + series
        if self.viz_type in self.CHART_CONFIGS["bubble_charts"]:
            entity = self.form_data.get("entity")
            series = self.form_data.get("series")
            if entity:
                columns.append(entity)
            if series:
                columns.append(series)
            return columns

        # 11. Graph chart - source + target columns
        if self.viz_type in self.CHART_CONFIGS["graph_charts"]:
            source = self.form_data.get("source")
            target = self.form_data.get("target")
            if source:
                columns.append(source)
            if target:
                columns.append(target)
            return columns

        # 12. Tree chart - id, parent, name columns
        if self.viz_type in self.CHART_CONFIGS["tree_charts"]:
            id_col = self.form_data.get("id")
            parent_col = self.form_data.get("parent")
            name_col = self.form_data.get("name")
            for col in [id_col, parent_col, name_col]:
                if col:
                    columns.append(col)
            return columns

        # 13. Word cloud - series column
        if self.viz_type in self.CHART_CONFIGS["word_cloud_charts"]:
            series = self.form_data.get("series")
            if series:
                columns.append(series)
            return columns

        # 14. Default handling - groupby, columns, all_columns

        # Add groupby
        groupby = self.form_data.get("groupby", [])
        if isinstance(groupby, str):
            if groupby and groupby not in columns:
                columns.append(groupby)
        elif isinstance(groupby, list):
            for g in groupby:
                if g and g not in columns:
                    columns.append(g)

        # Add columns field (if not already processed)
        form_columns = self.form_data.get("columns", [])
        if form_columns and self.viz_type not in self.CHART_CONFIGS["hierarchy_charts"]:
            if isinstance(form_columns, list):
                for col in form_columns:
                    if col not in columns:
                        columns.append(col)
            elif form_columns not in columns:
                columns.append(form_columns)

        # Add all_columns (Table chart)
        all_columns = self.form_data.get("all_columns", [])
        if all_columns:
            for col in all_columns:
                if col not in columns:
                    columns.append(col)

        return columns

    def _make_base_axis_column(self, column: Union[str, Dict]) -> Dict[str, Any]:
        """Convert column to BASE_AXIS format for timeseries/heatmap charts"""
        time_grain = self.form_data.get("time_grain_sqla")

        if isinstance(column, dict):
            result = {
                "columnType": "BASE_AXIS",
                "sqlExpression": column.get("sqlExpression") or column.get("column_name"),
                "label": column.get("label") or column.get("column_name"),
                "expressionType": column.get("expressionType", "SQL"),
            }
            if time_grain:
                result["timeGrain"] = time_grain
            return result
        else:
            result = {
                "columnType": "BASE_AXIS",
                "sqlExpression": column,
                "label": column,
                "expressionType": "SQL",
            }
            if time_grain:
                result["timeGrain"] = time_grain
            return result

    def _build_metrics(self) -> List[Any]:
        """Build metrics list"""
        metrics = []

        # Legacy bubble chart: metrics = [x, y, size] in that order
        if self.viz_type in self.CHART_CONFIGS.get("legacy_bubble_charts", set()):
            for field in ["x", "y", "size"]:
                m = self.form_data.get(field)
                if m and m not in metrics:
                    metrics.append(m)
            return metrics

        # Handle metrics (list)
        form_metrics = self.form_data.get("metrics")
        if form_metrics:
            if isinstance(form_metrics, list):
                metrics.extend(form_metrics)
            else:
                metrics.append(form_metrics)

        # Handle metric (single) - common in many charts
        metric = self.form_data.get("metric")
        if metric and metric not in metrics:
            metrics.append(metric)

        # Handle secondary_metric (sunburst_v2, world_map)
        secondary = self.form_data.get("secondary_metric")
        if secondary and secondary not in metrics:
            metrics.append(secondary)

        # Handle x, y, size for bubble_v2 charts
        if self.viz_type in self.CHART_CONFIGS.get("bubble_charts", set()):
            x_metric = self.form_data.get("x")
            y_metric = self.form_data.get("y")
            size_metric = self.form_data.get("size")

            if x_metric and x_metric not in metrics:
                metrics.append(x_metric)
            if y_metric and y_metric not in metrics:
                metrics.append(y_metric)
            if size_metric and size_metric not in metrics:
                metrics.append(size_metric)

        # Handle percent_metrics (Table)
        percent_metrics = self.form_data.get("percent_metrics", [])
        if percent_metrics:
            for m in percent_metrics:
                if m not in metrics:
                    metrics.append(m)

        return metrics

    def _build_orderby(self) -> List[List[Any]]:
        """Build sorting rules"""
        orderby = []

        # Metric-only charts typically don't need orderby
        if self.viz_type in self.CHART_CONFIGS["metric_only_charts"]:
            return []

        # big_number doesn't need orderby
        if self.viz_type == "big_number":
            return []

        # box_plot doesn't seem to have orderby
        if self.viz_type == "box_plot":
            return []

        # Hierarchy charts (sunburst, treemap) typically don't have orderby
        if self.viz_type in {"sunburst_v2", "treemap_v2"}:
            return []

        # Sankey doesn't have orderby
        if self.viz_type in self.CHART_CONFIGS["sankey_charts"]:
            return []

        # Legacy bubble and world_map don't have orderby
        if self.viz_type in self.CHART_CONFIGS.get("legacy_bubble_charts", set()):
            return []
        if self.viz_type in self.CHART_CONFIGS.get("world_map_charts", set()):
            return []

        # Heatmap: order by x_axis and groupby
        if self.viz_type == "heatmap_v2":
            x_axis = self.form_data.get("x_axis")
            groupby = self.form_data.get("groupby")
            if x_axis:
                orderby.append([x_axis, True])  # ascending
            if groupby:
                if isinstance(groupby, str):
                    orderby.append([groupby, True])
                elif isinstance(groupby, list) and groupby:
                    orderby.append([groupby[0], True])
            return orderby

        # Waterfall: order by x_axis
        if self.viz_type == "waterfall":
            x_axis = self.form_data.get("x_axis")
            if x_axis:
                orderby.append([x_axis, True])
            return orderby

        # Timeseries charts with groupby: order by first metric descending
        if self.viz_type in self.CHART_CONFIGS["pivot_post_processing_charts"]:
            metrics = self._build_metrics()
            if metrics:
                order_desc = self.form_data.get("order_desc", True)
                orderby.append([metrics[0], not order_desc])
            return orderby

        # pivot_table_v2: order by first metric
        if self.viz_type in self.CHART_CONFIGS["pivot_table_charts"]:
            metrics = self._build_metrics()
            if metrics:
                order_desc = self.form_data.get("order_desc", True)
                orderby.append([metrics[0], not order_desc])
            return orderby

        # Table, funnel, radar: order by metric
        if self.viz_type in {"table", "funnel", "radar"}:
            sort_by_metric = self.form_data.get("sort_by_metric", False)
            metrics = self._build_metrics()

            # Check for timeseries_limit_metric
            ts_metric = self.form_data.get("timeseries_limit_metric")
            if ts_metric:
                order_desc = self.form_data.get("order_desc", True)
                orderby.append([ts_metric, not order_desc])
            elif metrics:
                # For table: always order by first metric
                # For funnel/radar: order by metric (usually enabled)
                if self.viz_type == "table" or sort_by_metric or self.viz_type in {"funnel", "radar"}:
                    order_desc = self.form_data.get("order_desc", True)
                    orderby.append([metrics[0], not order_desc])

            return orderby

        # Default: use first metric if available and sort_by_metric is True
        sort_by_metric = self.form_data.get("sort_by_metric", False)
        if sort_by_metric:
            metrics = self._build_metrics()
            if metrics:
                order_desc = self.form_data.get("order_desc", True)
                orderby.append([metrics[0], not order_desc])

        return orderby

    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filters from adhoc_filters"""
        filters = []

        # Handle adhoc_filters
        for f in self.form_data.get("adhoc_filters", []):
            expr_type = f.get("expressionType")

            if expr_type == "SIMPLE":
                operator = f.get("operator")
                comparator = f.get("comparator")
                subject = f.get("subject")

                if operator == "TEMPORAL_RANGE":
                    filters.append({"col": subject, "op": "TEMPORAL_RANGE", "val": comparator})
                else:
                    # Handle single-item lists
                    if isinstance(comparator, list) and len(comparator) == 1:
                        comparator = comparator[0]

                    filters.append(
                        {
                            "col": subject,
                            "op": operator,
                            "val": comparator,
                        }
                    )

        # Handle legacy filters field
        old_filters = self.form_data.get("filters")
        if old_filters:
            filters.extend(old_filters)

        # Handle extra_filters (from dashboard)
        extra_filters = self.form_data.get("extra_filters", [])
        for ef in extra_filters:
            if ef.get("col") and ef.get("val") is not None:
                filters.append(
                    {
                        "col": ef["col"],
                        "op": ef.get("op", "=="),
                        "val": ef["val"],
                    }
                )

        return filters

    def _build_extras(self) -> Dict[str, Any]:
        """Build extras including SQL expressions from adhoc_filters"""
        extras = {
            "having": "",
            "where": "",
        }

        # Add time_grain_sqla if present
        time_grain = self.form_data.get("time_grain_sqla")
        if time_grain:
            extras["time_grain_sqla"] = time_grain

        # Process SQL expressions from adhoc_filters
        for f in self.form_data.get("adhoc_filters", []):
            if f.get("expressionType") == "SQL":
                clause = f.get("clause", "WHERE")
                sql_expr = f.get("sqlExpression", "")

                if clause == "WHERE":
                    if extras["where"]:
                        extras["where"] += f" AND ({sql_expr})"
                    else:
                        extras["where"] = sql_expr
                elif clause == "HAVING":
                    if extras["having"]:
                        extras["having"] += f" AND ({sql_expr})"
                    else:
                        extras["having"] = sql_expr

        # Handle explicit where/having
        if self.form_data.get("where"):
            if extras["where"]:
                extras["where"] += f" AND ({self.form_data['where']})"
            else:
                extras["where"] = self.form_data["where"]

        if self.form_data.get("having"):
            if extras["having"]:
                extras["having"] += f" AND ({self.form_data['having']})"
            else:
                extras["having"] = self.form_data["having"]

        return extras

    def _build_post_processing(self) -> List[Dict[str, Any]]:
        """Build post processing rules based on chart type"""
        post_processing = []

        # 1. big_number with trendline needs pivot + flatten
        if self.viz_type == "big_number":
            metrics = self._build_metrics()
            if metrics:
                aggregates = {}
                for m in metrics:
                    metric_label = self._get_metric_label(m)
                    aggregates[metric_label] = {"operator": "mean"}

                post_processing.append(
                    {
                        "operation": "pivot",
                        "options": {
                            "index": ["__timestamp"],
                            "columns": [],
                            "aggregates": aggregates,
                            "drop_missing_columns": True,
                        },
                    }
                )
                post_processing.append({"operation": "flatten"})
            return post_processing

        # 2. Timeseries charts need pivot + rename + flatten
        if self.viz_type in self.CHART_CONFIGS["pivot_post_processing_charts"]:
            metrics = self._build_metrics()
            groupby = self.form_data.get("groupby", [])
            x_axis = self.form_data.get("x_axis")

            # Determine index column
            if x_axis:
                index_col = x_axis
            else:
                index_col = "__timestamp"

            # Build aggregates
            aggregates = {}
            for m in metrics:
                metric_label = self._get_metric_label(m)
                aggregates[metric_label] = {"operator": "mean"}

            # Build columns (groupby)
            pivot_columns = []
            if isinstance(groupby, str) and groupby:
                pivot_columns = [groupby]
            elif isinstance(groupby, list):
                pivot_columns = list(groupby)

            post_processing.append(
                {
                    "operation": "pivot",
                    "options": {
                        "index": [index_col],
                        "columns": pivot_columns,
                        "aggregates": aggregates,
                        "drop_missing_columns": False,
                    },
                }
            )

            # Add rename operation
            rename_columns = {}
            for m in metrics:
                metric_label = self._get_metric_label(m)
                rename_columns[metric_label] = None

            post_processing.append(
                {"operation": "rename", "options": {"columns": rename_columns, "level": 0, "inplace": True}}
            )

            post_processing.append({"operation": "flatten"})

            return post_processing

        # 3. Box plot needs boxplot post_processing
        if self.viz_type == "box_plot":
            groupby = self.form_data.get("groupby", [])
            metrics = self._build_metrics()
            whisker_options = self.form_data.get("whiskerOptions", "Tukey")

            # Map whisker options to type
            whisker_type = "tukey"
            if "min" in whisker_options.lower() or "max" in whisker_options.lower():
                whisker_type = "min/max"
            elif "percentile" in whisker_options.lower():
                whisker_type = "percentile"

            post_processing.append(
                {
                    "operation": "boxplot",
                    "options": {
                        "whisker_type": whisker_type,
                        "groupby": groupby if isinstance(groupby, list) else [groupby] if groupby else [],
                        "metrics": [self._get_metric_label(m) for m in metrics],
                    },
                }
            )
            return post_processing

        # 4. Heatmap needs rank post_processing
        if self.viz_type == "heatmap_v2":
            metrics = self._build_metrics()
            if metrics:
                metric_label = self._get_metric_label(metrics[0])
                post_processing.append({"operation": "rank", "options": {"metric": metric_label}})
            return post_processing

        # 5. Metric-only charts don't need post processing
        if self.viz_type in self.CHART_CONFIGS["metric_only_charts"]:
            return []

        # 6. Rolling window if specified
        rolling_type = self.form_data.get("rolling_type")
        if rolling_type and rolling_type.lower() != "none":
            post_processing.append(
                {
                    "operation": "rolling",
                    "options": {
                        "rolling_type": rolling_type,
                        "window": self.form_data.get("rolling_periods", 1),
                        "min_periods": self.form_data.get("min_periods", 0),
                    },
                }
            )

        # 7. Contribution if specified
        if self.form_data.get("contribution"):
            post_processing.append({"operation": "contribution", "options": {}})

        return post_processing

    def _get_metric_label(self, metric: Any) -> str:
        """Extract label from metric object"""
        if isinstance(metric, str):
            return metric
        elif isinstance(metric, dict):
            # Check for label first
            if metric.get("label"):
                return metric["label"]
            # Try to build label from aggregate and column
            aggregate = metric.get("aggregate")
            column = metric.get("column", {})
            column_name = column.get("column_name", "") if isinstance(column, dict) else ""
            if aggregate and column_name:
                return f"{aggregate}({column_name})"
            return metric.get("column", {}).get("column_name", "") if isinstance(metric.get("column"), dict) else ""
        return ""

    def _enrich_form_data(self) -> Dict[str, Any]:
        """Enrich form_data with runtime required fields"""
        enriched = {**self.form_data}

        if "chart_id" not in enriched and "slice_id" in enriched:
            enriched["chart_id"] = enriched["slice_id"]

        if "url_params" not in enriched:
            enriched["url_params"] = {}

        enriched["result_format"] = enriched.get("result_format", "json")
        enriched["result_type"] = enriched.get("result_type", "full")

        if "extra_form_data" not in enriched:
            enriched["extra_form_data"] = {}

        return enriched
