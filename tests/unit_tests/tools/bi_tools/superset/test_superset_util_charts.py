# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for chart builder functions in superset_util.py (lines 1319+).

All tests are CI-level: zero external dependencies, fully deterministic.
Covers: build_boxplot_query, build_table_query, build_big_number_query,
build_big_number_trendline_query, build_pie_query, build_funnel_query,
build_gauge_query, build_heatmap_query, build_histogram_query,
build_bubble_query, build_waterfall_query, build_sankey_query,
build_sunburst_query, build_treemap_query, build_word_cloud_query,
build_graph_query, build_tree_query, build_radar_query,
build_pivot_table_query, build_mixed_timeseries_query,
build_gantt_query, build_pop_kpi_query, build_default_query,
build_timeseries_query (via build_query_context),
legacy chart builders, deck.gl builders,
ChartBuildQueryRegistry, build_query_context.
"""


from datus.tools.bi_tools.superset.superset_util import (
    ChartBuildQueryRegistry,
    QueryObject,
    build_big_number_query,
    build_big_number_trendline_query,
    build_boxplot_query,
    build_bubble_query,
    build_deck_arc_query,
    build_deck_grid_query,
    build_deck_path_query,
    build_deck_polygon_query,
    build_deck_scatter_query,
    build_default_query,
    build_funnel_query,
    build_gantt_query,
    build_gauge_query,
    build_graph_query,
    build_heatmap_query,
    build_histogram_query,
    build_legacy_bubble_query,
    build_legacy_bullet_query,
    build_legacy_cal_heatmap_query,
    build_legacy_chord_query,
    build_legacy_country_map_query,
    build_legacy_deck_geojson_query,
    build_legacy_deck_multi_query,
    build_legacy_mapbox_query,
    build_legacy_paired_ttest_query,
    build_legacy_parallel_coordinates_query,
    build_legacy_partition_query,
    build_legacy_time_pivot_query,
    build_legacy_time_table_query,
    build_legacy_timeseries_query,
    build_legacy_world_map_query,
    build_mixed_timeseries_query,
    build_pie_query,
    build_pivot_table_query,
    build_pop_kpi_query,
    build_query_context,
    build_radar_query,
    build_sankey_query,
    build_sunburst_query,
    build_table_query,
    build_timeseries_query,
    build_tree_query,
    build_treemap_query,
    build_waterfall_query,
    build_word_cloud_query,
    get_chart_build_query_registry,
    register_chart_build_query,
)

# =============================================================================
# Helpers
# =============================================================================


def _base_query(**kwargs) -> QueryObject:
    """Create a QueryObject with sensible defaults."""
    defaults = dict(columns=[], metrics=[], orderby=[], filters=[], extras={})
    defaults.update(kwargs)
    return QueryObject(**defaults)


# =============================================================================
# Tests: build_default_query
# =============================================================================


class TestBuildDefaultQuery:
    def test_wraps_in_list(self):
        q = _base_query(metrics=["count"])
        result = build_default_query(q, {})
        assert result == [q]

    def test_returns_same_object(self):
        q = _base_query()
        assert build_default_query(q, {"x": 1})[0] is q


# =============================================================================
# Tests: build_boxplot_query
# =============================================================================


class TestBuildBoxplotQuery:
    def test_basic_no_whisker(self):
        q = _base_query(metrics=["count"])
        result = build_boxplot_query(q, {"groupby": ["region"]})
        assert len(result) == 1
        assert result[0].columns  # columns set from groupby

    def test_with_whisker_adds_post_processing(self):
        q = _base_query(metrics=["count"])
        form = {"groupby": ["region"], "whiskerOptions": "Tukey"}
        result = build_boxplot_query(q, form)
        assert result[0].post_processing
        assert result[0].post_processing[0]["operation"] == "boxplot"

    def test_with_form_columns(self):
        q = _base_query(metrics=["count"])
        form = {"columns": ["date"], "groupby": ["region"]}
        result = build_boxplot_query(q, form)
        # columns should include form_columns + groupby
        cols = result[0].columns
        assert "region" in cols

    def test_with_time_grain_temporal_column(self):
        q = _base_query(metrics=["count"])
        form = {
            "columns": ["date"],
            "groupby": ["region"],
            "time_grain_sqla": "P1D",
            "temporal_columns_lookup": {"date": True},
        }
        result = build_boxplot_query(q, form)
        # date should be transformed to a dict with timeGrain
        cols = result[0].columns
        date_col = next((c for c in cols if isinstance(c, dict) and c.get("sqlExpression") == "date"), None)
        assert date_col is not None
        assert date_col["timeGrain"] == "P1D"

    def test_empty_groupby(self):
        q = _base_query(metrics=["count"])
        result = build_boxplot_query(q, {})
        assert result[0].series_columns is None


# =============================================================================
# Tests: build_table_query
# =============================================================================


class TestBuildTableQuery:
    def test_basic_aggregate_mode(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[])
        result = build_table_query(q, {"query_mode": "aggregate"})
        assert len(result) == 1
        assert result[0].metrics == ["count"]

    def test_all_columns_forces_raw_mode(self):
        q = _base_query(columns=[], metrics=["count"], orderby=[])
        result = build_table_query(q, {"all_columns": ["id", "name"]})
        assert len(result) == 1

    def test_sort_by_timeseries_limit_metric(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[])
        form = {"query_mode": "aggregate", "timeseries_limit_metric": "revenue", "order_desc": True}
        result = build_table_query(q, form)
        assert result[0].orderby == [("revenue", False)]

    def test_sort_falls_back_to_first_metric(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[])
        result = build_table_query(q, {"query_mode": "aggregate"})
        assert result[0].orderby == [("count", False)]

    def test_percent_metrics_added_to_post_processing(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[])
        form = {"query_mode": "aggregate", "percent_metrics": ["revenue"]}
        result = build_table_query(q, form)
        pp = result[0].post_processing
        assert any(p["operation"] == "contribution" for p in pp)

    def test_percent_metric_added_to_metrics(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[])
        form = {"query_mode": "aggregate", "percent_metrics": ["revenue"]}
        result = build_table_query(q, form)
        assert "revenue" in result[0].metrics

    def test_time_compare_adds_time_offsets(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[], time_range="last week")
        form = {"query_mode": "aggregate", "time_compare": ["1 year ago"]}
        result = build_table_query(q, form)
        assert result[0].time_offsets == ["1 year ago"]

    def test_custom_time_compare_with_start_offset(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[], time_range="last week")
        form = {
            "query_mode": "aggregate",
            "time_compare": ["custom"],
            "start_date_offset": "2023-01-01",
        }
        result = build_table_query(q, form)
        assert "2023-01-01" in result[0].time_offsets

    def test_inherit_time_compare(self):
        q = _base_query(columns=["region"], metrics=["count"], orderby=[], time_range="last week")
        form = {"query_mode": "aggregate", "time_compare": ["inherit"]}
        result = build_table_query(q, form)
        assert "inherit" in result[0].time_offsets

    def test_temporal_column_transformation(self):
        q = _base_query(columns=["date", "region"], metrics=["count"], orderby=[])
        form = {
            "query_mode": "aggregate",
            "time_grain_sqla": "P1D",
            "temporal_columns_lookup": {"date": True},
        }
        result = build_table_query(q, form)
        cols = result[0].columns
        date_col = next((c for c in cols if isinstance(c, dict) and c.get("sqlExpression") == "date"), None)
        assert date_col is not None
        assert date_col["columnType"] == "BASE_AXIS"


# =============================================================================
# Tests: build_big_number_query
# =============================================================================


class TestBuildBigNumberQuery:
    def test_returns_base_query_unchanged(self):
        q = _base_query(metrics=["count"])
        result = build_big_number_query(q, {})
        assert result == [q]
        assert result[0] is q


# =============================================================================
# Tests: build_big_number_trendline_query
# =============================================================================


class TestBuildBigNumberTrendlineQuery:
    def test_with_metrics_adds_pivot_and_flatten(self):
        q = _base_query(metrics=["count"])
        result = build_big_number_trendline_query(q, {})
        assert result[0].is_timeseries is True
        pp = result[0].post_processing
        ops = [p["operation"] for p in pp]
        assert "pivot" in ops
        assert "flatten" in ops

    def test_pivot_aggregates_metric_label(self):
        q = _base_query(metrics=["count"])
        result = build_big_number_trendline_query(q, {})
        pivot = next(p for p in result[0].post_processing if p["operation"] == "pivot")
        assert "count" in pivot["options"]["aggregates"]

    def test_no_metrics_skips_post_processing(self):
        q = _base_query(metrics=[])
        result = build_big_number_trendline_query(q, {})
        assert result[0].post_processing == []

    def test_dict_metric_label(self):
        q = _base_query(metrics=[{"label": "Revenue", "aggregate": "SUM", "column": {"column_name": "rev"}}])
        result = build_big_number_trendline_query(q, {})
        pivot = next(p for p in result[0].post_processing if p["operation"] == "pivot")
        assert "Revenue" in pivot["options"]["aggregates"]


# =============================================================================
# Tests: build_pie_query
# =============================================================================


class TestBuildPieQuery:
    def test_adds_contribution_post_processing(self):
        q = _base_query(metrics=["count"])
        result = build_pie_query(q, {})
        pp = result[0].post_processing
        assert len(pp) == 1
        assert pp[0]["operation"] == "contribution"
        assert "% count" in pp[0]["options"]["rename_columns"]

    def test_sort_by_metric_sets_orderby(self):
        q = _base_query(metrics=["revenue"])
        form = {"sort_by_metric": True, "metric": "revenue"}
        result = build_pie_query(q, form)
        assert result[0].orderby == [("revenue", False)]

    def test_no_metric_no_orderby(self):
        q = _base_query(metrics=["revenue"])
        form = {"sort_by_metric": True}
        result = build_pie_query(q, form)
        assert result[0].orderby == []

    def test_no_metrics_no_post_processing(self):
        q = _base_query(metrics=[])
        result = build_pie_query(q, {})
        assert result[0].post_processing == []


# =============================================================================
# Tests: build_funnel_query
# =============================================================================


class TestBuildFunnelQuery:
    def test_sort_by_metric_sets_orderby(self):
        q = _base_query(metrics=["count"])
        result = build_funnel_query(q, {"sort_by_metric": True, "metric": "count"})
        assert result[0].orderby == [("count", False)]

    def test_no_sort_no_orderby(self):
        q = _base_query(metrics=["count"], orderby=[])
        result = build_funnel_query(q, {})
        assert result[0].orderby == []


# =============================================================================
# Tests: build_gauge_query
# =============================================================================


class TestBuildGaugeQuery:
    def test_sort_by_metric_sets_orderby(self):
        q = _base_query(metrics=["count"])
        result = build_gauge_query(q, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_no_metrics_no_orderby(self):
        q = _base_query(metrics=[])
        result = build_gauge_query(q, {"sort_by_metric": True})
        assert result[0].orderby == []


# =============================================================================
# Tests: build_heatmap_query
# =============================================================================


class TestBuildHeatmapQuery:
    def test_basic_columns_from_x_axis(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"]}
        result = build_heatmap_query(q, form)
        cols = result[0].columns
        assert "region" in cols

    def test_sort_x_by_value_uses_metric(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "sort_x_axis": "value_asc"}
        result = build_heatmap_query(q, form)
        orderby = result[0].orderby
        assert any(ob[0] == "count" for ob in orderby)

    def test_sort_x_by_alpha_uses_column(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "sort_x_axis": "alpha_desc"}
        result = build_heatmap_query(q, form)
        orderby = result[0].orderby
        assert any(ob[0] == "date" for ob in orderby)

    def test_sort_y_by_value_uses_metric(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "sort_y_axis": "value_desc"}
        result = build_heatmap_query(q, form)
        orderby = result[0].orderby
        assert any(ob[0] == "count" for ob in orderby)

    def test_sort_y_by_alpha_uses_groupby(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "sort_y_axis": "alpha_asc"}
        result = build_heatmap_query(q, form)
        orderby = result[0].orderby
        assert any(ob[0] == "region" for ob in orderby)

    def test_normalize_across_x_adds_rank_post_processing(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "normalize_across": "x"}
        result = build_heatmap_query(q, form)
        pp = result[0].post_processing
        assert any(p["operation"] == "rank" for p in pp)

    def test_normalize_across_y_adds_rank_post_processing(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "normalize_across": "y"}
        result = build_heatmap_query(q, form)
        pp = result[0].post_processing
        assert any(p["operation"] == "rank" for p in pp)

    def test_no_normalize_no_post_processing(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"]}
        result = build_heatmap_query(q, form)
        assert result[0].post_processing == []


# =============================================================================
# Tests: build_histogram_query
# =============================================================================


class TestBuildHistogramQuery:
    def test_column_in_columns(self):
        q = _base_query()
        result = build_histogram_query(q, {"column": "price", "groupby": []})
        assert "price" in result[0].columns

    def test_groupby_appended_to_columns(self):
        q = _base_query()
        result = build_histogram_query(q, {"column": "price", "groupby": ["region"]})
        assert "region" in result[0].columns

    def test_metrics_cleared(self):
        q = _base_query(metrics=["count"])
        result = build_histogram_query(q, {"column": "price"})
        assert result[0].metrics == []

    def test_histogram_op_added_to_post_processing(self):
        q = _base_query()
        result = build_histogram_query(q, {"column": "price", "bins": 15})
        pp = result[0].post_processing
        assert pp[0]["operation"] == "histogram"
        assert pp[0]["options"]["bins"] == 15

    def test_no_column_no_post_processing(self):
        q = _base_query()
        result = build_histogram_query(q, {})
        assert result[0].post_processing == []


# =============================================================================
# Tests: build_bubble_query
# =============================================================================


class TestBuildBubbleQuery:
    def test_entity_and_series_in_columns(self):
        q = _base_query()
        form = {"entity": "country", "series": "category"}
        result = build_bubble_query(q, form)
        assert "country" in result[0].columns
        assert "category" in result[0].columns

    def test_only_entity(self):
        q = _base_query()
        result = build_bubble_query(q, {"entity": "country"})
        assert result[0].columns == ["country"]

    def test_orderby_inverted(self):
        q = _base_query(orderby=[("revenue", True)])
        result = build_bubble_query(q, {})
        assert result[0].orderby == [("revenue", False)]

    def test_no_entity_empty_columns(self):
        q = _base_query()
        result = build_bubble_query(q, {})
        assert result[0].columns == []


# =============================================================================
# Tests: build_waterfall_query
# =============================================================================


class TestBuildWaterfallQuery:
    def test_x_axis_in_columns(self):
        q = _base_query()
        form = {"x_axis": "date", "groupby": []}
        result = build_waterfall_query(q, form)
        # x_axis gets BASE_AXIS transform or just passed through
        assert len(result[0].columns) >= 1

    def test_orderby_by_x_axis_and_groupby(self):
        q = _base_query()
        form = {"x_axis": "date", "groupby": ["region"]}
        result = build_waterfall_query(q, form)
        orderby = result[0].orderby
        assert ("date", True) in orderby
        assert ("region", True) in orderby

    def test_no_x_axis_empty_columns(self):
        q = _base_query()
        result = build_waterfall_query(q, {})
        assert result[0].columns == []
        assert result[0].orderby == []


# =============================================================================
# Tests: build_sankey_query
# =============================================================================


class TestBuildSankeyQuery:
    def test_source_and_target_in_columns(self):
        q = _base_query()
        result = build_sankey_query(q, {"source": "from", "target": "to"})
        assert "from" in result[0].columns
        assert "to" in result[0].columns

    def test_sort_by_metric(self):
        q = _base_query(metrics=["count"])
        result = build_sankey_query(q, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_no_source_target(self):
        q = _base_query()
        result = build_sankey_query(q, {})
        assert result[0].columns == []


# =============================================================================
# Tests: build_sunburst_query
# =============================================================================


class TestBuildSunburstQuery:
    def test_sort_by_metric(self):
        q = _base_query(metrics=["count"])
        result = build_sunburst_query(q, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_no_sort_no_orderby(self):
        q = _base_query(metrics=["count"], orderby=[])
        result = build_sunburst_query(q, {})
        assert result[0].orderby == []


# =============================================================================
# Tests: build_treemap_query
# =============================================================================


class TestBuildTreemapQuery:
    def test_sort_by_metric(self):
        q = _base_query(metrics=["count"])
        result = build_treemap_query(q, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_no_metrics_no_orderby(self):
        q = _base_query(metrics=[])
        result = build_treemap_query(q, {"sort_by_metric": True})
        assert result[0].orderby == []


# =============================================================================
# Tests: build_word_cloud_query
# =============================================================================


class TestBuildWordCloudQuery:
    def test_sort_by_metric(self):
        q = _base_query(metrics=["count"])
        result = build_word_cloud_query(q, {"sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_no_sort_unchanged(self):
        q = _base_query(metrics=["count"], orderby=[("other", True)])
        result = build_word_cloud_query(q, {})
        assert result[0].orderby == [("other", True)]


# =============================================================================
# Tests: build_graph_query
# =============================================================================


class TestBuildGraphQuery:
    def test_source_target_and_categories(self):
        q = _base_query()
        form = {
            "source": "src",
            "target": "tgt",
            "source_category": "src_cat",
            "target_category": "tgt_cat",
        }
        result = build_graph_query(q, form)
        cols = result[0].columns
        assert "src" in cols
        assert "tgt" in cols
        assert "src_cat" in cols
        assert "tgt_cat" in cols

    def test_partial_fields(self):
        q = _base_query()
        result = build_graph_query(q, {"source": "from"})
        assert result[0].columns == ["from"]

    def test_no_fields_empty_columns(self):
        q = _base_query()
        result = build_graph_query(q, {})
        assert result[0].columns == []


# =============================================================================
# Tests: build_tree_query
# =============================================================================


class TestBuildTreeQuery:
    def test_id_parent_name(self):
        q = _base_query()
        form = {"id": "node_id", "parent": "parent_id", "name": "label"}
        result = build_tree_query(q, form)
        assert result[0].columns == ["node_id", "parent_id", "label"]

    def test_partial_fields(self):
        q = _base_query()
        result = build_tree_query(q, {"id": "node_id"})
        assert result[0].columns == ["node_id"]

    def test_no_fields(self):
        q = _base_query()
        result = build_tree_query(q, {})
        assert result[0].columns == []


# =============================================================================
# Tests: build_radar_query
# =============================================================================


class TestBuildRadarQuery:
    def test_x_axis_in_columns(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "category", "groupby": ["region"]}
        result = build_radar_query(q, form)
        cols = result[0].columns
        assert "category" in cols
        assert "region" in cols

    def test_groupby_deduped(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "region", "groupby": ["region"]}
        result = build_radar_query(q, form)
        assert result[0].columns.count("region") == 1

    def test_rank_op_with_normalize(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "category", "groupby": [], "normalize_across": "x"}
        result = build_radar_query(q, form)
        pp = result[0].post_processing
        assert any(p["operation"] == "rank" for p in pp)

    def test_no_x_axis_falls_back_to_groupby(self):
        q = _base_query(metrics=["count"])
        form = {"groupby": ["region", "category"]}
        result = build_radar_query(q, form)
        cols = result[0].columns
        assert "region" in cols


# =============================================================================
# Tests: build_pivot_table_query
# =============================================================================


class TestBuildPivotTableQuery:
    def test_groupby_columns_and_rows(self):
        q = _base_query(metrics=["count"])
        form = {"groupbyColumns": ["date"], "groupbyRows": ["region"]}
        result = build_pivot_table_query(q, form)
        assert "date" in result[0].columns
        assert "region" in result[0].columns

    def test_temporal_wrapping(self):
        q = _base_query(metrics=["count"])
        form = {
            "groupbyColumns": ["date"],
            "groupbyRows": [],
            "time_grain_sqla": "P1M",
            "temporal_columns_lookup": {"date": True},
        }
        result = build_pivot_table_query(q, form)
        date_col = next(
            (c for c in result[0].columns if isinstance(c, dict) and c.get("sqlExpression") == "date"), None
        )
        assert date_col is not None
        assert date_col["timeGrain"] == "P1M"

    def test_series_limit_metric_orderby(self):
        q = _base_query(metrics=["count"])
        form = {"groupbyColumns": [], "groupbyRows": [], "series_limit_metric": "revenue"}
        result = build_pivot_table_query(q, form)
        assert result[0].orderby[0][0] == "revenue"

    def test_orderby_falls_back_to_first_metric(self):
        q = _base_query(metrics=["count"])
        form = {"groupbyColumns": [], "groupbyRows": [], "order_desc": True}
        result = build_pivot_table_query(q, form)
        assert result[0].orderby[0][0] == "count"

    def test_no_metrics_no_orderby(self):
        q = _base_query(metrics=[])
        form = {"groupbyColumns": [], "groupbyRows": []}
        result = build_pivot_table_query(q, form)
        assert result[0].orderby == []


# =============================================================================
# Tests: build_mixed_timeseries_query
# =============================================================================


class TestBuildMixedTimeseriesQuery:
    def test_returns_at_least_one_query(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "metrics": ["count"]}
        result = build_mixed_timeseries_query(q, form)
        assert len(result) >= 1

    def test_with_metrics_b_returns_two_queries(self):
        q = _base_query(metrics=["count"])
        form = {
            "x_axis": "date",
            "groupby": ["region"],
            "metrics": ["count"],
            "metrics_b": ["revenue"],
            "groupby_b": ["category"],
        }
        result = build_mixed_timeseries_query(q, form)
        assert len(result) == 2

    def test_query_b_has_metrics_b(self):
        q = _base_query(metrics=["count"])
        form = {
            "x_axis": "date",
            "groupby": ["region"],
            "metrics": ["count"],
            "metrics_b": ["revenue"],
        }
        result = build_mixed_timeseries_query(q, form)
        assert result[1].metrics == ["revenue"]

    def test_no_metrics_b_returns_one_query(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": [], "metrics": ["count"]}
        result = build_mixed_timeseries_query(q, form)
        assert len(result) == 1


# =============================================================================
# Tests: build_gantt_query
# =============================================================================


class TestBuildGanttQuery:
    def test_basic_fields(self):
        q = _base_query()
        form = {"start_time": "start", "end_time": "end", "y_axis": "task"}
        result = build_gantt_query(q, form)
        cols = result[0].columns
        assert "start" in cols
        assert "end" in cols
        assert "task" in cols

    def test_series_columns_set(self):
        q = _base_query()
        form = {"start_time": "start", "end_time": "end", "y_axis": "task", "series": "group"}
        result = build_gantt_query(q, form)
        assert "group" in result[0].columns
        assert "group" in result[0].series_columns

    def test_tooltip_columns_included(self):
        q = _base_query()
        form = {"start_time": "start", "tooltip_columns": ["owner"]}
        result = build_gantt_query(q, form)
        assert "owner" in result[0].columns

    def test_tooltip_metrics(self):
        q = _base_query()
        form = {"start_time": "start", "tooltip_metrics": ["duration"]}
        result = build_gantt_query(q, form)
        assert result[0].metrics == ["duration"]

    def test_order_by_cols_json_parsed(self):
        import json

        q = _base_query()
        form = {
            "start_time": "start",
            "order_by_cols": [json.dumps(["start", True])],
        }
        result = build_gantt_query(q, form)
        assert result[0].orderby == [("start", True)]

    def test_invalid_order_by_cols_skipped(self):
        q = _base_query()
        form = {"start_time": "start", "order_by_cols": ["not_json"]}
        result = build_gantt_query(q, form)
        assert result[0].orderby == []


# =============================================================================
# Tests: build_pop_kpi_query
# =============================================================================


class TestBuildPopKpiQuery:
    def test_cols_set_as_columns(self):
        q = _base_query(metrics=["count"], time_range="last week")
        result = build_pop_kpi_query(q, {"cols": ["region"], "time_compare": ["1 year ago"]})
        assert result[0].columns == ["region"]

    def test_time_offsets_non_custom(self):
        q = _base_query(metrics=["count"], time_range="last week")
        form = {"cols": [], "time_compare": ["1 year ago", "1 month ago"]}
        result = build_pop_kpi_query(q, form)
        assert "1 year ago" in result[0].time_offsets
        assert "1 month ago" in result[0].time_offsets

    def test_custom_time_compare(self):
        q = _base_query(metrics=["count"], time_range="last week")
        form = {"cols": [], "time_compare": ["custom"], "start_date_offset": "2023-01-01"}
        result = build_pop_kpi_query(q, form)
        assert "2023-01-01" in result[0].time_offsets

    def test_inherit_time_compare(self):
        q = _base_query(metrics=["count"], time_range="last week")
        form = {"cols": [], "time_compare": ["inherit"]}
        result = build_pop_kpi_query(q, form)
        assert "inherit" in result[0].time_offsets

    def test_no_time_compare_empty_offsets(self):
        q = _base_query(metrics=["count"])
        result = build_pop_kpi_query(q, {"cols": ["region"]})
        assert result[0].time_offsets == []


# =============================================================================
# Tests: build_timeseries_query (directly)
# =============================================================================


class TestBuildTimeseriesQueryDirect:
    def test_x_axis_set_builds_columns(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"]}
        result = build_timeseries_query(q, form)
        cols = result[0].columns
        assert "region" in cols

    def test_no_x_axis_sets_is_timeseries(self):
        q = _base_query(metrics=["count"])
        form = {"groupby": ["region"]}
        result = build_timeseries_query(q, form)
        assert result[0].is_timeseries is True

    def test_with_time_comparison_adds_time_offsets(self):
        q = _base_query(metrics=["count"], time_range="last week")
        form = {"x_axis": "date", "groupby": [], "time_compare": ["1 year ago"]}
        result = build_timeseries_query(q, form)
        assert "1 year ago" in result[0].time_offsets

    def test_post_processing_includes_flatten(self):
        q = _base_query(metrics=["count"])
        form = {"x_axis": "date", "groupby": ["region"], "metrics": ["count"]}
        result = build_timeseries_query(q, form)
        pp = result[0].post_processing
        ops = [p["operation"] for p in pp]
        assert "flatten" in ops


# =============================================================================
# Tests: Legacy chart builders
# =============================================================================


class TestBuildLegacyBubbleQuery:
    def test_entity_series_as_columns(self):
        q = _base_query()
        form = {"entity": "country", "series": "category", "x": "price", "y": "sales", "size": "count"}
        result = build_legacy_bubble_query(q, form)
        assert "country" in result[0].columns
        assert "category" in result[0].columns

    def test_metrics_order_size_x_y(self):
        q = _base_query()
        form = {"entity": "country", "x": "price", "y": "sales", "size": "count"}
        result = build_legacy_bubble_query(q, form)
        assert result[0].metrics == ["count", "price", "sales"]

    def test_limit_sets_row_limit(self):
        q = _base_query()
        result = build_legacy_bubble_query(q, {"entity": "country", "limit": "50"})
        assert result[0].row_limit == 50

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_legacy_bubble_query(q, {})
        assert result[0].is_timeseries is False

    def test_entity_only_no_series_dedup(self):
        q = _base_query()
        result = build_legacy_bubble_query(q, {"entity": "country", "series": "country"})
        assert result[0].columns == ["country"]


class TestBuildLegacyBulletQuery:
    def test_sets_metric(self):
        q = _base_query()
        result = build_legacy_bullet_query(q, {"metric": "count"})
        assert result[0].metrics == ["count"]

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_legacy_bullet_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildLegacyTimeseriesQuery:
    def test_is_timeseries(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_timeseries_query(q, {})
        assert result[0].is_timeseries is True

    def test_sort_by_timeseries_limit_metric(self):
        q = _base_query(metrics=["count"])
        form = {"timeseries_limit_metric": "revenue", "order_desc": True}
        result = build_legacy_timeseries_query(q, form)
        assert "revenue" in result[0].metrics
        assert result[0].orderby == [("revenue", False)]

    def test_no_timeseries_limit_metric_uses_first_metric(self):
        q = _base_query(metrics=["count"])
        form = {"order_desc": False}
        result = build_legacy_timeseries_query(q, form)
        assert result[0].orderby == [("count", True)]


class TestBuildLegacyTimePivotQuery:
    def test_overrides_to_single_metric(self):
        q = _base_query(metrics=["count", "revenue"])
        result = build_legacy_time_pivot_query(q, {"metric": "revenue"})
        assert result[0].metrics == ["revenue"]

    def test_is_timeseries(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_time_pivot_query(q, {"metric": "count"})
        assert result[0].is_timeseries is True


class TestBuildLegacyChordQuery:
    def test_groupby_and_columns(self):
        q = _base_query()
        form = {"groupby": "source", "columns": "target", "metric": "count"}
        result = build_legacy_chord_query(q, form)
        assert "source" in result[0].columns
        assert "target" in result[0].columns
        assert result[0].metrics == ["count"]

    def test_sort_by_metric(self):
        q = _base_query()
        form = {"groupby": "source", "metric": "count", "sort_by_metric": True}
        result = build_legacy_chord_query(q, form)
        assert result[0].orderby == [("count", False)]

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_legacy_chord_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildLegacyCountryMapQuery:
    def test_sets_entity_and_metric(self):
        q = _base_query()
        result = build_legacy_country_map_query(q, {"entity": "country", "metric": "count"})
        assert result[0].columns == ["country"]
        assert result[0].metrics == ["count"]

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_legacy_country_map_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildLegacyWorldMapQuery:
    def test_sets_entity(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_world_map_query(q, {"entity": "country"})
        assert result[0].columns == ["country"]

    def test_sort_by_metric(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_world_map_query(q, {"entity": "country", "sort_by_metric": True})
        assert result[0].orderby == [("count", False)]

    def test_is_not_timeseries(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_world_map_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildLegacyParallelCoordinatesQuery:
    def test_sets_series_as_column(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_parallel_coordinates_query(q, {"series": "category"})
        assert result[0].columns == ["category"]

    def test_timeseries_limit_metric_added_to_metrics(self):
        q = _base_query(metrics=["count"])
        form = {"timeseries_limit_metric": "revenue", "order_desc": True}
        result = build_legacy_parallel_coordinates_query(q, form)
        assert "revenue" in result[0].metrics

    def test_is_not_timeseries(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_parallel_coordinates_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildLegacyMapboxQuery:
    def test_raw_mode_columns(self):
        q = _base_query()
        form = {
            "all_columns_x": "lon",
            "all_columns_y": "lat",
            "mapbox_label": ["city"],
            "point_radius": "radius_col",
        }
        result = build_legacy_mapbox_query(q, form)
        cols = result[0].columns
        assert "lon" in cols
        assert "lat" in cols
        assert "city" in cols
        assert "radius_col" in cols

    def test_count_label_not_included(self):
        q = _base_query()
        form = {"all_columns_x": "lon", "all_columns_y": "lat", "mapbox_label": ["count"]}
        result = build_legacy_mapbox_query(q, form)
        assert "count" not in result[0].columns

    def test_auto_radius_not_included(self):
        q = _base_query()
        form = {"all_columns_x": "lon", "point_radius": "Auto"}
        result = build_legacy_mapbox_query(q, form)
        assert "Auto" not in result[0].columns

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_legacy_mapbox_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildLegacyCalHeatmapQuery:
    def test_sets_metrics(self):
        q = _base_query()
        result = build_legacy_cal_heatmap_query(q, {"metrics": ["count"]})
        assert result[0].metrics == ["count"]

    def test_subdomain_granularity_mapping(self):
        _base_query()
        for subdomain, expected in [
            ("min", "PT1M"),
            ("hour", "PT1H"),
            ("day", "P1D"),
            ("week", "P1W"),
            ("month", "P1M"),
            ("year", "P1Y"),
        ]:
            q2 = _base_query()
            result = build_legacy_cal_heatmap_query(q2, {"subdomain_granularity": subdomain})
            assert result[0].extras["time_grain_sqla"] == expected

    def test_default_subdomain_is_pt1m(self):
        q = _base_query()
        result = build_legacy_cal_heatmap_query(q, {})
        assert result[0].extras["time_grain_sqla"] == "PT1M"

    def test_is_timeseries(self):
        q = _base_query()
        result = build_legacy_cal_heatmap_query(q, {})
        assert result[0].is_timeseries is True


class TestBuildLegacyPartitionQuery:
    def test_time_series_option_not_time(self):
        q = _base_query(metrics=["count"])
        form = {"time_series_option": "not_time"}
        result = build_legacy_partition_query(q, form)
        assert result[0].is_timeseries is False

    def test_time_series_option_value_axis(self):
        q = _base_query(metrics=["count"])
        form = {"time_series_option": "value_axis"}
        result = build_legacy_partition_query(q, form)
        assert result[0].is_timeseries is True


class TestBuildLegacyPairedTtestQuery:
    def test_is_timeseries(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_paired_ttest_query(q, {})
        assert result[0].is_timeseries is True

    def test_limit_metric_added(self):
        q = _base_query(metrics=["count"])
        form = {"timeseries_limit_metric": "revenue", "order_desc": True}
        result = build_legacy_paired_ttest_query(q, form)
        assert "revenue" in result[0].metrics


class TestBuildLegacyTimeTableQuery:
    def test_is_timeseries(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_time_table_query(q, {})
        assert result[0].is_timeseries is True

    def test_orderby_by_first_metric(self):
        q = _base_query(metrics=["count"])
        result = build_legacy_time_table_query(q, {"order_desc": True})
        assert result[0].orderby == [("count", False)]


class TestBuildLegacyDeckMultiQuery:
    def test_returns_empty_query(self):
        q = _base_query()
        result = build_legacy_deck_multi_query(q, {})
        assert len(result) == 1
        assert result[0].columns == []
        assert result[0].metrics == []
        assert result[0].is_timeseries is False


class TestBuildLegacyDeckGeojsonQuery:
    def test_geojson_column_added(self):
        q = _base_query()
        result = build_legacy_deck_geojson_query(q, {"geojson": "geometry"})
        assert "geometry" in result[0].columns
        assert result[0].metrics == []

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_legacy_deck_geojson_query(q, {})
        assert result[0].is_timeseries is False


# =============================================================================
# Tests: Deck.GL chart builders
# =============================================================================


class TestBuildDeckArcQuery:
    def test_spatial_columns_extracted(self):
        q = _base_query()
        form = {
            "start_spatial": {"type": "latlong", "latCol": "start_lat", "lonCol": "start_lon"},
            "end_spatial": {"type": "latlong", "latCol": "end_lat", "lonCol": "end_lon"},
        }
        result = build_deck_arc_query(q, form)
        cols = result[0].columns
        assert "start_lat" in cols
        assert "start_lon" in cols
        assert "end_lat" in cols
        assert "end_lon" in cols

    def test_null_filters_added(self):
        q = _base_query()
        form = {"start_spatial": {"type": "latlong", "latCol": "lat", "lonCol": "lon"}}
        result = build_deck_arc_query(q, form)
        filters = result[0].filters
        filter_cols = [f["col"] for f in filters]
        assert "lat" in filter_cols
        assert "lon" in filter_cols

    def test_dimension_added(self):
        q = _base_query()
        result = build_deck_arc_query(q, {"dimension": "color_col"})
        assert "color_col" in result[0].columns

    def test_js_columns_added(self):
        q = _base_query()
        result = build_deck_arc_query(q, {"js_columns": ["js_col"]})
        assert "js_col" in result[0].columns

    def test_tooltip_contents_added(self):
        q = _base_query()
        form = {"tooltip_contents": [{"column": "tooltip_col"}]}
        result = build_deck_arc_query(q, form)
        assert "tooltip_col" in result[0].columns

    def test_is_timeseries_when_time_grain(self):
        q = _base_query()
        result = build_deck_arc_query(q, {"time_grain_sqla": "P1D"})
        assert result[0].is_timeseries is True

    def test_not_timeseries_without_time_grain(self):
        q = _base_query()
        result = build_deck_arc_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildDeckScatterQuery:
    def test_spatial_columns_extracted(self):
        q = _base_query()
        form = {"spatial": {"type": "latlong", "latCol": "lat", "lonCol": "lon"}}
        result = build_deck_scatter_query(q, form)
        assert "lat" in result[0].columns
        assert "lon" in result[0].columns

    def test_delimited_spatial(self):
        q = _base_query()
        form = {"spatial": {"type": "delimited", "lonlatCol": "coords"}}
        result = build_deck_scatter_query(q, form)
        assert "coords" in result[0].columns

    def test_geohash_spatial(self):
        q = _base_query()
        form = {"spatial": {"type": "geohash", "geohashCol": "ghash"}}
        result = build_deck_scatter_query(q, form)
        assert "ghash" in result[0].columns

    def test_point_radius_fixed_metric(self):
        q = _base_query()
        form = {"point_radius_fixed": {"value": "size_metric"}}
        result = build_deck_scatter_query(q, form)
        assert result[0].metrics == ["size_metric"]
        assert result[0].orderby == [("size_metric", False)]

    def test_is_not_timeseries(self):
        q = _base_query()
        result = build_deck_scatter_query(q, {})
        assert result[0].is_timeseries is False


class TestBuildDeckGridQuery:
    def test_spatial_columns(self):
        q = _base_query()
        form = {"spatial": {"type": "latlong", "latCol": "lat", "lonCol": "lon"}}
        result = build_deck_grid_query(q, form)
        assert "lat" in result[0].columns

    def test_js_columns(self):
        q = _base_query()
        result = build_deck_grid_query(q, {"js_columns": ["js1"]})
        assert "js1" in result[0].columns

    def test_is_timeseries_when_time_grain(self):
        q = _base_query()
        result = build_deck_grid_query(q, {"time_grain_sqla": "P1D"})
        assert result[0].is_timeseries is True


class TestBuildDeckPathQuery:
    def test_line_column_added(self):
        q = _base_query()
        result = build_deck_path_query(q, {"line_column": "path_col"})
        assert "path_col" in result[0].columns

    def test_js_columns_added(self):
        q = _base_query()
        result = build_deck_path_query(q, {"js_columns": ["js1"]})
        assert "js1" in result[0].columns

    def test_is_timeseries_when_time_grain(self):
        q = _base_query()
        result = build_deck_path_query(q, {"time_grain_sqla": "P1D"})
        assert result[0].is_timeseries is True


class TestBuildDeckPolygonQuery:
    def test_line_column_added(self):
        q = _base_query()
        result = build_deck_polygon_query(q, {"line_column": "poly_col"})
        assert "poly_col" in result[0].columns


# =============================================================================
# Tests: ChartBuildQueryRegistry
# =============================================================================


class TestChartBuildQueryRegistry:
    def test_get_known_chart_type(self):
        registry = ChartBuildQueryRegistry()
        fn = registry.get("table")
        assert fn is build_table_query

    def test_get_unknown_returns_default(self):
        registry = ChartBuildQueryRegistry()
        fn = registry.get("unknown_chart")
        assert fn is build_default_query

    def test_register_custom(self):
        registry = ChartBuildQueryRegistry()

        def custom_fn(q, fd):
            return [q]

        registry.register("my_custom_chart", custom_fn)
        assert registry.get("my_custom_chart") is custom_fn

    def test_all_timeseries_types_registered(self):
        registry = ChartBuildQueryRegistry()
        for t in ["echarts_timeseries", "echarts_timeseries_bar", "echarts_area"]:
            fn = registry.get(t)
            assert fn is build_timeseries_query

    def test_pie_registered(self):
        assert ChartBuildQueryRegistry().get("pie") is build_pie_query

    def test_big_number_total_registered(self):
        assert ChartBuildQueryRegistry().get("big_number_total") is build_big_number_query

    def test_big_number_trendline_registered(self):
        assert ChartBuildQueryRegistry().get("big_number") is build_big_number_trendline_query

    def test_mixed_timeseries_registered(self):
        assert ChartBuildQueryRegistry().get("mixed_timeseries") is build_mixed_timeseries_query

    def test_legacy_bubble_registered(self):
        assert ChartBuildQueryRegistry().get("bubble") is build_legacy_bubble_query

    def test_deck_arc_registered(self):
        assert ChartBuildQueryRegistry().get("deck_arc") is build_deck_arc_query

    def test_histogram_v2_registered(self):
        assert ChartBuildQueryRegistry().get("histogram_v2") is build_histogram_query


class TestGetChartBuildQueryRegistry:
    def test_returns_registry_instance(self):
        registry = get_chart_build_query_registry()
        assert isinstance(registry, ChartBuildQueryRegistry)

    def test_global_registry_shared(self):
        r1 = get_chart_build_query_registry()
        r2 = get_chart_build_query_registry()
        assert r1 is r2


class TestRegisterChartBuildQuery:
    def test_registers_on_global_registry(self):
        def my_fn(q, fd):
            return [q]

        register_chart_build_query("test_chart_xyz", my_fn)
        registry = get_chart_build_query_registry()
        assert registry.get("test_chart_xyz") is my_fn


# =============================================================================
# Tests: build_query_context (main function)
# =============================================================================


class TestBuildQueryContext:
    def test_basic_pie_chart(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "pie",
            "metrics": ["count"],
            "groupby": ["category"],
        }
        result = build_query_context(form_data)
        assert result["datasource"] == {"id": 1, "type": "table"}
        assert len(result["queries"]) == 1
        pp = result["queries"][0].get("post_processing", [])
        assert any(p["operation"] == "contribution" for p in pp)

    def test_basic_table_chart(self):
        form_data = {
            "datasource": "2__table",
            "viz_type": "table",
            "groupby": ["region"],
            "metrics": ["count"],
            "query_mode": "aggregate",
        }
        result = build_query_context(form_data)
        assert result["datasource"]["id"] == 2
        assert len(result["queries"]) == 1

    def test_big_number_total(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "big_number_total",
            "metrics": ["count"],
        }
        result = build_query_context(form_data)
        assert len(result["queries"]) == 1

    def test_invalid_datasource_defaults(self):
        form_data = {"viz_type": "bar", "metrics": ["count"]}
        result = build_query_context(form_data)
        assert result["datasource"] == {"id": 0, "type": "table"}

    def test_custom_build_query_func(self):
        custom_called = []

        def custom_fn(q, fd):
            custom_called.append(True)
            return [q]

        form_data = {"datasource": "1__table", "viz_type": "bar", "metrics": ["count"]}
        build_query_context(form_data, build_query=custom_fn)
        assert custom_called

    def test_x_axis_triggers_normalize(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "echarts_timeseries_line",
            "x_axis": "date",
            "groupby": ["region"],
            "metrics": ["count"],
            "columns": ["date"],
        }
        result = build_query_context(form_data)
        # Should not raise and should return valid context
        assert "queries" in result

    def test_result_format_and_type_defaults(self):
        form_data = {"datasource": "1__table", "viz_type": "bar", "metrics": ["count"]}
        result = build_query_context(form_data)
        assert result["result_format"] == "json"

    def test_result_type_query_is_default(self):
        form_data = {"datasource": "1__table", "viz_type": "bar", "metrics": ["count"]}
        result = build_query_context(form_data)
        assert result["result_type"] == "query"

    def test_force_field_passed(self):
        form_data = {"datasource": "1__table", "viz_type": "bar", "metrics": ["count"], "force": True}
        result = build_query_context(form_data)
        assert result["force"] is True

    def test_mixed_timeseries_returns_two_queries(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "mixed_timeseries",
            "x_axis": "date",
            "metrics": ["count"],
            "groupby": [],
            "metrics_b": ["revenue"],
            "groupby_b": [],
        }
        result = build_query_context(form_data)
        assert len(result["queries"]) == 2

    def test_none_post_processing_filtered(self):
        """Ensure None items are removed from post_processing lists."""
        form_data = {
            "datasource": "1__table",
            "viz_type": "echarts_timeseries_bar",
            "x_axis": "date",
            "groupby": ["region"],
            "metrics": ["count"],
        }
        result = build_query_context(form_data)
        for q in result["queries"]:
            for p in q.get("post_processing", []):
                assert p is not None

    def test_histogram_chart(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "histogram",
            "column": "price",
            "bins": 20,
        }
        result = build_query_context(form_data)
        assert len(result["queries"]) == 1
        pp = result["queries"][0].get("post_processing", [])
        assert any(p["operation"] == "histogram" for p in pp)

    def test_waterfall_chart(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "waterfall",
            "x_axis": "date",
            "groupby": [],
            "metrics": ["revenue"],
        }
        result = build_query_context(form_data)
        assert len(result["queries"]) == 1

    def test_legacy_world_map(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "world_map",
            "entity": "country",
            "metrics": ["count"],
        }
        result = build_query_context(form_data)
        assert len(result["queries"]) == 1
        assert result["queries"][0]["columns"] == ["country"]

    def test_deck_arc_chart(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "deck_arc",
            "start_spatial": {"type": "latlong", "latCol": "slat", "lonCol": "slon"},
        }
        result = build_query_context(form_data)
        cols = result["queries"][0]["columns"]
        assert "slat" in cols

    def test_handlebars_uses_default_query(self):
        form_data = {
            "datasource": "1__table",
            "viz_type": "handlebars",
            "groupby": ["region"],
            "metrics": ["count"],
        }
        result = build_query_context(form_data)
        assert len(result["queries"]) == 1
