# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/tools/bi_tools/superset/superset_util.py

All tests are CI-level: zero external dependencies, fully deterministic.
"""

import pytest

from datus.tools.bi_tools.superset.superset_util import (
    LEGACY_VIZ_TYPES,
    ComparisonType,
    DatasourceKey,
    DatasourceType,
    QueryContext,
    QueryObject,
    boxplot_operator,
    build_query_object,
    build_timeseries_query,
    contribution_operator,
    ensure_list,
    extract_extras,
    extract_query_fields,
    flatten_operator,
    get_column_label,
    get_metric_label,
    get_x_axis_column,
    get_x_axis_column_with_time_grain,
    get_x_axis_label,
    histogram_operator,
    is_adhoc_column,
    is_physical_column,
    is_time_comparison,
    is_x_axis_set,
    normalize_orderby,
    normalize_time_column,
    pivot_operator,
    process_filters,
    prophet_operator,
    rank_operator,
    rename_operator,
    resample_operator,
    rolling_window_operator,
    sort_operator,
    time_compare_operator,
    uses_legacy_api,
)

# =============================================================================
# Tests: uses_legacy_api
# =============================================================================


class TestUsesLegacyApi:
    def test_deck_arc_is_legacy(self):
        assert uses_legacy_api("deck_arc") is True

    def test_world_map_is_legacy(self):
        assert uses_legacy_api("world_map") is True

    def test_bar_is_not_legacy(self):
        assert uses_legacy_api("bar") is False

    def test_echarts_timeseries_not_legacy(self):
        assert uses_legacy_api("echarts_timeseries") is False

    def test_pie_not_legacy(self):
        assert uses_legacy_api("pie") is False

    def test_all_legacy_types_recognized(self):
        for viz in LEGACY_VIZ_TYPES:
            assert uses_legacy_api(viz) is True


# =============================================================================
# Tests: ensure_list
# =============================================================================


class TestEnsureList:
    def test_none_returns_empty(self):
        assert ensure_list(None) == []

    def test_list_passthrough(self):
        assert ensure_list([1, 2, 3]) == [1, 2, 3]

    def test_single_value_wrapped(self):
        assert ensure_list("foo") == ["foo"]

    def test_int_wrapped(self):
        assert ensure_list(42) == [42]

    def test_empty_list(self):
        assert ensure_list([]) == []


# =============================================================================
# Tests: get_column_label
# =============================================================================


class TestGetColumnLabel:
    def test_string_column(self):
        assert get_column_label("region") == "region"

    def test_dict_with_label(self):
        assert get_column_label({"label": "Revenue", "sqlExpression": "SUM(x)"}) == "Revenue"

    def test_dict_with_sql_expression_only(self):
        assert get_column_label({"sqlExpression": "YEAR(date)"}) == "YEAR(date)"

    def test_dict_with_column_name_only(self):
        assert get_column_label({"column_name": "created_at"}) == "created_at"

    def test_non_string_non_dict(self):
        assert get_column_label(99) == "99"


# =============================================================================
# Tests: get_metric_label
# =============================================================================


class TestGetMetricLabel:
    def test_string_metric(self):
        assert get_metric_label("count") == "count"

    def test_dict_with_label(self):
        assert get_metric_label({"label": "Total Sales"}) == "Total Sales"

    def test_dict_with_aggregate_and_column(self):
        result = get_metric_label({"aggregate": "SUM", "column": {"column_name": "amount"}})
        assert result == "SUM(amount)"

    def test_dict_with_sql_expression(self):
        result = get_metric_label({"sqlExpression": "COUNT(DISTINCT id)"})
        assert result == "COUNT(DISTINCT id)"

    def test_non_string_non_dict(self):
        assert get_metric_label(10) == "10"

    def test_empty_dict(self):
        assert get_metric_label({}) == ""


# =============================================================================
# Tests: is_physical_column / is_adhoc_column
# =============================================================================


class TestIsPhysicalColumn:
    def test_string_is_physical(self):
        assert is_physical_column("date") is True

    def test_dict_is_not_physical(self):
        assert is_physical_column({"sqlExpression": "x"}) is False


class TestIsAdhocColumn:
    def test_valid_adhoc(self):
        col = {"sqlExpression": "YEAR(date)", "label": "year", "expressionType": "SQL"}
        assert is_adhoc_column(col) is True

    def test_missing_label(self):
        col = {"sqlExpression": "YEAR(date)", "expressionType": "SQL"}
        assert is_adhoc_column(col) is False

    def test_missing_sql_expression(self):
        col = {"label": "year", "expressionType": "SQL"}
        assert is_adhoc_column(col) is False

    def test_wrong_expression_type(self):
        col = {"sqlExpression": "x", "label": "x", "expressionType": "SIMPLE"}
        assert is_adhoc_column(col) is False

    def test_string_is_not_adhoc(self):
        assert is_adhoc_column("date") is False


# =============================================================================
# Tests: is_x_axis_set / get_x_axis_column / get_x_axis_label
# =============================================================================


class TestIsXAxisSet:
    def test_string_x_axis(self):
        assert is_x_axis_set({"x_axis": "date"}) is True

    def test_adhoc_x_axis(self):
        fd = {"x_axis": {"sqlExpression": "YEAR(date)"}}
        assert is_x_axis_set(fd) is True

    def test_none_x_axis(self):
        assert is_x_axis_set({"x_axis": None}) is False

    def test_missing_x_axis(self):
        assert is_x_axis_set({}) is False

    def test_dict_without_sql_expression(self):
        assert is_x_axis_set({"x_axis": {"label": "something"}}) is False


class TestGetXAxisColumn:
    def test_returns_x_axis_when_set(self):
        assert get_x_axis_column({"x_axis": "date"}) == "date"

    def test_falls_back_to_granularity_sqla(self):
        assert get_x_axis_column({"granularity_sqla": "ts"}) == "ts"

    def test_returns_none_when_neither_set(self):
        assert get_x_axis_column({}) is None

    def test_x_axis_takes_priority_over_granularity(self):
        fd = {"x_axis": "date_col", "granularity_sqla": "ts_col"}
        assert get_x_axis_column(fd) == "date_col"


class TestGetXAxisLabel:
    def test_physical_column(self):
        assert get_x_axis_label({"x_axis": "date"}) == "date"

    def test_adhoc_column(self):
        fd = {"x_axis": {"label": "my_date", "sqlExpression": "YEAR(d)", "expressionType": "SQL"}}
        # get_x_axis_label calls get_column_label which uses label key
        assert get_x_axis_label(fd) == "my_date"

    def test_none_when_missing(self):
        assert get_x_axis_label({}) is None


# =============================================================================
# Tests: get_x_axis_column_with_time_grain
# =============================================================================


class TestGetXAxisColumnWithTimeGrain:
    def test_physical_column_with_time_grain(self):
        fd = {"x_axis": "date", "time_grain_sqla": "P1D"}
        result = get_x_axis_column_with_time_grain(fd)
        assert isinstance(result, dict)
        assert result["sqlExpression"] == "date"
        assert result["timeGrain"] == "P1D"
        assert result["columnType"] == "BASE_AXIS"

    def test_physical_column_without_time_grain(self):
        fd = {"x_axis": "date"}
        result = get_x_axis_column_with_time_grain(fd)
        # No time grain - returns the string directly
        assert result == "date"

    def test_no_x_axis_falls_back_to_granularity_sqla(self):
        fd = {"granularity_sqla": "ts", "time_grain_sqla": "P1M"}
        result = get_x_axis_column_with_time_grain(fd)
        assert isinstance(result, dict)
        assert result["sqlExpression"] == "ts"
        assert result["timeGrain"] == "P1M"

    def test_adhoc_column_gets_base_axis(self):
        fd = {"x_axis": {"sqlExpression": "YEAR(d)", "label": "y", "expressionType": "SQL"}}
        result = get_x_axis_column_with_time_grain(fd)
        assert isinstance(result, dict)
        assert result["columnType"] == "BASE_AXIS"

    def test_returns_none_when_no_x_axis(self):
        assert get_x_axis_column_with_time_grain({}) is None


# =============================================================================
# Tests: DatasourceKey
# =============================================================================


class TestDatasourceKey:
    def test_from_string_table(self):
        dk = DatasourceKey.from_string("1__table")
        assert dk.id == 1
        assert dk.type == DatasourceType.TABLE

    def test_from_string_query(self):
        dk = DatasourceKey.from_string("42__query")
        assert dk.id == 42
        assert dk.type == DatasourceType.QUERY

    def test_from_dict(self):
        dk = DatasourceKey.from_string({"id": 10, "type": "table"})
        assert dk.id == 10
        assert dk.type == DatasourceType.TABLE

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            DatasourceKey.from_string("invalid_format")

    def test_to_dict(self):
        dk = DatasourceKey(id=5, type=DatasourceType.TABLE)
        d = dk.to_dict()
        assert d == {"id": 5, "type": "table"}


# =============================================================================
# Tests: QueryObject
# =============================================================================


class TestQueryObject:
    def test_to_dict_skips_none(self):
        q = QueryObject()
        d = q.to_dict()
        # None fields should not appear
        assert "time_range" not in d
        assert "since" not in d

    def test_to_dict_skips_empty_lists(self):
        q = QueryObject()
        d = q.to_dict()
        assert "columns" not in d
        assert "metrics" not in d

    def test_to_dict_includes_values(self):
        q = QueryObject(time_range="last week", columns=["date"])
        d = q.to_dict()
        assert d["time_range"] == "last week"
        assert d["columns"] == ["date"]

    def test_to_dict_includes_false_is_timeseries(self):
        q = QueryObject(is_timeseries=False)
        d = q.to_dict()
        # False bool: not None, not empty list/dict → should be included
        assert "is_timeseries" in d
        assert d["is_timeseries"] is False


# =============================================================================
# Tests: QueryContext
# =============================================================================


class TestQueryContext:
    def test_to_dict(self):
        q = QueryObject(time_range="last week")
        ctx = QueryContext(
            datasource={"id": 1, "type": "table"},
            force=False,
            queries=[q],
            form_data={"viz_type": "bar"},
        )
        d = ctx.to_dict()
        assert d["datasource"] == {"id": 1, "type": "table"}
        assert d["force"] is False
        assert d["result_format"] == "json"
        assert d["result_type"] == "full"
        assert len(d["queries"]) == 1


# =============================================================================
# Tests: is_time_comparison
# =============================================================================


class TestIsTimeComparison:
    def test_with_time_compare(self):
        q = QueryObject()
        assert is_time_comparison({"time_compare": ["1 year ago"]}, q) is True

    def test_without_time_compare(self):
        q = QueryObject()
        assert is_time_comparison({}, q) is False

    def test_empty_time_compare(self):
        q = QueryObject()
        assert is_time_comparison({"time_compare": []}, q) is False


# =============================================================================
# Tests: normalize_orderby
# =============================================================================


class TestNormalizeOrderby:
    def test_valid_orderby_passthrough(self):
        q = QueryObject(orderby=[("count", True)])
        result = normalize_orderby(q)
        assert result == [("count", True)]

    def test_falls_back_to_series_limit_metric(self):
        q = QueryObject(series_limit_metric="count", order_desc=True)
        result = normalize_orderby(q)
        assert result == [("count", False)]

    def test_falls_back_to_first_metric(self):
        q = QueryObject(metrics=["count", "sum"], order_desc=False)
        result = normalize_orderby(q)
        assert result == [("count", True)]

    def test_empty_when_no_metrics(self):
        q = QueryObject()
        result = normalize_orderby(q)
        assert result == []

    def test_json_string_orderby_in_valid_list(self):
        import json

        # json string items are parsed when there's already a valid tuple first item
        q = QueryObject(orderby=[("revenue", True), json.dumps(["extra_col", False])])
        result = normalize_orderby(q)
        assert ("revenue", True) in result


# =============================================================================
# Tests: normalize_time_column
# =============================================================================


class TestNormalizeTimeColumn:
    def test_physical_column_transformed(self):
        q = QueryObject(columns=["date", "region"])
        result = normalize_time_column({"x_axis": "date"}, q)
        # Find the date column
        date_col = next(c for c in result.columns if isinstance(c, dict) and c.get("sqlExpression") == "date")
        assert date_col["columnType"] == "BASE_AXIS"

    def test_no_x_axis_returns_unchanged(self):
        q = QueryObject(columns=["region"])
        result = normalize_time_column({}, q)
        assert result.columns == ["region"]

    def test_x_axis_not_in_columns(self):
        q = QueryObject(columns=["region"])
        result = normalize_time_column({"x_axis": "date"}, q)
        # date not in columns, nothing changed
        assert result.columns == ["region"]

    def test_with_time_grain(self):
        q = QueryObject(columns=["date"], extras={"time_grain_sqla": "P1D"})
        result = normalize_time_column({"x_axis": "date"}, q)
        date_col = result.columns[0]
        assert isinstance(date_col, dict)
        assert date_col.get("timeGrain") == "P1D"


# =============================================================================
# Tests: pivot_operator
# =============================================================================


class TestPivotOperator:
    def test_returns_pivot_rule(self):
        q = QueryObject(metrics=["count"], columns=["region"])
        form_data = {"x_axis": "date"}
        result = pivot_operator(form_data, q)
        assert result is not None
        assert result["operation"] == "pivot"
        assert "index" in result["options"]

    def test_no_metrics_returns_none(self):
        q = QueryObject()
        result = pivot_operator({"x_axis": "date"}, q)
        assert result is None

    def test_no_x_axis_returns_none(self):
        q = QueryObject(metrics=["count"])
        result = pivot_operator({}, q)
        assert result is None

    def test_drop_missing_columns_respects_show_empty(self):
        q = QueryObject(metrics=["count"])
        result = pivot_operator({"x_axis": "date", "show_empty_columns": True}, q)
        assert result is not None
        assert result["options"]["drop_missing_columns"] is False


# =============================================================================
# Tests: flatten_operator
# =============================================================================


class TestFlattenOperator:
    def test_returns_flatten(self):
        q = QueryObject()
        result = flatten_operator({}, q)
        assert result["operation"] == "flatten"


# =============================================================================
# Tests: rolling_window_operator
# =============================================================================


class TestRollingWindowOperator:
    def test_cumsum(self):
        q = QueryObject(metrics=["count"])
        result = rolling_window_operator({"rolling_type": "cumsum"}, q)
        assert result is not None
        assert result["operation"] == "cum"
        assert result["options"]["operator"] == "sum"

    def test_sum_rolling(self):
        q = QueryObject(metrics=["revenue"])
        result = rolling_window_operator({"rolling_type": "sum", "rolling_periods": 7, "min_periods": 1}, q)
        assert result["operation"] == "rolling"
        assert result["options"]["rolling_type"] == "sum"
        assert result["options"]["window"] == 7

    def test_none_rolling_type(self):
        q = QueryObject(metrics=["count"])
        assert rolling_window_operator({"rolling_type": "none"}, q) is None

    def test_missing_rolling_type(self):
        q = QueryObject(metrics=["count"])
        assert rolling_window_operator({}, q) is None


# =============================================================================
# Tests: resample_operator
# =============================================================================


class TestResampleOperator:
    def test_with_rule_and_method(self):
        q = QueryObject()
        result = resample_operator({"resample_rule": "1D", "resample_method": "ffill"}, q)
        assert result is not None
        assert result["operation"] == "resample"
        assert result["options"]["rule"] == "1D"
        assert result["options"]["method"] == "ffill"

    def test_missing_rule_returns_none(self):
        q = QueryObject()
        assert resample_operator({"resample_method": "ffill"}, q) is None

    def test_missing_method_returns_none(self):
        q = QueryObject()
        assert resample_operator({"resample_rule": "1D"}, q) is None


# =============================================================================
# Tests: rename_operator
# =============================================================================


class TestRenameOperator:
    def test_truncate_metric_single_metric(self):
        q = QueryObject(metrics=["count"])
        result = rename_operator({"x_axis": "date", "truncate_metric": True}, q)
        assert result is not None
        assert result["operation"] == "rename"
        assert "count" in result["options"]["columns"]

    def test_no_truncate_returns_none(self):
        q = QueryObject(metrics=["count"])
        result = rename_operator({"x_axis": "date", "truncate_metric": False}, q)
        assert result is None

    def test_multiple_metrics_returns_none(self):
        q = QueryObject(metrics=["count", "sum"])
        result = rename_operator({"x_axis": "date", "truncate_metric": True}, q)
        assert result is None

    def test_no_x_axis_returns_none(self):
        q = QueryObject(metrics=["count"])
        result = rename_operator({"truncate_metric": True}, q)
        assert result is None


# =============================================================================
# Tests: sort_operator
# =============================================================================


class TestSortOperator:
    def test_sort_by_x_axis(self):
        q = QueryObject()
        form_data = {"x_axis": "date", "x_axis_sort": "date", "x_axis_sort_asc": True}
        result = sort_operator(form_data, q)
        assert result is not None
        assert result["operation"] == "sort"
        assert result["options"]["is_sort_index"] is True

    def test_sort_by_other_column(self):
        q = QueryObject()
        form_data = {"x_axis": "date", "x_axis_sort": "revenue", "x_axis_sort_asc": False}
        result = sort_operator(form_data, q)
        assert result is not None
        assert result["options"]["by"] == "revenue"

    def test_missing_sort_returns_none(self):
        q = QueryObject()
        assert sort_operator({"x_axis": "date"}, q) is None

    def test_with_groupby_returns_none(self):
        q = QueryObject()
        form_data = {"x_axis": "date", "x_axis_sort": "date", "x_axis_sort_asc": True, "groupby": ["region"]}
        assert sort_operator(form_data, q) is None


# =============================================================================
# Tests: contribution_operator
# =============================================================================


class TestContributionOperator:
    def test_with_mode(self):
        q = QueryObject()
        result = contribution_operator({"contributionMode": "row"}, q)
        assert result is not None
        assert result["operation"] == "contribution"
        assert result["options"]["orientation"] == "row"

    def test_with_time_offsets(self):
        q = QueryObject()
        result = contribution_operator({"contributionMode": "column"}, q, time_offsets=["1 year ago"])
        assert result["options"]["time_shifts"] == ["1 year ago"]

    def test_no_mode_returns_none(self):
        q = QueryObject()
        assert contribution_operator({}, q) is None


# =============================================================================
# Tests: time_compare_operator
# =============================================================================


class TestTimeCompareOperator:
    def test_with_time_comparison(self):
        q = QueryObject()
        form_data = {"time_compare": ["1 year ago"], "comparison_type": "difference"}
        result = time_compare_operator(form_data, q)
        assert result is not None
        assert result["operation"] == "compare"
        assert result["options"]["compare_type"] == "difference"

    def test_without_time_comparison(self):
        q = QueryObject()
        assert time_compare_operator({}, q) is None

    def test_default_comparison_type(self):
        q = QueryObject()
        form_data = {"time_compare": ["1 year ago"]}
        result = time_compare_operator(form_data, q)
        assert result["options"]["compare_type"] == ComparisonType.VALUES.value


# =============================================================================
# Tests: boxplot_operator
# =============================================================================


class TestBoxplotOperator:
    def test_tukey_whisker(self):
        q = QueryObject(metrics=["count"])
        result = boxplot_operator({"whiskerOptions": "Tukey"}, q)
        assert result is not None
        assert result["operation"] == "boxplot"
        assert result["options"]["whisker_type"] == "tukey"

    def test_min_max_whisker(self):
        q = QueryObject(metrics=["count"])
        result = boxplot_operator({"whiskerOptions": "Min/max (no outliers)"}, q)
        assert result["options"]["whisker_type"] == "min/max"

    def test_percentile_whisker(self):
        q = QueryObject(metrics=["count"])
        result = boxplot_operator({"whiskerOptions": "10/90 percentiles"}, q)
        assert result["options"]["whisker_type"] == "percentile"
        assert result["options"]["percentiles"] == [10, 90]

    def test_no_whisker_returns_none(self):
        q = QueryObject(metrics=["count"])
        assert boxplot_operator({}, q) is None


# =============================================================================
# Tests: rank_operator
# =============================================================================


class TestRankOperator:
    def test_normalize_across_x(self):
        q = QueryObject(metrics=["count"])
        form_data = {"normalize_across": "x", "x_axis": "date"}
        result = rank_operator(form_data, q)
        assert result is not None
        assert result["operation"] == "rank"
        assert result["options"]["group_by"] == "date"

    def test_normalize_across_y(self):
        q = QueryObject(metrics=["count"])
        form_data = {"normalize_across": "y", "groupby": ["region"]}
        result = rank_operator(form_data, q)
        assert result is not None
        assert result["options"]["group_by"] == "region"

    def test_no_normalize_returns_none(self):
        q = QueryObject(metrics=["count"])
        assert rank_operator({}, q) is None

    def test_no_metrics_returns_none(self):
        assert rank_operator({"normalize_across": "x"}, QueryObject()) is None


# =============================================================================
# Tests: histogram_operator
# =============================================================================


class TestHistogramOperator:
    def test_basic_histogram(self):
        q = QueryObject()
        result = histogram_operator({"column": "amount", "bins": 20}, q)
        assert result is not None
        assert result["operation"] == "histogram"
        assert result["options"]["column"] == "amount"
        assert result["options"]["bins"] == 20

    def test_falls_back_to_all_columns(self):
        q = QueryObject()
        result = histogram_operator({"all_columns": ["price"]}, q)
        assert result is not None
        assert result["options"]["column"] == "price"

    def test_no_column_returns_none(self):
        q = QueryObject()
        assert histogram_operator({}, q) is None

    def test_defaults(self):
        q = QueryObject()
        result = histogram_operator({"column": "x"}, q)
        assert result["options"]["bins"] == 10
        assert result["options"]["cumulative"] is False
        assert result["options"]["normalize"] is False


# =============================================================================
# Tests: prophet_operator
# =============================================================================


class TestProphetOperator:
    def test_with_forecast_enabled(self):
        q = QueryObject()
        form_data = {
            "forecastEnabled": True,
            "x_axis": "date",
            "forecastPeriods": 30,
            "forecastInterval": 0.9,
        }
        result = prophet_operator(form_data, q)
        assert result is not None
        assert result["operation"] == "prophet"
        assert result["options"]["periods"] == 30
        assert result["options"]["confidence_interval"] == 0.9

    def test_forecast_not_enabled_returns_none(self):
        q = QueryObject()
        assert prophet_operator({"x_axis": "date", "forecastPeriods": 10}, q) is None

    def test_no_x_axis_returns_none(self):
        q = QueryObject()
        assert prophet_operator({"forecastEnabled": True}, q) is None

    def test_default_confidence_interval(self):
        q = QueryObject()
        form_data = {"forecastEnabled": True, "x_axis": "date"}
        result = prophet_operator(form_data, q)
        assert result["options"]["confidence_interval"] == 0.8


# =============================================================================
# Tests: extract_query_fields
# =============================================================================


class TestExtractQueryFields:
    def test_extracts_groupby_as_columns(self):
        form_data = {"groupby": ["region", "category"]}
        result = extract_query_fields(form_data)
        assert "region" in result["columns"] or result["columns"] == ["region", "category"]

    def test_extracts_metrics(self):
        form_data = {"metrics": ["count", "revenue"]}
        result = extract_query_fields(form_data)
        assert result["metrics"] == ["count", "revenue"]

    def test_raw_mode_skips_metrics(self):
        form_data = {"query_mode": "raw", "metrics": ["count"], "columns": ["id"]}
        result = extract_query_fields(form_data)
        assert result["metrics"] == []

    def test_aggregate_mode_skips_raw_columns(self):
        form_data = {"query_mode": "aggregate", "groupby": ["region"], "columns": ["id"]}
        result = extract_query_fields(form_data)
        # "columns" key under aggregate mode is skipped, only groupby used
        assert "region" in result["columns"]
        assert "id" not in result["columns"]

    def test_deduplicates_columns(self):
        form_data = {"groupby": ["region", "region"]}
        result = extract_query_fields(form_data)
        assert result["columns"].count("region") == 1

    def test_orderby_extracted(self):
        form_data = {"orderby": [("revenue", True)]}
        result = extract_query_fields(form_data)
        assert result["orderby"] == [("revenue", True)]

    def test_field_alias_metric(self):
        # "metric" is aliased to "metrics"
        form_data = {"metric": "count"}
        result = extract_query_fields(form_data)
        assert "count" in result["metrics"]


# =============================================================================
# Tests: extract_extras
# =============================================================================


class TestExtractExtras:
    def test_time_range(self):
        form_data = {"time_range": "last week"}
        result = extract_extras(form_data)
        assert result["time_range"] == "last week"

    def test_extra_filters_time_range(self):
        form_data = {"extra_filters": [{"col": "__time_range", "val": "last month", "op": "=="}]}
        result = extract_extras(form_data)
        assert result["time_range"] == "last month"
        assert "__time_range" in result["applied_time_extras"]

    def test_extra_filters_non_reserved(self):
        form_data = {"extra_filters": [{"col": "region", "val": "US", "op": "=="}]}
        result = extract_extras(form_data)
        assert len(result["filters"]) == 1
        assert result["filters"][0]["col"] == "region"

    def test_time_grain_sqla(self):
        form_data = {"time_grain_sqla": "P1D"}
        result = extract_extras(form_data)
        assert result["extras"]["time_grain_sqla"] == "P1D"

    def test_extra_filters_time_col(self):
        form_data = {"extra_filters": [{"col": "__time_col", "val": "created_at", "op": "=="}]}
        result = extract_extras(form_data)
        assert result["granularity_sqla"] == "created_at"


# =============================================================================
# Tests: process_filters
# =============================================================================


class TestProcessFilters:
    def test_simple_where_filter(self):
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "clause": "WHERE",
                    "subject": "region",
                    "operator": "==",
                    "comparator": "US",
                    "isExtra": False,
                }
            ]
        }
        result = process_filters(form_data)
        assert len(result["filters"]) == 1
        assert result["filters"][0]["col"] == "region"

    def test_sql_where_filter(self):
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SQL",
                    "clause": "WHERE",
                    "sqlExpression": "amount > 100",
                }
            ]
        }
        result = process_filters(form_data)
        assert "(amount > 100)" in result["extras"].get("where", "")

    def test_sql_having_filter(self):
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SQL",
                    "clause": "HAVING",
                    "sqlExpression": "COUNT(*) > 5",
                }
            ]
        }
        result = process_filters(form_data)
        assert "(COUNT(*) > 5)" in result["extras"].get("having", "")

    def test_existing_where_included(self):
        form_data = {"where": "status = 'active'"}
        result = process_filters(form_data)
        assert "(status = 'active')" in result["extras"].get("where", "")

    def test_multiple_where_joined_with_and(self):
        form_data = {
            "adhoc_filters": [
                {"expressionType": "SQL", "clause": "WHERE", "sqlExpression": "a > 1"},
                {"expressionType": "SQL", "clause": "WHERE", "sqlExpression": "b < 2"},
            ]
        }
        result = process_filters(form_data)
        where = result["extras"].get("where", "")
        assert "(a > 1) AND (b < 2)" in where


# =============================================================================
# Tests: build_query_object
# =============================================================================


class TestBuildQueryObject:
    def test_basic_build(self):
        form_data = {
            "groupby": ["region"],
            "metrics": ["count"],
            "time_range": "last week",
        }
        q = build_query_object(form_data)
        assert q.columns == ["region"]
        assert q.metrics == ["count"]
        assert q.time_range == "last week"

    def test_row_limit(self):
        form_data = {"row_limit": 100}
        q = build_query_object(form_data)
        assert q.row_limit == 100

    def test_invalid_row_limit_becomes_none(self):
        form_data = {"row_limit": "not_a_number"}
        q = build_query_object(form_data)
        assert q.row_limit is None

    def test_series_limit(self):
        form_data = {"series_limit": 10}
        q = build_query_object(form_data)
        assert q.series_limit == 10

    def test_extra_form_data_overrides(self):
        form_data = {
            "row_limit": 50,
            "extra_form_data": {"row_limit": 200},
        }
        q = build_query_object(form_data)
        assert q.row_limit == 200


# =============================================================================
# Tests: build_timeseries_query
# =============================================================================


class TestBuildTimeseriesQuery:
    def test_basic_timeseries(self):
        form_data = {
            "x_axis": "date",
            "groupby": ["region"],
            "metrics": ["count"],
        }
        base = build_query_object(form_data)
        queries = build_timeseries_query(base, form_data)
        assert len(queries) >= 1

    def test_with_time_comparison(self):
        form_data = {
            "x_axis": "date",
            "metrics": ["count"],
            "time_compare": ["1 year ago"],
        }
        base = build_query_object(form_data)
        queries = build_timeseries_query(base, form_data)
        # Time comparison adds extra query
        assert len(queries) >= 1
