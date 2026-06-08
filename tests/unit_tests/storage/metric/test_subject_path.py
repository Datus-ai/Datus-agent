# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.metric.subject_path."""

from datus.storage.metric.subject_path import (
    GENERIC_METRIC_SUBJECT_ROOTS,
    _normalize_part,
    _same_part,
    default_metric_subject_path,
    normalize_metric_subject_path,
    normalize_metric_subject_tree_tag,
)


class TestDefaultMetricSubjectPath:
    def test_with_datasource_and_table(self):
        assert default_metric_subject_path("myds", "orders") == ["myds", "orders"]

    def test_without_datasource_uses_metrics_root(self):
        assert default_metric_subject_path("", "orders") == ["Metrics", "orders"]

    def test_none_datasource_uses_metrics_root(self):
        assert default_metric_subject_path(None, "orders") == ["Metrics", "orders"]

    def test_empty_table_becomes_unknown(self):
        assert default_metric_subject_path("ds", "") == ["ds", "Unknown"]

    def test_both_empty(self):
        assert default_metric_subject_path("", "") == ["Metrics", "Unknown"]

    def test_whitespace_datasource_treated_as_empty(self):
        assert default_metric_subject_path("  ", "orders") == ["Metrics", "orders"]

    def test_whitespace_table_becomes_unknown(self):
        result = default_metric_subject_path("ds", "  ")
        assert result == ["ds", "Unknown"]


class TestNormalizeMetricSubjectPath:
    def test_empty_parts_falls_back_to_default(self):
        result = normalize_metric_subject_path([], datasource="ds", table_name="orders")
        assert result == ["ds", "orders"]

    def test_none_parts_falls_back_to_default(self):
        result = normalize_metric_subject_path(None, datasource="ds", table_name="orders")
        assert result == ["ds", "orders"]

    def test_no_datasource_returns_parts_unchanged(self):
        parts = ["Finance", "Revenue"]
        result = normalize_metric_subject_path(parts)
        assert result == ["Finance", "Revenue"]

    def test_generic_root_replaced_with_datasource(self):
        for root in GENERIC_METRIC_SUBJECT_ROOTS:
            result = normalize_metric_subject_path([root, "Orders"], datasource="myds")
            assert result[0] == "myds"
            assert result[1] == "Orders"

    def test_same_part_as_datasource_deduplicated(self):
        result = normalize_metric_subject_path(["myds", "Revenue"], datasource="myds")
        assert result == ["myds", "Revenue"]

    def test_datasource_repeated_in_tail_stripped(self):
        result = normalize_metric_subject_path(["myds", "myds", "Revenue"], datasource="myds")
        assert result == ["myds", "Revenue"]

    def test_non_generic_non_datasource_root_returned_as_is(self):
        result = normalize_metric_subject_path(["Finance", "Revenue"], datasource="myds")
        assert result == ["Finance", "Revenue"]

    def test_empty_tail_after_strip_uses_table_name(self):
        result = normalize_metric_subject_path(["metrics"], datasource="myds", table_name="orders")
        assert result == ["myds", "orders"]

    def test_empty_tail_after_strip_no_table_name_uses_unknown(self):
        result = normalize_metric_subject_path(["metrics"], datasource="myds")
        assert result == ["myds", "Unknown"]

    def test_case_insensitive_datasource_match(self):
        result = normalize_metric_subject_path(["MyDS", "Revenue"], datasource="myds")
        assert result[0] == "myds"
        assert result[1] == "Revenue"

    def test_whitespace_only_parts_filtered(self):
        result = normalize_metric_subject_path(["  ", "Finance"], datasource="ds", table_name="orders")
        # "  " gets stripped to "" and filtered, so only "Finance" remains
        assert result == ["Finance"]

    def test_multiple_parts_preserved(self):
        result = normalize_metric_subject_path(["metrics", "Finance", "Q1"], datasource="myds")
        assert result == ["myds", "Finance", "Q1"]


class TestNormalizeMetricSubjectTreeTag:
    def test_non_string_input_returned_as_is(self):
        assert normalize_metric_subject_tree_tag(42) == 42

    def test_tag_without_prefix_returned_as_is(self):
        assert normalize_metric_subject_tree_tag("some_tag") == "some_tag"

    def test_tag_with_prefix_normalized(self):
        result = normalize_metric_subject_tree_tag(
            "subject_tree: metrics/Revenue", datasource="myds", table_name="orders"
        )
        assert result.startswith("subject_tree:")
        assert "myds" in result
        assert "Revenue" in result

    def test_tag_with_prefix_no_datasource(self):
        result = normalize_metric_subject_tree_tag("subject_tree: Finance/Revenue")
        assert result == "subject_tree: Finance/Revenue"

    def test_empty_path_parts_handled(self):
        result = normalize_metric_subject_tree_tag("subject_tree: /Finance//Revenue/")
        assert "Finance" in result
        assert "Revenue" in result


class TestHelpers:
    def test_normalize_part_lowercases_and_strips(self):
        assert _normalize_part("  MyValue  ") == "myvalue"

    def test_normalize_part_empty(self):
        assert _normalize_part("") == ""

    def test_same_part_case_insensitive(self):
        assert _same_part("MyDS", "myds") is True
        assert _same_part("Finance", "Revenue") is False
