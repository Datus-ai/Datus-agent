# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/llms_tools/visualization_messages.py"""

import pytest

from datus.tools.llms_tools.visualization_messages import (
    _MESSAGES,
    empty_dataset_reason,
    normalize_language,
    reason_for_chart,
)

_KEYS = {"line", "bar", "pie", "scatter", "unknown", "empty"}
_CHART_TYPES = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Plot"]


class TestMessageTable:
    def test_every_language_defines_every_key(self):
        for code, table in _MESSAGES.items():
            assert set(table) == _KEYS, f"{code} is missing or has extra keys"

    def test_axis_templates_use_both_placeholders(self):
        for code, table in _MESSAGES.items():
            for key in ("line", "bar", "pie", "scatter"):
                assert "{x}" in table[key], f"{code}/{key} drops the dimension"
                assert "{y}" in table[key], f"{code}/{key} drops the metrics"

    def test_context_free_templates_take_no_placeholders(self):
        for code, table in _MESSAGES.items():
            for key in ("unknown", "empty"):
                assert "{" not in table[key], f"{code}/{key} has an unfilled placeholder"


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("en", "en"),
            ("zh", "zh"),
            ("zh-CN", "zh"),
            ("zh_CN", "zh"),
            ("ZH-cn", "zh"),
            ("  zh  ", "zh"),
            ("zh-TW", "zh"),
            ("zh-Hant", "zh"),
            ("en-GB", "en"),
        ],
    )
    def test_known_tags_and_regional_variants(self, code, expected):
        assert normalize_language(code) == expected

    @pytest.mark.parametrize("code", [None, "", "   ", "klingon", "ja", "pt-BR"])
    def test_unknown_or_missing_falls_back_to_english(self, code):
        assert normalize_language(code) == "en"


class TestReasonForChart:
    def test_renders_axes_into_the_requested_language(self):
        reason = reason_for_chart("Bar Chart", "zh-CN", "region", ["sales", "profit"])
        assert "region" in reason
        assert "sales, profit" in reason
        assert reason == _MESSAGES["zh"]["bar"].format(x="region", y="sales, profit")

    @pytest.mark.parametrize("chart_type", _CHART_TYPES)
    def test_every_chart_type_localizes(self, chart_type):
        en = reason_for_chart(chart_type, "en", "date", ["revenue"])
        zh = reason_for_chart(chart_type, "zh", "date", ["revenue"])
        assert en != zh
        assert "date" in en and "date" in zh
        assert "revenue" in en and "revenue" in zh
        assert "{" not in zh

    def test_unknown_language_falls_back_to_english_wording(self):
        assert reason_for_chart("Line Chart", "klingon", "date", ["revenue"]) == reason_for_chart(
            "Line Chart", "en", "date", ["revenue"]
        )

    @pytest.mark.parametrize(
        "chart_type,x_col,y_cols",
        [
            ("Unknown", "date", ["revenue"]),
            ("Heatmap", "date", ["revenue"]),
            ("Bar Chart", "", ["revenue"]),
            ("Bar Chart", "region", []),
        ],
    )
    def test_incomplete_picks_use_the_unknown_wording(self, chart_type, x_col, y_cols):
        assert reason_for_chart(chart_type, "zh", x_col, y_cols) == _MESSAGES["zh"]["unknown"]

    def test_non_string_labels_render_instead_of_raising(self):
        """pandas column labels are not necessarily str — joining them blindly
        used to raise TypeError before the caller could report anything."""
        reason = reason_for_chart("Scatter Plot", "en", 1, [2])
        assert "1" in reason and "2" in reason
        assert reason == _MESSAGES["en"]["scatter"].format(x="1", y="2")

    def test_a_label_named_zero_is_a_label_not_a_missing_axis(self):
        reason = reason_for_chart("Bar Chart", "zh", 0, [0])
        assert reason == _MESSAGES["zh"]["bar"].format(x="0", y="0")
        assert reason != _MESSAGES["zh"]["unknown"]

    @pytest.mark.parametrize("x_col,y_cols", [(None, ["v"]), ("", ["v"]), ("region", [None]), ("region", [])])
    def test_absent_axes_still_use_the_unknown_wording(self, x_col, y_cols):
        assert reason_for_chart("Bar Chart", "en", x_col, y_cols) == _MESSAGES["en"]["unknown"]

    def test_defaults_to_english_when_language_omitted(self):
        assert reason_for_chart("Pie Chart", x_col="region", y_cols=["sales"]) == _MESSAGES["en"]["pie"].format(
            x="region", y="sales"
        )


class TestEmptyDatasetReason:
    def test_localized(self):
        assert empty_dataset_reason("zh-CN") == _MESSAGES["zh"]["empty"]

    def test_defaults_to_english(self):
        assert empty_dataset_reason() == _MESSAGES["en"]["empty"]
