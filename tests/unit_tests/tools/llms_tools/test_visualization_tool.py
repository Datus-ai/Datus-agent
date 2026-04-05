# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/llms_tools/visualization_tool.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from datus.schemas.visualization import VisualizationInput, VisualizationOutput
from datus.tools.llms_tools.visualization_tool import VisualizationTool


def _make_tool(model=None):
    tool = VisualizationTool(model=model)
    return tool


def _make_input(data):
    return VisualizationInput(data=data)


class TestVisualizationToolInit:
    def test_init_with_explicit_model(self):
        mock_model = MagicMock()
        tool = VisualizationTool(model=mock_model)
        assert tool.model is mock_model

    def test_init_without_model_sets_none(self):
        tool = VisualizationTool()
        assert tool.model is None

    def test_init_with_agent_config_creates_model(self):
        mock_config = MagicMock()
        mock_model = MagicMock()

        with patch("datus.tools.llms_tools.visualization_tool.LLMBaseModel") as mock_base:
            mock_base.create_model.return_value = mock_model
            tool = VisualizationTool(agent_config=mock_config)

        assert tool.model is mock_model

    def test_init_with_agent_config_failing_model_sets_none(self):
        mock_config = MagicMock()

        with patch("datus.tools.llms_tools.visualization_tool.LLMBaseModel") as mock_base:
            mock_base.create_model.side_effect = Exception("no key")
            tool = VisualizationTool(agent_config=mock_config)

        assert tool.model is None

    def test_custom_parameters(self):
        tool = VisualizationTool(preview_rows=3, max_preview_char=500, max_y_cols=2, max_pie_categories=5)
        assert tool.preview_rows == 3
        assert tool.max_preview_char == 500
        assert tool.max_y_cols == 2
        assert tool.max_pie_categories == 5


class TestConvertToDataframe:
    def test_none_returns_none(self):
        tool = _make_tool()
        assert tool._convert_to_dataframe(None) is None

    def test_dataframe_returns_copy(self):
        tool = _make_tool()
        df = pd.DataFrame({"a": [1, 2]})
        result = tool._convert_to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert result is not df  # should be a copy

    def test_pyarrow_table_converted(self):
        tool = _make_tool()
        arrow = pa.table({"a": [1, 2], "b": [3, 4]})
        result = tool._convert_to_dataframe(arrow)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_list_of_dicts_converted(self):
        tool = _make_tool()
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = tool._convert_to_dataframe(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_empty_list_returns_empty_dataframe(self):
        tool = _make_tool()
        result = tool._convert_to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_unsupported_type_returns_none(self):
        tool = _make_tool()
        result = tool._convert_to_dataframe("not supported")
        assert result is None


class TestNormalizeChartType:
    def test_empty_returns_unknown(self):
        tool = _make_tool()
        assert tool._normalize_chart_type("") == "Unknown"

    def test_bar_normalized(self):
        tool = _make_tool()
        assert tool._normalize_chart_type("bar") == "Bar Chart"
        assert tool._normalize_chart_type("Bar Chart") == "Bar Chart"

    def test_line_normalized(self):
        tool = _make_tool()
        assert tool._normalize_chart_type("line") == "Line Chart"
        assert tool._normalize_chart_type("Line Chart") == "Line Chart"

    def test_scatter_normalized(self):
        tool = _make_tool()
        assert tool._normalize_chart_type("scatter") == "Scatter Plot"
        assert tool._normalize_chart_type("Scatter Plot") == "Scatter Plot"

    def test_pie_normalized(self):
        tool = _make_tool()
        assert tool._normalize_chart_type("pie") == "Pie Chart"
        assert tool._normalize_chart_type("Pie Chart") == "Pie Chart"

    def test_unknown_value_returns_unknown(self):
        tool = _make_tool()
        assert tool._normalize_chart_type("heatmap") == "Unknown"


class TestRuleBasedRecommendation:
    def test_line_chart_with_datetime_and_numeric(self):
        tool = _make_tool()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "revenue": [100, 200, 300],
            }
        )
        result = tool._rule_based_recommendation(df)
        assert result.chart_type == "Line Chart"
        assert result.x_col == "date"
        assert "revenue" in result.y_cols

    def test_pie_chart_with_small_categories(self):
        tool = _make_tool()
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "value": [10, 20, 30],
            }
        )
        result = tool._rule_based_recommendation(df)
        assert result.chart_type == "Pie Chart"

    def test_bar_chart_with_categorical_and_numeric(self):
        tool = _make_tool()
        df = pd.DataFrame(
            {
                "region": [
                    "North",
                    "South",
                    "East",
                    "West",
                    "Center",
                    "Northeast",
                    "Southwest",
                    "Northwest",
                    "Southeast",
                ],
                "sales": [100, 200, 300, 400, 500, 600, 700, 800, 900],
            }
        )
        result = tool._rule_based_recommendation(df)
        assert result.chart_type == "Bar Chart"

    def test_scatter_plot_with_two_numerics(self):
        tool = _make_tool()
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
            }
        )
        result = tool._rule_based_recommendation(df)
        assert result.chart_type == "Scatter Plot"

    def test_unknown_chart_for_single_column(self):
        tool = _make_tool()
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        result = tool._rule_based_recommendation(df)
        assert result.chart_type == "Unknown"

    def test_result_is_visualization_output(self):
        tool = _make_tool()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = tool._rule_based_recommendation(df)
        assert isinstance(result, VisualizationOutput)
        assert result.success is True


class TestExecute:
    def test_raises_type_error_for_wrong_input(self):
        tool = _make_tool()
        with pytest.raises(TypeError):
            tool.execute("not a VisualizationInput")

    def test_returns_error_for_none_data(self):
        tool = _make_tool()
        # VisualizationInput doesn't accept None directly; test via _convert_to_dataframe
        assert tool._convert_to_dataframe(None) is None
        # And for unsupported types the tool's error output path
        result = tool._error_output("no data", "Unknown")
        assert result.success is False

    def test_returns_error_for_empty_dataframe(self):
        tool = _make_tool()
        inp = _make_input(pd.DataFrame())
        result = tool.execute(inp)
        assert result.success is False

    def test_uses_heuristic_when_no_model(self):
        tool = _make_tool(model=None)
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = tool.execute(_make_input(df))
        assert isinstance(result, VisualizationOutput)
        assert result.success is True

    def test_uses_llm_when_model_available(self):
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "chart_type": "Line Chart",
            "x_col": "date",
            "y_cols": ["revenue"],
            "reason": "temporal trend",
        }
        tool = _make_tool(model=mock_model)
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "revenue": [100, 200],
            }
        )

        with patch("datus.tools.llms_tools.visualization_tool.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt text"
            result = tool.execute(_make_input(df))

        assert result.success is True

    def test_falls_back_to_heuristic_when_llm_fails(self):
        mock_model = MagicMock()
        mock_model.generate_with_json_output.side_effect = Exception("LLM error")
        tool = _make_tool(model=mock_model)
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        with patch("datus.tools.llms_tools.visualization_tool.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt"
            result = tool.execute(_make_input(df))

        assert result.success is True

    def test_list_data_input(self):
        tool = _make_tool()
        data = [{"cat": "A", "val": 10}, {"cat": "B", "val": 20}, {"cat": "C", "val": 30}]
        result = tool.execute(_make_input(data))
        assert isinstance(result, VisualizationOutput)
        assert result.success is True

    def test_pyarrow_table_input(self):
        tool = _make_tool()
        arrow = pa.table({"a": [1, 2], "b": [3, 4]})
        result = tool.execute(_make_input(arrow))
        assert result.success is True


class TestFormatHelpers:
    def test_format_columns_info(self):
        tool = _make_tool()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        info = tool._format_columns_info(df)
        assert "a" in info
        assert "b" in info

    def test_format_data_preview_truncation(self):
        tool = VisualizationTool(max_preview_char=20)
        df = pd.DataFrame({"col": list(range(100))})
        preview = tool._format_data_preview(df)
        assert "(truncated)" in preview

    def test_serialize_value_numpy_int(self):
        tool = _make_tool()
        val = np.int64(42)
        result = tool._serialize_value(val)
        assert result == 42
        assert isinstance(result, int)

    def test_serialize_value_bytes(self):
        tool = _make_tool()
        result = tool._serialize_value(b"hello")
        assert result == "hello"

    def test_serialize_value_set(self):
        tool = _make_tool()
        result = tool._serialize_value({1, 2, 3})
        assert isinstance(result, list)

    def test_serialize_value_timestamp(self):
        tool = _make_tool()
        ts = pd.Timestamp("2024-01-01")
        result = tool._serialize_value(ts)
        assert "2024" in result

    def test_default_reason_line_chart(self):
        tool = _make_tool()
        reason = tool._default_reason("Line Chart", "date", ["revenue"])
        assert "Line chart" in reason

    def test_default_reason_unknown(self):
        tool = _make_tool()
        reason = tool._default_reason("Unknown", "", [])
        assert "Unable" in reason


class TestSelectHelpers:
    def test_select_numeric_metrics_respects_max_y_cols(self):
        tool = VisualizationTool(max_y_cols=2)
        cols = ["a", "b", "c", "d"]
        result = tool._select_numeric_metrics(cols)
        assert len(result) == 2

    def test_select_numeric_metrics_excludes_specified(self):
        tool = _make_tool()
        result = tool._select_numeric_metrics(["a", "b", "c"], exclude={"a"})
        assert "a" not in result

    def test_sanitize_y_cols_returns_valid_cols(self):
        tool = _make_tool()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = tool._sanitize_y_cols(df, ["a", "b", "missing"])
        assert "missing" not in result
        assert "a" in result

    def test_sanitize_y_cols_falls_back_to_numeric(self):
        tool = _make_tool()
        df = pd.DataFrame({"num": [1, 2], "cat": ["x", "y"]})
        result = tool._sanitize_y_cols(df, ["missing_col"])
        # Should fall back to numeric columns
        assert "num" in result

    def test_select_dimension_prefers_categorical(self):
        tool = _make_tool()
        df = pd.DataFrame({"num": [1, 2], "cat": ["x", "y"]})
        result = tool._select_dimension(df)
        assert result == "cat"

    def test_select_dimension_excludes_specified(self):
        tool = _make_tool()
        df = pd.DataFrame({"cat": ["x", "y"], "other": ["a", "b"]})
        result = tool._select_dimension(df, exclude={"cat"})
        assert result != "cat"


class TestLlmBasedRecommendation:
    def test_llm_non_dict_response_returns_none(self):
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = "not a dict"
        tool = _make_tool(model=mock_model)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with patch("datus.tools.llms_tools.visualization_tool.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt"
            result = tool._llm_based_recommendation(df)

        assert result is None

    def test_llm_response_with_string_y_cols(self):
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "chart_type": "Bar Chart",
            "x_col": "cat",
            "y_cols": "val",  # string instead of list
            "reason": "bar",
        }
        tool = _make_tool(model=mock_model)
        df = pd.DataFrame({"cat": ["A", "B"], "val": [1, 2]})

        with patch("datus.tools.llms_tools.visualization_tool.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt"
            result = tool._llm_based_recommendation(df)

        assert result is not None
        assert isinstance(result.y_cols, list)

    def test_llm_invalid_x_col_falls_back(self):
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "chart_type": "Bar Chart",
            "x_col": "nonexistent_col",
            "y_cols": ["val"],
            "reason": "bar",
        }
        tool = _make_tool(model=mock_model)
        df = pd.DataFrame({"cat": ["A", "B"], "val": [1, 2]})

        with patch("datus.tools.llms_tools.visualization_tool.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt"
            result = tool._llm_based_recommendation(df)

        # x_col should fall back to a column that exists
        assert result.x_col in df.columns or result.x_col == ""
