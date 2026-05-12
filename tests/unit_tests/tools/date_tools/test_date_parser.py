# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for DateParserTool.

CI-level: zero external deps, zero network, zero API keys.
Mocks LLM model to avoid actual API calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from datus.tools.date_tools.date_parser import DateParserTool


def _make_tool():
    return DateParserTool(language="en")


def _make_zh_tool():
    return DateParserTool(language="zh")


# ---------------------------------------------------------------------------
# TestExecute
# ---------------------------------------------------------------------------


class TestDateParserToolExecute:
    def test_execute_returns_extracted_dates(self):
        tool = _make_tool()
        mock_model = MagicMock()
        mock_dates = [MagicMock()]

        with patch.object(tool, "extract_and_parse_dates", return_value=mock_dates):
            result = tool.execute(
                task_text="Show data for last month",
                current_date="2024-01-15",
                model=mock_model,
            )

        assert result == mock_dates

    def test_execute_returns_empty_list_on_exception(self):
        tool = _make_tool()
        mock_model = MagicMock()

        with patch.object(tool, "extract_and_parse_dates", side_effect=RuntimeError("parse error")):
            result = tool.execute(
                task_text="Show data",
                current_date="2024-01-15",
                model=mock_model,
            )

        assert result == []


# ---------------------------------------------------------------------------
# TestExtractAndParseDates
# ---------------------------------------------------------------------------


class TestExtractAndParseDates:
    def test_returns_empty_when_no_expressions(self):
        tool = _make_tool()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {"expressions": []}

        with patch("datus.tools.date_tools.date_parser.get_date_extraction_prompt", return_value="prompt"):
            with patch("datus.tools.date_tools.date_parser.parse_date_extraction_response", return_value=[]):
                result = tool.extract_and_parse_dates("hello world", "2024-01-15", mock_model)

        assert result == []

    def test_returns_empty_on_exception(self):
        tool = _make_tool()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.side_effect = RuntimeError("LLM error")

        with patch("datus.tools.date_tools.date_parser.get_date_extraction_prompt", return_value="prompt"):
            result = tool.extract_and_parse_dates("hello world", "2024-01-15", mock_model)

        assert result == []

    def test_parses_extracted_expressions(self):
        tool = _make_tool()
        mock_model = MagicMock()

        expressions = [{"original_text": "last month", "date_type": "relative", "confidence": 0.9}]
        mock_extracted_date = MagicMock()

        with patch("datus.tools.date_tools.date_parser.get_date_extraction_prompt", return_value="prompt"):
            with patch("datus.tools.date_tools.date_parser.parse_date_extraction_response", return_value=expressions):
                with patch.object(tool, "parse_temporal_expression", return_value=mock_extracted_date):
                    result = tool.extract_and_parse_dates("last month data", "2024-01-15", mock_model)

        assert len(result) == 1
        assert result[0] == mock_extracted_date


# ---------------------------------------------------------------------------
# TestParseTemporalExpression
# ---------------------------------------------------------------------------


class TestParseTemporalExpression:
    def test_returns_extracted_date_on_success(self):
        tool = _make_tool()
        mock_model = MagicMock()
        reference_date = datetime(2024, 1, 15)
        expression = {"original_text": "last month", "date_type": "relative", "confidence": 0.9}

        start = datetime(2023, 12, 1)
        end = datetime(2023, 12, 31)

        with patch.object(tool, "parse_with_llm", return_value=(start, end)):
            with patch.object(tool, "create_extracted_date") as mock_create:
                mock_create.return_value = MagicMock()
                result = tool.parse_temporal_expression(expression, reference_date, mock_model)

        assert result is mock_create.return_value
        mock_create.assert_called_once_with("last month", "relative", 0.9, start, end)

    def test_returns_none_when_llm_fails(self):
        tool = _make_tool()
        mock_model = MagicMock()
        reference_date = datetime(2024, 1, 15)
        expression = {"original_text": "last year", "date_type": "relative", "confidence": 0.8}

        with patch.object(tool, "parse_with_llm", return_value=None):
            result = tool.parse_temporal_expression(expression, reference_date, mock_model)

        assert result is None

    def test_zh_previous_week_uses_calendar_rule_without_llm(self):
        tool = _make_zh_tool()
        mock_model = MagicMock()
        reference_date = datetime(2025, 2, 15)
        expression = {"original_text": "上周的会议记录", "date_type": "relative", "confidence": 0.9}

        with patch.object(tool, "parse_with_llm") as mock_parse:
            result = tool.parse_temporal_expression(expression, reference_date, mock_model)

        mock_parse.assert_not_called()
        assert result.start_date == "2025-02-03"
        assert result.end_date == "2025-02-09"
        assert result.date_type == "range"

    def test_zh_next_week_uses_calendar_rule_without_llm(self):
        tool = _make_zh_tool()
        mock_model = MagicMock()
        reference_date = datetime(2025, 2, 15)
        expression = {"original_text": "下周的计划", "date_type": "relative", "confidence": 0.9}

        with patch.object(tool, "parse_with_llm") as mock_parse:
            result = tool.parse_temporal_expression(expression, reference_date, mock_model)

        mock_parse.assert_not_called()
        assert result.start_date == "2025-02-17"
        assert result.end_date == "2025-02-23"
        assert result.date_type == "range"

    def test_zh_compound_week_expression_falls_back_to_llm(self):
        tool = _make_zh_tool()
        mock_model = MagicMock()
        reference_date = datetime(2025, 2, 15)
        expression = {"original_text": "从上周到下周", "date_type": "range", "confidence": 0.9}
        start = datetime(2025, 2, 3)
        end = datetime(2025, 2, 23)

        with patch.object(tool, "parse_with_llm", return_value=(start, end)) as mock_parse:
            result = tool.parse_temporal_expression(expression, reference_date, mock_model)

        mock_parse.assert_called_once_with("从上周到下周", reference_date, mock_model)
        assert result.start_date == "2025-02-03"
        assert result.end_date == "2025-02-23"
        assert result.date_type == "range"

    def test_zh_weekday_specific_week_expression_falls_back_to_llm(self):
        tool = _make_zh_tool()
        mock_model = MagicMock()
        reference_date = datetime(2025, 2, 15)
        start = datetime(2025, 2, 17)
        end = datetime(2025, 2, 17)

        for original_text in ("下周一的会议", "上星期三的记录", "下周的周一计划"):
            expression = {"original_text": original_text, "date_type": "relative", "confidence": 0.9}
            with patch.object(tool, "parse_with_llm", return_value=(start, end)) as mock_parse:
                result = tool.parse_temporal_expression(expression, reference_date, mock_model)

            mock_parse.assert_called_once_with(original_text, reference_date, mock_model)
            assert result.parsed_date == "2025-02-17"
            assert result.start_date is None
            assert result.end_date is None


# ---------------------------------------------------------------------------
# TestParseWithLlm
# ---------------------------------------------------------------------------


class TestParseWithLlm:
    def test_returns_start_end_dates_on_success(self):
        tool = _make_tool()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = {
            "start_date": "2023-12-01",
            "end_date": "2023-12-31",
        }
        reference_date = datetime(2024, 1, 15)

        with patch("datus.tools.date_tools.date_parser.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt text"
            result = tool.parse_with_llm("last month", reference_date, mock_model)

        start, end = result
        assert start == datetime(2023, 12, 1)
        assert end == datetime(2023, 12, 31)

    def test_returns_none_when_model_returns_non_dict(self):
        tool = _make_tool()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.return_value = "not a dict"
        reference_date = datetime(2024, 1, 15)

        with patch("datus.tools.date_tools.date_parser.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt"
            result = tool.parse_with_llm("last month", reference_date, mock_model)

        assert result is None

    def test_returns_none_on_exception(self):
        tool = _make_tool()
        mock_model = MagicMock()
        mock_model.generate_with_json_output.side_effect = RuntimeError("LLM error")
        reference_date = datetime(2024, 1, 15)

        with patch("datus.tools.date_tools.date_parser.get_prompt_manager") as mock_gpm:
            mock_pm = mock_gpm.return_value
            mock_pm.render_template.return_value = "prompt"
            result = tool.parse_with_llm("last month", reference_date, mock_model)

        assert result is None


# ---------------------------------------------------------------------------
# TestCreateExtractedDate
# ---------------------------------------------------------------------------


class TestCreateExtractedDate:
    def test_creates_specific_date_when_same_start_and_end(self):
        tool = _make_tool()
        d = datetime(2024, 1, 15)
        result = tool.create_extracted_date("today", "specific", 1.0, d, d)

        assert result.parsed_date == "2024-01-15"
        assert result.start_date is None
        assert result.end_date is None

    def test_creates_range_when_different_start_and_end(self):
        tool = _make_tool()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        result = tool.create_extracted_date("this month", "range", 0.9, start, end)

        assert result.start_date == "2024-01-01"
        assert result.end_date == "2024-01-31"
        assert result.parsed_date is None
        assert result.date_type == "range"

    def test_converts_range_type_to_specific_for_same_dates(self):
        tool = _make_tool()
        d = datetime(2024, 1, 15)
        result = tool.create_extracted_date("2024-01-15", "range", 1.0, d, d)
        # When start==end, date_type should be changed from "range" to "specific"
        assert result.date_type == "specific"


# ---------------------------------------------------------------------------
# TestGenerateDateContext
# ---------------------------------------------------------------------------


class TestGenerateDateContext:
    def test_returns_empty_string_for_no_dates(self):
        tool = _make_tool()
        result = tool.generate_date_context([])
        assert result == ""

    def test_formats_range_date(self):
        from datus.schemas.date_parser_node_models import ExtractedDate

        tool = _make_tool()
        date = ExtractedDate(
            original_text="last month",
            parsed_date=None,
            date_type="range",
            start_date="2023-12-01",
            end_date="2023-12-31",
            confidence=0.9,
        )
        result = tool.generate_date_context([date])
        assert "last month" in result
        assert "2023-12-01" in result
        assert "2023-12-31" in result

    def test_formats_specific_date(self):
        from datus.schemas.date_parser_node_models import ExtractedDate

        tool = _make_tool()
        date = ExtractedDate(
            original_text="today",
            date_type="specific",
            parsed_date="2024-01-15",
            start_date=None,
            end_date=None,
            confidence=1.0,
        )
        result = tool.generate_date_context([date])
        assert "today" in result
        assert "2024-01-15" in result
