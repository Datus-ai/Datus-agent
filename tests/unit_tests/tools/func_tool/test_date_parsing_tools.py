# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for DateParsingTools - CI level, zero external dependencies."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_agent_config():
    config = Mock()
    config.nodes = {}
    return config


@pytest.fixture
def mock_model():
    return Mock()


@pytest.fixture
def date_parsing_tools(mock_agent_config, mock_model):
    with patch("datus.tools.func_tool.date_parsing_tools.DateParserTool") as mock_parser_cls:
        mock_parser_cls.return_value = Mock()
        from datus.tools.func_tool.date_parsing_tools import DateParsingTools

        tool = DateParsingTools(agent_config=mock_agent_config, model=mock_model)
        tool.date_parser_tool = mock_parser_cls.return_value
        return tool


class TestGetLanguageSetting:
    def test_returns_en_when_no_nodes(self, mock_agent_config, mock_model):
        with patch("datus.tools.func_tool.date_parsing_tools.DateParserTool"):
            from datus.tools.func_tool.date_parsing_tools import DateParsingTools

            tool = DateParsingTools(agent_config=mock_agent_config, model=mock_model)
        assert tool._get_language_setting() == "en"

    def test_returns_en_when_no_date_parser_key(self, mock_model):
        config = Mock()
        config.nodes = {"other_node": Mock()}
        with patch("datus.tools.func_tool.date_parsing_tools.DateParserTool"):
            from datus.tools.func_tool.date_parsing_tools import DateParsingTools

            tool = DateParsingTools(agent_config=config, model=mock_model)
        assert tool._get_language_setting() == "en"

    def test_returns_language_from_config(self, mock_model):
        config = Mock()
        date_parser_cfg = Mock()
        date_parser_cfg.input = Mock()
        date_parser_cfg.input.language = "zh"
        config.nodes = {"date_parser": date_parser_cfg}
        with patch("datus.tools.func_tool.date_parsing_tools.DateParserTool"):
            from datus.tools.func_tool.date_parsing_tools import DateParsingTools

            tool = DateParsingTools(agent_config=config, model=mock_model)
        assert tool._get_language_setting() == "zh"

    def test_returns_en_when_agent_config_none(self, mock_model):
        with patch("datus.tools.func_tool.date_parsing_tools.DateParserTool"):
            from datus.tools.func_tool.date_parsing_tools import DateParsingTools

            tool = DateParsingTools(agent_config=None, model=mock_model)
        assert tool._get_language_setting() == "en"


class TestSetReferenceDate:
    def test_set_reference_date(self, date_parsing_tools):
        assert date_parsing_tools.reference_date is None
        date_parsing_tools.set_reference_date("2024-01-15")
        assert date_parsing_tools.reference_date == "2024-01-15"

    def test_set_reference_date_none(self, date_parsing_tools):
        date_parsing_tools.set_reference_date("2024-01-15")
        date_parsing_tools.set_reference_date(None)
        assert date_parsing_tools.reference_date is None


class TestAvailableTools:
    def test_available_tools_returns_one_tool(self, date_parsing_tools):
        with patch("datus.tools.func_tool.date_parsing_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = date_parsing_tools.available_tools()
        assert len(tools) == 1


class TestParseTemporalExpressions:
    def test_success(self, date_parsing_tools):
        mock_date = Mock()
        mock_date.model_dump.return_value = {"expression": "last month", "start": "2024-01-01", "end": "2024-01-31"}
        date_parsing_tools.date_parser_tool.execute.return_value = [mock_date]
        date_parsing_tools.date_parser_tool.generate_date_context.return_value = "From 2024-01-01 to 2024-01-31"

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-02-01"):
            result = date_parsing_tools.parse_temporal_expressions("last month", "2024-02-01")

        assert result.success == 1
        assert result.error is None
        assert "extracted_dates" in result.result
        assert "date_context" in result.result
        assert len(result.result["extracted_dates"]) == 1

    def test_success_no_current_date(self, date_parsing_tools):
        date_parsing_tools.date_parser_tool.execute.return_value = []
        date_parsing_tools.date_parser_tool.generate_date_context.return_value = ""

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-02-15"):
            result = date_parsing_tools.parse_temporal_expressions("today")

        assert result.success == 1

    def test_uses_reference_date_as_fallback(self, date_parsing_tools):
        """When current_date is None, parse_temporal_expressions should use reference_date."""
        date_parsing_tools.set_reference_date("2023-06-15")
        date_parsing_tools.date_parser_tool.execute.return_value = []
        date_parsing_tools.date_parser_tool.generate_date_context.return_value = ""

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2023-06-15") as mock_get_date:
            result = date_parsing_tools.parse_temporal_expressions("last month", current_date=None)

        mock_get_date.assert_called_once_with("2023-06-15")
        assert result.success == 1

    def test_exception_returns_failure(self, date_parsing_tools):
        date_parsing_tools.date_parser_tool.execute.side_effect = Exception("parser error")

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-02-01"):
            result = date_parsing_tools.parse_temporal_expressions("last month")

        assert result.success == 0
        assert "parser error" in result.error


class TestGetCurrentDate:
    def test_returns_current_date_no_reference(self, date_parsing_tools):
        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-03-15"):
            result = date_parsing_tools.get_current_date()

        assert result.success == 1
        assert result.result["current_date"] == "2024-03-15"
        assert result.result["is_reference_date"] is False

    def test_returns_reference_date_when_set(self, date_parsing_tools):
        date_parsing_tools.set_reference_date("2023-12-31")

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2023-12-31"):
            result = date_parsing_tools.get_current_date()

        assert result.success == 1
        assert result.result["current_date"] == "2023-12-31"
        assert result.result["is_reference_date"] is True

    def test_exception_returns_failure(self, date_parsing_tools):
        with patch(
            "datus.utils.time_utils.get_default_current_date",
            side_effect=Exception("time error"),
        ):
            result = date_parsing_tools.get_current_date()

        assert result.success == 0
        assert "time error" in result.error
