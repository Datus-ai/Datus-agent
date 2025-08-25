from typing import Any, Dict, List

import pytest

from datus.agent.node.date_parser_node import DateParserNode
from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)


@pytest.fixture
def chinese_expressions_test_cases() -> List[Dict[str, Any]]:
    """Test cases for Chinese temporal expressions"""
    return [
        {
            "text": "未来三个月内的数据",
            "reference": "2025-01-01",
            "expected_start": "2025-01-01",
            "expected_end": "2025-04-01",
            "description": "未来三个月内",
        },
        {
            "text": "最近三个月的销售情况",
            "reference": "2025-01-01",
            "expected_start": "2024-10-01",
            "expected_end": "2025-01-01",
            "description": "最近三个月",
        },
        {
            "text": "下个月的业绩目标",
            "reference": "2025-01-15",
            "expected_start": "2025-02-01",
            "expected_end": "2025-02-28",
            "description": "下个月",
        },
        {
            "text": "上周的会议记录",
            "reference": "2025-02-15",
            "expected_start": "2025-02-03",
            "expected_end": "2025-02-09",
            "description": "上周",
        },
        {
            "text": "今年的营收报告",
            "reference": "2025-06-15",
            "expected_start": "2025-01-01",
            "expected_end": "2025-12-31",
            "description": "今年",
        },
        {
            "text": "接下来两周的计划",
            "reference": "2025-02-15",
            "expected_start": "2025-02-15",
            "expected_end": "2025-03-01",
            "description": "接下来两周",
        },
    ]


@pytest.fixture
def english_expressions_test_cases() -> List[Dict[str, Any]]:
    """Test cases for English temporal expressions"""
    return [
        {
            "text": "next 3 months performance data",
            "reference": "2025-01-01",
            "expected_start": "2025-01-01",
            "expected_end": "2025-04-01",
            "description": "next 3 months",
        },
        {
            "text": "last 6 months sales report",
            "reference": "2025-01-01",
            "expected_start": "2024-07-01",
            "expected_end": "2025-01-01",
            "description": "last 6 months",
        },
        {
            "text": "yesterday's meeting notes",
            "reference": "2025-01-15",
            "expected_start": "2025-01-14",
            "expected_end": "2025-01-14",
            "description": "yesterday",
        },
        {
            "text": "this year's budget",
            "reference": "2025-06-15",
            "expected_start": "2025-01-01",
            "expected_end": "2025-12-31",
            "description": "this year",
        },
    ]


@pytest.fixture
def mixed_expressions_test_cases() -> List[Dict[str, Any]]:
    """Test cases for mixed or complex expressions"""
    return [
        {
            "text": "从上个月到下个月的趋势分析",
            "reference": "2025-01-15",
            "expected_start": "2024-12-01",
            "expected_end": "2025-02-28",
            "description": "从上个月到下个月",
        },
        {
            "text": "Q1 2025财报",
            "reference": "2025-06-01",
            "expected_start": "2025-01-01",
            "expected_end": "2025-03-31",
            "description": "Q1 2025",
        },
        {
            "text": "2024年底到现在的数据对比",
            "reference": "2025-01-15",
            "expected_start": "2024-12-31",
            "expected_end": "2025-01-15",
            "description": "2024年底到现在",
        },
    ]


@pytest.fixture
def agent_config():
    """Load agent configuration"""
    return load_acceptance_config()


def _create_date_parser(agent_config, language: str):
    """Helper function to create date parser instance with specified language"""
    try:
        model = LLMBaseModel.create_model(agent_config)
        # Set language for date parsing
        if not hasattr(agent_config, "nodes"):
            agent_config.nodes = {}

        # Create a mock NodeConfig with input containing language setting
        mock_input = type("DateParserInput", (), {"language": language})()
        mock_node_config = type("NodeConfig", (), {"input": mock_input})()
        agent_config.nodes["date_parser"] = mock_node_config

        parser = DateParserNode(
            node_id="test_date_parser",
            description="Test date parser node",
            node_type="date_parser",
            agent_config=agent_config,
        )
        parser.model = model
        return parser
    except Exception as e:
        pytest.skip(f"Date parser initialization failed: {e}")


@pytest.fixture
def date_parser_en(agent_config):
    """Create English date parser instance"""
    return _create_date_parser(agent_config, "en")


@pytest.fixture
def date_parser(agent_config):
    """Create Chinese date parser instance"""
    return _create_date_parser(agent_config, "cn")


class TestDateParser:
    """Test suite for DateParser class"""

    def test_chinese_expressions(self, chinese_expressions_test_cases, date_parser):
        """Test Chinese temporal expressions parsing"""
        for test_case in chinese_expressions_test_cases:
            results = date_parser._extract_and_parse_dates(test_case["text"], test_case["reference"])

            assert results is not None, f"No results for: {test_case['description']}"
            assert len(results) > 0, f"Empty results for: {test_case['description']}"

            result = results[0]

            if result.date_type == "range":
                actual_start = result.start_date
                actual_end = result.end_date
            else:
                actual_start = result.parsed_date
                actual_end = result.parsed_date

            assert actual_start == test_case["expected_start"], (
                f"Start date mismatch for {test_case['description']}: "
                f"expected {test_case['expected_start']}, got {actual_start}"
            )
            assert actual_end == test_case["expected_end"], (
                f"End date mismatch for {test_case['description']}: "
                f"expected {test_case['expected_end']}, got {actual_end}"
            )

    def test_english_expressions(self, english_expressions_test_cases, date_parser_en):
        """Test English temporal expressions parsing"""
        for test_case in english_expressions_test_cases:
            results = date_parser_en._extract_and_parse_dates(test_case["text"], test_case["reference"])

            assert results is not None, f"No results for: {test_case['description']}"
            assert len(results) > 0, f"Empty results for: {test_case['description']}"

            result = results[0]

            if result.date_type == "range":
                actual_start = result.start_date
                actual_end = result.end_date
            else:
                actual_start = result.parsed_date
                actual_end = result.parsed_date

            assert actual_start == test_case["expected_start"], (
                f"Start date mismatch for {test_case['description']}: "
                f"expected {test_case['expected_start']}, got {actual_start}"
            )
            assert actual_end == test_case["expected_end"], (
                f"End date mismatch for {test_case['description']}: "
                f"expected {test_case['expected_end']}, got {actual_end}"
            )

    def test_mixed_expressions(self, mixed_expressions_test_cases, date_parser):
        """Test mixed temporal expressions parsing"""
        for test_case in mixed_expressions_test_cases:
            results = date_parser._extract_and_parse_dates(test_case["text"], test_case["reference"])

            assert results is not None, f"No results for: {test_case['description']}"
            assert len(results) > 0, f"Empty results for: {test_case['description']}"

            result = results[0]

            if result.date_type == "range":
                actual_start = result.start_date
                actual_end = result.end_date
            else:
                actual_start = result.parsed_date
                actual_end = result.parsed_date

            assert actual_start == test_case["expected_start"], (
                f"Start date mismatch for {test_case['description']}: "
                f"expected {test_case['expected_start']}, got {actual_start}"
            )
            assert actual_end == test_case["expected_end"], (
                f"End date mismatch for {test_case['description']}: "
                f"expected {test_case['expected_end']}, got {actual_end}"
            )
