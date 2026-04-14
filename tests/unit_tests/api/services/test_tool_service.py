"""Tests for datus.api.services.tool_service — direct tool dispatch."""

from unittest.mock import MagicMock, patch

import pytest

from datus.api.services.tool_service import ToolService
from datus.tools.func_tool.base import FuncToolResult


@pytest.fixture
def mock_agent_config():
    """Create a mock AgentConfig."""
    return MagicMock()


@pytest.fixture
def mock_context_search_tools():
    """Create a mock ContextSearchTools with expected methods."""
    mock = MagicMock()
    mock.list_subject_tree = MagicMock(return_value=FuncToolResult(success=1, result={"domain": {}}))
    mock.search_metrics = MagicMock(return_value=FuncToolResult(success=1, result=[]))
    mock.get_metrics = MagicMock(return_value=FuncToolResult(success=1, result={}))
    mock.search_reference_sql = MagicMock(return_value=FuncToolResult(success=1, result=[]))
    mock.get_reference_sql = MagicMock(return_value=FuncToolResult(success=1, result={}))
    mock.search_semantic_objects = MagicMock(return_value=FuncToolResult(success=1, result=[]))
    mock.search_knowledge = MagicMock(return_value=FuncToolResult(success=1, result=[]))
    mock.get_knowledge = MagicMock(return_value=FuncToolResult(success=1, result=[]))
    return mock


@pytest.fixture
def tool_service(mock_agent_config, mock_context_search_tools):
    """Create ToolService with mocked dependencies."""
    with patch("datus.api.services.tool_service.ContextSearchTools", return_value=mock_context_search_tools):
        return ToolService(mock_agent_config)


class TestToolRegistry:
    """Tests for tool registry completeness."""

    def test_all_context_tools_registered(self, tool_service):
        """All expected ContextSearchTools methods are registered."""
        expected = [
            "get_knowledge",
            "get_metrics",
            "get_reference_sql",
            "list_subject_tree",
            "search_knowledge",
            "search_metrics",
            "search_reference_sql",
            "search_semantic_objects",
        ]
        assert tool_service.registered_tools == expected

    def test_registered_tools_is_sorted(self, tool_service):
        """registered_tools returns sorted list."""
        tools = tool_service.registered_tools
        assert tools == sorted(tools)


class TestToolServiceExecute:
    """Tests for execute() dispatch."""

    def test_execute_list_subject_tree_no_params(self, tool_service, mock_context_search_tools):
        """execute list_subject_tree with no params succeeds."""
        result = tool_service.execute("list_subject_tree", {})
        assert result.success is True
        assert result.data is not None
        mock_context_search_tools.list_subject_tree.assert_called_once_with()

    def test_execute_search_metrics_with_params(self, tool_service, mock_context_search_tools):
        """execute search_metrics with query_text succeeds."""
        result = tool_service.execute("search_metrics", {"query_text": "revenue"})
        assert result.success is True
        mock_context_search_tools.search_metrics.assert_called_once_with(query_text="revenue")

    def test_execute_search_reference_sql(self, tool_service, mock_context_search_tools):
        """execute search_reference_sql with query_text succeeds."""
        result = tool_service.execute("search_reference_sql", {"query_text": "sales query"})
        assert result.success is True
        mock_context_search_tools.search_reference_sql.assert_called_once_with(query_text="sales query")

    def test_execute_unknown_tool_returns_error(self, tool_service):
        """execute with unknown tool_name returns TOOL_NOT_FOUND error."""
        result = tool_service.execute("nonexistent_tool", {})
        assert result.success is False
        assert result.errorCode == "TOOL_NOT_FOUND"
        assert "nonexistent_tool" in result.errorMessage

    def test_execute_invalid_params_returns_error(self, tool_service):
        """execute with wrong params returns INVALID_PARAMETERS error."""

        # Replace a registered tool with a real function that has strict signature
        def strict_tool(query_text: str) -> FuncToolResult:
            return FuncToolResult(success=1, result=[])

        tool_service._registry["strict_tool"] = strict_tool
        result = tool_service.execute("strict_tool", {"wrong_param": "value"})
        assert result.success is False
        assert result.errorCode == "INVALID_PARAMETERS"

    def test_execute_tool_exception_returns_error(self, tool_service, mock_context_search_tools):
        """execute wraps tool exception in error Result."""
        mock_context_search_tools.search_metrics.side_effect = RuntimeError("boom")
        result = tool_service.execute("search_metrics", {"query_text": "test"})
        assert result.success is False
        assert result.errorCode == "TOOL_EXECUTION_ERROR"
        assert "boom" in result.errorMessage

    def test_execute_search_metrics_with_optional_params(self, tool_service, mock_context_search_tools):
        """execute search_metrics with all optional params succeeds."""
        result = tool_service.execute(
            "search_metrics",
            {"query_text": "revenue", "subject_path": ["Finance"], "top_n": 3},
        )
        assert result.success is True
        mock_context_search_tools.search_metrics.assert_called_once_with(
            query_text="revenue", subject_path=["Finance"], top_n=3
        )

    def test_execute_get_knowledge_with_paths(self, tool_service, mock_context_search_tools):
        """execute get_knowledge with paths param succeeds."""
        paths = [["Finance", "Revenue", "knowledge1"]]
        result = tool_service.execute("get_knowledge", {"paths": paths})
        assert result.success is True
        mock_context_search_tools.get_knowledge.assert_called_once_with(paths=paths)

    def test_execute_returns_func_tool_result(self, tool_service, mock_context_search_tools):
        """execute returns Result[FuncToolResult] with correct data."""
        expected = FuncToolResult(success=1, result={"key": "value"})
        mock_context_search_tools.list_subject_tree.return_value = expected
        result = tool_service.execute("list_subject_tree", {})
        assert result.success is True
        assert result.data.success == 1
        assert result.data.result == {"key": "value"}
