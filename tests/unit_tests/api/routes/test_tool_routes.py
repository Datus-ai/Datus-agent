"""Tests for datus/api/routes/tool_routes.py — direct tool dispatch endpoint."""

from unittest.mock import MagicMock

from datus.api.models.base_models import Result
from datus.api.routes.tool_routes import execute_tool
from datus.tools.func_tool.base import FuncToolResult


def _mock_svc(execute_return=None):
    """Build a mock DatusService with tool service."""
    svc = MagicMock()
    if execute_return is None:
        execute_return = Result(success=True, data=FuncToolResult(success=1, result=[]))
    svc.tool.execute.return_value = execute_return
    return svc


class TestExecuteToolRoute:
    """Tests for POST /api/v1/tools/{tool_name}."""

    def test_search_metrics_with_body(self):
        """POST /tools/search_metrics with body calls tool service."""
        svc = _mock_svc()
        result = execute_tool("search_metrics", svc, {"query_text": "revenue"})
        assert result.success is True
        svc.tool.execute.assert_called_once_with("search_metrics", {"query_text": "revenue"})

    def test_unknown_tool_returns_result(self):
        """POST /tools/unknown_tool returns error result from service."""
        error_result = Result(
            success=False,
            errorCode="TOOL_NOT_FOUND",
            errorMessage="Tool 'unknown_tool' not found.",
        )
        svc = _mock_svc(execute_return=error_result)
        result = execute_tool("unknown_tool", svc, {})
        assert result.success is False
        svc.tool.execute.assert_called_once_with("unknown_tool", {})

    def test_list_subject_tree_empty_body(self):
        """POST /tools/list_subject_tree with empty body passes empty dict."""
        svc = _mock_svc()
        result = execute_tool("list_subject_tree", svc, None)
        assert result.success is True
        svc.tool.execute.assert_called_once_with("list_subject_tree", {})

    def test_passes_params_through(self):
        """Body params are passed directly to tool service."""
        svc = _mock_svc()
        params = {"query_text": "test", "top_n": 3, "subject_path": ["Finance"]}
        execute_tool("search_metrics", svc, params)
        svc.tool.execute.assert_called_once_with("search_metrics", params)
