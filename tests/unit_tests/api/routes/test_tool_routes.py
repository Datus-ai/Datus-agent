"""Tests for datus/api/routes/tool_routes.py — direct tool dispatch endpoint."""

from unittest.mock import MagicMock

from datus.api.models.base_models import Result
from datus.api.routes.tool_routes import execute_tool
from datus.tools.func_tool.base import FuncToolResult


def _mock_svc(execute_return=None):
    """Build a mock DatusService whose tool_for() returns a stub tool service."""
    svc = MagicMock()
    if execute_return is None:
        execute_return = Result(success=True, data=FuncToolResult(success=1, result=[]))
    svc.tool_for.return_value.execute.return_value = execute_return
    return svc


class TestExecuteToolRoute:
    """Tests for POST /api/v1/tools/{tool_name}."""

    def test_search_metrics_with_body(self):
        """POST /tools/search_metrics with body calls tool service."""
        svc = _mock_svc()
        result = execute_tool("search_metrics", svc, None, {"query_text": "revenue"})
        assert result.success is True
        svc.tool_for.return_value.execute.assert_called_once_with("search_metrics", {"query_text": "revenue"})

    def test_unknown_tool_returns_result(self):
        """POST /tools/unknown_tool returns error result from service."""
        error_result = Result(
            success=False,
            errorCode="TOOL_NOT_FOUND",
            errorMessage="Tool 'unknown_tool' not found.",
        )
        svc = _mock_svc(execute_return=error_result)
        result = execute_tool("unknown_tool", svc, None, {})
        assert result.success is False
        svc.tool_for.return_value.execute.assert_called_once_with("unknown_tool", {})

    def test_list_subject_tree_empty_body(self):
        """POST /tools/list_subject_tree with empty body passes empty dict."""
        svc = _mock_svc()
        result = execute_tool("list_subject_tree", svc, None, None)
        assert result.success is True
        svc.tool_for.return_value.execute.assert_called_once_with("list_subject_tree", {})

    def test_passes_params_through(self):
        """Body params are passed directly to tool service."""
        svc = _mock_svc()
        params = {"query_text": "test", "top_n": 3, "subject_path": ["Finance"]}
        execute_tool("search_metrics", svc, None, params)
        svc.tool_for.return_value.execute.assert_called_once_with("search_metrics", params)

    def test_unscoped_request_asks_for_the_unscoped_service(self):
        """No sub-agent on the request means the same unscoped service as before."""
        svc = _mock_svc()

        execute_tool("search_metrics", svc, None, {})

        svc.tool_for.assert_called_once_with(None)

    def test_request_sub_agent_selects_the_scoped_service(self):
        """The whole point: the request's sub-agent decides which scope the tool
        reads through. Passing it to `tool_for` is what keeps a consumer of one
        published sub-agent from searching another one's knowledge base.

        `SubAgentDep` has already validated the name by this point, so the route
        never has to consider an unknown one."""
        svc = _mock_svc()

        execute_tool("search_metrics", svc, "revenue_analyst", {})

        svc.tool_for.assert_called_once_with("revenue_analyst")
