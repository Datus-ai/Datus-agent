"""
Test cases for datus/tools/func_tool/base.py
Focuses on trans_to_function_tool parameter filtering for LLM-hallucinated arguments.
"""

import json

import pytest

from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, trans_to_function_tool


class TestTransToFunctionTool:
    """Tests for trans_to_function_tool and its parameter filtering logic."""

    def _make_tool_from_method(self, method):
        """Helper to create a FunctionTool from a bound method."""
        return trans_to_function_tool(method)

    @pytest.mark.asyncio
    async def test_filters_unexpected_parameters(self):
        """LLM-hallucinated parameters should be filtered out silently."""

        class FakeTool:
            def search_table(self, query_text: str, top_n: int = 5) -> FuncToolResult:
                return FuncToolResult(result={"query_text": query_text, "top_n": top_n})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_table)

        # Simulate LLM sending an extra 'database_type' parameter
        args = json.dumps({"query_text": "test query", "database_type": "sqlite"})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"]["query_text"] == "test query"
        assert result["result"]["top_n"] == 5

    @pytest.mark.asyncio
    async def test_valid_parameters_pass_through(self):
        """All valid parameters should be passed through correctly."""

        class FakeTool:
            def search_table(self, query_text: str, top_n: int = 5) -> FuncToolResult:
                return FuncToolResult(result={"query_text": query_text, "top_n": top_n})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_table)

        args = json.dumps({"query_text": "hello", "top_n": 10})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"]["query_text"] == "hello"
        assert result["result"]["top_n"] == 10

    @pytest.mark.asyncio
    async def test_empty_args(self):
        """Empty arguments should work without errors."""

        class FakeTool:
            def no_args_method(self) -> FuncToolResult:
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.no_args_method)

        result = await tool.on_invoke_tool(None, "")
        assert result["success"] == 1
        assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self):
        """Invalid JSON should return an error result."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, "not-valid-json{")
        assert result["success"] == 0
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_multiple_extra_params_all_filtered(self):
        """Multiple hallucinated parameters should all be filtered out."""

        class FakeTool:
            def simple(self, name: str) -> FuncToolResult:
                return FuncToolResult(result=name)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.simple)

        args = json.dumps({"name": "test", "fake1": 1, "fake2": "x", "fake3": True})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"] == "test"


class TestFuncToolListResult:
    """Tests for the canonical list-shaped envelope."""

    def test_defaults_empty_items_and_none_pagination(self):
        env = FuncToolListResult()
        assert env.items == []
        assert env.total is None
        assert env.has_more is None
        assert env.extra is None

    def test_serialization_round_trips_through_funcresult(self):
        env = FuncToolListResult(
            items=[{"id": "1", "name": "foo"}, {"id": "2", "name": "bar"}],
            total=137,
            has_more=True,
            extra={"next_offset": 20},
        )
        outer = FuncToolResult(result=env.model_dump())
        dumped = outer.model_dump(mode="json")

        assert dumped["success"] == 1
        assert dumped["error"] is None
        assert dumped["result"] == {
            "items": [{"id": "1", "name": "foo"}, {"id": "2", "name": "bar"}],
            "total": 137,
            "has_more": True,
            "extra": {"next_offset": 20},
        }

    def test_items_stay_a_list_when_none_passed(self):
        # Pydantic rejects items=None (default_factory returns []), so the
        # "always a list" invariant is enforced at construction time.
        with pytest.raises(ValueError):
            FuncToolListResult(items=None)

    def test_extra_accepts_arbitrary_tool_metadata(self):
        env = FuncToolListResult(
            items=[{"k": "v"}],
            extra={"next_offset": 5, "cursor": "abc", "filters_applied": ["x"]},
        )
        assert env.extra["cursor"] == "abc"
        assert env.extra["filters_applied"] == ["x"]

    def test_empty_items_is_empty_list_not_missing(self):
        env = FuncToolListResult(items=[], total=0, has_more=False)
        dumped = env.model_dump()
        assert dumped["items"] == []
        assert dumped["total"] == 0
        assert dumped["has_more"] is False
