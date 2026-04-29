"""
Test cases for datus/tools/func_tool/base.py
Focuses on trans_to_function_tool parameter filtering for LLM-hallucinated arguments.
"""

import json
import logging

import pytest

from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, parse_tool_args, trans_to_function_tool


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
        """Truly unrepairable JSON should return an error result."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, "totally broken garbage @@!!")
        assert result["success"] == 0
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_malformed_json_repaired_by_json_repair(self):
        """GLM-style malformed JSON (unquoted string values) should be repaired."""

        class FakeTool:
            def search_document(self, keywords: str, platform: str = "") -> FuncToolResult:
                return FuncToolResult(result={"keywords": keywords, "platform": platform})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_document)

        malformed = '{"keywords": PERCENTILE function, "platform": "starrocks"}'
        result = await tool.on_invoke_tool(None, malformed)

        assert result["success"] == 1
        assert result["result"]["keywords"] == "PERCENTILE function"
        assert result["result"]["platform"] == "starrocks"

    @pytest.mark.asyncio
    async def test_malformed_json_repair_logs_warning(self, caplog):
        """json_repair fallback should log a warning when repairing."""

        class FakeTool:
            def search_document(self, keywords: str, platform: str = "") -> FuncToolResult:
                return FuncToolResult(result={"keywords": keywords, "platform": platform})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_document)

        malformed = '{"keywords": PERCENTILE function, "platform": "starrocks"}'
        with caplog.at_level(logging.WARNING, logger="datus.tools.func_tool.base"):
            result = await tool.on_invoke_tool(None, malformed)

        assert result["success"] == 1
        assert any("Repaired malformed JSON" in msg for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_repair_not_valid_json_brace(self):
        """'not-valid-json{' repairs to {} but should fail on missing required params."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, "not-valid-json{")
        assert result["success"] == 0
        assert "missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_repair_truncated_key_value(self):
        """'{"x":' repairs to {"x": ""} but should fail on missing required params."""

        class FakeTool:
            def some_method(self, name: str) -> FuncToolResult:
                return FuncToolResult(result=name)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, '{"x":')
        assert result["success"] == 0
        assert "missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_repair_empty_dict(self):
        """'{}' is valid JSON but should fail when required params are missing."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, "{}")
        assert result["success"] == 0
        assert "missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_repair_array_not_dict(self):
        """'[1,2,3]' is valid JSON but not a dict — should return error."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)

        result = await tool.on_invoke_tool(None, "[1,2,3]")
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


class TestParseToolArgs:
    """Direct tests for parse_tool_args covering branches not reachable via trans_to_function_tool."""

    def test_empty_args_with_required_fields(self):
        """Empty args_str with required_fields should return error."""
        result, error = parse_tool_args("", required_fields={"x"}, tool_name="test")
        assert error is not None
        assert "missing required fields" in error
        assert result == {}

    def test_none_args_with_required_fields(self):
        """None args_str with required_fields should return error."""
        result, error = parse_tool_args(None, required_fields={"x"}, tool_name="test")
        assert error is not None
        assert "missing required fields" in error

    def test_whitespace_only_args_with_required_fields(self):
        """Whitespace-only string with required_fields should return error."""
        result, error = parse_tool_args("   ", required_fields={"x"}, tool_name="test")
        assert error is not None
        assert "missing required fields" in error

    def test_non_string_dict_input(self):
        """Non-string dict-like input should be converted."""
        result, error = parse_tool_args({"key": "val"}, tool_name="test")
        assert error is None
        assert result == {"key": "val"}

    def test_non_string_dict_input_missing_required(self):
        """Non-string dict input missing required fields should return error."""
        result, error = parse_tool_args({"a": 1}, required_fields={"b"}, tool_name="test")
        assert error is not None
        assert "missing required fields" in error

    def test_non_string_invalid_input(self):
        """Non-string, non-dict input should return error."""
        result, error = parse_tool_args(12345, tool_name="test")
        assert error is not None
        assert "Invalid arguments" in error

    def test_json_repair_exception_falls_through(self, monkeypatch):
        """When json_repair itself raises, should fall through to Invalid JSON error."""
        import json_repair as _jr

        monkeypatch.setattr(_jr, "loads", lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom")))
        result, error = parse_tool_args("{bad json}", tool_name="test")
        assert error is not None
        assert "Invalid JSON" in error

    def test_truncated_json_hint(self, monkeypatch):
        """Truncated JSON not ending in } or ] should include truncation hint."""
        import json_repair as _jr

        monkeypatch.setattr(_jr, "loads", lambda *a, **kw: None)
        result, error = parse_tool_args('{"key": "val', tool_name="test")
        assert error is not None
        assert "truncated" in error.lower()

    def test_repair_succeeds_no_truncation_hint(self):
        """json_repair fixes truncated JSON — no error expected."""
        result, error = parse_tool_args('{"key": "val', tool_name="test")
        assert error is None
        assert result == {"key": "val"}


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
