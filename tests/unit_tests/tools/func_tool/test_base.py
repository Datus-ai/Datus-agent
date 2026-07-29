"""
Test cases for datus/tools/func_tool/base.py
Focuses on trans_to_function_tool parameter filtering for LLM-hallucinated arguments.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest
from agents import Agent, Runner, SQLiteSession
from agents.items import ToolCallItem
from agents.models.interface import Model
from agents.run import RunConfig
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, trans_to_function_tool
from datus.tools.permission.permission_hooks import PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.permission.profiles import get_profile
from datus.tools.registry.tool_registry import ToolRegistry


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
    async def test_repairs_arguments_for_replay_without_executing(self):
        """Repair malformed arguments for replay, then execute only a valid retry."""

        class FakeTool:
            def __init__(self):
                self.received = None

            def execute_sql(
                self,
                sql: str,
                datasource: str = "",
                database: str = "",
                min_rows: Optional[int] = None,
                max_rows: Optional[int] = None,
            ) -> FuncToolResult:
                self.received = {
                    "sql": sql,
                    "datasource": datasource,
                    "database": database,
                    "min_rows": min_rows,
                    "max_rows": max_rows,
                }
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.execute_sql)
        raw_args = '{"sql":"SELECT 1","datasource":"njtest","database":"njyh","min_rows":,"max_rows":}'
        raw_tool_call = ResponseFunctionToolCall(
            arguments=raw_args,
            call_id="call_repaired",
            name="execute_sql",
            type="function_call",
        )
        tool_ctx = SimpleNamespace(tool_arguments=raw_args, tool_call=raw_tool_call)

        result = await tool.on_invoke_tool(tool_ctx, raw_args)

        assert result["success"] == 0
        assert "not executed" in result["error"]
        assert "Retry" in result["error"]
        assert fake.received is None
        written_args = json.loads(tool_ctx.tool_arguments)
        assert written_args["sql"] == "SELECT 1"
        assert {"min_rows", "max_rows"} <= written_args.keys()
        assert json.loads(raw_tool_call.arguments) == written_args

        replayed = ToolCallItem(agent=Agent(name="test"), raw_item=raw_tool_call).to_input_item()
        assert json.loads(replayed["arguments"])["sql"] == "SELECT 1"

        retry_args = json.dumps(
            {
                "sql": "SELECT 1",
                "datasource": "njtest",
                "database": "njyh",
                "min_rows": None,
                "max_rows": None,
            }
        )
        retry_tool_call = ResponseFunctionToolCall(
            arguments=retry_args,
            call_id="call_retry",
            name="execute_sql",
            type="function_call",
        )
        retry_ctx = SimpleNamespace(tool_arguments=retry_args, tool_call=retry_tool_call)

        retry_result = await tool.on_invoke_tool(retry_ctx, retry_args)

        assert retry_result["success"] == 1
        assert fake.received == {
            "sql": "SELECT 1",
            "datasource": "njtest",
            "database": "njyh",
            "min_rows": None,
            "max_rows": None,
        }

    @pytest.mark.asyncio
    async def test_valid_json_preserves_original_arguments_string(self):
        """A valid call should not be rewritten merely to canonicalize formatting."""

        class FakeTool:
            def some_method(self, x: str) -> FuncToolResult:
                return FuncToolResult(result=x)

        tool = self._make_tool_from_method(FakeTool().some_method)
        raw_args = '{\n  "x": "value"\n}'
        raw_tool_call = ResponseFunctionToolCall(
            arguments=raw_args,
            call_id="call_valid",
            name="some_method",
            type="function_call",
        )
        tool_ctx = SimpleNamespace(tool_arguments=raw_args, tool_call=raw_tool_call)

        result = await tool.on_invoke_tool(tool_ctx, raw_args)

        assert result["success"] == 1
        assert tool_ctx.tool_arguments == raw_args
        assert raw_tool_call.arguments == raw_args

    @pytest.mark.asyncio
    async def test_runner_replays_repaired_args_and_executes_only_valid_retry(self):
        """Exercise the real Runner turn and session persistence path."""

        class FakeTool:
            def __init__(self):
                self.calls = []

            def execute_sql(
                self,
                sql: str,
                min_rows: Optional[int] = None,
                max_rows: Optional[int] = None,
            ) -> FuncToolResult:
                self.calls.append((sql, min_rows, max_rows))
                return FuncToolResult(result="ok")

        class SequenceModel(Model):
            def __init__(self):
                self.inputs = []

            async def get_response(self, system_instructions, input, *_args, **_kwargs):
                from agents.items import ModelResponse

                del system_instructions
                self.inputs.append(input)
                if len(self.inputs) == 1:
                    output = [
                        ResponseFunctionToolCall(
                            arguments='{"sql":"SELECT 1","min_rows":,"max_rows":}',
                            call_id="call_malformed",
                            name="execute_sql",
                            type="function_call",
                        )
                    ]
                elif len(self.inputs) == 2:
                    replayed_call = next(
                        item for item in input if isinstance(item, dict) and item.get("call_id") == "call_malformed"
                    )
                    assert json.loads(replayed_call["arguments"])["sql"] == "SELECT 1"
                    output = [
                        ResponseFunctionToolCall(
                            arguments='{"sql":"SELECT 1","min_rows":null,"max_rows":null}',
                            call_id="call_retry",
                            name="execute_sql",
                            type="function_call",
                        )
                    ]
                else:
                    output = [
                        ResponseOutputMessage(
                            id="message_done",
                            content=[
                                ResponseOutputText(
                                    annotations=[],
                                    text="done",
                                    type="output_text",
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message",
                        )
                    ]
                return ModelResponse(output=output, usage=Usage(), response_id=None)

            async def stream_response(self, *_args, **_kwargs):
                if False:
                    yield None

        fake = FakeTool()
        model = SequenceModel()
        session = SQLiteSession("tool-args-repair", ":memory:")
        tool = self._make_tool_from_method(fake.execute_sql)
        registry = ToolRegistry()
        registry.register_tools("db_tools", [tool])
        broker = MagicMock()
        permission_hooks = PermissionHooks(
            broker=broker,
            permission_manager=PermissionManager(
                global_config=get_profile("normal"),
                active_profile="normal",
            ),
            node_name="chat",
            tool_registry=registry,
        )
        agent = Agent(
            name="test",
            model=model,
            tools=[tool],
            hooks=permission_hooks,
        )

        result = await Runner.run(
            agent,
            input="run a query",
            max_turns=5,
            session=session,
            run_config=RunConfig(tracing_disabled=True),
        )

        assert result.final_output == "done"
        assert fake.calls == [("SELECT 1", None, None)]
        broker.request.assert_not_called()
        saved_calls = [
            item for item in await session.get_items() if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        assert len(saved_calls) == 2
        assert all(isinstance(json.loads(item["arguments"]), dict) for item in saved_calls)

    @pytest.mark.asyncio
    async def test_unrepairable_json_is_not_executed_and_writes_back_empty_object(self):
        """An unrecoverable call must leave valid JSON for the Runner's next turn."""

        class FakeTool:
            def __init__(self):
                self.called = False

            def some_method(self, x: str) -> FuncToolResult:
                self.called = True
                return FuncToolResult(result=x)

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.some_method)
        raw_tool_call = ResponseFunctionToolCall(
            arguments="not-valid-json{",
            call_id="call_invalid",
            name="some_method",
            type="function_call",
        )
        tool_ctx = SimpleNamespace(tool_arguments=raw_tool_call.arguments, tool_call=raw_tool_call)

        result = await tool.on_invoke_tool(tool_ctx, raw_tool_call.arguments)

        assert result["success"] == 0
        assert "Invalid JSON" in result["error"]
        assert fake.called is False
        assert tool_ctx.tool_arguments == "{}"
        assert raw_tool_call.arguments == "{}"

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

    @pytest.mark.asyncio
    async def test_excluded_parameters_are_not_exposed(self):
        """Excluded parameters should be omitted from the function-tool schema."""

        class FakeTool:
            def list_tables(self, catalog: str = "", database: str = "") -> FuncToolResult:
                return FuncToolResult(result={"catalog": catalog, "database": database})

        fake = FakeTool()
        tool = trans_to_function_tool(fake.list_tables, excluded_params={"catalog"})

        assert "catalog" not in tool.params_json_schema.get("properties", {})
        assert "database" in tool.params_json_schema.get("properties", {})

        args = json.dumps({"catalog": "cat", "database": "db"})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 0
        assert result["error"] == "Unsupported parameters for this tool: catalog"

    @pytest.mark.asyncio
    async def test_missing_required_parameter_returns_recoverable_error(self):
        """Omitting a required arg must yield a recoverable error, not a crash.

        Regression: a tool call that drops a required argument (e.g. ``edit_file``
        without ``path``) previously reached ``method_to_call(**args_dict)`` and
        raised a raw ``TypeError`` at bind time, aborting the whole agent
        interaction. The dispatcher now rejects it up front so the model can retry.
        """

        class FakeTool:
            def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:
                return FuncToolResult(result="edited")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.edit_file)

        args = json.dumps({"old_string": "a", "new_string": "b"})  # path omitted
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 0
        assert "Missing required parameter(s)" in result["error"]
        assert "path" in result["error"]
        assert "edit_file" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_multiple_required_parameters_are_all_reported(self):
        class FakeTool:
            def combine(self, a: str, b: str, c: int = 0) -> FuncToolResult:
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.combine)

        result = await tool.on_invoke_tool(None, json.dumps({"c": 5}))  # a, b omitted

        assert result["success"] == 0
        assert "a" in result["error"] and "b" in result["error"]

    @pytest.mark.asyncio
    async def test_all_required_present_still_runs(self):
        """The guard must not block a well-formed call (no false positives)."""

        class FakeTool:
            def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:
                return FuncToolResult(result={"path": path})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.edit_file)

        args = json.dumps({"path": "orders.yml", "old_string": "a", "new_string": "b"})
        result = await tool.on_invoke_tool(None, args)

        assert result["success"] == 1
        assert result["result"] == {"path": "orders.yml"}

    @pytest.mark.asyncio
    async def test_optional_parameters_may_be_omitted(self):
        """Parameters with defaults are not required and may be omitted."""

        class FakeTool:
            def search_table(self, query_text: str, top_n: int = 5) -> FuncToolResult:
                return FuncToolResult(result={"query_text": query_text, "top_n": top_n})

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.search_table)

        result = await tool.on_invoke_tool(None, json.dumps({"query_text": "hello"}))

        assert result["success"] == 1
        assert result["result"] == {"query_text": "hello", "top_n": 5}

    @pytest.mark.asyncio
    async def test_sync_tool_runs_off_the_event_loop_thread(self):
        """Synchronous tool methods must be offloaded to a worker thread.

        ``final_invoker`` is awaited directly on the asyncio event-loop thread.
        If a blocking sync tool (e.g. a StarRocks ``list_tables`` metadata query)
        ran inline, it would freeze the loop and make the whole server
        unresponsive. The fix dispatches sync methods via ``asyncio.to_thread``;
        here we assert the method body executed on a different thread.
        """
        loop_thread_id = threading.get_ident()
        ran_on: dict = {}

        class FakeTool:
            def list_tables(self) -> FuncToolResult:
                ran_on["thread_id"] = threading.get_ident()
                return FuncToolResult(result="ok")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.list_tables)

        result = await tool.on_invoke_tool(None, "{}")

        assert result["success"] == 1
        assert ran_on["thread_id"] != loop_thread_id

    @pytest.mark.asyncio
    async def test_blocking_sync_tool_does_not_stall_the_event_loop(self):
        """A blocking sync tool must not prevent other coroutines from running."""
        proceed = threading.Event()

        class FakeTool:
            def slow_io(self) -> FuncToolResult:
                # Blocks the worker thread until the event loop releases it.
                proceed.wait(2)
                return FuncToolResult(result="done")

        fake = FakeTool()
        tool = self._make_tool_from_method(fake.slow_io)

        task = asyncio.create_task(tool.on_invoke_tool(None, "{}"))
        # If slow_io blocked the loop, this yield would never resume and the
        # event below would never be set (deadlock). Reaching here with the task
        # still pending proves the loop stayed free while slow_io blocked.
        await asyncio.sleep(0.05)
        assert not task.done()

        proceed.set()
        result = await task
        assert result["success"] == 1
        assert result["result"] == "done"


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
