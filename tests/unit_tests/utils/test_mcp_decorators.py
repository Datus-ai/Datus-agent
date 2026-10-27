# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import threading
from contextvars import ContextVar
from types import SimpleNamespace

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from datus.tools.func_tool.base import FuncToolResult
from datus.utils.mcp_decorators import (
    MCPToolConfig,
    create_dynamic_tool_wrapper,
    create_static_tool_wrapper,
)
from datus.utils.mcp_execution import MCPToolExecutor


def make_wrapper(instance, name, mode, **options):
    method = getattr(instance, name)
    common = dict(method_name=name, config=MCPToolConfig(), format_result=lambda result: result)
    common.update(options)
    if mode == "static":
        return create_static_tool_wrapper(bound_method=method, **common)
    context = ContextVar("tool_context")
    context.set(SimpleNamespace(tool=instance, available=True))
    return create_dynamic_tool_wrapper(
        method=getattr(type(instance), name),
        context_getter=context.get,
        instance_attr="tool",
        availability_attr="available",
        **common,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dynamic", "static"])
async def test_slow_mcp_tool_keeps_health_responsive_and_copies_context(mode):
    """Exercise SDK dispatch: synchronous MCP functions run inline in MCP 1.30."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    request_id = ContextVar("request_id")
    request_id.set("request-1529")
    seen = {}

    class Tools:
        def list_tables(self, catalog: str = "default") -> dict:
            seen.update(request_id=request_id.get(), thread=threading.get_ident(), catalog=catalog)
            loop.call_soon_threadsafe(started.set)
            release.wait(1)
            return {"tables": ["orders"]}

    mcp = FastMCP("regression")
    mcp.tool()(make_wrapper(Tools(), "list_tables", mode))

    async def health(request):
        return JSONResponse({"status": "ok", "tool_running": not release.is_set() and not task.done()})

    app = Starlette(routes=[Route("/health", health)])
    task = asyncio.create_task(mcp.call_tool("list_tables", {"catalog": "sales"}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "tool_running": True}
        assert seen["thread"] != threading.get_ident()
        assert seen["request_id"] == "request-1529"
        assert seen["catalog"] == "sales"
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dynamic", "static"])
async def test_async_tools_are_awaited_on_request_loop(mode):
    loop = asyncio.get_running_loop()
    request_id = ContextVar("async_request_id", default=None)
    request_id.set("async-request")

    class Tools:
        async def query(self, value: str = "orders") -> dict:
            await asyncio.sleep(0)
            assert asyncio.get_running_loop() is loop
            assert request_id.get() == "async-request"
            return {"value": value}

    wrapper = make_wrapper(Tools(), "query", mode)
    assert await wrapper(value="sales") == {"value": "sales"}


@pytest.mark.asyncio
async def test_shared_dynamic_wrapper_keeps_concurrent_request_contexts_isolated():
    context = ContextVar("datasource_context")
    request_id = ContextVar("isolated_request_id")

    class Tools:
        def __init__(self, datasource):
            self.datasource = datasource

        def query(self) -> dict:
            return {"datasource": self.datasource, "request": request_id.get()}

    wrapper = create_dynamic_tool_wrapper(
        method_name="query",
        method=Tools.query,
        config=MCPToolConfig(),
        context_getter=context.get,
        instance_attr="tool",
        availability_attr="available",
        format_result=lambda result: result,
    )

    async def call(datasource):
        context.set(SimpleNamespace(tool=Tools(datasource), available=True))
        request_id.set(f"request-{datasource}")
        return await wrapper()

    assert await asyncio.gather(call("sales"), call("finance")) == [
        {"datasource": "sales", "request": "request-sales"},
        {"datasource": "finance", "request": "request-finance"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dynamic", "static"])
@pytest.mark.parametrize("stage", ["transformer", "format"])
async def test_blocking_plugin_and_formatter_run_off_loop(mode, stage, monkeypatch):
    from datus.tools.middleware import tool_middleware

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    request_id = ContextVar("plugin_request", default=None)
    request_id.set("policy-context")
    calls = []

    def blocking_stage():
        assert threading.get_ident() != loop_thread
        assert request_id.get() == "policy-context"
        loop.call_soon_threadsafe(started.set)
        release.wait(1)

    def transformer(name, args, context):
        if stage == "transformer":
            blocking_stage()
        return {**args, "value": "filtered"}

    def format_result(result):
        if stage == "format":
            blocking_stage()
        return result.model_dump()

    class Tools:
        def query(self, value: str) -> FuncToolResult:
            calls.append(value)
            return FuncToolResult(success=1, result=value)

    loop_thread = threading.get_ident()
    monkeypatch.setattr(
        tool_middleware, "collect_plugin_tool_transformers", lambda active=None: {"query": [transformer]}
    )
    wrapper = make_wrapper(Tools(), "query", mode, format_result=format_result)
    task = asyncio.create_task(wrapper(value="unfiltered"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not task.done()
    finally:
        release.set()
    assert (await task)["result"] == "filtered"
    assert calls == ["filtered"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dynamic", "static"])
async def test_cancelled_mcp_call_retains_capacity_until_worker_finishes(mode):
    from datus.utils.exceptions import DatusException

    executor = MCPToolExecutor(max_workers=1, max_queue_size=1)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    calls = []

    class Tools:
        def query(self, value: str) -> dict:
            calls.append(value)
            if value == "running":
                loop.call_soon_threadsafe(started.set)
                release.wait(2)
                raise ValueError("late driver failure")
            return {"value": value}

    wrapper = make_wrapper(Tools(), "query", mode, executor=executor)
    first = asyncio.create_task(wrapper(value="running"))
    queued = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        queued = asyncio.create_task(wrapper(value="queued"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        # Both submitted calls keep their slots, including the queued call.
        for _ in range(3):
            with pytest.raises(DatusException, match="capacity exhausted"):
                await wrapper(value="excess")
        assert calls == ["running"]
        # The default executor remains usable while the MCP pool is full.
        assert await asyncio.to_thread(lambda: "healthy") == "healthy"
    finally:
        release.set()
        executor.shutdown()
