# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/mcp_server.py

All tests are CI-level: zero external dependencies, zero network access.
All external calls (AgentConfig, tools, FastMCP) are mocked.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.mcp_server import DatusMCPServer, LightweightDynamicMCPServer, ToolContext, ToolContextManager
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.mcp_decorators import MCPToolConfig, create_dynamic_tool_wrapper
from datus.utils.mcp_execution import MCPToolExecutor

# =============================================================================
# Fixtures
# =============================================================================


def _make_agent_config(datasource="default"):
    cfg = MagicMock()
    cfg.datasource = datasource
    return cfg


def _make_tool_context(datasource="default", subagent=None):
    ctx = ToolContext(
        datasource=datasource,
        subagent=subagent,
        agent_config=_make_agent_config(datasource),
        tools={},
    )
    return ctx


# =============================================================================
# Tests: ToolContext
# =============================================================================


class TestMCPToolContext:
    def test_datasource(self):
        ctx = _make_tool_context("sales")
        assert ctx.datasource == "sales"

    def test_subagent_default_none(self):
        ctx = _make_tool_context()
        assert ctx.subagent is None

    def test_has_db_tools_false_when_no_db_tool(self):
        ctx = _make_tool_context()
        assert ctx.has_db_tools is False

    def test_has_context_tools_false_when_no_context_tool(self):
        ctx = _make_tool_context()
        assert ctx.has_context_tools is False

    def test_has_db_tools_true_when_db_tool_present(self):
        ctx = _make_tool_context()
        ctx.tools["db_tool"] = MagicMock()
        assert ctx.has_db_tools is True

    def test_has_context_tools_true_when_context_tool_present(self):
        ctx = _make_tool_context()
        ctx.tools["context_tool"] = MagicMock()
        assert ctx.has_context_tools is True

    def test_db_tool_property_returns_correct_tool(self):
        ctx = _make_tool_context()
        mock_db = MagicMock()
        ctx.tools["db_tool"] = mock_db
        assert ctx.db_tool is mock_db

    def test_context_tool_property_returns_correct_tool(self):
        ctx = _make_tool_context()
        mock_ct = MagicMock()
        ctx.tools["context_tool"] = mock_ct
        assert ctx.context_tool is mock_ct

    def test_close_clears_tools(self):
        ctx = _make_tool_context()
        mock_tool = MagicMock()
        mock_tool.connector = None
        ctx.tools["db_tool"] = mock_tool
        ctx.close()
        assert ctx.tools == {}

    def test_close_closes_connector_if_present(self):
        ctx = _make_tool_context()
        mock_connector = MagicMock()
        mock_tool = MagicMock()
        mock_tool.connector = mock_connector
        ctx.tools["db_tool"] = mock_tool
        ctx.close()
        mock_connector.close.assert_called_once()

    def test_close_handles_connector_error_gracefully(self):
        ctx = _make_tool_context()
        mock_connector = MagicMock()
        mock_connector.close.side_effect = RuntimeError("close failed")
        mock_tool = MagicMock()
        mock_tool.connector = mock_connector
        ctx.tools["db_tool"] = mock_tool
        ctx.close()  # Should not raise
        assert ctx.tools == {}


# =============================================================================
# Tests: ToolContextManager
# =============================================================================


@pytest.fixture
def mock_config_manager():
    mgr = MagicMock()
    mgr.get.side_effect = lambda key, default=None: (
        {"datasources": {"ns1": {}, "ns2": {}}}
        if key == "services"
        else {"agent1": {}}
        if key == "agentic_nodes"
        else default
    )
    return mgr


@pytest.fixture
def context_manager(mock_config_manager):
    with patch("datus.mcp_server.configuration_manager", return_value=mock_config_manager):
        mgr = ToolContextManager(config_path=None, max_size=5)
    try:
        yield mgr
    finally:
        mgr.close_all()
        mgr.tool_executor.shutdown()


class TestToolContextManagerInit:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["dynamic", "static"])
    async def test_startup_applies_quoted_execution_capacity(self, mode):
        settings = {"max_workers": "1", "max_queue_size": "0"}
        if mode == "dynamic":
            manager_config = SimpleNamespace(
                get=lambda key, default=None: settings if key == "mcp_tool_execution" else default
            )
            with patch("datus.mcp_server.configuration_manager", return_value=manager_config):
                owner = ToolContextManager()
            executor = owner.tool_executor
            close = owner.close_all
        else:
            with (
                patch("datus.mcp_server.load_agent_config", return_value=SimpleNamespace(mcp_tool_execution=settings)),
                patch.object(DatusMCPServer, "_init_tools"),
                patch.object(DatusMCPServer, "_register_tools"),
            ):
                owner = DatusMCPServer("test")
            executor = owner._tool_executor
            close = owner.close
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def block():
            loop.call_soon_threadsafe(started.set)
            release.wait(2)

        task = asyncio.create_task(executor.run(block))
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(DatusException) as caught:
                await executor.run(lambda: "excess")
            assert caught.value.code == ErrorCode.TOOL_EXECUTION_FAILED
        finally:
            release.set()
            await task
            close()
            executor.shutdown()

    @pytest.mark.parametrize("mode", ["dynamic", "static"])
    @pytest.mark.parametrize("execution", [{"workers": 2}, {"max_workers": "invalid"}, {"max_workers": True}])
    def test_startup_reports_invalid_execution_settings(self, mode, execution):
        with pytest.raises(DatusException) as caught:
            if mode == "dynamic":
                manager_config = SimpleNamespace(
                    get=lambda key, default=None: execution if key == "mcp_tool_execution" else default
                )
                with patch("datus.mcp_server.configuration_manager", return_value=manager_config):
                    ToolContextManager()
            else:
                with patch(
                    "datus.mcp_server.load_agent_config", return_value=SimpleNamespace(mcp_tool_execution=execution)
                ):
                    DatusMCPServer("test")
        assert caught.value.code == ErrorCode.COMMON_CONFIG_ERROR

    def test_available_datasources(self, context_manager):
        datasources = context_manager.available_datasources
        assert "ns1" in datasources
        assert "ns2" in datasources

    def test_available_subagents(self, context_manager):
        subagents = context_manager.available_subagents
        assert "agent1" in subagents

    def test_validate_datasource_known(self, context_manager):
        assert context_manager.validate_datasource("ns1") is True

    def test_validate_datasource_unknown(self, context_manager):
        assert context_manager.validate_datasource("unknown") is False

    def test_validate_subagent_known(self, context_manager):
        assert context_manager.validate_subagent("agent1") is True

    def test_validate_subagent_unknown(self, context_manager):
        assert context_manager.validate_subagent("missing") is False

    def test_default_max_size(self, mock_config_manager):
        with patch("datus.mcp_server.configuration_manager", return_value=mock_config_manager):
            mgr = ToolContextManager()
        assert mgr._max_size == ToolContextManager.DEFAULT_MAX_SIZE

    def test_custom_max_size(self, mock_config_manager):
        with patch("datus.mcp_server.configuration_manager", return_value=mock_config_manager):
            mgr = ToolContextManager(max_size=10)
        assert mgr._max_size == 10

    def test_cache_key_without_subagent(self, context_manager):
        key = context_manager._get_cache_key("ns1")
        assert key == "ns1:"

    def test_cache_key_with_subagent(self, context_manager):
        key = context_manager._get_cache_key("ns1", "agent1")
        assert key == "ns1:agent1"


class TestToolContextManagerGetOrCreate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("lookup", ["get", "lease", "worker"])
    async def test_cached_context_does_not_wait_for_stalled_initialization(self, context_manager, lookup):
        cached = _make_tool_context("healthy")
        context_manager._contexts["healthy:"] = cached
        context_manager.tool_executor.shutdown()
        context_manager.tool_executor = MCPToolExecutor(max_workers=1, max_queue_size=0)
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def create(datasource, subagent):
            loop.call_soon_threadsafe(started.set)
            release.wait(2)
            return _make_tool_context(datasource)

        async def query_cached():
            if lookup == "get":
                return await context_manager.get_or_create_context("healthy")
            if lookup == "lease":
                async with context_manager.use_context("healthy") as context:
                    return context
            context = await asyncio.to_thread(context_manager._get_or_create_context, "healthy", None, True)
            context.release()
            return context

        with patch.object(context_manager, "_create_context", side_effect=create):
            initialization = asyncio.create_task(context_manager.get_or_create_context("slow"))
            cached_task = None
            try:
                await asyncio.wait_for(started.wait(), 2)
                cached_task = asyncio.create_task(query_cached())
                done, _ = await asyncio.wait([cached_task], timeout=0.2)
                assert cached_task in done, "cached context waited on stalled initialization"
                assert await cached_task is cached
            finally:
                release.set()
                await initialization
                if cached_task is not None:
                    await asyncio.gather(cached_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_context_initialization_does_not_block_and_survives_cancellation(self, context_manager):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        ctx = _make_tool_context("slow")

        def create(datasource, subagent):
            loop.call_soon_threadsafe(started.set)
            release.wait(1)
            return ctx

        with patch.object(context_manager, "_create_context", side_effect=create) as factory:
            task = asyncio.create_task(context_manager.get_or_create_context("slow"))
            try:
                await asyncio.wait_for(started.wait(), 2)
                assert not task.done(), "context initialization blocked the event loop"
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
            assert await context_manager.get_or_create_context("slow") is ctx
            assert factory.call_count == 1

    @pytest.mark.asyncio
    async def test_creates_context_on_miss(self, context_manager):
        mock_ctx = _make_tool_context("ns1")
        with patch.object(context_manager, "_create_context", return_value=mock_ctx):
            result = await context_manager.get_or_create_context(datasource="ns1")
        assert result is mock_ctx

    @pytest.mark.asyncio
    async def test_returns_cached_context_on_hit(self, context_manager):
        mock_ctx = _make_tool_context("ns1")
        with patch.object(context_manager, "_create_context", return_value=mock_ctx):
            result1 = await context_manager.get_or_create_context(datasource="ns1")
            result2 = await context_manager.get_or_create_context(datasource="ns1")
        # _create_context should only be called once
        assert result1 is result2

    @pytest.mark.asyncio
    async def test_lru_eviction_when_full(self, context_manager):
        """When cache is full, oldest entry is evicted."""
        context_manager._max_size = 2
        contexts = [_make_tool_context(f"ns{i}") for i in range(3)]

        call_count = 0

        def make_ctx(ns, sa=None):
            nonlocal call_count
            ctx = contexts[call_count]
            call_count += 1
            return ctx

        with patch.object(context_manager, "_create_context", side_effect=make_ctx):
            await context_manager.get_or_create_context(datasource="ns0")
            await context_manager.get_or_create_context(datasource="ns1")
            # This should evict ns0
            await context_manager.get_or_create_context(datasource="ns2")

        assert len(context_manager._contexts) == 2
        assert "ns0:" not in context_manager._contexts

    @pytest.mark.asyncio
    async def test_lru_moves_to_end_on_access(self, context_manager):
        """Accessing an existing context moves it to end (most recent)."""
        context_manager._max_size = 3
        ctx_a = _make_tool_context("ns_a")
        ctx_b = _make_tool_context("ns_b")

        def make_ctx(ns, sa=None):
            return {"ns_a": ctx_a, "ns_b": ctx_b}[ns]

        with patch.object(context_manager, "_create_context", side_effect=make_ctx):
            await context_manager.get_or_create_context(datasource="ns_a")
            await context_manager.get_or_create_context(datasource="ns_b")
            # Access ns_a again — should move to end
            await context_manager.get_or_create_context(datasource="ns_a")

        keys = list(context_manager._contexts.keys())
        assert keys[-1] == "ns_a:"


class TestToolContextManagerCloseAll:
    def test_close_all_clears_contexts(self, context_manager):
        ctx = _make_tool_context("ns1")
        context_manager._contexts["ns1:"] = ctx
        context_manager.close_all()
        assert context_manager._contexts == {}

    @pytest.mark.asyncio
    async def test_close_all_handles_close_error(self, context_manager):
        closed = threading.Event()
        ctx = _make_tool_context("ns1")

        def fail_close():
            closed.set()
            raise RuntimeError("connector close failed")

        ctx.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=fail_close))
        context_manager._contexts["ns1:"] = ctx
        context_manager.close_all()
        assert context_manager._contexts == {}
        assert await asyncio.to_thread(closed.wait, 2)


class TestDynamicContextLifetime:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("transport", ["http", "sse"])
    async def test_transport_keeps_context_alive_before_tool_dispatch(self, context_manager, transport):
        entered = asyncio.Event()
        release = asyncio.Event()
        closed = threading.Event()
        context = _make_tool_context("a")
        tool = SimpleNamespace(connector=SimpleNamespace(close=closed.set))
        context.tools["db_tool"] = tool
        context_manager._max_size = 1
        server = LightweightDynamicMCPServer.__new__(LightweightDynamicMCPServer)
        server._context_manager = context_manager
        server._started = True
        server._sse_sessions = {}

        async def send(message):
            pass

        async def dispatch(scope, receive, send):
            if server._get_context().datasource != "a":
                return
            if transport == "sse":
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"event: endpoint\ndata: /messages/?session_id=abc123\n\n",
                    }
                )
            entered.set()
            await release.wait()
            assert server._get_context().db_tool is tool
            assert not closed.is_set()

        server._session_manager = SimpleNamespace(handle_request=dispatch)
        server._sse_app = dispatch
        task = None
        with patch.object(
            context_manager,
            "_create_context",
            side_effect=lambda ds, sa: context if ds == "a" else _make_tool_context(ds),
        ):
            try:
                task = asyncio.create_task(server.handle_request({}, None, send, "a", transport=transport))
                await asyncio.wait_for(entered.wait(), 2)
                await server.handle_request({}, None, send, "b", transport=transport)
                assert not closed.is_set(), "eviction closed a context still owned by its transport"
                assert server._sse_sessions == {"http": {}, "sse": {"abc123": context}}[transport]
                release.set()
                await task
                assert await asyncio.to_thread(closed.wait, 2)
                assert server._sse_sessions == {}
            finally:
                release.set()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_eviction_keeps_cancelled_requests_worker_context_alive(self, context_manager):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        closed = threading.Event()
        finished = threading.Event()
        healthy_at_return = []

        class Tools:
            connector = SimpleNamespace(close=closed.set)

            def query(self) -> dict:
                loop.call_soon_threadsafe(started.set)
                release.wait(2)
                healthy_at_return.append(not closed.is_set())
                finished.set()
                return {"ok": True}

        context_manager._max_size = 1
        context_a = _make_tool_context("a")
        context_a.tools["db_tool"] = Tools()
        context_b = _make_tool_context("b")
        server = LightweightDynamicMCPServer.__new__(LightweightDynamicMCPServer)
        server._context_manager = context_manager
        server._started = True
        wrapper = create_dynamic_tool_wrapper(
            "query",
            Tools.query,
            MCPToolConfig(),
            server._get_context,
            "db_tool",
            "has_db_tools",
            lambda result: result,
            executor=context_manager.tool_executor,
        )

        async def dispatch(scope, receive, send):
            if server._get_context().datasource == "a":
                await wrapper()

        server._session_manager = SimpleNamespace(handle_request=dispatch)
        task = None
        with patch.object(
            context_manager, "_create_context", side_effect=lambda ds, sa: {"a": context_a, "b": context_b}[ds]
        ):
            try:
                task = asyncio.create_task(server.handle_request({}, None, None, "a"))
                await asyncio.wait_for(started.wait(), 2)
                await server.handle_request({}, None, None, "b")
                assert not closed.is_set(), "eviction closed a connector still used by a worker"
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not closed.is_set(), "request cancellation closed the running worker's connector"
                release.set()
                assert await asyncio.to_thread(finished.wait, 2)
                assert await asyncio.to_thread(closed.wait, 2)
                assert healthy_at_return == [True]
            finally:
                release.set()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                context_manager.close_all()
                context_manager.tool_executor.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transport", ["http", "sse"])
    async def test_shutdown_does_not_wait_for_cancelled_initialization(self, mock_config_manager, transport):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        started = asyncio.Event()
        stop = asyncio.Event()
        release = threading.Event()
        closed = threading.Event()
        context = _make_tool_context("slow")
        context.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=closed.set))

        with (
            patch("datus.mcp_server.configuration_manager", return_value=mock_config_manager),
            patch.object(LightweightDynamicMCPServer, "_register_tools"),
        ):
            server = LightweightDynamicMCPServer()

        def create(datasource, subagent):
            loop.call_soon_threadsafe(started.set)
            release.wait(2)
            return context

        async def serve():
            async with server.lifespan_context(transport):
                entered.set()
                await stop.wait()

        lifecycle = asyncio.create_task(serve())
        initialization = None
        with patch.object(server._context_manager, "_create_context", side_effect=create):
            try:
                await asyncio.wait_for(entered.wait(), 2)
                initialization = asyncio.create_task(server._context_manager.get_or_create_context("slow"))
                await asyncio.wait_for(started.wait(), 2)
                initialization.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await initialization
                stop.set()
                done, _ = await asyncio.wait([lifecycle], timeout=0.2)
                assert lifecycle in done, "shutdown waited for the blocked initialization lock"
                await lifecycle
            finally:
                stop.set()
                release.set()
                await asyncio.wait_for(lifecycle, 2)
                if initialization is not None:
                    await asyncio.gather(initialization, return_exceptions=True)
            assert await asyncio.to_thread(closed.wait, 2), "late initialization leaked its connector"
            assert not server._context_manager._contexts

    @pytest.mark.asyncio
    async def test_cancelled_context_acquisition_releases_its_late_pin(self, context_manager):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        closed = threading.Event()
        context = _make_tool_context("a")
        context.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=closed.set))
        context_manager._max_size = 1

        def create(datasource, subagent):
            if datasource == "a":
                loop.call_soon_threadsafe(started.set)
                release.wait(2)
                return context
            return _make_tool_context(datasource)

        async def request():
            async with context_manager.use_context("a"):
                raise AssertionError("cancelled request must not resume")

        task = asyncio.create_task(request())
        with patch.object(context_manager, "_create_context", side_effect=create):
            try:
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                release.set()
                await context_manager.get_or_create_context("b")
                assert await asyncio.to_thread(closed.wait, 2), "abandoned request retained its context pin"
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                context_manager.close_all()
                context_manager.tool_executor.shutdown()

    @pytest.mark.asyncio
    async def test_async_tool_keeps_context_pinned_after_its_request_finishes(self, context_manager):
        started = asyncio.Event()
        release = asyncio.Event()
        closed = threading.Event()

        class Tools:
            connector = SimpleNamespace(close=closed.set)

            async def query(self) -> dict:
                started.set()
                await release.wait()
                assert not closed.is_set()
                return {"ok": True}

        context = _make_tool_context("a")
        context.tools["db_tool"] = Tools()
        context_manager._max_size = 1
        from datus.mcp_server import _current_tool_context

        wrapper = create_dynamic_tool_wrapper(
            "query",
            Tools.query,
            MCPToolConfig(),
            _current_tool_context.get,
            "db_tool",
            "has_db_tools",
            lambda result: result,
            executor=context_manager.tool_executor,
        )
        task = None
        with patch.object(
            context_manager,
            "_create_context",
            side_effect=lambda ds, sa: context if ds == "a" else _make_tool_context(ds),
        ):
            try:
                async with context_manager.use_context("a") as current:
                    token = _current_tool_context.set(current)
                    try:
                        task = asyncio.create_task(wrapper())
                        await asyncio.wait_for(started.wait(), 2)
                    finally:
                        _current_tool_context.reset(token)
                await context_manager.get_or_create_context("b")
                assert not closed.is_set()
                release.set()
                assert await task == {"ok": True}
                assert await asyncio.to_thread(closed.wait, 2)
            finally:
                release.set()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                context_manager.close_all()
                context_manager.tool_executor.shutdown()


# =============================================================================
# Tests: LightweightDynamicMCPServer._format_result (static method)
# =============================================================================


class TestFormatResult:
    def test_func_tool_result_converted(self):
        from datus.mcp_server import LightweightDynamicMCPServer

        result = FuncToolResult(success=1, error=None, result="data")
        formatted = LightweightDynamicMCPServer._format_result(result)
        assert formatted["success"] == 1
        assert formatted["result"] == "data"
        assert formatted["error"] is None

    def test_plain_value_wrapped(self):
        from datus.mcp_server import LightweightDynamicMCPServer

        formatted = LightweightDynamicMCPServer._format_result("plain text")
        assert formatted == {"success": 1, "error": None, "result": "plain text"}

    def test_dict_wrapped(self):
        from datus.mcp_server import LightweightDynamicMCPServer

        formatted = LightweightDynamicMCPServer._format_result({"key": "val"})
        assert formatted["result"] == {"key": "val"}


# =============================================================================
# Tests: LightweightDynamicMCPServer._get_context
# =============================================================================


class TestGetContext:
    def test_raises_when_no_context(self):
        from datus.mcp_server import LightweightDynamicMCPServer, _current_tool_context

        token = _current_tool_context.set(None)
        try:
            with pytest.raises(RuntimeError, match="No tool context"):
                LightweightDynamicMCPServer._get_context()
        finally:
            _current_tool_context.reset(token)

    def test_returns_context_when_set(self):
        from datus.mcp_server import LightweightDynamicMCPServer, _current_tool_context

        ctx = _make_tool_context("ns1")
        token = _current_tool_context.set(ctx)
        try:
            result = LightweightDynamicMCPServer._get_context()
            assert result is ctx
        finally:
            _current_tool_context.reset(token)


class _PublicationManagerContract:
    """Legacy publication cache API (Datus-backend 987e92e, mcp_mount.py).

    In particular, eviction and shutdown call ToolContext.close() directly.
    Configuration/policy loading remains asynchronous and request-scoped.
    """

    def __init__(self, contexts):
        self.contexts = contexts
        self.cached = None
        self.lookups = []
        self.closed = threading.Event()

    async def get_or_create_context(self, datasource, subagent=None):
        self.lookups.append((datasource, subagent))
        await asyncio.sleep(0)
        context = self.contexts[datasource]
        if self.cached is not None and self.cached is not context:
            self.cached.close()
        self.cached = context
        return context

    def close_all(self):
        if self.cached is not None:
            self.cached.close()
        self.cached = None
        self.closed.set()


def _build_publication_server(manager):
    # The deployed backend swaps this constructor and checks manager identity.
    with patch("datus.mcp_server.ToolContextManager", return_value=manager):
        server = LightweightDynamicMCPServer()
    assert server._context_manager is manager
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "sse"])
async def test_publication_manager_constructs_and_dispatches_requests(transport):
    context = _make_tool_context("version:user", "analyst")
    closed = threading.Event()
    context.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=closed.set))
    manager = _PublicationManagerContract({context.datasource: context})
    server = _build_publication_server(manager)
    observed = []

    async def dispatch(scope, receive, send):
        observed.append(server._get_context())

    async with server.lifespan_context(transport):
        if transport == "http":
            server._session_manager = SimpleNamespace(handle_request=dispatch)
        else:
            server._sse_app = dispatch
        await server.handle_request({}, None, None, "version:user", "analyst", transport)
    assert observed == [context]
    assert manager.lookups == [("version:user", "analyst")]
    assert await asyncio.to_thread(manager.closed.wait, 2)
    assert await asyncio.to_thread(closed.wait, 2)


@pytest.mark.asyncio
async def test_static_close_retains_cancelled_workers_connector():
    from datus.utils.mcp_decorators import ToolClassConfig, mcp_tool

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    closed = threading.Event()
    healthy = []

    class Tools:
        connector = SimpleNamespace(close=closed.set)

        @classmethod
        def create_static(cls, *args):
            return cls()

        @mcp_tool()
        def query(self) -> dict:
            loop.call_soon_threadsafe(started.set)
            release.wait(3)
            healthy.append(not closed.is_set())
            return {"ok": True}

    with (
        patch("datus.mcp_server.load_agent_config", return_value=SimpleNamespace(mcp_tool_execution={})),
        patch("datus.mcp_server.get_tool_registry", return_value=[ToolClassConfig("db_tool", Tools, "has_db_tools")]),
    ):
        server = DatusMCPServer("test")
    task = asyncio.create_task(server.mcp.call_tool("query", {}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        server.close()
        server.close()
        assert not closed.is_set(), "shutdown closed a connector still used by a cancelled worker"
        release.set()
        assert await asyncio.to_thread(closed.wait, 2)
        assert healthy == [True]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        server.close()


@pytest.mark.asyncio
async def test_static_connector_cleanup_does_not_block_request_loop():
    loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    release_close = threading.Event()
    closed = threading.Event()

    def close():
        loop.call_soon_threadsafe(close_started.set)
        release_close.wait(2)
        closed.set()

    with (
        patch("datus.mcp_server.load_agent_config", return_value=SimpleNamespace(mcp_tool_execution={})),
        patch.object(DatusMCPServer, "_init_tools"),
        patch.object(DatusMCPServer, "_register_tools"),
    ):
        server = DatusMCPServer("test")
    server.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=close))
    try:
        server.close()
        await asyncio.wait_for(close_started.wait(), 2)
        assert not closed.is_set(), "close blocked the event loop until connector cleanup finished"
    finally:
        release_close.set()
        assert await asyncio.to_thread(closed.wait, 2)


@pytest.mark.asyncio
async def test_publication_eviction_and_shutdown_retain_cancelled_worker():
    from datus.utils.mcp_decorators import ToolClassConfig, mcp_tool

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    closed = threading.Event()
    healthy = []

    class Tools:
        connector = SimpleNamespace(close=closed.set)

        @mcp_tool()
        def query(self) -> dict:
            loop.call_soon_threadsafe(started.set)
            release.wait(3)
            healthy.append(not closed.is_set())
            return {"ok": True}

    context = _make_tool_context("version:user-a")
    context.tools["db_tool"] = Tools()
    other = _make_tool_context("version:user-b")
    manager = _PublicationManagerContract({context.datasource: context, other.datasource: other})
    with patch("datus.mcp_server.get_tool_registry", return_value=[ToolClassConfig("db_tool", Tools, "has_db_tools")]):
        server = _build_publication_server(manager)

    async def dispatch(scope, receive, send):
        if server._get_context() is context:
            await server.mcp.call_tool("query", {})

    task = None
    try:
        async with server.lifespan_context("sse"):
            server._sse_app = dispatch
            task = asyncio.create_task(server.handle_request({}, None, None, context.datasource, transport="sse"))
            await asyncio.wait_for(started.wait(), 2)
            await server.handle_request({}, None, None, other.datasource, transport="sse")
            assert not closed.is_set(), "legacy LRU eviction invalidated a running worker"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not closed.is_set(), "legacy shutdown invalidated a cancelled worker"
        release.set()
        assert await asyncio.to_thread(closed.wait, 2)
        assert await asyncio.to_thread(manager.closed.wait, 2)
        assert healthy == [True]
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_publication_cancelled_lookup_is_drained_after_shutdown(fail):
    started = asyncio.Event()
    release = asyncio.Event()
    closed = threading.Event()
    context = _make_tool_context("version:user")
    context.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=closed.set))
    manager = _PublicationManagerContract({context.datasource: context})
    get_context = manager.get_or_create_context

    async def delayed_lookup(datasource, subagent=None):
        started.set()
        await release.wait()
        if fail:
            raise ValueError("publication lookup failed")
        return await get_context(datasource, subagent)

    manager.get_or_create_context = delayed_lookup
    server = _build_publication_server(manager)
    task = None
    try:
        async with server.lifespan_context("sse"):
            task = asyncio.create_task(server.handle_request({}, None, None, context.datasource, transport="sse"))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not manager.closed.is_set(), "close_all raced an unfinished legacy cache lookup"
        release.set()
        assert await asyncio.to_thread(manager.closed.wait, 2)
        assert closed.is_set() == (not fail), "late lookup did not release its connector"
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_publication_rejects_lookup_after_shutdown():
    manager = _PublicationManagerContract({})
    server = _build_publication_server(manager)
    async with server.lifespan_context("sse"):
        pass
    with pytest.raises(RuntimeError, match="closed"):
        async with server._execution_manager.use_context("version:user"):
            pytest.fail("closed manager admitted a new request")
    assert manager.lookups == []
    assert await asyncio.to_thread(manager.closed.wait, 2)


@pytest.mark.asyncio
async def test_publication_shutdown_keeps_cached_context_alive_during_policy_refresh():
    started = asyncio.Event()
    release = asyncio.Event()
    closed = threading.Event()
    healthy = []
    context = _make_tool_context("version:user")
    context.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=closed.set))
    manager = _PublicationManagerContract({context.datasource: context})
    get_context = manager.get_or_create_context

    async def refresh_lookup(datasource, subagent=None):
        if manager.cached is not None:
            started.set()
            await release.wait()
            healthy.append(not closed.is_set())
        return await get_context(datasource, subagent)

    manager.get_or_create_context = refresh_lookup
    server = _build_publication_server(manager)
    async with server.lifespan_context("sse"):
        async with server._execution_manager.use_context(context.datasource):
            pass
        task = asyncio.create_task(server.handle_request({}, None, None, context.datasource, transport="sse"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    try:
        # Drain any cleanup already queued by shutdown before resuming refresh.
        barrier = server._execution_manager._lifetimes._pool.submit(lambda: None)
        await asyncio.wrap_future(barrier)
        release.set()
        assert await asyncio.to_thread(manager.closed.wait, 2)
        assert await asyncio.to_thread(closed.wait, 2)
        assert healthy == [True], "shutdown closed the context during legacy policy refresh"
    finally:
        release.set()


@pytest.mark.asyncio
async def test_publication_cancelled_lookup_releases_pin_before_later_eviction():
    started = asyncio.Event()
    release = asyncio.Event()
    closed = threading.Event()
    context = _make_tool_context("version:user-a")
    context.tools["db_tool"] = SimpleNamespace(connector=SimpleNamespace(close=closed.set))
    other = _make_tool_context("version:user-b")
    manager = _PublicationManagerContract({context.datasource: context, other.datasource: other})
    get_context = manager.get_or_create_context

    async def delayed_lookup(datasource, subagent=None):
        if datasource == context.datasource:
            started.set()
            await release.wait()
        return await get_context(datasource, subagent)

    manager.get_or_create_context = delayed_lookup
    server = _build_publication_server(manager)
    async with server.lifespan_context("sse"):
        task = asyncio.create_task(server.handle_request({}, None, None, context.datasource, transport="sse"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            async with server._execution_manager.use_context(other.datasource):
                pass
            assert await asyncio.to_thread(closed.wait, 2), "abandoned lookup leaked a context pin"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    assert await asyncio.to_thread(manager.closed.wait, 2)
