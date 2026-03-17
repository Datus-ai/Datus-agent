# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/mcp_server.py

All tests are CI-level: zero external dependencies, zero network access.
All external calls (AgentConfig, tools, FastMCP) are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.mcp_server import ToolContext, ToolContextManager
from datus.tools.func_tool.base import FuncToolResult

# =============================================================================
# Fixtures
# =============================================================================


def _make_agent_config(namespace="default"):
    cfg = MagicMock()
    cfg.namespace = namespace
    return cfg


def _make_tool_context(namespace="default", subagent=None):
    ctx = ToolContext(
        namespace=namespace,
        subagent=subagent,
        agent_config=_make_agent_config(namespace),
        tools={},
    )
    return ctx


# =============================================================================
# Tests: ToolContext
# =============================================================================


class TestToolContext:
    def test_namespace(self):
        ctx = _make_tool_context("sales")
        assert ctx.namespace == "sales"

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
        {"ns1": {}, "ns2": {}} if key == "namespace" else {"agent1": {}} if key == "agentic_nodes" else default
    )
    return mgr


@pytest.fixture
def context_manager(mock_config_manager):
    with patch("datus.mcp_server.configuration_manager", return_value=mock_config_manager):
        mgr = ToolContextManager(config_path=None, max_size=5)
    return mgr


class TestToolContextManagerInit:
    def test_available_namespaces(self, context_manager):
        namespaces = context_manager.available_namespaces
        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_available_subagents(self, context_manager):
        subagents = context_manager.available_subagents
        assert "agent1" in subagents

    def test_validate_namespace_known(self, context_manager):
        assert context_manager.validate_namespace("ns1") is True

    def test_validate_namespace_unknown(self, context_manager):
        assert context_manager.validate_namespace("unknown") is False

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
    async def test_creates_context_on_miss(self, context_manager):
        mock_ctx = _make_tool_context("ns1")
        with patch.object(context_manager, "_create_context", return_value=mock_ctx):
            result = await context_manager.get_or_create_context("ns1")
        assert result is mock_ctx

    @pytest.mark.asyncio
    async def test_returns_cached_context_on_hit(self, context_manager):
        mock_ctx = _make_tool_context("ns1")
        with patch.object(context_manager, "_create_context", return_value=mock_ctx):
            result1 = await context_manager.get_or_create_context("ns1")
            result2 = await context_manager.get_or_create_context("ns1")
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
            await context_manager.get_or_create_context("ns0")
            await context_manager.get_or_create_context("ns1")
            # This should evict ns0
            await context_manager.get_or_create_context("ns2")

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
            await context_manager.get_or_create_context("ns_a")
            await context_manager.get_or_create_context("ns_b")
            # Access ns_a again — should move to end
            await context_manager.get_or_create_context("ns_a")

        keys = list(context_manager._contexts.keys())
        assert keys[-1] == "ns_a:"


class TestToolContextManagerCloseAll:
    def test_close_all_clears_contexts(self, context_manager):
        ctx = _make_tool_context("ns1")
        context_manager._contexts["ns1:"] = ctx
        context_manager.close_all()
        assert context_manager._contexts == {}

    def test_close_all_handles_close_error(self, context_manager):
        ctx = MagicMock()
        ctx.close.side_effect = Exception("error")
        context_manager._contexts["ns1:"] = ctx
        context_manager.close_all()  # Should not raise
        assert context_manager._contexts == {}


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
