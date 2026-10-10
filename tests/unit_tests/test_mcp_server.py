# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/mcp_server.py

All tests are CI-level: zero external dependencies, zero network access.
All external calls (AgentConfig, tools, FastMCP) are mocked.
"""

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from datus.mcp_server import DatusMCPServer, ToolContext, ToolContextManager
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools
from datus.utils.mcp_decorators import ToolClassConfig

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
    return mgr


class TestToolContextManagerInit:
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


@pytest.fixture(params=["dynamic", "static"])
def initialize_mcp_tools(
    request: pytest.FixtureRequest, context_manager: ToolContextManager
) -> Iterator[Callable[[list[ToolClassConfig]], dict[str, Any]]]:
    agent_config = _make_agent_config("ns1")

    def initialize(registry: list[ToolClassConfig]) -> dict[str, Any]:
        with patch("datus.mcp_server.get_tool_registry", return_value=registry):
            if request.param == "dynamic":
                return context_manager._create_context("ns1", subagent="agent1").tools
            return DatusMCPServer("ns1", sub_agent="agent1").tools

    with (
        patch("datus.mcp_server.load_agent_config", return_value=agent_config),
        patch("datus.mcp_server.FastMCP"),
        patch.object(DatusMCPServer, "_register_tools"),
    ):
        yield initialize


@pytest.fixture
def reference_template_dependencies() -> Iterator[tuple[MagicMock, list[ToolClassConfig]]]:
    database_tool = MagicMock(spec=DBFuncTool)
    database_tool.read_query.return_value = FuncToolResult(result=[{"product_id": 42}])
    registry = [
        ToolClassConfig("db_tool", DBFuncTool, "has_db_tools"),
        ToolClassConfig("reference_template_tool", ReferenceTemplateTools, "has_reference_template_tools"),
    ]
    with (
        patch("datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG") as store_class,
        patch.object(DBFuncTool, "create_dynamic", return_value=database_tool),
        patch.object(DBFuncTool, "create_static", return_value=database_tool),
    ):
        store_class.return_value.get_reference_template_size.return_value = 1
        store_class.return_value.get_reference_template_detail.return_value = [
            {
                "name": "product_lookup",
                "template": "SELECT {{ product_id }} AS product_id",
                "parameters": '[{"name": "product_id"}]',
            }
        ]
        yield database_tool, registry


class TestMCPReferenceTemplateExecution:
    @pytest.mark.parametrize("template_first", [False, True], ids=["database-first", "template-first"])
    def test_initialized_template_executes_with_database_tool(
        self,
        initialize_mcp_tools: Callable[[list[ToolClassConfig]], dict[str, Any]],
        reference_template_dependencies: tuple[MagicMock, list[ToolClassConfig]],
        template_first: bool,
    ) -> None:
        database_tool, registry = reference_template_dependencies
        if template_first:
            registry = list(reversed(registry))
        tools = initialize_mcp_tools(registry)
        template_tool = tools["reference_template_tool"]

        result = template_tool.execute_reference_template(
            subject_path=["retail", "products"],
            name="product_lookup",
            params='{"product_id": 42}',
            datasource="ns1",
        )

        assert result.success == 1, result.error
        assert result.result["rendered_sql"] == "SELECT 42 AS product_id"
        assert result.result["query_result"] == [{"product_id": 42}]
        assert template_tool.db_func_tool is tools["db_tool"] is database_tool
        database_tool.read_query.assert_called_once_with("SELECT 42 AS product_id", datasource="ns1")
        assert "execute_reference_template" in {tool.name for tool in template_tool.available_tools()}

    @pytest.mark.parametrize("missing_tool", ["db_tool", "reference_template_tool"])
    @pytest.mark.parametrize("failure_mode", ["unregistered", "initialization-failed"])
    def test_unavailable_dependency_preserves_remaining_tools(
        self,
        initialize_mcp_tools: Callable[[list[ToolClassConfig]], dict[str, Any]],
        reference_template_dependencies: tuple[MagicMock, list[ToolClassConfig]],
        missing_tool: str,
        failure_mode: str,
    ) -> None:
        database_tool, registry = reference_template_dependencies
        missing_class = next(entry.tool_class for entry in registry if entry.name == missing_tool)
        if failure_mode == "unregistered":
            registry = [entry for entry in registry if entry.name != missing_tool]
        with (
            patch.object(missing_class, "create_dynamic", side_effect=RuntimeError("Tool unavailable")),
            patch.object(missing_class, "create_static", side_effect=RuntimeError("Tool unavailable")),
        ):
            tools = initialize_mcp_tools(registry)

        assert tools.get(missing_tool) is None
        if missing_tool == "db_tool":
            template_tool = tools["reference_template_tool"]
            assert template_tool.db_func_tool is None
            result = template_tool.render_reference_template(
                ["retail", "products"], "product_lookup", '{"product_id": 42}'
            )
            assert result.success == 1, result.error
            assert result.result["rendered_sql"] == "SELECT 42 AS product_id"
        else:
            assert tools["db_tool"] is database_tool
        database_tool.read_query.assert_not_called()


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
