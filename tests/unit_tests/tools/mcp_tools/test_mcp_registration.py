# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-09: MCP Server tool registration tests.

Verifies MCP tool discovery, registration, schema generation, and
decorator mechanics without starting a real server or loading external config.
"""

from unittest.mock import MagicMock

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.utils.mcp_decorators import (
    MCPToolConfig,
    create_dynamic_tool_wrapper,
    create_static_tool_wrapper,
    get_mcp_tools,
    get_tool_registry,
    mcp_tool,
    mcp_tool_class,
)

# ---------------------------------------------------------------------------
# Test @mcp_tool decorator
# ---------------------------------------------------------------------------


class TestMCPToolDecorator:
    def test_decorator_attaches_config(self):
        @mcp_tool()
        def my_tool(self, query: str) -> FuncToolResult:
            """Search something."""
            pass

        assert hasattr(my_tool, "_mcp_config")
        assert isinstance(my_tool._mcp_config, MCPToolConfig)
        assert my_tool._mcp_config.availability_check is None

    def test_decorator_with_availability_check(self):
        @mcp_tool(availability_check="has_feature")
        def my_tool(self, query: str) -> FuncToolResult:
            """Feature-gated tool."""
            pass

        assert my_tool._mcp_config.availability_check == "has_feature"

    def test_decorator_preserves_function(self):
        @mcp_tool()
        def list_tables(self, include_views: bool = True) -> FuncToolResult:
            """List all tables."""
            pass

        assert list_tables.__name__ == "list_tables"
        assert "List all tables" in list_tables.__doc__


# ---------------------------------------------------------------------------
# Test get_mcp_tools discovery
# ---------------------------------------------------------------------------


class TestGetMCPTools:
    def test_discovers_decorated_methods(self):
        class MyTools:
            @mcp_tool()
            def tool_a(self, x: str):
                """Tool A."""
                pass

            @mcp_tool(availability_check="has_x")
            def tool_b(self, y: int):
                """Tool B."""
                pass

            def not_a_tool(self):
                pass

        tools = get_mcp_tools(MyTools)
        names = [name for name, _, _ in tools]
        assert "tool_a" in names
        assert "tool_b" in names
        assert "not_a_tool" not in names
        assert len(tools) == 2

    def test_returns_empty_for_no_tools(self):
        class EmptyClass:
            def regular_method(self):
                pass

        assert get_mcp_tools(EmptyClass) == []


# ---------------------------------------------------------------------------
# Test @mcp_tool_class decorator
# ---------------------------------------------------------------------------


class TestMCPToolClassDecorator:
    def test_registers_class_in_global_registry(self):
        initial_registry = list(get_tool_registry())

        try:

            @mcp_tool_class(name="test_tool_xyz", availability_property="has_test_xyz")
            class TestToolXYZ:
                @classmethod
                def create_dynamic(cls, agent_config, sub_agent_name=None):
                    return cls()

                @classmethod
                def create_static(cls, agent_config, sub_agent_name=None, database_name=None):
                    return cls()

                @mcp_tool()
                def do_something(self, q: str):
                    """Do something."""
                    pass

            registry = get_tool_registry()
            assert len(registry) > len(initial_registry)

            # Find our registered class
            registered = [c for c in registry if c.name == "test_tool_xyz"]
            assert len(registered) == 1
            assert registered[0].tool_class is TestToolXYZ
            assert registered[0].availability_property == "has_test_xyz"
        finally:
            # Restore the registry to avoid polluting other tests
            registry = get_tool_registry()
            registry[:] = initial_registry

    def test_requires_create_dynamic(self):
        with pytest.raises(TypeError, match="create_dynamic"):

            @mcp_tool_class(name="bad_tool", availability_property="has_bad")
            class BadTool:
                @classmethod
                def create_static(cls, agent_config, sub_agent_name=None, database_name=None):
                    return cls()

    def test_requires_create_static(self):
        with pytest.raises(TypeError, match="create_static"):

            @mcp_tool_class(name="bad_tool2", availability_property="has_bad2")
            class BadTool2:
                @classmethod
                def create_dynamic(cls, agent_config, sub_agent_name=None):
                    return cls()


# ---------------------------------------------------------------------------
# Test global tool registry contains real tool classes
# ---------------------------------------------------------------------------


class TestGlobalToolRegistry:
    def test_registry_contains_db_tool(self):
        registry = get_tool_registry()
        db_entries = [c for c in registry if c.name == "db_tool"]
        assert len(db_entries) == 1
        assert db_entries[0].availability_property == "has_db_tools"

    def test_registry_contains_context_tool(self):
        registry = get_tool_registry()
        ctx_entries = [c for c in registry if c.name == "context_tool"]
        assert len(ctx_entries) == 1
        assert ctx_entries[0].availability_property == "has_context_tools"

    def test_db_tool_has_expected_methods(self):
        from datus.tools.func_tool.database import DBFuncTool

        tools = get_mcp_tools(DBFuncTool)
        tool_names = [name for name, _, _ in tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names
        assert "list_databases" in tool_names

    def test_context_tool_has_expected_methods(self):
        from datus.tools.func_tool.context_search import ContextSearchTools

        tools = get_mcp_tools(ContextSearchTools)
        tool_names = [name for name, _, _ in tools]
        assert "list_subject_tree" in tool_names
        assert "search_metrics" in tool_names

    def test_all_tools_have_docstrings(self):
        """All MCP tools must have docstrings (used as tool descriptions)."""
        from datus.tools.func_tool.context_search import ContextSearchTools
        from datus.tools.func_tool.database import DBFuncTool

        for cls in [DBFuncTool, ContextSearchTools]:
            for name, method, _config in get_mcp_tools(cls):
                assert method.__doc__, f"{cls.__name__}.{name} is missing a docstring"


# ---------------------------------------------------------------------------
# Test dynamic tool wrapper
# ---------------------------------------------------------------------------


class TestDynamicToolWrapper:
    def test_wrapper_calls_tool_method(self):
        """Wrapper should call the actual tool method with correct args."""

        class FakeTool:
            @mcp_tool()
            def my_method(self, query: str, top_n: int = 5) -> FuncToolResult:
                """Search for items."""
                return FuncToolResult(success=1, result=f"found:{query}:{top_n}")

        fake_instance = FakeTool()
        ctx = MagicMock()
        ctx.has_db_tools = True
        ctx.db_tool = fake_instance

        wrapper = create_dynamic_tool_wrapper(
            method_name="my_method",
            method=FakeTool.my_method,
            config=FakeTool.my_method._mcp_config,
            context_getter=lambda: ctx,
            instance_attr="db_tool",
            availability_attr="has_db_tools",
            format_result=lambda r: r.model_dump() if isinstance(r, FuncToolResult) else r,
        )

        result = wrapper(query="test", top_n=3)
        assert result["success"] == 1
        assert "found:test:3" in str(result["result"])

    def test_wrapper_returns_error_when_unavailable(self):
        class FakeTool:
            @mcp_tool()
            def my_method(self, query: str) -> FuncToolResult:
                """Tool."""
                pass

        ctx = MagicMock()
        ctx.has_db_tools = False

        wrapper = create_dynamic_tool_wrapper(
            method_name="my_method",
            method=FakeTool.my_method,
            config=FakeTool.my_method._mcp_config,
            context_getter=lambda: ctx,
            instance_attr="db_tool",
            availability_attr="has_db_tools",
            format_result=lambda r: r,
        )

        result = wrapper(query="test")
        assert result["success"] == 0
        assert "not available" in result["error"]

    def test_wrapper_preserves_signature(self):
        """Wrapper should preserve the original method signature (minus self)."""
        import inspect

        class FakeTool:
            @mcp_tool()
            def search(self, query_text: str, top_n: int = 5, include_views: bool = True) -> FuncToolResult:
                """Search."""
                pass

        wrapper = create_dynamic_tool_wrapper(
            method_name="search",
            method=FakeTool.search,
            config=FakeTool.search._mcp_config,
            context_getter=lambda: None,
            instance_attr="db_tool",
            availability_attr="has_db_tools",
            format_result=lambda r: r,
        )

        sig = inspect.signature(wrapper)
        param_names = list(sig.parameters.keys())
        assert "self" not in param_names
        assert "query_text" in param_names
        assert "top_n" in param_names
        assert "include_views" in param_names
        assert sig.parameters["top_n"].default == 5

    def test_wrapper_checks_feature_availability(self):
        class FakeTool:
            has_schema = False

            @mcp_tool(availability_check="has_schema")
            def search_table(self, query: str) -> FuncToolResult:
                """Search."""
                pass

        fake_instance = FakeTool()
        ctx = MagicMock()
        ctx.has_db_tools = True
        ctx.db_tool = fake_instance

        wrapper = create_dynamic_tool_wrapper(
            method_name="search_table",
            method=FakeTool.search_table,
            config=FakeTool.search_table._mcp_config,
            context_getter=lambda: ctx,
            instance_attr="db_tool",
            availability_attr="has_db_tools",
            format_result=lambda r: r,
        )

        result = wrapper(query="test")
        assert result["success"] == 0
        assert "not available" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test static tool wrapper
# ---------------------------------------------------------------------------


class TestStaticToolWrapper:
    def test_wrapper_calls_bound_method(self):
        class FakeTool:
            @mcp_tool()
            def get_info(self) -> FuncToolResult:
                """Get info."""
                return FuncToolResult(success=1, result="info_data")

        instance = FakeTool()
        wrapper = create_static_tool_wrapper(
            method_name="get_info",
            bound_method=instance.get_info,
            config=instance.get_info._mcp_config,
            format_result=lambda r: r.model_dump() if isinstance(r, FuncToolResult) else r,
        )

        result = wrapper()
        assert result["success"] == 1
        assert result["result"] == "info_data"


# ---------------------------------------------------------------------------
# Test ToolContext dataclass
# ---------------------------------------------------------------------------


class TestToolContext:
    def test_tool_context_properties(self):
        from datus.mcp_server import ToolContext

        mock_db = MagicMock()
        mock_ctx_tool = MagicMock()

        context = ToolContext(
            namespace="test_ns",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": mock_db, "context_tool": mock_ctx_tool},
        )

        assert context.namespace == "test_ns"
        assert context.subagent is None
        assert context.db_tool is mock_db
        assert context.context_tool is mock_ctx_tool
        assert context.has_db_tools is True
        assert context.has_context_tools is True

    def test_tool_context_without_tools(self):
        from datus.mcp_server import ToolContext

        context = ToolContext(
            namespace="test_ns",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": None, "context_tool": None},
        )

        assert context.has_db_tools is False
        assert context.has_context_tools is False

    def test_tool_context_close(self):
        from datus.mcp_server import ToolContext

        mock_tool = MagicMock()
        mock_tool.connector = MagicMock()

        context = ToolContext(
            namespace="test_ns",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": mock_tool},
        )

        context.close()
        mock_tool.connector.close.assert_called_once()
        assert len(context.tools) == 0
