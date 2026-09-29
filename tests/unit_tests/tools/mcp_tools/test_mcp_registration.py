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
from datus.utils import mcp_decorators
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

        assert hasattr(my_tool, "_mcp_config")
        assert isinstance(my_tool._mcp_config, MCPToolConfig)
        assert my_tool._mcp_config.availability_check is None

    def test_decorator_with_availability_check(self):
        @mcp_tool(availability_check="has_feature")
        def my_tool(self, query: str) -> FuncToolResult:
            """Feature-gated tool."""

        assert my_tool._mcp_config.availability_check == "has_feature"

    def test_decorator_preserves_function(self):
        @mcp_tool()
        def list_tables(self, include_views: bool = True) -> FuncToolResult:
            """List all tables."""

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

            @mcp_tool(availability_check="has_x")
            def tool_b(self, y: int):
                """Tool B."""

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
        # Snapshot the real registry; ``get_tool_registry()`` returns a copy
        # so we must mutate the underlying module-level list to restore it.
        initial_registry = list(mcp_decorators._GLOBAL_TOOL_REGISTRY)

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

            registry = get_tool_registry()
            assert len(registry) > len(initial_registry)

            # Find our registered class
            registered = [c for c in registry if c.name == "test_tool_xyz"]
            assert len(registered) == 1
            assert registered[0].tool_class is TestToolXYZ
            assert registered[0].availability_property == "has_test_xyz"
        finally:
            mcp_decorators._GLOBAL_TOOL_REGISTRY[:] = initial_registry

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
        assert "execute_sql" in tool_names
        assert "list_databases" in tool_names

    def test_context_tool_has_expected_methods(self):
        from datus.tools.func_tool.context_search import ContextSearchTools

        tools = get_mcp_tools(ContextSearchTools)
        tool_names = {name for name, _, _ in tools}
        assert tool_names == {
            "list_subject_tree",
            "search_metrics",
            "search_reference_sql",
            "get_reference_sql",
            "search_semantic_objects",
        }

    def test_semantic_mcp_tools_use_builtin_dosi(self, monkeypatch):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        calls = []

        def fake_init(self, agent_config, sub_agent_name=None):
            calls.append((agent_config, sub_agent_name))

        monkeypatch.setattr(SemanticTools, "__init__", fake_init)
        config = object()
        assert isinstance(SemanticTools.create_dynamic(config, "analyst"), SemanticTools)
        assert calls == [(config, "analyst")]
        assert all(tool_config.availability_check is None for _, _, tool_config in get_mcp_tools(SemanticTools))

    def test_all_tools_have_docstrings(self):
        """All MCP tools must have docstrings (used as tool descriptions)."""
        import datus.mcp_server  # noqa: F401  (import registers every tool class)

        for tool_config in get_tool_registry():
            for name, method, _config in get_mcp_tools(tool_config.tool_class):
                assert method.__doc__, f"{tool_config.tool_class.__name__}.{name} is missing a docstring"

    @pytest.mark.parametrize(
        ("name", "availability_property", "category", "expected"),
        [
            (
                "semantic_tool",
                "has_semantic_tools",
                "semantic_tools",
                {
                    "list_metrics",
                    "get_metric",
                    "query_metrics",
                    "get_query_metrics_result",
                    "validate_semantic",
                    "attribution_analyze",
                },
            ),
            ("date_parsing_tool", "has_date_parsing_tools", "date_parsing_tools", {"parse_temporal_expressions"}),
            (
                "platform_doc_tool",
                "has_platform_doc_tools",
                "platform_doc_tools",
                {"list_document_nav", "get_document", "search_document"},
            ),
            (
                "filesystem_tool",
                "has_filesystem_tools",
                "filesystem_tools",
                {"glob", "grep", "read_file", "read_image"},
            ),
        ],
    )
    def test_read_tool_classes_are_registered(self, name, availability_property, category, expected):
        import datus.mcp_server  # noqa: F401  (import registers every tool class)

        entries = [c for c in get_tool_registry() if c.name == name]
        assert len(entries) == 1
        assert entries[0].availability_property == availability_property
        assert entries[0].tool_class.permission_category == category
        assert {n for n, _, _ in get_mcp_tools(entries[0].tool_class)} == expected

    def test_filesystem_writers_stay_off_mcp(self):
        """Only the read methods carry @mcp_tool; an MCP call runs no permission hooks."""
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        names = {n for n, _, _ in get_mcp_tools(FilesystemFuncTool)}
        assert not names & {"write_file", "edit_file", "delete_file"}

    def test_db_tool_registers_the_migration_helpers(self):
        from datus.tools.func_tool.database import DBFuncTool

        names = {n for n, _, _ in get_mcp_tools(DBFuncTool)}
        assert {"get_migration_capabilities", "suggest_table_layout", "validate_ddl"} <= names
        assert "transfer_query_result" not in names


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
            datasource="test_ns",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": mock_db, "context_tool": mock_ctx_tool},
        )

        assert context.datasource == "test_ns"
        assert context.subagent is None
        assert context.db_tool is mock_db
        assert context.context_tool is mock_ctx_tool
        assert context.has_db_tools is True
        assert context.has_context_tools is True

    def test_tool_context_without_tools(self):
        from datus.mcp_server import ToolContext

        context = ToolContext(
            datasource="test_ns",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": None, "context_tool": None},
        )

        assert context.has_db_tools is False
        assert context.has_context_tools is False

    def test_every_registered_class_has_its_context_properties(self):
        """The dynamic wrapper reads ``name`` and ``availability_property`` off the
        context; a class without its pair here would answer "not available"."""
        import datus.mcp_server  # noqa: F401  (import registers every tool class)
        from datus.mcp_server import ToolContext

        for tool_config in get_tool_registry():
            instance = MagicMock()
            context = ToolContext(
                datasource="test_ns",
                subagent=None,
                agent_config=MagicMock(),
                tools={tool_config.name: instance},
            )
            assert getattr(context, tool_config.name) is instance, tool_config.name
            assert getattr(context, tool_config.availability_property) is True, tool_config.name

            empty = ToolContext(datasource="test_ns", subagent=None, agent_config=MagicMock(), tools={})
            assert getattr(empty, tool_config.availability_property) is False, tool_config.name

    def test_tool_context_close(self):
        from datus.mcp_server import ToolContext

        mock_tool = MagicMock()
        mock_tool.connector = MagicMock()

        context = ToolContext(
            datasource="test_ns",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": mock_tool},
        )

        context.close()
        mock_tool.connector.close.assert_called_once()
        assert len(context.tools) == 0


# ---------------------------------------------------------------------------
# Plugin tool transformers on the MCP path
# ---------------------------------------------------------------------------


class _PolicyConfig:
    """The slice of AgentConfig the transformer step reads."""

    def __init__(self, policy_context=None):
        self.policy_context = policy_context or {}
        self.project_root = "/proj"

    def active_plugin_names(self):
        return None


class _FakeSemanticTools:
    permission_category = "semantic_tools"
    sub_agent_name = "analyst"

    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.calls = []

    def metric_datasets(self):
        return {"revenue": ["orders"]}

    @mcp_tool()
    def query_metrics(self, metrics: list, where: str = "") -> FuncToolResult:
        """Query metrics."""
        self.calls.append({"metrics": metrics, "where": where})
        return FuncToolResult(success=1, result="ok")


def _patch_transformers(monkeypatch, by_pattern):
    from datus.tools.middleware import tool_middleware

    monkeypatch.setattr(tool_middleware, "collect_plugin_tool_transformers", lambda active=None: by_pattern)


def _dynamic_query_metrics(instance):
    ctx = MagicMock()
    ctx.has_semantic_tools = True
    ctx.semantic_tool = instance
    return create_dynamic_tool_wrapper(
        method_name="query_metrics",
        method=_FakeSemanticTools.query_metrics,
        config=_FakeSemanticTools.query_metrics._mcp_config,
        context_getter=lambda: ctx,
        instance_attr="semantic_tool",
        availability_attr="has_semantic_tools",
        format_result=lambda r: r.model_dump() if isinstance(r, FuncToolResult) else r,
    )


class TestPluginTransformersOnMCP:
    """A metric row policy lives in the node's tool wrapper, not in SemanticTools.

    An MCP call has no node, so the wrapper has to run the same chain or the
    policy never applies to it.
    """

    def test_dynamic_wrapper_hands_the_method_the_transformed_args(self, monkeypatch):
        seen = {}

        def narrow(tool_name, args, context):
            seen.update(tool_name=tool_name, context=context)
            return {**args, "where": "region = 'EU'"}

        _patch_transformers(monkeypatch, {"semantic_tools.query_metrics": [narrow]})
        instance = _FakeSemanticTools(_PolicyConfig({"row_filter": "x"}))

        result = _dynamic_query_metrics(instance)(metrics=["revenue"], where="")

        assert result["success"] == 1
        assert instance.calls == [{"metrics": ["revenue"], "where": "region = 'EU'"}]
        # The category comes from the instance, so a category-qualified pattern matches.
        assert seen["tool_name"] == "query_metrics"
        assert seen["context"]["policy_context"] == {"row_filter": "x"}
        assert seen["context"]["metric_datasets"] == {"revenue": ["orders"]}
        assert seen["context"]["node_name"] == "analyst"
        assert seen["context"]["agent_config"] is instance.agent_config

    def test_a_refusing_transformer_denies_the_call(self, monkeypatch):
        def refuse(tool_name, args, context):
            raise ValueError("no datasets resolved for revenue")

        _patch_transformers(monkeypatch, {"semantic_tools.*": [refuse]})
        instance = _FakeSemanticTools(_PolicyConfig())

        result = _dynamic_query_metrics(instance)(metrics=["revenue"])

        assert result["success"] == 0
        assert "Denied by policy" in result["error"]
        assert "no datasets resolved" in result["error"]
        assert instance.calls == [], "a denied call must not reach the tool"

    def test_a_pattern_for_another_category_does_not_match(self, monkeypatch):
        def narrow(tool_name, args, context):
            return {**args, "where": "narrowed"}

        _patch_transformers(monkeypatch, {"db_tools.query_metrics": [narrow]})
        instance = _FakeSemanticTools(_PolicyConfig())

        _dynamic_query_metrics(instance)(metrics=["revenue"], where="as-is")

        assert instance.calls == [{"metrics": ["revenue"], "where": "as-is"}]

    def test_static_wrapper_runs_the_same_chain(self, monkeypatch):
        def narrow(tool_name, args, context):
            return {**args, "where": "region = 'EU'"}

        _patch_transformers(monkeypatch, {"semantic_tools.query_metrics": [narrow]})
        instance = _FakeSemanticTools(_PolicyConfig())
        wrapper = create_static_tool_wrapper(
            method_name="query_metrics",
            bound_method=instance.query_metrics,
            config=instance.query_metrics._mcp_config,
            format_result=lambda r: r.model_dump() if isinstance(r, FuncToolResult) else r,
        )

        wrapper(metrics=["revenue"])

        assert instance.calls == [{"metrics": ["revenue"], "where": "region = 'EU'"}]


# ---------------------------------------------------------------------------
# Filesystem tool construction for MCP
# ---------------------------------------------------------------------------


class _FsConfig:
    def __init__(self, project_root, node_config=None):
        self.project_root = project_root
        self.filesystem_strict = False  # the CLI default; MCP must not inherit it
        self.filesystem_allowlist = None
        self.path_manager = None
        self._node_config = node_config or {}

    def sub_agent_config(self, name):
        return self._node_config


class TestFilesystemToolForMCP:
    def test_create_dynamic_is_strict_even_when_the_config_is_not(self, tmp_path):
        """Outside strict mode an EXTERNAL path is left to PermissionHooks to
        confirm, and an MCP call runs no hooks."""
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        root = tmp_path / "project"
        root.mkdir()
        (root / "inside.txt").write_text("inside")
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")

        tool = FilesystemFuncTool.create_dynamic(_FsConfig(str(root)))

        assert tool.strict is True
        assert tool.read_file("inside.txt").success == 1
        refused = tool.read_file(str(outside))
        assert refused.success == 0
        assert "secret" not in str(refused.result or "")

    def test_create_dynamic_keeps_what_the_transformer_step_reads(self, tmp_path):
        """Without ``agent_config`` the MCP wrapper would fall back to every
        installed plugin and an empty policy context."""
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        config = _FsConfig(str(tmp_path))
        tool = FilesystemFuncTool.create_dynamic(config, sub_agent_name="analyst")

        assert tool.agent_config is config
        assert tool.sub_agent_name == "analyst"

    def test_read_image_reaches_the_client_as_image_content(self, tmp_path):
        """Through a real FastMCP: the image must arrive as an image block, not
        as base64 inside a JSON text block."""
        import asyncio

        from mcp.server.fastmcp import FastMCP
        from mcp.types import ImageContent, TextContent
        from PIL import Image

        from datus.mcp_server import DatusMCPServer
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
        from datus.utils.mcp_decorators import register_static_tools

        Image.new("RGB", (4, 3), "red").save(tmp_path / "chart.png")
        tool = FilesystemFuncTool.create_dynamic(_FsConfig(str(tmp_path)))
        mcp = FastMCP(name="test")
        register_static_tools(mcp, tool, DatusMCPServer._format_result)

        result = asyncio.run(mcp.call_tool("read_image", {"path": "chart.png"}))
        content = result[0] if isinstance(result, tuple) else result

        images = [block for block in content if isinstance(block, ImageContent)]
        assert len(images) == 1
        assert images[0].mimeType == "image/png"
        assert images[0].data and not images[0].data.startswith("data:")
        texts = [block for block in content if isinstance(block, TextContent)]
        assert texts and '"width": 4' in texts[0].text

    def test_create_dynamic_roots_at_the_sub_agents_workspace(self, tmp_path):
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        workspace = tmp_path / "ws"
        workspace.mkdir()
        tool = FilesystemFuncTool.create_dynamic(
            _FsConfig(str(tmp_path / "project"), node_config={"workspace_root": str(workspace)}),
            sub_agent_name="analyst",
        )

        assert tool.root_path == str(workspace)
