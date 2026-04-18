# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for Datus MCP Server.

Tests that require real database connections or knowledge base data
are in tests/integration/tools/test_mcp_server.py (marked nightly).
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.mcp_server import (
    DatusMCPServer,
    LightweightDynamicMCPServer,
    ToolContext,
    ToolContextManager,
    create_dynamic_app,
    create_server,
)
from tests.conftest import TEST_CONF_DIR

TEST_CONF_PATH = str(TEST_CONF_DIR / "agent.yml")


# =============================================================================
# Static Mode: Server Creation & ASGI App (mocked tools)
# =============================================================================


class TestMCPServerCreation:
    """Test MCP server initialization with mocked tools."""

    @pytest.fixture
    def server(self):
        with patch.object(DatusMCPServer, "_init_tools"), patch.object(DatusMCPServer, "_register_tools"):
            server = create_server(namespace="bird_sqlite", config_path=TEST_CONF_PATH)
            server.tools = {
                "db_tool": MagicMock(),
                "context_tool": MagicMock(),
            }
            yield server

    def test_server_creation(self, server):
        """Test that server can be created."""
        assert server is not None
        assert server.namespace == "bird_sqlite"
        assert server.mcp is not None

    def test_server_has_tools(self, server):
        """Test that tools are initialized."""
        assert server.db_tool is not None or server.context_tool is not None

    def test_db_tools_initialized(self, server):
        """Test database tools are available."""
        assert server.db_tool is not None

    def test_context_tools_initialized(self, server):
        """Test context tools are available."""
        assert server.context_tool is not None


class TestMCPServerASGIApp:
    """Test ASGI app creation with mocked tools."""

    @pytest.fixture
    def server(self):
        with patch.object(DatusMCPServer, "_init_tools"), patch.object(DatusMCPServer, "_register_tools"):
            server = create_server(namespace="bird_sqlite", config_path=TEST_CONF_PATH)
            server.tools = {}
            yield server

    def test_get_sse_app(self, server):
        """Test SSE ASGI app creation."""
        app = server.get_sse_app()
        assert app is not None

    def test_get_streamable_http_app(self, server):
        """Test streamable HTTP ASGI app creation."""
        app = server.get_streamable_http_app()
        assert app is not None


# =============================================================================
# Dynamic Mode: ToolContextManager (mocked context creation)
# =============================================================================


class TestToolContextManager:
    """Test ToolContextManager caching and validation logic."""

    @pytest.fixture
    def manager(self):
        manager = ToolContextManager(config_path=TEST_CONF_PATH, max_size=3)
        yield manager
        manager.close_all()

    def test_manager_creation(self, manager):
        """Test that manager can be created."""
        assert manager is not None
        assert len(manager.available_namespaces) > 0

    def test_validate_namespace(self, manager):
        """Test namespace validation."""
        assert manager.validate_namespace("bird_sqlite") is True
        assert manager.validate_namespace("non_existent_namespace") is False

    @pytest.mark.asyncio
    async def test_get_or_create_context(self, manager):
        """Test context creation and caching."""
        mock_context = ToolContext(
            namespace="bird_sqlite",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": MagicMock()},
        )
        with patch.object(manager, "_create_context", return_value=mock_context):
            context1 = await manager.get_or_create_context("bird_sqlite")
            assert context1 is not None
            assert isinstance(context1, ToolContext)
            assert context1.namespace == "bird_sqlite"

            # Second call returns cached context
            context2 = await manager.get_or_create_context("bird_sqlite")
            assert context2 is context1

    @pytest.mark.asyncio
    async def test_context_has_tools(self, manager):
        """Test that context has tools initialized."""
        mock_context = ToolContext(
            namespace="bird_sqlite",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": MagicMock(), "context_tool": MagicMock()},
        )
        with patch.object(manager, "_create_context", return_value=mock_context):
            context = await manager.get_or_create_context("bird_sqlite")
            assert context.has_db_tools or context.has_context_tools

    @pytest.mark.asyncio
    async def test_lru_eviction(self, manager):
        """Test LRU cache eviction when max_size is exceeded."""
        namespaces = list(manager.available_namespaces)[:4]
        if len(namespaces) < 4:
            pytest.skip("Need at least 4 namespaces for LRU eviction test")

        def mock_create(ns, subagent=None):
            return ToolContext(
                namespace=ns,
                subagent=subagent,
                agent_config=MagicMock(),
                tools={"db_tool": MagicMock()},
            )

        with patch.object(manager, "_create_context", side_effect=mock_create):
            for ns in namespaces:
                await manager.get_or_create_context(ns)

            # After creating 4 contexts with max_size=3, first one should be evicted
            assert len(manager._contexts) == 3
            assert manager._get_cache_key(namespaces[0]) not in manager._contexts

    @pytest.mark.asyncio
    async def test_context_with_subagent(self, manager):
        """Test context creation with subagent parameter."""
        mock_context = ToolContext(
            namespace="bird_sqlite",
            subagent=None,
            agent_config=MagicMock(),
            tools={},
        )
        with patch.object(manager, "_create_context", return_value=mock_context):
            context = await manager.get_or_create_context("bird_sqlite", subagent=None)
            assert context.subagent is None

        # Different cache keys for different subagents
        key1 = manager._get_cache_key("bird_sqlite", None)
        key2 = manager._get_cache_key("bird_sqlite", "test_agent")
        assert key1 != key2


# =============================================================================
# Dynamic Mode: LightweightDynamicMCPServer (lazy init, no mocking needed)
# =============================================================================


class TestLightweightDynamicMCPServer:
    """Test LightweightDynamicMCPServer."""

    @pytest.fixture
    def server(self):
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH, max_cache_size=10)
        yield server
        server._context_manager.close_all()

    def test_server_creation(self, server):
        """Test that dynamic server can be created."""
        assert server is not None
        assert server.mcp is not None
        assert len(server.available_namespaces) > 0

    def test_validate_namespace(self, server):
        """Test namespace validation."""
        assert server.validate_namespace("bird_sqlite") is True
        assert server.validate_namespace("invalid_ns") is False

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test that tools are registered with FastMCP."""
        tools = await server.mcp.list_tools()
        assert len(tools) > 0

        tool_names = [t.name for t in tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names


# =============================================================================
# Dynamic Mode: HTTP Endpoints (lazy init, no mocking needed)
# =============================================================================


class TestDynamicModeHTTPEndpoints:
    """Test dynamic mode HTTP endpoints (no DB connection needed)."""

    @pytest.fixture
    def http_app(self):
        return create_dynamic_app(config_path=TEST_CONF_PATH, transport="http", max_cache_size=10)

    @pytest.mark.asyncio
    async def test_root_endpoint(self, http_app):
        """Test root endpoint returns server info."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert data["service"] == "Datus MCP Server"
            assert data["mode"] == "lightweight-dynamic"
            assert data["transport"] == "http"
            assert "available_namespaces" in data
            assert "bird_sqlite" in data["available_namespaces"]

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_app):
        """Test health endpoint."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["transport"] == "http"

    @pytest.mark.asyncio
    async def test_invalid_namespace_returns_404(self, http_app):
        """Test that invalid namespace returns 404."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_app),
            base_url="http://test",
        ) as client:
            response = await client.post("/mcp/invalid_namespace_xyz")
            assert response.status_code == 404
            assert "not available" in response.json()["error"]


# =============================================================================
# Dynamic Mode: SSE Endpoints (lazy init, no mocking needed)
# =============================================================================


class TestDynamicModeSSEEndpoints:
    """Test dynamic mode SSE endpoints (no DB connection needed)."""

    @pytest.fixture
    def sse_app(self):
        return create_dynamic_app(config_path=TEST_CONF_PATH, transport="sse", max_cache_size=10)

    @pytest.mark.asyncio
    async def test_root_endpoint_sse(self, sse_app):
        """Test root endpoint returns SSE-specific info."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sse_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert data["transport"] == "sse"
            assert "sse" in data["endpoints"]
            assert "/sse/{namespace}" in data["endpoints"]["sse"]

    @pytest.mark.asyncio
    async def test_health_endpoint_sse(self, sse_app):
        """Test health endpoint for SSE mode."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sse_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["transport"] == "sse"

    def test_sse_app_has_correct_transport(self, sse_app):
        """Test that SSE app is configured with correct transport."""
        from starlette.routing import Mount, Route

        routes = sse_app.routes
        route_paths = []
        for route in routes:
            if isinstance(route, Route):
                route_paths.append(route.path)
            elif isinstance(route, Mount):
                route_paths.append(route.path)

        assert "/sse" in route_paths, f"Expected /sse route, got {route_paths}"
        assert "/messages/" in route_paths, f"Expected /messages/ route, got {route_paths}"
        assert "/messages" in route_paths, f"Expected /messages route, got {route_paths}"

    @pytest.mark.asyncio
    async def test_sse_invalid_namespace_returns_404(self, sse_app):
        """Test that invalid namespace returns 404 for SSE."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sse_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/sse/invalid_namespace_xyz")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_messages_endpoint_requires_session_id(self, sse_app):
        """Test that /messages/ endpoint requires session_id."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sse_app),
            base_url="http://test",
        ) as client:
            response = await client.post("/messages/", json={})
            assert response.status_code == 400
            assert "session_id" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_messages_endpoint_unknown_session_returns_404(self, sse_app):
        """Test that /messages/ with unknown session returns 404."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sse_app),
            base_url="http://test",
        ) as client:
            response = await client.post("/messages/?session_id=unknown123", json={})
            assert response.status_code == 404
            assert "Unknown session_id" in response.json()["error"]


# =============================================================================
# Dynamic Mode: Router Path Parsing (pure logic, no mocking needed)
# =============================================================================


class TestDynamicRouterPathParsing:
    """Test DynamicRouter path parsing logic."""

    def test_parse_simple_namespace(self):
        """Test parsing simple namespace path."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="http")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/mcp":
                router = route.app
                break

        scope = {"path": "/bird_sqlite", "query_string": b""}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subagent is None
        assert subpath == "/"

        server._context_manager.close_all()

    def test_parse_namespace_with_subpath(self):
        """Test parsing namespace with subpath."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="sse")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/sse":
                router = route.app
                break

        scope = {"path": "/bird_sqlite/messages", "query_string": b""}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subpath == "/messages"

        server._context_manager.close_all()

    def test_parse_namespace_with_subagent(self):
        """Test parsing namespace with subagent query param."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="http")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/mcp":
                router = route.app
                break

        scope = {"path": "/bird_sqlite", "query_string": b"subagent=my_agent"}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subagent == "my_agent"

        server._context_manager.close_all()

    def test_parse_with_mount_prefix_not_stripped(self):
        """Test parsing when mount prefix is not stripped by Starlette."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="sse")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/sse":
                router = route.app
                break

        scope = {"path": "/sse/bird_sqlite", "query_string": b""}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subpath == "/"

        server._context_manager.close_all()
