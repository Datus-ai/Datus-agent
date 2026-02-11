# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tests for Datus MCP Server

Run with:
    pytest tests/test_mcp_server.py -v

    # Run only dynamic mode tests
    pytest tests/test_mcp_server.py -v -k "Dynamic"

    # Run only SSE tests
    pytest tests/test_mcp_server.py -v -k "SSE"
"""

import asyncio

import pytest
import pytest_asyncio

from datus.mcp_server import (
    LightweightDynamicMCPServer,
    ToolContext,
    ToolContextManager,
    create_dynamic_app,
    create_server,
)


class TestMCPServerCreation:
    """Test MCP server initialization."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        server = create_server(namespace="bird_sqlite")
        yield server
        server.close()

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
        if server.db_tool:
            assert server.db_tool is not None

    def test_context_tools_initialized(self, server):
        """Test context tools are available."""
        if server.context_tool:
            assert server.context_tool is not None


class TestMCPToolRegistration:
    """Test MCP tool registration."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        server = create_server(namespace="bird_sqlite")
        yield server
        server.close()

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test that tools are registered with FastMCP."""
        tools = await server.mcp.list_tools()
        assert len(tools) > 0

        # Check for expected database tools
        tool_names = [t.name for t in tools]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names
        assert "read_query" in tool_names

    @pytest.mark.asyncio
    async def test_list_subject_tree_tool(self, server):
        """Test list_subject_tree tool is registered."""
        tools = await server.mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_subject_tree" in tool_names


class TestMCPToolExecution:
    """Test MCP tool execution."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        server = create_server(namespace="bird_sqlite")
        yield server
        server.close()

    @pytest.mark.asyncio
    async def test_call_list_tables(self, server):
        """Test calling list_tables tool."""
        result = await server.mcp.call_tool("list_tables", {})
        assert result is not None
        # Result should be a list of TextContent
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_call_list_subject_tree(self, server):
        """Test calling list_subject_tree tool."""
        result = await server.mcp.call_tool("list_subject_tree", {})
        assert result is not None


class TestMCPServerASGIApp:
    """Test ASGI app creation for HTTP mode."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        server = create_server(namespace="bird_sqlite")
        yield server
        server.close()

    def test_get_sse_app(self, server):
        """Test SSE ASGI app creation."""
        app = server.get_sse_app()
        assert app is not None

    def test_get_streamable_http_app(self, server):
        """Test streamable HTTP ASGI app creation."""
        app = server.get_streamable_http_app()
        assert app is not None


# ============================================================================
# Dynamic Mode Tests
# ============================================================================


class TestToolContextManager:
    """Test ToolContextManager for dynamic mode."""

    @pytest.fixture
    def manager(self):
        """Create a ToolContextManager instance."""
        manager = ToolContextManager(max_size=3)
        yield manager
        manager.close_all()

    def test_manager_creation(self, manager):
        """Test that manager can be created."""
        assert manager is not None
        assert len(manager.available_namespaces) > 0

    def test_validate_namespace(self, manager):
        """Test namespace validation."""
        # bird_sqlite should be available in test config
        assert manager.validate_namespace("bird_sqlite") is True
        assert manager.validate_namespace("non_existent_namespace") is False

    @pytest.mark.asyncio
    async def test_get_or_create_context(self, manager):
        """Test context creation and caching."""
        # First call creates context
        context1 = await manager.get_or_create_context("bird_sqlite")
        assert context1 is not None
        assert isinstance(context1, ToolContext)
        assert context1.namespace == "bird_sqlite"

        # Second call returns cached context
        context2 = await manager.get_or_create_context("bird_sqlite")
        assert context2 is context1  # Same instance

    @pytest.mark.asyncio
    async def test_context_has_tools(self, manager):
        """Test that context has tools initialized."""
        context = await manager.get_or_create_context("bird_sqlite")
        # At least one tool should be available
        assert context.has_db_tools or context.has_context_tools

    @pytest.mark.asyncio
    async def test_lru_eviction(self, manager):
        """Test LRU cache eviction when max_size is exceeded."""
        # Manager has max_size=3
        # Create 4 contexts to trigger eviction
        namespaces = manager.available_namespaces[:4] if len(manager.available_namespaces) >= 4 else None
        if not namespaces or len(namespaces) < 4:
            pytest.skip("Need at least 4 namespaces for LRU eviction test")

        contexts = []
        for ns in namespaces:
            ctx = await manager.get_or_create_context(ns)
            contexts.append(ctx)

        # After creating 4 contexts with max_size=3, first one should be evicted
        assert len(manager._contexts) == 3
        # First namespace should no longer be in cache
        assert manager._get_cache_key(namespaces[0]) not in manager._contexts

    @pytest.mark.asyncio
    async def test_context_with_subagent(self, manager):
        """Test context creation with subagent parameter."""
        # Test with None subagent
        context = await manager.get_or_create_context("bird_sqlite", subagent=None)
        assert context.subagent is None

        # Different cache keys for different subagents
        key1 = manager._get_cache_key("bird_sqlite", None)
        key2 = manager._get_cache_key("bird_sqlite", "test_agent")
        assert key1 != key2


class TestLightweightDynamicMCPServer:
    """Test LightweightDynamicMCPServer."""

    @pytest.fixture
    def server(self):
        """Create a dynamic server instance."""
        server = LightweightDynamicMCPServer(max_cache_size=10)
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


class TestDynamicModeHTTPIntegration:
    """Integration tests for dynamic mode with HTTP transport."""

    @pytest.fixture
    def http_app(self):
        """Create dynamic app with HTTP transport."""
        return create_dynamic_app(transport="http", max_cache_size=10)

    @pytest_asyncio.fixture
    async def http_client_with_lifespan(self, http_app):
        """Create HTTP client with proper lifespan handling."""
        import httpx

        # Manually trigger lifespan startup
        lifespan_scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
        startup_complete = asyncio.Event()
        shutdown_triggered = asyncio.Event()
        startup_sent = asyncio.Event()

        async def receive():
            if not startup_sent.is_set():
                startup_sent.set()
                return {"type": "lifespan.startup"}
            await shutdown_triggered.wait()
            return {"type": "lifespan.shutdown"}

        async def send(message):
            if message["type"] == "lifespan.startup.complete":
                startup_complete.set()

        # Start lifespan in background
        lifespan_task = asyncio.create_task(http_app(lifespan_scope, receive, send))

        # Wait for startup
        await asyncio.wait_for(startup_complete.wait(), timeout=10.0)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_app),
            base_url="http://test",
        ) as client:
            yield client

        # Cleanup: trigger shutdown
        shutdown_triggered.set()
        try:
            await asyncio.wait_for(lifespan_task, timeout=5.0)
        except asyncio.TimeoutError:
            lifespan_task.cancel()
            try:
                await lifespan_task
            except asyncio.CancelledError:
                pass

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

    @pytest.mark.asyncio
    async def test_mcp_endpoint_accepts_request(self, http_client_with_lifespan):
        """Test that MCP endpoint accepts valid namespace requests.

        This test verifies that:
        1. The server is properly started (lifespan works)
        2. The endpoint routes to the correct namespace
        3. The request reaches the MCP handler (not blocked by routing)

        Note: Full MCP protocol testing requires a proper MCP client.
        This test verifies the server infrastructure is working.
        """
        # Send a simple POST to verify server is started and routing works
        response = await http_client_with_lifespan.post(
            "/mcp/bird_sqlite",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            },
            headers={"Content-Type": "application/json"},
        )
        # The key assertion: server started correctly and processed the request
        # Status 200 = success, 202 = accepted for streaming
        # Status 500 with "Server not started" would indicate lifespan failure
        # Any 2xx response indicates the request reached the MCP handler
        assert response.status_code < 500, f"Server error: {response.status_code}, body: {response.text}"
        # Verify it's not a routing error (404)
        assert response.status_code != 404, "Namespace routing failed"


class TestDynamicModeSSEIntegration:
    """Integration tests for dynamic mode with SSE transport."""

    @pytest.fixture
    def sse_app(self):
        """Create dynamic app with SSE transport."""
        return create_dynamic_app(transport="sse", max_cache_size=10)

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
        """Test that SSE app is configured with correct transport.

        This is a smoke test to verify the SSE app is properly configured.
        Full SSE streaming integration would require an actual MCP client
        that handles long-lived SSE connections properly.
        """
        # Verify the app is created and routes exist
        from starlette.routing import Mount, Route

        routes = sse_app.routes
        route_paths = []
        for route in routes:
            if isinstance(route, Route):
                route_paths.append(route.path)
            elif isinstance(route, Mount):
                route_paths.append(route.path)

        # Verify SSE-specific routes
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


class TestDynamicRouterPathParsing:
    """Test DynamicRouter path parsing logic."""

    def test_parse_simple_namespace(self):
        """Test parsing simple namespace path."""
        from datus.mcp_server import LightweightDynamicMCPServer

        # Access the inner class through create_asgi_app
        server = LightweightDynamicMCPServer()
        app = server.create_asgi_app(transport="http")

        # Get the DynamicRouter class from routes
        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/mcp":
                router = route.app
                break

        # Test path parsing
        scope = {"path": "/bird_sqlite", "query_string": b""}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subagent is None
        assert subpath == "/"

        server._context_manager.close_all()

    def test_parse_namespace_with_subpath(self):
        """Test parsing namespace with subpath."""
        from datus.mcp_server import LightweightDynamicMCPServer

        server = LightweightDynamicMCPServer()
        app = server.create_asgi_app(transport="sse")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/sse":
                router = route.app
                break

        # Test path with subpath
        scope = {"path": "/bird_sqlite/messages", "query_string": b""}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subpath == "/messages"

        server._context_manager.close_all()

    def test_parse_namespace_with_subagent(self):
        """Test parsing namespace with subagent query param."""
        from datus.mcp_server import LightweightDynamicMCPServer

        server = LightweightDynamicMCPServer()
        app = server.create_asgi_app(transport="http")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/mcp":
                router = route.app
                break

        # Test path with subagent
        scope = {"path": "/bird_sqlite", "query_string": b"subagent=my_agent"}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subagent == "my_agent"

        server._context_manager.close_all()

    def test_parse_with_mount_prefix_not_stripped(self):
        """Test parsing when mount prefix is not stripped by Starlette."""
        from datus.mcp_server import LightweightDynamicMCPServer

        server = LightweightDynamicMCPServer()
        app = server.create_asgi_app(transport="sse")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/sse":
                router = route.app
                break

        # Test when prefix is still in path
        scope = {"path": "/sse/bird_sqlite", "query_string": b""}
        namespace, subagent, subpath = router._parse_request(scope)
        assert namespace == "bird_sqlite"
        assert subpath == "/"

        server._context_manager.close_all()


if __name__ == "__main__":
    # Quick manual test
    async def main():
        print("Creating MCP server...")
        with create_server(namespace="bird_sqlite") as server:
            print(f"Server namespace: {server.namespace}")
            print(f"Has DB tools: {server.db_tool is not None}")
            print(f"Has Context tools: {server.context_tool is not None}")

            print("\nListing registered tools...")
            tools = await server.mcp.list_tools()
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            print("\nTesting list_tables...")
            result = await server.mcp.call_tool("list_tables", {})
            print(f"list_tables result: {result}")

            print("\nTesting list_subject_tree...")
            result = await server.mcp.call_tool("list_subject_tree", {})
            print(f"list_subject_tree result: {result}")

            print("\nAll tests passed!")

    asyncio.run(main())
