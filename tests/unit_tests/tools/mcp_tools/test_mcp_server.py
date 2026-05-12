# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for Datus MCP Server.

Tests that require real database connections or knowledge base data
are in tests/integration/tools/test_mcp_server.py (marked nightly).
"""

import sys
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
from datus.tools.mcp_tools.mcp_server import SilentMCPServerStdio, find_mcp_directory
from tests.conftest import TEST_CONF_DIR

TEST_CONF_PATH = str(TEST_CONF_DIR / "agent.yml")


def _make_stdio_params(command="python", args=None, env=None):
    """Build a simple MCPServerStdioParams-like object via mock."""
    params = MagicMock()
    params.command = command
    params.args = args or []
    params.env = env or {}
    return params


# =============================================================================
# Static Mode: Server Creation & ASGI App (mocked tools)
# =============================================================================


class TestMCPServerCreation:
    """Test MCP server initialization with mocked tools."""

    @pytest.fixture
    def server(self):
        with patch.object(DatusMCPServer, "_init_tools"), patch.object(DatusMCPServer, "_register_tools"):
            server = create_server(datasource="bird_sqlite", config_path=TEST_CONF_PATH)
            server.tools = {
                "db_tool": MagicMock(),
                "context_tool": MagicMock(),
            }
            yield server

    def test_server_creation(self, server):
        """Test that server can be created."""
        assert server.datasource == "bird_sqlite"
        assert server._stateless_http is False

    def test_server_has_tools(self, server):
        """Test that tools are initialized."""
        assert set(server.tools) == {"db_tool", "context_tool"}

    def test_db_tools_initialized(self, server):
        """Test database tools are available."""
        assert server.db_tool is server.tools["db_tool"]

    def test_context_tools_initialized(self, server):
        """Test context tools are available."""
        assert server.context_tool is server.tools["context_tool"]


class TestMCPServerASGIApp:
    """Test ASGI app creation with mocked tools."""

    @pytest.fixture
    def server(self):
        with patch.object(DatusMCPServer, "_init_tools"), patch.object(DatusMCPServer, "_register_tools"):
            server = create_server(datasource="bird_sqlite", config_path=TEST_CONF_PATH)
            server.tools = {}
            yield server

    def test_get_sse_app(self, server):
        """Test SSE ASGI app creation."""
        app = server.get_sse_app()
        assert hasattr(app, "routes")

    def test_get_streamable_http_app(self, server):
        """Test streamable HTTP ASGI app creation."""
        app = server.get_streamable_http_app()
        assert hasattr(app, "routes")


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
        assert manager.config_path == TEST_CONF_PATH
        assert "bird_sqlite" in manager.available_datasources

    def test_validate_datasource(self, manager):
        """Test datasource validation."""
        assert manager.validate_datasource("bird_sqlite") is True
        assert manager.validate_datasource("non_existent_datasource") is False

    @pytest.mark.asyncio
    async def test_get_or_create_context(self, manager):
        """Test context creation and caching."""
        mock_context = ToolContext(
            datasource="bird_sqlite",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": MagicMock()},
        )
        with patch.object(manager, "_create_context", return_value=mock_context):
            context1 = await manager.get_or_create_context(datasource="bird_sqlite")
            assert isinstance(context1, ToolContext)
            assert context1.datasource == "bird_sqlite"

            # Second call returns cached context
            context2 = await manager.get_or_create_context(datasource="bird_sqlite")
            assert context2 is context1

    @pytest.mark.asyncio
    async def test_context_has_tools(self, manager):
        """Test that context has tools initialized."""
        mock_context = ToolContext(
            datasource="bird_sqlite",
            subagent=None,
            agent_config=MagicMock(),
            tools={"db_tool": MagicMock(), "context_tool": MagicMock()},
        )
        with patch.object(manager, "_create_context", return_value=mock_context):
            context = await manager.get_or_create_context(datasource="bird_sqlite")
            assert context.has_db_tools or context.has_context_tools

    @pytest.mark.asyncio
    async def test_lru_eviction(self, manager):
        """Test LRU cache eviction when max_size is exceeded."""
        datasources = list(manager.available_datasources)[:4]
        if len(datasources) < 4:
            pytest.skip("Need at least 4 datasources for LRU eviction test")

        def mock_create(ds, subagent=None):
            return ToolContext(
                datasource=ds,
                subagent=subagent,
                agent_config=MagicMock(),
                tools={"db_tool": MagicMock()},
            )

        with patch.object(manager, "_create_context", side_effect=mock_create):
            for ds in datasources:
                await manager.get_or_create_context(datasource=ds)

            # After creating 4 contexts with max_size=3, first one should be evicted
            assert len(manager._contexts) == 3
            assert manager._get_cache_key(datasources[0]) not in manager._contexts

    @pytest.mark.asyncio
    async def test_context_with_subagent(self, manager):
        """Test context creation with subagent parameter."""
        mock_context = ToolContext(
            datasource="bird_sqlite",
            subagent=None,
            agent_config=MagicMock(),
            tools={},
        )
        with patch.object(manager, "_create_context", return_value=mock_context):
            context = await manager.get_or_create_context(datasource="bird_sqlite", subagent=None)
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
        assert server.config_path == TEST_CONF_PATH
        assert "bird_sqlite" in server.available_datasources

    def test_validate_datasource(self, server):
        """Test datasource validation."""
        assert server.validate_datasource("bird_sqlite") is True
        assert server.validate_datasource("invalid_ns") is False

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test that tools are registered with FastMCP."""
        tools = await server.mcp.list_tools()

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
            assert "available_datasources" in data
            assert "bird_sqlite" in data["available_datasources"]

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
    async def test_invalid_datasource_returns_404(self, http_app):
        """Test that invalid datasource returns 404."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_app),
            base_url="http://test",
        ) as client:
            response = await client.post("/mcp/invalid_datasource_xyz")
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
            assert "/sse/{datasource}" in data["endpoints"]["sse"]

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
    async def test_sse_invalid_datasource_returns_404(self, sse_app):
        """Test that invalid datasource returns 404 for SSE."""
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sse_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/sse/invalid_datasource_xyz")
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

    def test_parse_simple_datasource(self):
        """Test parsing simple datasource path."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="http")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/mcp":
                router = route.app
                break

        scope = {"path": "/bird_sqlite", "query_string": b""}
        datasource, subagent, subpath = router._parse_request(scope)
        assert datasource == "bird_sqlite"
        assert subagent is None
        assert subpath == "/"

        server._context_manager.close_all()

    def test_parse_datasource_with_subpath(self):
        """Test parsing datasource with subpath."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="sse")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/sse":
                router = route.app
                break

        scope = {"path": "/bird_sqlite/messages", "query_string": b""}
        datasource, subagent, subpath = router._parse_request(scope)
        assert datasource == "bird_sqlite"
        assert subpath == "/messages"

        server._context_manager.close_all()

    def test_parse_datasource_with_subagent(self):
        """Test parsing datasource with subagent query param."""
        server = LightweightDynamicMCPServer(config_path=TEST_CONF_PATH)
        app = server.create_asgi_app(transport="http")

        from starlette.routing import Mount

        for route in app.routes:
            if isinstance(route, Mount) and route.path == "/mcp":
                router = route.app
                break

        scope = {"path": "/bird_sqlite", "query_string": b"subagent=my_agent"}
        datasource, subagent, subpath = router._parse_request(scope)
        assert datasource == "bird_sqlite"
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
        datasource, subagent, subpath = router._parse_request(scope)
        assert datasource == "bird_sqlite"
        assert subpath == "/"

        server._context_manager.close_all()


# =============================================================================
# MCP server module helpers merged from test_mcp_server_module.py
# =============================================================================


class TestSilentMCPServerStdio:
    """Test that SilentMCPServerStdio correctly wraps command with stderr redirection."""

    def test_unix_wraps_with_sh(self):
        params = _make_stdio_params(command="uvicorn", args=["app:app"], env={})
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch.object(sys, "platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params.command == "sh"
        assert params.args[0] == "-c"
        assert "2>/dev/null" in params.args[1]
        assert params.env is None

    def test_windows_wraps_with_cmd(self):
        params = _make_stdio_params(command="node", args=["server.js"], env={})
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "win32"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params.command == "cmd"
        assert params.args[0] == "/c"
        assert "2>nul" in params.args[1]

    def test_env_vars_excluded_from_shell_env(self):
        env = {"API_KEY": "secret", "BASH_FUNC_xyz": "bad", "SHLVL": "2", "PATH": "/usr/bin"}
        params = _make_stdio_params(command="python", args=[], env=env)
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params.env is None
        cmd_str = params.args[1]
        assert "API_KEY" in cmd_str
        assert "BASH_FUNC_xyz" not in cmd_str
        assert "SHLVL" not in cmd_str

    def test_dict_params_also_handled(self):
        params = {"command": "echo", "args": ["hello"], "env": {}}
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        assert params["command"] == "sh"
        assert params["env"] is None

    def test_args_quoted_properly(self):
        params = _make_stdio_params(command="my server", args=["--flag with space"], env={})
        with patch("datus.tools.mcp_tools.mcp_server.MCPServerStdio.__init__", return_value=None):
            with patch("sys.platform", "linux"):
                srv = SilentMCPServerStdio.__new__(SilentMCPServerStdio)
                SilentMCPServerStdio.__init__(srv, params)
        cmd_str = params.args[1]
        assert "'my server'" in cmd_str


class TestFindMcpDirectory:
    def test_finds_relative_path_when_exists(self, tmp_path, monkeypatch):
        mcp_dir = tmp_path / "mcp" / "my-server"
        mcp_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        result = find_mcp_directory("my-server")
        assert "my-server" in result

    def test_finds_via_sys_path_site_packages(self, tmp_path, monkeypatch):
        site_pkg = tmp_path / "site-packages"
        mcp_dir = site_pkg / "mcp" / "test-server"
        mcp_dir.mkdir(parents=True)

        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        with patch("sys.path", [str(site_pkg)]):
            result = find_mcp_directory("test-server")

        assert "test-server" in result, f"Expected 'test-server' in result path, got: {result!r}"
        assert "site-packages" in result, f"Expected result to come from site-packages, got: {result!r}"

    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("sys.path", []):
            with pytest.raises(FileNotFoundError, match="not found"):
                find_mcp_directory("nonexistent-server-xyz")
