# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
End-to-End Integration Tests for Datus MCP Server

Tests the full MCP protocol stack by starting real servers and connecting
with MCP SDK clients. Covers:
  - Static mode with HTTP Streamable transport
  - Static mode with SSE transport
  - Static mode with stdio transport
  - Dynamic mode with HTTP Streamable transport (multi-namespace)
  - Dynamic mode with SSE transport (multi-namespace)

Namespaces tested:
  - ssb_sqlite: SQLite database with SSB benchmark tables
  - duckdb: DuckDB database with MetricFlow demo tables
"""

import asyncio
import json
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from datus.mcp_server import DatusMCPServer, create_dynamic_app

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "conf" / "agent.yml")

# Expected tool names for assertion reuse
STATIC_EXPECTED_TOOLS = {
    "list_tables",
    "describe_table",
    "read_query",
    "list_databases",
    "get_table_ddl",
    "list_subject_tree",
}
DYNAMIC_EXPECTED_TOOLS = {"list_tables", "describe_table", "read_query"}


# =============================================================================
# Helpers
# =============================================================================


def find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def start_uvicorn(app, port: int) -> tuple:
    """Start a uvicorn server in a background asyncio task.

    Returns (server, task) tuple. The server is ready to accept connections
    when this function returns.
    """
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for server to start accepting connections
    for _ in range(200):  # up to 10 seconds
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"uvicorn server failed to start on port {port}")
    return server, task


@asynccontextmanager
async def mcp_http_session(url: str):
    """Context manager that yields an initialized MCP ClientSession over HTTP Streamable."""
    async with streamablehttp_client(url=url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def mcp_sse_session(url: str):
    """Context manager that yields an initialized MCP ClientSession over SSE."""
    async with sse_client(url=url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def mcp_stdio_session(server_params: StdioServerParameters):
    """Context manager that yields an initialized MCP ClientSession over stdio."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def parse_tool_result(result) -> dict:
    """Parse a CallToolResult into a dict with success/error/result keys."""
    assert not result.isError, f"Tool call returned error: {result}"
    assert len(result.content) > 0, "Tool call returned empty content"
    data = json.loads(result.content[0].text)
    return data


# =============================================================================
# Static Mode Base Class
# =============================================================================


class StaticModeTestBase:
    """Base test class for static-mode MCP server tests across transports.

    Subclasses must implement ``_session()`` returning an async context manager
    that yields an initialized ``ClientSession``.
    """

    def _session(self):
        """Return an async context manager yielding an initialized ClientSession."""
        raise NotImplementedError

    async def test_list_tools(self):
        """Verify that expected tools are registered and discoverable."""
        async with self._session() as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            for expected in STATIC_EXPECTED_TOOLS:
                assert expected in tool_names, f"Missing expected tool: {expected}"

    async def test_list_tables(self):
        """Verify list_tables returns the SSB benchmark tables."""
        async with self._session() as session:
            result = await session.call_tool("list_tables", {})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"list_tables failed: {data.get('error')}"
            tables_text = str(data["result"]).lower()
            assert "lineorder" in tables_text
            assert "customer" in tables_text

    async def test_describe_table(self):
        """Verify describe_table returns column information for the customer table."""
        async with self._session() as session:
            result = await session.call_tool("describe_table", {"table_name": "customer"})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"describe_table failed: {data.get('error')}"
            assert data["result"] is not None

    async def test_read_query(self):
        """Verify read_query executes SQL and returns results."""
        async with self._session() as session:
            result = await session.call_tool("read_query", {"sql": "SELECT COUNT(*) AS cnt FROM customer"})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"read_query failed: {data.get('error')}"
            assert data["result"] is not None

    async def test_list_databases(self):
        """Verify list_databases returns database info."""
        async with self._session() as session:
            result = await session.call_tool("list_databases", {})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"list_databases failed: {data.get('error')}"

    async def test_get_table_ddl(self):
        """Verify get_table_ddl returns DDL for a known table."""
        async with self._session() as session:
            result = await session.call_tool("get_table_ddl", {"table_name": "customer"})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"get_table_ddl failed: {data.get('error')}"
            ddl_text = str(data["result"]).upper()
            assert "CREATE" in ddl_text or "TABLE" in ddl_text

    async def test_list_subject_tree(self):
        """Verify list_subject_tree is callable and does not error."""
        async with self._session() as session:
            result = await session.call_tool("list_subject_tree", {})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"list_subject_tree failed: {data.get('error')}"


# =============================================================================
# Dynamic Mode Base Class
# =============================================================================


class DynamicModeTestBase:
    """Base test class for dynamic-mode MCP server tests across transports.

    Subclasses must implement ``_ssb_session()`` and ``_duckdb_session()``
    returning async context managers that yield initialized ``ClientSession``s.
    """

    def _ssb_session(self):
        """Return an async context manager for the ssb_sqlite namespace."""
        raise NotImplementedError

    def _duckdb_session(self):
        """Return an async context manager for the duckdb namespace."""
        raise NotImplementedError

    async def test_list_tools_ssb(self):
        """Verify tools are discoverable on ssb_sqlite namespace."""
        async with self._ssb_session() as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            for expected in DYNAMIC_EXPECTED_TOOLS:
                assert expected in tool_names, f"Missing expected tool: {expected}"

    async def test_list_tools_duckdb(self):
        """Verify tools are discoverable on duckdb namespace."""
        async with self._duckdb_session() as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            for expected in DYNAMIC_EXPECTED_TOOLS:
                assert expected in tool_names, f"Missing expected tool: {expected}"

    async def test_list_tables_ssb(self):
        """Verify list_tables on ssb_sqlite returns SSB tables."""
        async with self._ssb_session() as session:
            result = await session.call_tool("list_tables", {})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"list_tables ssb failed: {data.get('error')}"
            tables_text = str(data["result"]).lower()
            assert "lineorder" in tables_text
            assert "customer" in tables_text

    async def test_list_tables_duckdb(self):
        """Verify list_tables on duckdb returns MetricFlow demo tables."""
        async with self._duckdb_session() as session:
            result = await session.call_tool("list_tables", {})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"list_tables duckdb failed: {data.get('error')}"
            tables_text = str(data["result"]).lower()
            assert "mf_demo" in tables_text

    async def test_describe_table_ssb(self):
        """Verify describe_table on ssb_sqlite returns column info."""
        async with self._ssb_session() as session:
            result = await session.call_tool("describe_table", {"table_name": "supplier"})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"describe_table ssb failed: {data.get('error')}"
            assert data["result"] is not None

    async def test_read_query_ssb(self):
        """Verify read_query on ssb_sqlite executes SQL."""
        async with self._ssb_session() as session:
            result = await session.call_tool("read_query", {"sql": "SELECT COUNT(*) AS cnt FROM supplier"})
            data = parse_tool_result(result)
            assert data["success"] == 1, f"read_query ssb failed: {data.get('error')}"
            assert data["result"] is not None

    async def test_read_query_duckdb(self):
        """Verify read_query on duckdb executes SQL."""
        async with self._duckdb_session() as session:
            result = await session.call_tool(
                "read_query", {"sql": "SELECT COUNT(*) AS cnt FROM mf_demo.mf_demo_customers"}
            )
            data = parse_tool_result(result)
            assert data["success"] == 1, f"read_query duckdb failed: {data.get('error')}"
            assert data["result"] is not None

    async def test_multi_namespace_isolation(self):
        """Verify that ssb_sqlite and duckdb return different table sets."""
        async with self._ssb_session() as ssb_session:
            ssb_result = await ssb_session.call_tool("list_tables", {})
            ssb_data = parse_tool_result(ssb_result)

        async with self._duckdb_session() as duck_session:
            duck_result = await duck_session.call_tool("list_tables", {})
            duck_data = parse_tool_result(duck_result)

        ssb_text = str(ssb_data["result"]).lower()
        duck_text = str(duck_data["result"]).lower()

        # SSB has lineorder, duckdb does not
        assert "lineorder" in ssb_text
        assert "lineorder" not in duck_text

        # DuckDB has mf_demo tables, SSB does not
        assert "mf_demo" in duck_text
        assert "mf_demo" not in ssb_text


# =============================================================================
# Static Mode: HTTP Streamable
# =============================================================================


@pytest.mark.asyncio
class TestStaticModeHTTPStreamable(StaticModeTestBase):
    """Test DatusMCPServer in static mode using HTTP Streamable transport."""

    @pytest_asyncio.fixture(autouse=True)
    async def static_server(self):
        """Start a static-mode MCP server for ssb_sqlite namespace."""
        port = find_free_port()
        server = DatusMCPServer(namespace="ssb_sqlite", config_path=CONFIG_PATH)
        app = server.get_streamable_http_app()
        uvi_server, task = await start_uvicorn(app, port)
        self.port = port
        self.url = f"http://127.0.0.1:{port}/mcp"
        yield
        uvi_server.should_exit = True
        await task
        server.close()

    def _session(self):
        return mcp_http_session(self.url)


# =============================================================================
# Static Mode: SSE
# =============================================================================


@pytest.mark.asyncio
class TestStaticModeSSE(StaticModeTestBase):
    """Test DatusMCPServer in static mode using SSE transport."""

    @pytest_asyncio.fixture(autouse=True)
    async def static_sse_server(self):
        """Start a static-mode MCP server with SSE transport for ssb_sqlite."""
        port = find_free_port()
        server = DatusMCPServer(namespace="ssb_sqlite", config_path=CONFIG_PATH)
        app = server.get_sse_app()
        uvi_server, task = await start_uvicorn(app, port)
        self.port = port
        self.url = f"http://127.0.0.1:{port}/sse"
        yield
        uvi_server.should_exit = True
        await task
        server.close()

    def _session(self):
        return mcp_sse_session(self.url)


# =============================================================================
# Static Mode: stdio
# =============================================================================


@pytest.mark.asyncio
class TestStaticModeStdio(StaticModeTestBase):
    """Test DatusMCPServer in static mode using stdio transport."""

    @staticmethod
    def _server_params() -> StdioServerParameters:
        """Build StdioServerParameters to launch the MCP server as a subprocess."""
        return StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "datus.mcp_server",
                "--namespace",
                "ssb_sqlite",
                "--transport",
                "stdio",
                "--config",
                CONFIG_PATH,
            ],
        )

    def _session(self):
        return mcp_stdio_session(self._server_params())


# =============================================================================
# Dynamic Mode: HTTP Streamable
# =============================================================================


@pytest.mark.asyncio
class TestDynamicModeHTTPStreamable(DynamicModeTestBase):
    """Test LightweightDynamicMCPServer with HTTP Streamable transport."""

    @pytest_asyncio.fixture(autouse=True)
    async def dynamic_http_server(self):
        """Start a dynamic-mode MCP server with HTTP transport."""
        port = find_free_port()
        app = create_dynamic_app(config_path=CONFIG_PATH, transport="http")
        uvi_server, task = await start_uvicorn(app, port)
        self.port = port
        self.ssb_url = f"http://127.0.0.1:{port}/mcp/ssb_sqlite"
        self.duckdb_url = f"http://127.0.0.1:{port}/mcp/duckdb"
        yield
        uvi_server.should_exit = True
        await task

    def _ssb_session(self):
        return mcp_http_session(self.ssb_url)

    def _duckdb_session(self):
        return mcp_http_session(self.duckdb_url)


# =============================================================================
# Dynamic Mode: SSE
# =============================================================================


@pytest.mark.asyncio
class TestDynamicModeSSE(DynamicModeTestBase):
    """Test LightweightDynamicMCPServer with SSE transport."""

    @pytest_asyncio.fixture(autouse=True)
    async def dynamic_sse_server(self):
        """Start a dynamic-mode MCP server with SSE transport."""
        port = find_free_port()
        app = create_dynamic_app(config_path=CONFIG_PATH, transport="sse")
        uvi_server, task = await start_uvicorn(app, port)
        self.port = port
        self.ssb_url = f"http://127.0.0.1:{port}/sse/ssb_sqlite"
        self.duckdb_url = f"http://127.0.0.1:{port}/sse/duckdb"
        yield
        uvi_server.should_exit = True
        await task

    def _ssb_session(self):
        return mcp_sse_session(self.ssb_url)

    def _duckdb_session(self):
        return mcp_sse_session(self.duckdb_url)
