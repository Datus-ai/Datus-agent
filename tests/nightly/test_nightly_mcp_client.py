"""
Nightly MCP Client integration tests (N10).

Tests MCP protocol interactions from the client perspective:
context tool calls, error handling, large result sets, and concurrent calls.
"""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
from mcp.client.streamable_http import streamablehttp_client

from datus.mcp_server import DatusMCPServer
from mcp import ClientSession

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "conf" / "agent.yml")


def find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def start_uvicorn(app, port: int):
    """Start a uvicorn server in a background asyncio task."""
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
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


def parse_tool_result(result) -> dict:
    """Parse a CallToolResult into a dict."""
    assert not result.isError, f"Tool call returned error: {result}"
    assert len(result.content) > 0, "Tool call returned empty content"
    data = json.loads(result.content[0].text)
    return data


@pytest.mark.asyncio
@pytest.mark.nightly
class TestNightlyMCPClient:
    """N10: MCP Client integration tests."""

    @pytest_asyncio.fixture(autouse=True)
    async def mcp_server(self):
        """Start a static-mode MCP server for ssb_sqlite namespace."""
        port = find_free_port()
        server = DatusMCPServer(namespace="ssb_sqlite", config_path=CONFIG_PATH)
        app = server.get_streamable_http_app()
        uvi_server, task = await start_uvicorn(app, port)
        self.url = f"http://127.0.0.1:{port}/mcp"
        yield
        uvi_server.should_exit = True
        await task
        server.close()

    async def test_context_tool_list_subject_tree(self):
        """N10-05: list_subject_tree via MCP client returns valid response."""
        async with mcp_http_session(self.url) as session:
            result = await session.call_tool("list_subject_tree", {})
            data = parse_tool_result(result)

            assert data["success"] == 1, f"list_subject_tree should succeed, got error: {data.get('error')}"
            assert data["result"] is not None, "list_subject_tree should return a result"

    async def test_error_handling_invalid_sql(self):
        """N10-07a: read_query with invalid SQL returns proper error via MCP."""
        async with mcp_http_session(self.url) as session:
            result = await session.call_tool("read_query", {"sql": "SELECT * FROM nonexistent_xyz_table"})
            data = parse_tool_result(result)

            assert data["success"] == 0, "read_query with invalid table should return success=0"
            assert data.get("error") is not None, "Should have error message"
            assert len(data["error"]) > 0, "Error message should not be empty"

    async def test_error_handling_nonexistent_table_describe(self):
        """N10-07b: describe_table for nonexistent table returns empty columns."""
        async with mcp_http_session(self.url) as session:
            result = await session.call_tool("describe_table", {"table_name": "nonexistent_xyz_table"})
            data = parse_tool_result(result)

            # SQLite returns success with 0 columns for nonexistent tables
            assert (
                data["success"] == 1
            ), f"describe_table should return a valid response, got error: {data.get('error')}"
            assert isinstance(data["result"], dict), f"Result should be a dict, got {type(data['result'])}"
            columns = data["result"].get("columns", [])
            assert len(columns) == 0, f"Nonexistent table should have 0 columns, got {len(columns)}"

    async def test_large_result_set(self):
        """N10-08: Large query result is properly handled (compressed/truncated)."""
        async with mcp_http_session(self.url) as session:
            result = await session.call_tool("read_query", {"sql": "SELECT * FROM lineorder LIMIT 500"})
            data = parse_tool_result(result)

            assert data["success"] == 1, f"read_query should succeed, got error: {data.get('error')}"
            assert data["result"] is not None, "Should have result data"
            # Result should contain data in some form
            result_str = str(data["result"])
            assert len(result_str) > 100, f"Large result should have substantial content, got len={len(result_str)}"

    async def test_concurrent_tool_calls(self):
        """N10-09: Multiple concurrent tool calls all succeed."""
        async with mcp_http_session(self.url) as session:
            # Run 3 different tool calls concurrently
            results = await asyncio.gather(
                session.call_tool("list_tables", {}),
                session.call_tool("describe_table", {"table_name": "customer"}),
                session.call_tool("read_query", {"sql": "SELECT COUNT(*) as cnt FROM supplier"}),
            )

            assert len(results) == 3, f"Should have 3 results, got {len(results)}"

            # Verify all succeeded
            for i, result in enumerate(results):
                data = parse_tool_result(result)
                assert data["success"] == 1, f"Concurrent call {i} should succeed, got error: {data.get('error')}"
                assert data["result"] is not None, f"Concurrent call {i} should have result"
