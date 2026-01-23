# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tests for Datus MCP Server

Run with:
    pytest tests/test_mcp_server.py -v
"""

import asyncio

import pytest

from datus.mcp_server import create_server


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


if __name__ == "__main__":
    # Quick manual test
    async def main():
        print("Creating MCP server...")
        with create_server(namespace="bird_sqlite") as server:
            print(f"Server namespace: {server.namespace}")
            print(f"Has DB tools: {server._has_db_tools}")
            print(f"Has Context tools: {server._has_context_tools}")

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
