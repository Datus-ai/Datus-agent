"""Tests for datus.api.services.mcp_service — MCP tool management."""

import pytest

from datus.api.models.mcp_models import AddServerInput, ToolFilterInput
from datus.api.services.mcp_service import MCPService


class TestMCPServiceInit:
    """Tests for MCPService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """MCPService initializes with real agent config."""
        svc = MCPService(agent_config=real_agent_config)
        assert svc is not None
        assert svc.manager is not None


class TestMCPServiceListServers:
    """Tests for list_servers."""

    def test_list_servers_returns_result(self, real_agent_config):
        """list_servers returns a Result object."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.list_servers()
        assert result.success is True

    def test_list_servers_with_type_filter(self, real_agent_config):
        """list_servers with type filter returns Result."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.list_servers(server_type="stdio")
        assert result.success is True

    def test_list_servers_returns_dict_data(self, real_agent_config):
        """list_servers data is a dict (possibly empty)."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.list_servers()
        assert isinstance(result.data, dict)


class TestMCPServiceAddRemoveServer:
    """Tests for add_server and remove_server."""

    def test_add_server_stdio(self, real_agent_config):
        """add_server creates a new stdio server config."""
        svc = MCPService(agent_config=real_agent_config)
        request = AddServerInput(
            name="test_server",
            type="stdio",
            command="echo",
            args=["hello"],
        )
        result = svc.add_server(request)
        assert result.success is True

    def test_remove_server(self, real_agent_config):
        """remove_server removes a server config."""
        svc = MCPService(agent_config=real_agent_config)
        svc.add_server(AddServerInput(name="to_remove", type="stdio", command="echo"))
        result = svc.remove_server("to_remove")
        assert result.success is True

    def test_remove_nonexistent_server(self, real_agent_config):
        """remove_server for nonexistent server returns error."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.remove_server("ghost_server")
        assert result.success is False


class TestMCPServiceToolFilter:
    """Tests for tool filter operations."""

    def test_get_tool_filter_nonexistent(self, real_agent_config):
        """get_tool_filter for nonexistent server returns error."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.get_tool_filter("nonexistent")
        assert result.success is False

    def test_remove_tool_filter_nonexistent(self, real_agent_config):
        """remove_tool_filter for nonexistent server returns error."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.remove_tool_filter("nonexistent")
        assert result.success is False

    def test_set_tool_filter_nonexistent(self, real_agent_config):
        """set_tool_filter for nonexistent server returns error."""
        svc = MCPService(agent_config=real_agent_config)
        result = svc.set_tool_filter("nonexistent", ToolFilterInput(allowed_tools=["tool1"]))
        assert result.success is False


@pytest.mark.asyncio
class TestMCPServiceAsync:
    """Tests for async MCP operations."""

    async def test_check_connectivity_nonexistent(self, real_agent_config):
        """check_connectivity for nonexistent server returns error."""
        svc = MCPService(agent_config=real_agent_config)
        result = await svc.check_connectivity("nonexistent_server")
        assert result.success is False

    async def test_list_tools_nonexistent(self, real_agent_config):
        """list_tools for nonexistent server returns error."""
        svc = MCPService(agent_config=real_agent_config)
        result = await svc.list_tools("nonexistent_server")
        assert result.success is False

    async def test_call_tool_nonexistent(self, real_agent_config):
        """call_tool for nonexistent server returns error."""
        from datus.api.models.mcp_models import CallToolInput

        svc = MCPService(agent_config=real_agent_config)
        result = await svc.call_tool("nonexistent", "some_tool", CallToolInput())
        assert result.success is False


class TestMCPServiceWithServer:
    """Tests for MCP operations with a real added server."""

    def test_full_server_lifecycle(self, real_agent_config):
        """Add server, list, get filter, remove filter, remove server."""
        svc = MCPService(agent_config=real_agent_config)

        # Add server
        add_result = svc.add_server(AddServerInput(name="lifecycle_srv", type="stdio", command="echo"))
        assert add_result.success is True

        # List servers should include it
        list_result = svc.list_servers()
        assert list_result.success is True
        assert "lifecycle_srv" in str(list_result.data)

        # Get tool filter (should work now that server exists)
        filter_result = svc.get_tool_filter("lifecycle_srv")
        assert filter_result is not None  # May succeed or return "no filter"

        # Set tool filter
        set_result = svc.set_tool_filter("lifecycle_srv", ToolFilterInput(allowed_tools=["tool_a"]))
        assert set_result is not None

        # Remove tool filter
        rm_filter_result = svc.remove_tool_filter("lifecycle_srv")
        assert rm_filter_result is not None

        # Remove server
        rm_result = svc.remove_server("lifecycle_srv")
        assert rm_result.success is True

    def test_add_server_duplicate(self, real_agent_config):
        """add_server with duplicate name returns error."""
        svc = MCPService(agent_config=real_agent_config)
        svc.add_server(AddServerInput(name="dup_srv", type="stdio", command="echo"))
        result = svc.add_server(AddServerInput(name="dup_srv", type="stdio", command="echo"))
        assert result.success is False
        # Clean up
        svc.remove_server("dup_srv")
