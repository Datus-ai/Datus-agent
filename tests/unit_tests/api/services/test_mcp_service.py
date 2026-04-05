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
