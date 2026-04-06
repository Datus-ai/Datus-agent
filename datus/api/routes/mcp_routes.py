"""
API routes for MCP (Model Context Protocol) endpoints.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Path, Query

from datus.api.deps import ServiceDep
from datus.api.models.base_models import Result
from datus.api.models.mcp_models import (
    AddServerInput,
    CallToolInput,
    ToolFilterInput,
)

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])

# Pre-configured parameters to avoid definition-time evaluation in defaults
SERVER_TYPE_QUERY = Query(None, description="Filter by server type (stdio, sse, http)")
REMOVE_SERVER_NAME_PATH = Path(..., description="Name of the server to remove")
SERVER_NAME_CHECK_PATH = Path(..., description="Name of the server to check")
SERVER_NAME_PATH = Path(..., description="Name of the server")
TOOL_NAME_PATH = Path(..., description="Name of the tool to call")
APPLY_FILTER_QUERY = Query(True, description="Whether to apply tool filtering")


@router.get(
    "/servers",
    response_model=Result[Dict[str, Any]],
    summary="List MCP Servers",
    description="List all MCP servers with optional filtering by type",
)
async def list_servers(
    svc: ServiceDep,
    server_type: Optional[str] = SERVER_TYPE_QUERY,
) -> Result[Dict[str, Any]]:
    """List all MCP servers."""
    return svc.mcp.list_servers(server_type=server_type)


@router.post(
    "/servers",
    response_model=Result[Dict[str, Any]],
    summary="Add MCP Server",
    description="Add a new MCP server configuration",
)
async def add_server(
    server_config: AddServerInput,
    svc: ServiceDep,
) -> Result[Dict[str, Any]]:
    """Add a new MCP server."""
    return svc.mcp.add_server(server_config)


@router.delete(
    "/servers/{server_name}",
    response_model=Result[Dict[str, Any]],
    summary="Remove MCP Server",
    description="Remove an MCP server configuration",
)
async def remove_server(
    svc: ServiceDep,
    server_name: str = REMOVE_SERVER_NAME_PATH,
) -> Result[Dict[str, Any]]:
    """Remove an MCP server."""
    return svc.mcp.remove_server(server_name)


@router.get(
    "/servers/{server_name}/connectivity",
    response_model=Result[Dict[str, Any]],
    summary="Check Server Connectivity",
    description="Check connectivity status of an MCP server",
)
async def check_connectivity(
    svc: ServiceDep,
    server_name: str = SERVER_NAME_CHECK_PATH,
) -> Result[Dict[str, Any]]:
    """Check server connectivity status."""
    return await svc.mcp.check_connectivity(server_name)


@router.get(
    "/servers/{server_name}/tools",
    response_model=Result[Dict[str, Any]],
    summary="List Server Tools",
    description="List tools available on an MCP server",
)
async def list_tools(
    svc: ServiceDep,
    server_name: str = SERVER_NAME_PATH,
    apply_filter: bool = APPLY_FILTER_QUERY,
) -> Result[Dict[str, Any]]:
    """List tools available on an MCP server."""
    return await svc.mcp.list_tools(server_name, apply_filter)


@router.post(
    "/servers/{server_name}/tools/{tool_name}/call",
    response_model=Result[Dict[str, Any]],
    summary="Call Tool",
    description="Call a tool on an MCP server",
)
async def call_tool(
    request: CallToolInput,
    svc: ServiceDep,
    server_name: str = SERVER_NAME_PATH,
    tool_name: str = TOOL_NAME_PATH,
) -> Result[Dict[str, Any]]:
    """Call a tool on an MCP server."""
    return await svc.mcp.call_tool(server_name, tool_name, request)


@router.get(
    "/servers/{server_name}/filters",
    response_model=Result[Dict[str, Any]],
    summary="Get Tool Filter",
    description="Get tool filter configuration for an MCP server",
)
async def get_tool_filter(
    svc: ServiceDep,
    server_name: str = SERVER_NAME_PATH,
) -> Result[Dict[str, Any]]:
    """Get tool filter configuration."""
    return svc.mcp.get_tool_filter(server_name)


@router.put(
    "/servers/{server_name}/filters",
    response_model=Result[Dict[str, Any]],
    summary="Set Tool Filter",
    description="Set tool filter configuration for an MCP server",
)
async def set_tool_filter(
    filter_config: ToolFilterInput,
    svc: ServiceDep,
    server_name: str = SERVER_NAME_PATH,
) -> Result[Dict[str, Any]]:
    """Set tool filter configuration."""
    return svc.mcp.set_tool_filter(server_name, filter_config)


@router.delete(
    "/servers/{server_name}/filters",
    response_model=Result[Dict[str, Any]],
    summary="Remove Tool Filter",
    description="Remove tool filter configuration from an MCP server",
)
async def remove_tool_filter(
    svc: ServiceDep,
    server_name: str = SERVER_NAME_PATH,
) -> Result[Dict[str, Any]]:
    """Remove tool filter configuration."""
    return svc.mcp.remove_tool_filter(server_name)
