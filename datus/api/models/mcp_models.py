"""Pydantic models for MCP (Model Context Protocol) API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Server management models
class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str = Field(..., description="Server name/identifier")
    type: str = Field(..., description="Server type (stdio, sse, http)")
    status: str = Field(..., description="Server status (available, unavailable)")
    command: Optional[str] = Field(None, description="Command for stdio servers")
    args: Optional[List[str]] = Field(None, description="Arguments for stdio servers")
    url: Optional[str] = Field(None, description="URL for sse/http servers")
    headers: Optional[Dict[str, str]] = Field(None, description="Headers for sse/http servers")
    timeout: Optional[float] = Field(None, description="Timeout for sse/http servers")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables for stdio servers")
    cwd: Optional[str] = Field(None, description="Working directory for stdio servers")


class AddServerInput(BaseModel):
    """Input model for adding a new MCP server."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "new_server", "type": "stdio", "command": "python", "args": ["-m", "new_server"]}
        }
    )

    name: str = Field(..., description="Server name/identifier")
    type: str = Field(..., description="Server type (stdio, sse, http)")
    command: Optional[str] = Field(None, description="Command for stdio servers")
    args: Optional[List[str]] = Field(None, description="Arguments for stdio servers")
    url: Optional[str] = Field(None, description="URL for sse/http servers")
    headers: Optional[Dict[str, str]] = Field(None, description="Headers for sse/http servers")
    timeout: Optional[float] = Field(None, description="Timeout for sse/http servers")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables for stdio servers")
    cwd: Optional[str] = Field(None, description="Working directory for stdio servers")


# Connectivity models
class ConnectivityDetails(BaseModel):
    """Details about server connectivity."""

    type: str = Field(..., description="Server type")
    tools_count: Optional[int] = Field(None, description="Number of available tools")
    tools_available: Optional[bool] = Field(None, description="Whether tools are available")
    tool_names: Optional[List[str]] = Field(None, description="List of available tool names")
    connected: Optional[bool] = Field(None, description="Whether connection was successful")
    command: Optional[str] = Field(None, description="Command for stdio servers")
    args: Optional[List[str]] = Field(None, description="Arguments for stdio servers")
    url: Optional[str] = Field(None, description="URL for sse/http servers")
    headers_count: Optional[int] = Field(None, description="Number of headers")
    timeout: Optional[float] = Field(None, description="Connection timeout")
    env_count: Optional[int] = Field(None, description="Number of environment variables")


# Tool management models
class ToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    inputSchema: Dict[str, Any] = Field(..., description="Tool input schema")


class CallToolInput(BaseModel):
    """Input model for calling a tool on an MCP server."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"parameters": {"query": "SELECT * FROM users", "limit": 10}}}
    )

    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


# Filter management models
class ToolFilterInput(BaseModel):
    """Input model for setting tool filters."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"enabled": True, "allowed_tools": ["query", "analyze"], "blocked_tools": ["delete", "drop"]}
        }
    )

    enabled: bool = Field(True, description="Whether filtering is enabled")
    allowed_tools: Optional[List[str]] = Field(None, description="List of allowed tool names")
    blocked_tools: Optional[List[str]] = Field(None, description="List of blocked tool names")


class ToolFilterConfig(BaseModel):
    """Tool filter configuration."""

    enabled: bool = Field(..., description="Whether filtering is enabled")
    allowed_tool_names: Optional[List[str]] = Field(None, description="List of allowed tool names")
    blocked_tool_names: Optional[List[str]] = Field(None, description="List of blocked tool names")
