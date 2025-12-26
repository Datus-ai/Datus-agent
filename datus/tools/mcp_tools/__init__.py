# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MCP Tools - Model Context Protocol server management tools.

This package provides comprehensive functionality for managing MCP servers
including stdio, sse, and streamable communication types.

All connections are managed through the OpenAI agents SDK for SSE and streamable servers.
"""

from .mcp_config import (
    AnyMCPServerConfig,
    HTTPServerConfig,
    MCPServerConfig,
    MCPServerType,
    SSEServerConfig,
    STDIOServerConfig,
)
from .mcp_manager import MCPManager
from .mcp_server import MCPServer, MCPServerStdioParams, SilentMCPServerStdio, find_mcp_directory
from .mcp_tool import MCPTool, parse_command_string

__all__ = [
    "AnyMCPServerConfig",
    "HTTPServerConfig",
    "MCPServerConfig",
    "MCPManager",
    "MCPServerType",
    "MCPTool",
    "SSEServerConfig",
    "STDIOServerConfig",
    "parse_command_string",
    "SilentMCPServerStdio",
    "MCPServer",
    "MCPServerStdioParams",
    "find_mcp_directory",
]
