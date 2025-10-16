# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MCP Config - Pydantic models for MCP server config management.

This module provides data models for MCP server configs with validation
and serialization capabilities.
"""

import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MCPServerType(str, Enum):
    """Enumeration of MCP server communication types."""

    STDIO = "stdio"  # Standard input/output communication
    SSE = "sse"  # Server-sent events communication
    HTTP = "http"  # HTTP communication protocol


class ToolFilterConfig(BaseModel):
    """Configuration for tool filtering on MCP servers."""

    allowed_tool_names: Optional[List[str]] = Field(
        None, description="List of allowed tool names (whitelist). If specified, only these tools are allowed."
    )
    blocked_tool_names: Optional[List[str]] = Field(
        None, description="List of blocked tool names (blacklist). These tools are excluded."
    )
    enabled: bool = Field(default=True, description="Whether tool filtering is enabled")

    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Check if a tool is allowed based on filter configuration.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is allowed, False otherwise
        """
        if not self.enabled:
            return True

        # First apply allowlist (if configured)
        if self.allowed_tool_names is not None:
            if tool_name not in self.allowed_tool_names:
                return False

        # Then apply blocklist to remaining tools
        if self.blocked_tool_names is not None:
            if tool_name in self.blocked_tool_names:
                return False

        return True


# Type alias for any MCP server config subclass
AnyMCPServerConfig = Union["STDIOServerConfig", "SSEServerConfig", "HTTPServerConfig"]


def expand_env_vars(value: str) -> str:
    """
    Expand env variables in a string.

    Supports format: ${VAR} and ${VAR:-default}

    Args:
        value: String that may contain env variables

    Returns:
        String with env variables expanded
    """

    def replace_var(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default_value = var_expr.split(":-", 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(var_expr, match.group(0))  # Return original if not found

    # Pattern to match ${VAR} or ${VAR:-default}
    pattern = r"\$\{([^}]+)\}"
    return re.sub(pattern, replace_var, value)


def expand_config_env_vars(config_dict: dict) -> dict:
    """
    Expand environment variables in config dictionary recursively.

    Args:
        config_dict: Dictionary that may contain env variables in string values

    Returns:
        Dictionary with env variables expanded
    """
    expanded = {}
    for key, value in config_dict.items():
        if isinstance(value, str):
            expanded[key] = expand_env_vars(value)
        elif isinstance(value, dict):
            # Recursively expand env variables in nested dicts (like headers)
            expanded_dict = {}
            for k, v in value.items():
                if isinstance(v, str):
                    expanded_dict[k] = expand_env_vars(v)
                else:
                    expanded_dict[k] = v
            expanded[key] = expanded_dict
        elif isinstance(value, list):
            # Handle list values (like args)
            expanded_list = []
            for item in value:
                if isinstance(item, str):
                    expanded_list.append(expand_env_vars(item))
                else:
                    expanded_list.append(item)
            expanded[key] = expanded_list
        else:
            expanded[key] = value
    return expanded


class MCPServerConfig(BaseModel):
    """Base config for an MCP server instance."""

    name: str = Field(..., description="Server name/identifier")
    type: MCPServerType = Field(..., description="Server communication type")
    tool_filter: Optional[ToolFilterConfig] = Field(None, description="Tool filtering configuration")

    class Config:
        use_enum_values = True

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v):
        """Convert string to MCPServerType enum."""
        if isinstance(v, str):
            try:
                return MCPServerType(v)
            except ValueError:
                raise ValueError(f"Invalid server type: {v}")
        return v

    @classmethod
    def from_config_format(cls, name: str, config: Dict[str, Any]) -> "AnyMCPServerConfig":
        """
        Create appropriate MCPServerConfig subclass from config format.

        Args:
            name: Server name
            config: Config in standard format

        Returns:
            MCPServerConfig instance (appropriate subclass)
        """
        server_type = config.get("type", "stdio")

        # Handle env variables expansion
        expanded_config = {}
        for key, value in config.items():
            if isinstance(value, str):
                expanded_config[key] = expand_env_vars(value)
            elif isinstance(value, dict):
                # Recursively expand env variables in nested dicts
                expanded_dict = {}
                for k, v in value.items():
                    if isinstance(v, str):
                        expanded_dict[k] = expand_env_vars(v)
                    else:
                        expanded_dict[k] = v
                expanded_config[key] = expanded_dict
            else:
                expanded_config[key] = value

        # Parse tool filter configuration if present
        tool_filter = None
        if "tool_filter" in expanded_config:
            filter_config = expanded_config["tool_filter"]
            if isinstance(filter_config, dict):
                tool_filter = ToolFilterConfig(**filter_config)
            elif isinstance(filter_config, ToolFilterConfig):
                tool_filter = filter_config

        # Create appropriate subclass based on server type
        if server_type == MCPServerType.STDIO:
            return STDIOServerConfig(
                name=name,
                command=expanded_config.get("command"),
                args=expanded_config.get("args"),
                env=expanded_config.get("env"),
                tool_filter=tool_filter,
            )
        elif server_type == MCPServerType.SSE:
            return SSEServerConfig(
                name=name,
                url=expanded_config.get("url"),
                headers=expanded_config.get("headers"),
                timeout=expanded_config.get("timeout"),
                tool_filter=tool_filter,
            )
        elif server_type == MCPServerType.HTTP:
            return HTTPServerConfig(
                name=name,
                url=expanded_config.get("url"),
                headers=expanded_config.get("headers"),
                timeout=expanded_config.get("timeout"),
                tool_filter=tool_filter,
            )
        else:
            raise ValueError(f"Unknown server type: {server_type}")


class STDIOServerConfig(MCPServerConfig):
    """Config for STDIO MCP servers."""

    type: MCPServerType = Field(default=MCPServerType.STDIO, description="Server communication type")
    command: str = Field(..., description="Command to execute")
    args: Optional[List[str]] = Field(None, description="Command arguments")
    env: Optional[Dict[str, str]] = Field(None, description="Env variables")

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for STDIO server."""
        return {
            "type": "stdio",
            "command": self.command,
            "args": self.args or [],
            "env": self.env or {},
        }


class SSEServerConfig(MCPServerConfig):
    """Config for SSE (Server-Sent Events) MCP servers."""

    type: MCPServerType = Field(default=MCPServerType.SSE, description="Server communication type")
    url: str = Field(..., description="Server URL")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    timeout: Optional[float] = Field(None, description="Connection timeout")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for SSE server."""
        return {
            "type": "sse",
            "url": self.url,
            "headers": self.headers or {},
            "timeout": self.timeout,
        }


class HTTPServerConfig(MCPServerConfig):
    """Config for HTTP MCP servers."""

    type: MCPServerType = Field(default=MCPServerType.HTTP, description="Server communication type")
    url: str = Field(..., description="Server URL")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    timeout: Optional[float] = Field(None, description="Connection timeout")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for HTTP server."""
        return {
            "type": "http",
            "url": self.url,
            "headers": self.headers or {},
            "timeout": self.timeout,
        }


class MCPConfig(BaseModel):
    """Root config containing all MCP servers."""

    version: str = Field(default="1.0", description="Config version")
    servers: Dict[str, AnyMCPServerConfig] = Field(default_factory=dict, description="MCP servers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True

    def add_server(self, config: AnyMCPServerConfig) -> None:
        """Add a server config."""
        self.servers[config.name] = config

    def remove_server(self, name: str) -> bool:
        """Remove a server config."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False

    def get_server(self, name: str) -> Optional[AnyMCPServerConfig]:
        """Get server config by name."""
        return self.servers.get(name)

    def list_servers(self, server_type: Optional[MCPServerType] = None) -> List[AnyMCPServerConfig]:
        """List server configs with optional filtering."""
        servers = list(self.servers.values())

        if server_type:
            servers = [s for s in servers if s.type == server_type]

        return servers

    @classmethod
    def from_config_format(cls, config: Dict[str, Any]) -> "MCPConfig":
        """
        Create MCPConfig from config format.

        Args:
            config: Config with "mcpServers" key

        Returns:
            MCPConfig instance
        """
        mcp_config = cls()

        if "mcpServers" in config:
            for name, server_config in config["mcpServers"].items():
                server = MCPServerConfig.from_config_format(name, server_config)
                mcp_config.add_server(server)

        return mcp_config

    def to_config_format(self) -> Dict[str, Any]:
        """
        Convert to config format.

        Returns:
            Dictionary in standard config format
        """
        config = {"mcpServers": {}}

        for name, server in self.servers.items():
            server_config = {"type": server.type}

            # Add tool filter configuration if present
            if server.tool_filter:
                filter_dict = server.tool_filter.model_dump(exclude_none=True)
                if filter_dict:  # Only add if not empty
                    server_config["tool_filter"] = filter_dict

            if server.type == MCPServerType.STDIO:
                if server.command:
                    server_config["command"] = server.command
                if server.args:
                    server_config["args"] = server.args
                if server.env:
                    server_config["env"] = server.env

            elif server.type == MCPServerType.SSE:
                if server.url:
                    server_config["url"] = server.url
                if server.headers:
                    server_config["headers"] = server.headers
                if server.timeout:
                    server_config["timeout"] = server.timeout

            elif server.type == MCPServerType.HTTP:
                if server.url:
                    server_config["url"] = server.url
                if server.headers:
                    server_config["headers"] = server.headers
                if server.timeout:
                    server_config["timeout"] = server.timeout

            config["mcpServers"][name] = server_config

        return config
