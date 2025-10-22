# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MCP Manager - Core business logic for managing MCP servers.

This module provides the main MCPManager class that handles all MCP server
lifecycle operations including config management, server control,
and status monitoring.
"""

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, RunContextWrapper, Usage
from agents.mcp import MCPServerStdioParams
from agents.mcp.server import MCPServerSse, MCPServerSseParams, MCPServerStreamableHttp, MCPServerStreamableHttpParams

from datus.tools.mcp_tools.mcp_config import (
    AnyMCPServerConfig,
    MCPConfig,
    MCPServerType,
    STDIOServerConfig,
    ToolFilterConfig,
    expand_config_env_vars,
)
from datus.tools.mcp_tools.mcp_server import SilentMCPServerStdio
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def create_static_tool_filter(
    allowed_tool_names: Optional[List[str]] = None,
    blocked_tool_names: Optional[List[str]] = None,
    enabled: bool = True,
) -> ToolFilterConfig:
    """
    Create a static tool filter configuration.

    Args:
        allowed_tool_names: List of allowed tool names (whitelist)
        blocked_tool_names: List of blocked tool names (blacklist)
        enabled: Whether filtering is enabled

    Returns:
        ToolFilterConfig instance
    """
    return ToolFilterConfig(
        allowed_tool_names=allowed_tool_names,
        blocked_tool_names=blocked_tool_names,
        enabled=enabled,
    )


def _validate_server_exists(manager, server_name: str) -> Tuple[bool, str, Optional[AnyMCPServerConfig]]:
    """Validate that a server exists and return its config."""
    config = manager.get_server_config(server_name)
    if not config:
        return False, f"Server '{server_name}' not found", None
    return True, "", config


class MCPManager:
    """
    Main manager class for MCP server operations.

    Provides functionality for:
    - Config management (CRUD operations)
    - Server lifecycle management (start/stop/restart)
    - Status monitoring and health checks
    - Config file persistence

    Configuration path:
    - Fixed at {agent.home}/conf/.mcp.json (default: ~/.datus/conf/.mcp.json)
    """

    def __init__(self):
        """
        Initialize the MCP manager.

        MCP configuration is fixed at {agent.home}/conf/.mcp.json.
        Configure agent.home in agent.yml to change the root directory.
        The path cannot be overridden to ensure consistent configuration management.
        """
        from datus.utils.path_manager import get_path_manager

        path_manager = get_path_manager()
        path_manager.ensure_dirs("conf")
        self.config_path = path_manager.mcp_config_path()

        self.config: MCPConfig = MCPConfig()
        self._lock = threading.Lock()

        # Load existing config
        self.load_config()

    def load_config(self) -> bool:
        """
        Load config from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if self.config_path.exists() and self.config_path.stat().st_size > 0:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "mcpServers" in data:
                    self.config = MCPConfig.from_config_format(data)
                    logger.info(f"Loaded MCP config from {self.config_path}")
                else:
                    # Invalid format, use defaults
                    self.config = MCPConfig()
                    logger.info("Config file format not recognized, using defaults")

                return True
            else:
                if self.config_path.exists():
                    logger.info(f"Config file at {self.config_path} is empty, using defaults")
                else:
                    logger.info(f"No config file found at {self.config_path}, using defaults")
                return True
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False

    def save_config(self) -> bool:
        """
        Save config to file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, "w", encoding="utf-8") as f:
                config_data = self.config.to_config_format()
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved MCP config to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def add_server(self, config: AnyMCPServerConfig) -> Tuple[bool, str]:
        """
        Add a new MCP server config.

        Args:
            config: Server config

        Returns:
            Tuple of (success, message)
        """
        try:
            with self._lock:
                if config.name in self.config.servers:
                    return False, f"Server '{config.name}' already exists"

                self.config.add_server(config)

                if self.save_config():
                    logger.info(f"Added MCP server: {config.name} ({config.type})")
                    return True, f"Successfully added server '{config.name}'"
                else:
                    return False, "Failed to save config"

        except Exception as e:
            logger.error(f"Error adding server {config.name}: {e}")
            return False, f"Error adding server: {e}"

    def remove_server(self, name: str) -> Tuple[bool, str]:
        """
        Remove an MCP server config.

        Args:
            name: Server name

        Returns:
            Tuple of (success, message)
        """
        try:
            with self._lock:
                if name not in self.config.servers:
                    return False, f"Server '{name}' not found"

                self.config.remove_server(name)

                if self.save_config():
                    logger.info(f"Removed MCP server: {name}")
                    return True, f"Successfully removed server '{name}'"
                else:
                    return False, "Failed to save config"

        except Exception as e:
            logger.error(f"Error removing server {name}: {e}")
            return False, f"Error removing server: {e}"

    def list_servers(self, server_type: Optional[MCPServerType] = None) -> List[AnyMCPServerConfig]:
        """
        List MCP servers with optional filtering.

        Args:
            server_type: Filter by server type
            status: Filter by status

        Returns:
            List of server configs
        """
        servers = self.config.list_servers(server_type=server_type)

        return servers

    def get_server_config(self, name: str) -> Optional[AnyMCPServerConfig]:
        """
        Get server config by name.

        Args:
            name: Server name

        Returns:
            Server config or None if not found
        """
        return self.config.get_server(name)

    async def check_connectivity(self, name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check server connectivity by attempting to connect and list tools."""
        valid, error_msg, config = _validate_server_exists(self, name)
        if not valid:
            return False, error_msg, {}

        # Create server instance
        server_instance, connectivity_details = self._create_server_instance(config)
        if not server_instance:
            error_msg = connectivity_details.get("error", "Failed to create server instance")
            return False, f"Failed to create server instance for '{name}': {error_msg}", connectivity_details

        # Run connectivity test
        success, tools_data = await self._run_tools_operation_async(server_instance, name, "connectivity_test")
        connectivity_details.update(tools_data)
        connectivity_details["type"] = config.type

        if success:
            return True, f"Server '{name}' connectivity test passed", connectivity_details
        else:
            error_msg = tools_data.get("error", "Connectivity test failed")
            return False, f"Server '{name}' connectivity test failed: {error_msg}", connectivity_details

    async def list_tools(self, server_name: str, apply_filter: bool = True) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """List tools available on the specified MCP server."""
        try:
            valid, error_msg, config = _validate_server_exists(self, server_name)
            if not valid:
                return False, error_msg, []

            # Create server instance
            server_instance, details = self._create_server_instance(config)
            if not server_instance:
                error_msg = details.get("error", "Failed to create server instance")
                return False, f"Failed to connect to server '{server_name}': {error_msg}", []

            # Run list_tools operation
            success, tools_data = await self._run_tools_operation_async(server_instance, server_name, "list_tools")

            if success:
                tools_list = tools_data.get("tools", [])

                # Apply tool filtering if enabled and requested
                if apply_filter and config.tool_filter:
                    filtered_tools = []
                    for tool in tools_list:
                        tool_name = tool.get("name", "")
                        if config.tool_filter.is_tool_allowed(tool_name):
                            filtered_tools.append(tool)
                    tools_list = filtered_tools

                return True, f"Found {len(tools_list)} tools on server '{server_name}'", tools_list
            else:
                error_msg = tools_data.get("error", "Failed to list tools")
                return False, f"Failed to list tools on server '{server_name}': {error_msg}", []

        except Exception as e:
            logger.error(f"Error listing tools for server {server_name}: {e}")
            return False, f"Error listing tools: {e}", []

    async def list_filtered_tools(self, server_name: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """List tools available on the specified MCP server with filtering applied."""
        return await self.list_tools(server_name, apply_filter=True)

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Call a tool on the specified MCP server."""
        try:
            valid, error_msg, config = _validate_server_exists(self, server_name)
            if not valid:
                return False, error_msg, {}

            # Check if tool is allowed by filter
            if config.tool_filter and not config.tool_filter.is_tool_allowed(tool_name):
                return False, f"Tool '{tool_name}' is blocked by server filter configuration", {}

            # Create server instance
            server_instance, details = self._create_server_instance(config)
            if not server_instance:
                error_msg = details.get("error", "Failed to create server instance")
                return False, f"Failed to connect to server '{server_name}': {error_msg}", {}

            # Run call_tool operation
            success, result_data = await self._run_tools_operation_async(
                server_instance, server_name, "call_tool", tool_name=tool_name, arguments=arguments
            )

            if success:
                return True, f"Successfully called tool '{tool_name}' on server '{server_name}'", result_data
            else:
                error_msg = result_data.get("error", "Failed to call tool")
                return False, f"Failed to call tool '{tool_name}' on server '{server_name}': {error_msg}", {}

        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on server {server_name}: {e}")
            return False, f"Error calling tool: {e}", {}

    def set_tool_filter(self, server_name: str, tool_filter: ToolFilterConfig) -> Tuple[bool, str]:
        """Set tool filter configuration for a server."""
        try:
            with self._lock:
                config = self.get_server_config(server_name)
                if not config:
                    return False, f"Server '{server_name}' not found"

                # Update the tool filter
                config.tool_filter = tool_filter

                if self.save_config():
                    logger.info(f"Updated tool filter for server: {server_name}")
                    return True, f"Successfully updated tool filter for server '{server_name}'"
                else:
                    return False, "Failed to save config"

        except Exception as e:
            logger.error(f"Error setting tool filter for server {server_name}: {e}")
            return False, f"Error setting tool filter: {e}"

    def get_tool_filter(self, server_name: str) -> Tuple[bool, str, Optional[ToolFilterConfig]]:
        """Get tool filter configuration for a server."""
        try:
            config = self.get_server_config(server_name)
            if not config:
                return False, f"Server '{server_name}' not found", None

            return True, f"Retrieved tool filter for server '{server_name}'", config.tool_filter

        except Exception as e:
            logger.error(f"Error getting tool filter for server {server_name}: {e}")
            return False, f"Error getting tool filter: {e}", None

    def _create_server_instance(self, config: AnyMCPServerConfig) -> Tuple[Any, Dict[str, Any]]:
        """Create MCP server instance based on config type."""
        try:
            # Expand environment variables in config
            config_dict = config.model_dump()
            expanded_config = expand_config_env_vars(config_dict)

            if config.type == MCPServerType.STDIO:
                return self._create_stdio_server(config, expanded_config)
            elif config.type == MCPServerType.SSE:
                return self._create_sse_server(expanded_config)
            elif config.type == MCPServerType.HTTP:
                return self._create_http_server(expanded_config)
            else:
                return None, {"error": f"Unsupported server type: {config.type}"}
        except Exception as e:
            logger.error(f"Failed to create server instance: {e}")
            return None, {"error": str(e)}

    def _create_stdio_server(self, config: STDIOServerConfig, expanded_config: Dict[str, Any]):
        """Create STDIO server instance."""
        env_vars = config.env or {}

        server_params = MCPServerStdioParams(
            command=expanded_config.get("command"),
            args=expanded_config.get("args", []),
            env=env_vars,
        )

        server_instance = SilentMCPServerStdio(params=server_params, client_session_timeout_seconds=60)
        details = {
            "command": expanded_config.get("command"),
            "args": expanded_config.get("args", []),
            "env_count": len(env_vars),
        }
        return server_instance, details

    def _create_sse_server(self, expanded_config: Dict[str, Any]):
        """Create SSE server instance."""
        url = expanded_config.get("url")
        if not url:
            return None, {"error": "URL is required for SSE server"}

        headers = expanded_config.get("headers") or {}
        timeout = 30.0 if not expanded_config.get("timeout") else float(expanded_config.get("timeout"))
        headers["Accept"] = "text/event-stream"

        server_params = MCPServerSseParams(url=url, headers=headers, timeout=timeout, sse_read_timeout=timeout)
        server_instance = MCPServerSse(params=server_params, client_session_timeout_seconds=60)
        details = {"url": url, "headers_count": len(headers) if headers else 0, "timeout": timeout}
        return server_instance, details

    def _create_http_server(self, expanded_config: Dict[str, Any]):
        """Create HTTP server instance."""
        url = expanded_config.get("url")
        if not url:
            return None, {"error": "URL is required for HTTP server"}

        headers = expanded_config.get("headers", {}) or {}
        timeout = float(expanded_config.get("timeout")) if expanded_config.get("timeout") else 30.0

        # Merge default headers with user-provided headers
        merged_headers = {"Accept": "application/json, text/event-stream", **headers}

        server_params = MCPServerStreamableHttpParams(
            url=url,
            headers=merged_headers,
            timeout=timeout,
            sse_read_timeout=timeout,
            terminate_on_close=True,
        )
        server_instance = MCPServerStreamableHttp(params=server_params, client_session_timeout_seconds=60)
        details = {"url": url, "headers_count": len(merged_headers), "timeout": timeout}
        return server_instance, details

    async def _run_tools_operation_async(
        self, server_instance: SilentMCPServerStdio, server_name: str, operation: str, timeout=30.0, **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run tools operation with proper async handling."""
        try:
            # Connect to the server
            await server_instance.connect()
            logger.info(f"Successfully connected to MCP server '{server_name}' for {operation}")

            # Create minimal agent and run context
            agent = Agent(name=f"tools-agent-{server_name}")
            run_context = RunContextWrapper(context=None, usage=Usage())
            async with asyncio.timeout(timeout):
                return await self._dispatch_operation(
                    server_instance, server_name, operation, agent, run_context, **kwargs
                )
        except Exception as e:
            error_msg = str(e)
            if "ConnectError" in str(type(e)):
                error_msg = "Failed to connect to server. Please check the server address and network connectivity."

            logger.error(f"Error executing {operation} on server '{server_name}': {error_msg}")
            return False, {"error": error_msg}
        except (asyncio.CancelledError, ExceptionGroup) as e:
            error_msg = str(e)
            if "ConnectError" in str(type(e)):
                error_msg = "Failed to connect to server. Please check the server address and network connectivity."

            logger.error(f"Error executing {operation} on server '{server_name}': {error_msg}")
            return False, {"error": error_msg}
        finally:
            await self._cleanup_server_instance(server_instance)

    async def _dispatch_operation(
        self, server_instance, server_name: str, operation: str, agent, run_context, **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """Dispatch operation to appropriate handler."""
        if operation == "list_tools":
            return await self._handle_list_tools(server_instance, server_name, run_context, agent)
        elif operation == "call_tool":
            return await self._handle_call_tool(server_instance, server_name, **kwargs)
        elif operation == "connectivity_test":
            return await self._handle_connectivity_test(server_instance, server_name, run_context, agent)
        else:
            return False, {"error": f"Unknown operation: {operation}"}

    async def _handle_list_tools(self, server_instance, server_name: str, run_context, agent):
        """Handle list_tools operation."""
        tools = await server_instance.list_tools(run_context, agent)
        tools_list = []

        if tools:
            for tool in tools:
                input_schema = {}
                if tool.inputSchema:
                    if hasattr(tool.inputSchema, "model_dump"):
                        input_schema = tool.inputSchema.model_dump()
                    elif isinstance(tool.inputSchema, dict):
                        input_schema = tool.inputSchema

                tools_list.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": input_schema,
                    }
                )

        logger.info(f"Found {len(tools_list)} tools on server '{server_name}'")
        return True, {"tools": tools_list}

    async def _handle_call_tool(self, server_instance, server_name: str, **kwargs):
        """Handle call_tool operation."""
        tool_name = kwargs.get("tool_name")
        arguments = kwargs.get("arguments", {})

        if not tool_name:
            return False, {"error": "Tool name is required"}

        result = await server_instance.call_tool(tool_name, arguments)
        result_dict = {"content": [], "isError": getattr(result, "isError", False)}

        if hasattr(result, "content") and result.content:
            for content_item in result.content:
                content_dict = {"type": content_item.type}
                if hasattr(content_item, "text"):
                    content_dict["text"] = content_item.text
                elif hasattr(content_item, "data"):
                    content_dict["data"] = content_item.data
                result_dict["content"].append(content_dict)

        logger.info(f"Successfully called tool '{tool_name}' on server '{server_name}'")
        return True, result_dict

    async def _handle_connectivity_test(self, server_instance, server_name: str, run_context, agent):
        """Handle connectivity_test operation."""
        logger.info(f"Successfully connected to MCP server '{server_name}'")
        tools_info = {"connected": True}

        if hasattr(server_instance, "list_tools"):
            try:
                tools = await server_instance.list_tools(run_context, agent)
                tool_count = len(tools) if tools else 0
                tool_names = [tool.name for tool in tools] if tools else []

                logger.info(f"MCP server '{server_name}' has {tool_count} tools available: {tool_names}")
                tools_info.update({"tools_available": True, "tool_count": tool_count, "tool_names": tool_names})
            except Exception as e:
                logger.warning(f"Failed to list tools on server '{server_name}': {e}")
                tools_info.update({"tools_available": False, "tools_error": str(e)})
        else:
            tools_info["tools_available"] = False
            logger.warning(f"MCP server '{server_name}' does not support list_tools")

        return True, tools_info

    async def _cleanup_server_instance(self, server_instance):
        """Clean up server instance resources."""
        try:
            if hasattr(server_instance, "cleanup"):
                await server_instance.cleanup()
            elif hasattr(server_instance, "disconnect"):
                await server_instance.disconnect()
            elif hasattr(server_instance, "close"):
                await server_instance.close()
        except Exception as cleanup_error:
            logger.debug(f"Cleanup error (suppressed): {cleanup_error}")
        except (asyncio.CancelledError, ExceptionGroup) as cleanup_error:
            logger.debug(f"Cleanup error (suppressed): {cleanup_error}")

    def cleanup(self) -> None:
        """Clean up MCP Manager."""
        logger.info("MCP Manager cleanup complete")
