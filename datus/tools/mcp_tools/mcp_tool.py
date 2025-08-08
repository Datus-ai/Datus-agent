"""
MCP Tool - Tool class for MCP server management operations.

This module provides the MCPTool class that implements the BaseTool interface
for MCP server management operations. It acts as a wrapper around MCPManager
providing a standardized tool interface.
"""

from typing import Any, Dict, Optional

from datus.tools.base import BaseTool, BaseToolExecResult, ToolAction
from datus.utils.loggings import get_logger

from .mcp_manager import MCPManager

logger = get_logger(__name__)


class MCPTool(BaseTool):
    """Tool class for MCP server management operations."""

    tool_name = "mcp_tool"
    tool_description = "Management tool for MCP (Model Context Protocol) servers"

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the MCP tool.

        Args:
            config_path: Path to MCP config file
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.manager = MCPManager(config_path=config_path)
        logger.info(f"Initialized MCP Tool with config path: {self.manager.config_path}")

    @ToolAction(description="Add a new MCP server config")
    def add_server(
        self,
        name: str,
        server_type: str,
        **config_params,
    ) -> BaseToolExecResult:
        """
        Add a new MCP server config.

        Args:
            name: Server name/identifier
            server_type: Server type (stdio, sse, http)
            **config_params: Server type specific config parameters

        Returns:
            BaseToolExecResult with operation result
        """
        try:
            # Prepare config data for server creation
            config_data = {"type": server_type, **config_params}

            # Create server config using the factory method
            from .mcp_config import MCPServerConfig

            server_config = MCPServerConfig.from_config_format(name, config_data)

            success, message = self.manager.add_server(server_config)

            result_data = {}
            if success:
                result_data.update(server_config.model_dump())

            return BaseToolExecResult(success=success, message=message, result=result_data if success else None)

        except Exception as e:
            logger.error(f"Error in add_server: {e}")
            return BaseToolExecResult(success=False, message=f"Error adding server: {e}")

    @ToolAction(description="Remove an MCP server config")
    def remove_server(self, name: str) -> BaseToolExecResult:
        """
        Remove an MCP server config.

        Args:
            name: Server name

        Returns:
            BaseToolExecResult with operation result
        """
        try:
            success, message = self.manager.remove_server(name)

            return BaseToolExecResult(
                success=success,
                message=message,
                result={"removed_server": name} if success else None,
            )

        except Exception as e:
            logger.error(f"Error in remove_server: {e}")
            return BaseToolExecResult(success=False, message=f"Error removing server: {e}")

    @ToolAction(description="List MCP server configs")
    def list_servers(self, server_type: Optional[str] = None) -> BaseToolExecResult:
        """
        List MCP server configs with optional filtering.

        Args:
            server_type: Filter by server type (stdio, sse, http)

        Returns:
            BaseToolExecResult with list of server configs
        """
        try:
            servers = self.manager.list_servers(server_type=server_type)

            # Convert to dict format for result
            server_list = []
            for server in servers:
                server_dict = server.model_dump()
                server_list.append(server_dict)

            return BaseToolExecResult(
                success=True,
                message=f"Found {len(server_list)} servers",
                result={"servers": server_list, "total_count": len(server_list)},
            )

        except Exception as e:
            logger.error(f"Error in list_servers: {e}")
            return BaseToolExecResult(success=False, message=f"Error listing servers: {e}")

    @ToolAction(description="Get MCP server config")
    def get_server(self, name: str) -> BaseToolExecResult:
        """
        Get MCP server config by name.

        Args:
            name: Server name

        Returns:
            BaseToolExecResult with server config
        """
        try:
            config = self.manager.get_server_config(name)

            if not config:
                return BaseToolExecResult(success=False, message=f"Server '{name}' not found")

            server_dict = config.model_dump()

            return BaseToolExecResult(success=True, message=f"Retrieved config for server '{name}'", result=server_dict)

        except Exception as e:
            logger.error(f"Error in get_server: {e}")
            return BaseToolExecResult(success=False, message=f"Error getting server: {e}")

    @ToolAction(description="Check connectivity to an MCP server")
    def check_connectivity(self, name: str) -> BaseToolExecResult:
        """
        Check connectivity to an MCP server by attempting to connect and list tools.

        Args:
            name: Server name

        Returns:
            BaseToolExecResult with connectivity status
        """
        try:
            success, message, details = self.manager.check_connectivity(name)

            return BaseToolExecResult(
                success=success,
                message=message,
                result={
                    "name": name,
                    "connectivity": success,
                    "details": details,
                },
            )

        except Exception as e:
            logger.error(f"Error in check_connectivity: {e}")
            return BaseToolExecResult(success=False, message=f"Error checking connectivity: {e}")

    @ToolAction(description="List tools available on an MCP server")
    def list_tools(self, server_name: str) -> BaseToolExecResult:
        """
        List tools available on an MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            BaseToolExecResult with list of available tools
        """
        try:
            success, message, tools_list = self.manager.list_tools(server_name)

            if success:
                return BaseToolExecResult(
                    success=True,
                    message=message,
                    result={"server_name": server_name, "tools_count": len(tools_list), "tools": tools_list},
                )
            else:
                return BaseToolExecResult(
                    success=False, message=message, result={"server_name": server_name, "tools_count": 0, "tools": []}
                )

        except Exception as e:
            logger.error(f"Error in list_tools: {e}")
            return BaseToolExecResult(success=False, message=f"Error listing tools: {e}")

    @ToolAction(description="Call a tool on an MCP server")
    def call_tool(
        self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> BaseToolExecResult:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool (optional)

        Returns:
            BaseToolExecResult with tool execution result
        """
        try:
            if arguments is None:
                arguments = {}

            success, message, result_data = self.manager.call_tool(server_name, tool_name, arguments)

            if success:
                return BaseToolExecResult(
                    success=True,
                    message=message,
                    result={
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result_data,
                    },
                )
            else:
                return BaseToolExecResult(
                    success=False,
                    message=message,
                    result={
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "error": result_data,
                    },
                )

        except Exception as e:
            logger.error(f"Error in call_tool: {e}")
            return BaseToolExecResult(success=False, message=f"Error calling tool: {e}")

    def cleanup(self) -> None:
        """Clean up the MCP tool and manager."""
        if self.manager:
            self.manager.cleanup()
