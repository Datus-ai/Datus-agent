import asyncio
import inspect
from typing import Dict, List, Type

from fastmcp import MCPServer

from datus.tools.base import BaseTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# untested
class ToolsMCPServer:
    """FastMCP-based tool server for exposing registered tools"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize MCP server
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server = MCPServer()
        self.tools: Dict[str, BaseTool] = {}  # Mapping of tool names to instances

    def register_tool(self, tool_instance: BaseTool) -> None:
        """Register a tool instance to the server
        Args:
            tool_instance: Tool instance to register
        """
        tool_name = tool_instance.__class__.__name__
        self.tools[tool_name] = tool_instance

        # Register all public methods of the tool
        for name, method in inspect.getmembers(tool_instance):
            if callable(method) and not name.startswith("_"):
                # Skip base class methods
                if name in ("validate_input", "format_output", "handle_error"):
                    continue

                # Get method signature
                inspect.signature(method)

                # Create async wrapper
                async def wrapper(handle_method=method, handle_name=name, *args, **kwargs):
                    try:
                        # Remove first argument (self)
                        result = handle_method(*args[1:], **kwargs)
                        return result
                    except Exception as e:
                        logger.error(f"Error executing {tool_name}.{handle_name}: {str(e)}")
                        return {"success": False, "error": str(e)}

                # Register to MCP server
                self.server.register_function(f"{tool_name}.{name}", wrapper)
                logger.info(f"Registered function: {tool_name}.{name}")

    def register_tool_class(self, tool_class: Type[BaseTool], **kwargs) -> None:
        """Register a tool class to the server
        Args:
            tool_class: Tool class to register
            **kwargs: Initialization arguments for the tool class
        """
        tool_instance = tool_class(**kwargs)
        self.register_tool(tool_instance)

    def start(self) -> None:
        """Start the MCP server"""
        logger.info(f"Starting MCP server at {self.host}:{self.port}")
        asyncio.run(self.server.start(self.host, self.port))

    def stop(self) -> None:
        """Stop the MCP server"""
        logger.info("Stopping MCP server")
        asyncio.run(self.server.stop())

    def get_registered_tools(self) -> List[str]:
        """Get names of all registered tools"""
        return list(self.tools.keys())

    def get_tool_methods(self, tool_name: str) -> List[str]:
        """Get all methods of a specific tool
        Args:
            tool_name: Name of the tool
        Returns:
            List of method names
        """
        if tool_name not in self.tools:
            return []

        methods = []
        for name, method in inspect.getmembers(self.tools[tool_name]):
            if callable(method) and not name.startswith("_"):
                if name not in ("validate_input", "format_output", "handle_error"):
                    methods.append(name)

        return methods
