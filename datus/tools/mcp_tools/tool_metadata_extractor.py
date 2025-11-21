# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tool Metadata Extractor for MCP Tools

This module provides functionality to extract, cache, and manage metadata
for MCP tools, including descriptions, parameters, and usage information.
"""

import asyncio
from typing import Any, Dict, List

from cachetools import TTLCache

from datus.tools.mcp_tools.mcp_manager import MCPManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ToolMetadataExtractor:
    """
    Tool metadata extractor that retrieves and caches detailed information
    about available MCP tools including descriptions, parameters, and usage.
    """

    def __init__(self, cache_ttl_seconds: int = 600):
        """
        Initialize the tool metadata extractor.

        Args:
            cache_ttl_seconds: TTL for metadata cache in seconds (default: 10 minutes)
        """
        self.mcp_manager = MCPManager()

        # Cache for tool metadata to reduce MCP calls
        self.metadata_cache = TTLCache(maxsize=500, ttl=cache_ttl_seconds)

        # Cache for tool lists to avoid repeated calls
        self.tool_list_cache = TTLCache(maxsize=10, ttl=cache_ttl_seconds)

        logger.info("Initialized Tool Metadata Extractor")

    def get_tool_metadata(self, tool_name: str, mcp_server: str = "clickzetta_mcp_sse") -> Dict[str, Any]:
        """
        Get detailed metadata for a specific MCP tool.

        Args:
            tool_name: Name of the tool to get metadata for
            mcp_server: MCP server name

        Returns:
            Dictionary containing tool metadata (description, parameters, etc.)
        """
        cache_key = f"{mcp_server}:{tool_name}"

        # Check cache first
        if cache_key in self.metadata_cache:
            logger.debug(f"Using cached metadata for tool: {tool_name}")
            return self.metadata_cache[cache_key]

        try:
            # Get metadata from MCP server
            metadata = asyncio.run(self._fetch_tool_metadata(tool_name, mcp_server))

            # Cache the result
            self.metadata_cache[cache_key] = metadata

            logger.debug(f"Retrieved metadata for tool: {tool_name}")
            return metadata

        except Exception as e:
            logger.error(f"Error getting metadata for tool {tool_name}: {e}")
            return self._get_default_metadata(tool_name)

    async def _fetch_tool_metadata(self, tool_name: str, mcp_server: str) -> Dict[str, Any]:
        """
        Fetch tool metadata from MCP server.

        Args:
            tool_name: Name of the tool
            mcp_server: MCP server name

        Returns:
            Tool metadata dictionary
        """
        try:
            # First, get the list of all tools to find detailed information
            success, result, tools = await self.mcp_manager.list_tools(server_name=mcp_server, apply_filter=True)

            if success and tools:
                # Find the specific tool in the list
                for tool in tools:
                    if isinstance(tool, dict):
                        if tool.get("name") == tool_name:
                            return self._extract_metadata_from_tool_dict(tool)
                    elif isinstance(tool, str) and tool == tool_name:
                        # If only tool name is available, try to get more info
                        return await self._fetch_detailed_tool_info(tool_name, mcp_server)

            # If tool not found in list, return basic metadata
            logger.warning(f"Tool {tool_name} not found in server {mcp_server}")
            return self._get_default_metadata(tool_name)

        except Exception as e:
            logger.error(f"Error fetching metadata for {tool_name}: {e}")
            return self._get_default_metadata(tool_name)

    async def _fetch_detailed_tool_info(self, tool_name: str, mcp_server: str) -> Dict[str, Any]:
        """
        Fetch detailed information for a tool when only the name is available.

        Args:
            tool_name: Name of the tool
            mcp_server: MCP server name

        Returns:
            Detailed tool metadata
        """
        # Try to get tool schema or description
        try:
            # Attempt to call the tool with no arguments to get parameter info
            success, result, data = await self.mcp_manager.call_tool(
                server_name=mcp_server, tool_name=tool_name, arguments={}
            )

            metadata = self._get_default_metadata(tool_name)

            # If the call failed but provided useful error information, extract it
            if not success and result:
                # Try to extract parameter information from error messages
                if "required" in str(result).lower() or "parameter" in str(result).lower():
                    metadata["parameter_info"] = str(result)

            return metadata

        except Exception as e:
            logger.debug(f"Could not get detailed info for {tool_name}: {e}")
            return self._get_default_metadata(tool_name)

    def _extract_metadata_from_tool_dict(self, tool_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a tool dictionary returned by MCP server.

        Args:
            tool_dict: Tool dictionary from MCP server

        Returns:
            Extracted metadata
        """
        metadata = {
            "name": tool_dict.get("name", ""),
            "description": tool_dict.get("description", "No description available"),
            "parameters": tool_dict.get("parameters", {}),
            "schema": tool_dict.get("inputSchema", {}),
            "category": self._infer_tool_category(tool_dict),
            "complexity": self._assess_tool_complexity(tool_dict),
            "source": "mcp_server",
        }

        # Extract additional fields if available
        if "inputSchema" in tool_dict:
            schema = tool_dict["inputSchema"]
            if "properties" in schema:
                metadata["parameter_details"] = schema["properties"]
            if "required" in schema:
                metadata["required_parameters"] = schema["required"]

        return metadata

    def _get_default_metadata(self, tool_name: str) -> Dict[str, Any]:
        """
        Get default metadata for a tool when detailed information is not available.

        Args:
            tool_name: Name of the tool

        Returns:
            Default metadata dictionary
        """
        # Infer basic information from tool name
        description = self._infer_description_from_name(tool_name)
        category = self._infer_category_from_name(tool_name)

        return {
            "name": tool_name,
            "description": description,
            "parameters": {},
            "schema": {},
            "category": category,
            "complexity": "unknown",
            "source": "inferred",
        }

    def _infer_description_from_name(self, tool_name: str) -> str:
        """
        Infer a description from the tool name.

        Args:
            tool_name: Name of the tool

        Returns:
            Inferred description
        """
        # Simple name-based description inference
        name_patterns = {
            "list": "Lists available items",
            "get": "Retrieves information",
            "describe": "Provides detailed description",
            "discover": "Discovers and analyzes",
            "analyze": "Performs analysis",
            "switch": "Switches configuration",
            "check": "Checks status or health",
            "monitor": "Monitors system state",
            "create": "Creates new items",
            "generate": "Generates content",
            "execute": "Executes operations",
        }

        tool_lower = tool_name.lower()
        for pattern, description in name_patterns.items():
            if pattern in tool_lower:
                return f"{description} - {tool_name}"

        return f"Tool for {tool_name.replace('_', ' ')}"

    def _infer_category_from_name(self, tool_name: str) -> str:
        """
        Infer tool category from its name.

        Args:
            tool_name: Name of the tool

        Returns:
            Inferred category
        """
        name_lower = tool_name.lower()

        # Category inference patterns
        if any(keyword in name_lower for keyword in ["list", "get", "describe", "show"]):
            return "exploration"
        elif any(keyword in name_lower for keyword in ["analyze", "discover", "relationship", "statistics"]):
            return "analysis"
        elif any(keyword in name_lower for keyword in ["switch", "manage", "admin", "configure"]):
            return "management"
        elif any(keyword in name_lower for keyword in ["create", "generate", "build", "make"]):
            return "generation"
        elif any(keyword in name_lower for keyword in ["monitor", "check", "track", "watch"]):
            return "monitoring"
        else:
            return "general"

    def _infer_tool_category(self, tool_dict: Dict[str, Any]) -> str:
        """
        Infer tool category from tool dictionary.

        Args:
            tool_dict: Tool dictionary

        Returns:
            Inferred category
        """
        # First try to infer from description
        description = tool_dict.get("description", "").lower()
        name = tool_dict.get("name", "").lower()

        # Analysis keywords in description
        if any(keyword in description for keyword in ["relationship", "analyze", "discover", "statistics"]):
            return "analysis"

        # Exploration keywords
        if any(keyword in description for keyword in ["list", "retrieve", "get", "browse", "explore"]):
            return "exploration"

        # Management keywords
        if any(keyword in description for keyword in ["switch", "manage", "configure", "admin"]):
            return "management"

        # Generation keywords
        if any(keyword in description for keyword in ["generate", "create", "build", "produce"]):
            return "generation"

        # Monitoring keywords
        if any(keyword in description for keyword in ["monitor", "check", "track", "health"]):
            return "monitoring"

        # Fall back to name-based inference
        return self._infer_category_from_name(name)

    def _assess_tool_complexity(self, tool_dict: Dict[str, Any]) -> str:
        """
        Assess the complexity of a tool based on its parameters and description.

        Args:
            tool_dict: Tool dictionary

        Returns:
            Complexity assessment ("simple", "medium", "complex")
        """
        # Count parameters
        param_count = 0
        if "inputSchema" in tool_dict and "properties" in tool_dict["inputSchema"]:
            param_count = len(tool_dict["inputSchema"]["properties"])
        elif "parameters" in tool_dict:
            param_count = len(tool_dict["parameters"])

        # Check required parameters
        required_count = 0
        if "inputSchema" in tool_dict and "required" in tool_dict["inputSchema"]:
            required_count = len(tool_dict["inputSchema"]["required"])

        # Assess complexity
        if param_count <= 2 and required_count <= 1:
            return "simple"
        elif param_count <= 5 and required_count <= 3:
            return "medium"
        else:
            return "complex"

    def get_all_tools_metadata(self, mcp_server: str = "clickzetta_mcp_sse") -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all available tools from an MCP server.

        Args:
            mcp_server: MCP server name

        Returns:
            Dictionary mapping tool names to their metadata
        """
        cache_key = f"all_tools:{mcp_server}"

        # Check cache first
        if cache_key in self.tool_list_cache:
            logger.debug(f"Using cached all tools metadata for server: {mcp_server}")
            return self.tool_list_cache[cache_key]

        try:
            all_metadata = asyncio.run(self._fetch_all_tools_metadata(mcp_server))

            # Cache the result
            self.tool_list_cache[cache_key] = all_metadata

            logger.info(f"Retrieved metadata for {len(all_metadata)} tools from {mcp_server}")
            return all_metadata

        except Exception as e:
            logger.error(f"Error getting all tools metadata: {e}")
            return {}

    async def _fetch_all_tools_metadata(self, mcp_server: str) -> Dict[str, Dict[str, Any]]:
        """
        Fetch metadata for all tools from MCP server.

        Args:
            mcp_server: MCP server name

        Returns:
            Dictionary of all tools metadata
        """
        try:
            # Get list of all tools
            success, result, tools = await self.mcp_manager.list_tools(server_name=mcp_server, apply_filter=True)

            if not success or not tools:
                logger.warning(f"Failed to get tools from {mcp_server}: {result}")
                return {}

            all_metadata = {}

            # Extract metadata for each tool
            for tool in tools:
                if isinstance(tool, dict):
                    tool_name = tool.get("name")
                    if tool_name:
                        metadata = self._extract_metadata_from_tool_dict(tool)
                        all_metadata[tool_name] = metadata
                elif isinstance(tool, str):
                    tool_name = tool
                    metadata = self._get_default_metadata(tool_name)
                    all_metadata[tool_name] = metadata

            return all_metadata

        except Exception as e:
            logger.error(f"Error fetching all tools metadata: {e}")
            return {}

    def get_tools_by_category(self, category: str, mcp_server: str = "clickzetta_mcp_sse") -> List[str]:
        """
        Get list of tools by category.

        Args:
            category: Tool category to filter by
            mcp_server: MCP server name

        Returns:
            List of tool names in the specified category
        """
        all_metadata = self.get_all_tools_metadata(mcp_server)

        return [tool_name for tool_name, metadata in all_metadata.items() if metadata.get("category") == category]

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.metadata_cache.clear()
        self.tool_list_cache.clear()
        logger.info("Cleared all metadata caches")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "metadata_cache": {
                "size": len(self.metadata_cache),
                "maxsize": self.metadata_cache.maxsize,
                "currsize": self.metadata_cache.currsize,
            },
            "tool_list_cache": {
                "size": len(self.tool_list_cache),
                "maxsize": self.tool_list_cache.maxsize,
                "currsize": self.tool_list_cache.currsize,
            },
        }
