# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Intelligent Tool Selector for MCP Tools

This module provides intelligent tool selection capabilities for MCP tools.
It analyzes user requests and recommends the most appropriate tools based on
intent, context, and tool capabilities.
"""

import asyncio
from typing import Dict, List, Optional, Set
from cachetools import TTLCache

from datus.tools.mcp_tools.mcp_manager import MCPManager
from datus.tools.mcp_tools.tool_metadata_extractor import ToolMetadataExtractor
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class IntelligentToolSelector:
    """
    Intelligent tool selector that provides smart recommendations for MCP tools.

    This class analyzes user requests and provides intelligent tool recommendations
    based on various strategies including keyword matching, semantic analysis,
    and category-based selection.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize the intelligent tool selector.

        Args:
            cache_ttl_seconds: TTL for tool recommendation cache in seconds
        """
        self.mcp_manager = MCPManager()
        self.metadata_extractor = ToolMetadataExtractor()

        # Cache for tool recommendations to improve performance
        self.recommendation_cache = TTLCache(maxsize=1000, ttl=cache_ttl_seconds)

        # Tool category mappings
        self.category_tool_mappings = {
            "analysis": [
                "discover_table_relationships",
                "analyze_schema_statistics",
                "get_table_lineage",
                "analyze_data_distribution",
            ],
            "exploration": ["list_tables", "describe_table", "list_schemas", "get_table_sample"],
            "management": ["switch_instance", "get_job_history", "check_system_status", "get_instance_info"],
            "generation": ["generate_sql", "create_semantic_model", "build_dashboard"],
            "monitoring": ["get_performance_metrics", "check_job_status", "monitor_system_health"],
        }

        # Keyword-to-tool mappings for fast lookup
        self.keyword_mappings = {
            # Relationship analysis keywords
            "relationship": ["discover_table_relationships", "get_table_lineage"],
            "ER": ["discover_table_relationships"],
            "关系": ["discover_table_relationships"],
            "foreign key": ["discover_table_relationships"],
            "primary key": ["discover_table_relationships"],
            "join": ["discover_table_relationships", "get_table_lineage"],
            # Schema analysis keywords
            "schema": ["list_schemas", "describe_table", "analyze_schema_statistics"],
            "table": ["list_tables", "describe_table", "get_table_sample"],
            "column": ["describe_table", "analyze_schema_statistics"],
            "structure": ["describe_table", "list_tables"],
            # Data analysis keywords
            "analysis": ["analyze_schema_statistics", "analyze_data_distribution"],
            "statistics": ["analyze_schema_statistics", "get_performance_metrics"],
            "distribution": ["analyze_data_distribution"],
            "sample": ["get_table_sample"],
            # Management keywords
            "instance": ["switch_instance", "get_instance_info"],
            "job": ["get_job_history", "check_job_status"],
            "history": ["get_job_history"],
            "status": ["check_system_status", "check_job_status"],
            # Chinese keywords
            "分析": ["analyze_schema_statistics", "analyze_data_distribution"],
            "表": ["list_tables", "describe_table"],
            "模式": ["list_schemas"],
            "结构": ["describe_table"],
        }

        logger.info("Initialized Intelligent Tool Selector")

    def get_tool_recommendations(
        self, category: str, user_message: str, mcp_server: str = "clickzetta_mcp_sse", max_recommendations: int = 5
    ) -> List[str]:
        """
        Get intelligent tool recommendations based on category and user message.

        Args:
            category: The request category (e.g., "analysis", "exploration")
            user_message: The user's original message
            mcp_server: MCP server name to get tools from
            max_recommendations: Maximum number of tools to recommend

        Returns:
            List of recommended tool names, ordered by relevance
        """
        cache_key = f"{category}:{hash(user_message)}:{mcp_server}"

        # Check cache first
        if cache_key in self.recommendation_cache:
            logger.debug(f"Using cached recommendations for: {cache_key}")
            return self.recommendation_cache[cache_key]

        try:
            # Get available tools from MCP server
            available_tools = asyncio.run(self._get_available_tools(mcp_server))

            if not available_tools:
                logger.warning(f"No tools available from MCP server: {mcp_server}")
                return []

            # Generate recommendations using multiple strategies
            recommendations = self._generate_recommendations(
                category=category,
                user_message=user_message,
                available_tools=available_tools,
                max_recommendations=max_recommendations,
            )

            # Cache the results
            self.recommendation_cache[cache_key] = recommendations

            logger.info(f"Generated {len(recommendations)} tool recommendations for category: {category}")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating tool recommendations: {e}")
            return []

    async def _get_available_tools(self, mcp_server: str) -> List[Dict]:
        """
        Get list of available tools from MCP server.

        Args:
            mcp_server: MCP server name

        Returns:
            List of tool dictionaries with metadata
        """
        try:
            success, result, tools = await self.mcp_manager.call_tool(
                server_name=mcp_server, tool_name="list_tools", arguments={}
            )

            if success and tools:
                return tools
            else:
                logger.warning(f"Failed to get tools from {mcp_server}: {result}")
                return []

        except Exception as e:
            logger.error(f"Error getting available tools: {e}")
            return []

    def _generate_recommendations(
        self, category: str, user_message: str, available_tools: List[Dict], max_recommendations: int
    ) -> List[str]:
        """
        Generate tool recommendations using multiple strategies.

        Args:
            category: Request category
            user_message: User's message
            available_tools: Available tools from MCP server
            max_recommendations: Maximum recommendations to return

        Returns:
            List of recommended tool names
        """
        # Extract tool names from available tools
        available_tool_names = set()
        for tool in available_tools:
            if isinstance(tool, dict) and "name" in tool:
                available_tool_names.add(tool["name"])
            elif isinstance(tool, str):
                available_tool_names.add(tool)

        # Strategy 1: Category-based recommendations
        category_recommendations = self._get_category_recommendations(category, available_tool_names)

        # Strategy 2: Keyword-based recommendations
        keyword_recommendations = self._get_keyword_recommendations(user_message, available_tool_names)

        # Strategy 3: Semantic similarity recommendations
        semantic_recommendations = self._get_semantic_recommendations(user_message, available_tools)

        # Combine and rank recommendations
        final_recommendations = self._combine_and_rank_recommendations(
            category_recs=category_recommendations,
            keyword_recs=keyword_recommendations,
            semantic_recs=semantic_recommendations,
            max_recommendations=max_recommendations,
        )

        return final_recommendations

    def _get_category_recommendations(self, category: str, available_tools: Set[str]) -> List[str]:
        """Get recommendations based on category mappings."""
        category_tools = self.category_tool_mappings.get(category.lower(), [])
        return [tool for tool in category_tools if tool in available_tools]

    def _get_keyword_recommendations(self, user_message: str, available_tools: Set[str]) -> List[str]:
        """Get recommendations based on keyword matching."""
        message_lower = user_message.lower()
        recommendations = []

        for keyword, tools in self.keyword_mappings.items():
            if keyword.lower() in message_lower:
                for tool in tools:
                    if tool in available_tools and tool not in recommendations:
                        recommendations.append(tool)

        return recommendations

    def _get_semantic_recommendations(self, user_message: str, available_tools: List[Dict]) -> List[str]:
        """Get recommendations based on semantic similarity with tool descriptions."""
        recommendations = []

        # Simple semantic matching based on description similarity
        message_words = set(user_message.lower().split())

        for tool in available_tools:
            if isinstance(tool, dict):
                tool_name = tool.get("name", "")
                tool_description = tool.get("description", "").lower()

                # Calculate simple word overlap score
                description_words = set(tool_description.split())
                overlap_score = len(message_words.intersection(description_words))

                if overlap_score > 0:
                    recommendations.append((tool_name, overlap_score))

        # Sort by overlap score and return tool names
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [tool_name for tool_name, _ in recommendations]

    def _combine_and_rank_recommendations(
        self, category_recs: List[str], keyword_recs: List[str], semantic_recs: List[str], max_recommendations: int
    ) -> List[str]:
        """
        Combine recommendations from different strategies and rank them.

        Args:
            category_recs: Category-based recommendations
            keyword_recs: Keyword-based recommendations
            semantic_recs: Semantic-based recommendations
            max_recommendations: Maximum number to return

        Returns:
            Final ranked list of tool recommendations
        """
        # Score-based ranking system
        tool_scores = {}

        # Category recommendations get highest priority (score 3)
        for tool in category_recs:
            tool_scores[tool] = tool_scores.get(tool, 0) + 3

        # Keyword recommendations get medium priority (score 2)
        for tool in keyword_recs:
            tool_scores[tool] = tool_scores.get(tool, 0) + 2

        # Semantic recommendations get lower priority (score 1)
        for tool in semantic_recs:
            tool_scores[tool] = tool_scores.get(tool, 0) + 1

        # Sort by score and return top recommendations
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        final_recommendations = [tool for tool, score in sorted_tools[:max_recommendations]]

        logger.debug(f"Final tool rankings: {dict(sorted_tools[:max_recommendations])}")
        return final_recommendations

    def get_tool_usage_statistics(self) -> Dict[str, int]:
        """
        Get usage statistics for recommended tools.

        Returns:
            Dictionary mapping tool names to usage counts
        """
        # This would be implemented with proper usage tracking
        # For now, return empty dict
        return {}

    def update_tool_mappings(self, new_mappings: Dict[str, List[str]]) -> None:
        """
        Update tool category mappings with new entries.

        Args:
            new_mappings: New category to tool mappings
        """
        for category, tools in new_mappings.items():
            if category in self.category_tool_mappings:
                # Merge with existing mappings
                existing_tools = set(self.category_tool_mappings[category])
                existing_tools.update(tools)
                self.category_tool_mappings[category] = list(existing_tools)
            else:
                # Add new category mapping
                self.category_tool_mappings[category] = tools

        logger.info(f"Updated tool mappings for categories: {list(new_mappings.keys())}")

    def clear_cache(self) -> None:
        """Clear the recommendation cache."""
        self.recommendation_cache.clear()
        logger.info("Cleared tool recommendation cache")
