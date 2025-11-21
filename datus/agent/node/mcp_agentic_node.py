# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MCP Agentic Node - Universal MCP Tool Orchestrator

This module implements an intelligent MCP (Model Context Protocol) node that provides
universal tool orchestration capabilities. It automatically detects user intent,
categorizes requests, and intelligently selects and executes appropriate MCP tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.base import BaseInput, BaseResult
from datus.tools.mcp_tools.intelligent_tool_selector import IntelligentToolSelector
from datus.tools.mcp_tools.mcp_manager import MCPManager
from datus.tools.mcp_tools.tool_categorizer import ToolCategorizer
from datus.tools.mcp_tools.tool_metadata_extractor import ToolMetadataExtractor
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.workflow import Workflow

logger = get_logger(__name__)


class MCPAgenticNode(AgenticNode):
    """
    Universal MCP Tool Orchestrator Node

    This node provides intelligent orchestration of MCP tools with the following capabilities:
    - Automatic user intent detection and categorization
    - Intelligent tool selection based on user requests
    - Dynamic tool recommendation with caching
    - Comprehensive tool execution with error handling
    - Streaming support for real-time feedback
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Any]] = None,
        node_name: Optional[str] = None,
    ):
        super().__init__(node_id, description, node_type, input_data, agent_config, tools, node_name)

        # Initialize MCP components
        self.mcp_manager = MCPManager()
        self.tool_categorizer = ToolCategorizer()
        self.tool_selector = IntelligentToolSelector()
        self.metadata_extractor = ToolMetadataExtractor()

        # Get MCP server name from node configuration
        if node_name and agent_config and hasattr(agent_config, "agentic_nodes"):
            node_config = agent_config.agentic_nodes.get(node_name)
            if node_config and hasattr(node_config, "mcp"):
                self.mcp_server_name = node_config.mcp
            else:
                self.mcp_server_name = "clickzetta_mcp_sse"  # fallback
        else:
            self.mcp_server_name = "clickzetta_mcp_sse"

        logger.info(f"Initialized MCP Agentic Node with server: {self.mcp_server_name}")

    def _detect_user_intent_and_category(self, user_message: str) -> Tuple[str, str]:
        """
        Detect user intent and categorize the request for intelligent tool selection.

        Args:
            user_message: The user's request message

        Returns:
            Tuple of (intent_description, category)
        """
        try:
            # Use the tool categorizer to determine the request category
            category = self.tool_categorizer.categorize_user_request(user_message)

            # Extract intent from the user message
            intent = user_message.strip()

            logger.debug(f"Detected intent: {intent}, category: {category}")
            return intent, category

        except Exception as e:
            logger.error(f"Error detecting user intent: {e}")
            return user_message, "general"

    def _get_tool_recommendations_for_category(self, category: str, user_message: str) -> List[str]:
        """
        Get intelligent tool recommendations based on category and user message.

        Args:
            category: The request category (e.g., "analysis", "generation", "management")
            user_message: The original user message

        Returns:
            List of recommended tool names
        """
        try:
            # Use the intelligent tool selector to get recommendations
            recommendations = self.tool_selector.get_tool_recommendations(
                category=category, user_message=user_message, mcp_server=self.mcp_server_name
            )

            logger.debug(f"Tool recommendations for category '{category}': {recommendations}")
            return recommendations

        except Exception as e:
            logger.error(f"Error getting tool recommendations: {e}")
            return []

    def _get_recommended_tools_prompt(self, recommended_tools: List[str]) -> str:
        """
        Generate a prompt section with recommended tools for the LLM.

        Args:
            recommended_tools: List of recommended tool names

        Returns:
            Formatted prompt section with tool recommendations
        """
        if not recommended_tools:
            return ""

        tools_section = "## Recommended MCP Tools\n\n"
        tools_section += "Based on your request, the following specialized tools are recommended:\n\n"

        for tool_name in recommended_tools:
            # Get tool metadata for better descriptions
            metadata = self.metadata_extractor.get_tool_metadata(tool_name, self.mcp_server_name)
            description = metadata.get("description", "No description available")

            tools_section += f"- **{tool_name}**: {description}\n"

        tools_section += "\nPlease prioritize using these specialized tools over basic tools when possible.\n\n"

        return tools_section

    async def _execute_mcp_tool_intelligently(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Any, Any]:
        """
        Execute an MCP tool with intelligent error handling and fallbacks.

        Args:
            tool_name: Name of the MCP tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tuple of (success, result, data)
        """
        try:
            # Execute the tool
            success, result, data = await self.mcp_manager.call_tool(
                server_name=self.mcp_server_name, tool_name=tool_name, arguments=tool_args
            )

            if success:
                logger.info(f"Successfully executed MCP tool: {tool_name}")
                return success, result, data
            else:
                logger.warning(f"MCP tool execution failed: {tool_name}, error: {result}")
                return success, result, data

        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return False, str(e), None

    def update_context(self, workflow: "Workflow") -> Dict:
        """Update context with MCP-specific information."""
        context = super().update_context(workflow)

        # Add MCP server information
        context.update({"mcp_server": self.mcp_server_name, "mcp_capabilities": "intelligent_tool_orchestration"})

        return context

    def setup_input(self, workflow: "Workflow") -> Dict:
        """Setup input with enhanced MCP tool orchestration."""
        input_dict = super().setup_input(workflow)

        # Get user message from input
        user_message = getattr(self.input, "user_message", "")

        if user_message:
            # Detect user intent and category
            intent, category = self._detect_user_intent_and_category(user_message)

            # Get intelligent tool recommendations
            recommended_tools = self._get_tool_recommendations_for_category(category, user_message)

            # Generate enhanced prompt with tool recommendations
            recommended_tools_prompt = self._get_recommended_tools_prompt(recommended_tools)

            # Update input with enhanced context
            input_dict.update(
                {
                    "user_intent": intent,
                    "request_category": category,
                    "recommended_tools": recommended_tools,
                    "tools_prompt_section": recommended_tools_prompt,
                }
            )

            logger.info(f"Enhanced MCP input for category: {category}, tools: {recommended_tools}")

        return input_dict

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute MCP node with streaming support and intelligent tool orchestration.

        This method provides real-time feedback during MCP tool execution and
        implements intelligent fallback strategies when tools fail.
        """
        try:
            # Yield initial status
            if action_history_manager:
                yield ActionHistory(
                    action="mcp_orchestration_start",
                    description="Starting intelligent MCP tool orchestration",
                    status="running",
                )

            # Get user message
            user_message = getattr(self.input, "user_message", "")

            if user_message:
                # Step 1: Intent detection and categorization
                intent, category = self._detect_user_intent_and_category(user_message)

                if action_history_manager:
                    yield ActionHistory(
                        action="intent_detection",
                        description=f"Detected intent category: {category}",
                        status="completed",
                    )

                # Step 2: Tool recommendation
                recommended_tools = self._get_tool_recommendations_for_category(category, user_message)

                if action_history_manager:
                    tools_list = ", ".join(recommended_tools) if recommended_tools else "none"
                    yield ActionHistory(
                        action="tool_recommendation",
                        description=f"Recommended tools: {tools_list}",
                        status="completed",
                    )

            # Execute the main agentic workflow with enhanced context
            async for action in super().execute_stream(action_history_manager):
                yield action

            # Yield completion status
            if action_history_manager:
                yield ActionHistory(
                    action="mcp_orchestration_complete",
                    description="MCP tool orchestration completed successfully",
                    status="completed",
                )

        except Exception as e:
            logger.error(f"Error in MCP node streaming execution: {e}")
            if action_history_manager:
                yield ActionHistory(
                    action="mcp_orchestration_error", description=f"MCP orchestration failed: {str(e)}", status="failed"
                )
            raise

    def execute(self) -> BaseResult:
        """Execute MCP node with intelligent tool orchestration."""
        try:
            # Setup enhanced context
            if hasattr(self, "workflow") and self.workflow:
                enhanced_input = self.setup_input(self.workflow)

                # Add enhanced context to the model if available
                if hasattr(self.model, "add_context"):
                    self.model.add_context(enhanced_input)

            # Execute the main agentic workflow
            result = super().execute()

            logger.info(f"MCP node execution completed with success: {result.success}")
            return result

        except Exception as e:
            logger.error(f"Error in MCP node execution: {e}")
            return BaseResult(success=False, error=str(e))
