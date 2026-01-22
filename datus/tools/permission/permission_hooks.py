# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Permission hooks for unified permission checking on all tools.

This module provides AgentHooks implementation that intercepts all tool calls
and performs permission checking before execution. It supports:
- Native Tools (db_tools, context_search_tools, filesystem_tools, etc.)
- MCP Tools (mcp.{server}.{tool})
- Skills (skills.{skill_name})

The hooks integrate with the InteractionBroker for async user interactions
when prompting users for permission confirmation.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agents.lifecycle import AgentHooks

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.tools.permission.permission_config import PermissionLevel

if TYPE_CHECKING:
    from datus.tools.permission.permission_manager import PermissionManager

logger = logging.getLogger(__name__)

# Global lock to prevent multiple permission prompts at once
_permission_prompt_lock = asyncio.Lock()


class PermissionDeniedException(Exception):
    """Exception raised when a tool call is denied by permission rules."""

    def __init__(self, message: str, tool_category: str = "", tool_name: str = ""):
        super().__init__(message)
        self.tool_category = tool_category
        self.tool_name = tool_name


class CompositeHooks(AgentHooks):
    """Combines multiple AgentHooks into one.

    This class allows multiple hooks to be applied in sequence,
    enabling composition of permission hooks with other hooks
    (e.g., PlanModeHooks).
    """

    def __init__(self, hooks_list: List[Optional[AgentHooks]]):
        """Initialize with a list of hooks.

        Args:
            hooks_list: List of AgentHooks instances (None values are filtered out)
        """
        self.hooks_list = [h for h in hooks_list if h is not None]

    async def on_start(self, context, agent) -> None:
        """Called when agent starts."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_start"):
                await hooks.on_start(context, agent)

    async def on_tool_start(self, context, agent, tool) -> None:
        """Called before a tool is executed."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_tool_start"):
                await hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Called after a tool completes."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_tool_end"):
                await hooks.on_tool_end(context, agent, tool, result)

    async def on_llm_end(self, context, agent, response) -> None:
        """Called when LLM finishes a turn."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_llm_end"):
                await hooks.on_llm_end(context, agent, response)

    async def on_end(self, context, agent, output) -> None:
        """Called when agent ends."""
        for hooks in self.hooks_list:
            if hasattr(hooks, "on_end"):
                await hooks.on_end(context, agent, output)


class PermissionHooks(AgentHooks):
    """AgentHooks implementation for unified permission checking on all tools.

    This class intercepts all tool calls and checks permissions before execution.
    It follows the existing tool classification structure:
    - Native Tools: Uses tool_registry to map tool_name -> category
    - MCP Tools: Parses "mcp__{server}__{tool}" format
    - Skills: Uses "skills" category with skill_name as pattern

    Example usage:
        permission_hooks = PermissionHooks(
            broker=interaction_broker,
            permission_manager=manager,
            node_name="chat",
        )

        # Register tools during setup
        permission_hooks.register_tools("db_tools", db_func_tool.available_tools())

        # Use in execution config
        config["hooks"] = CompositeHooks([existing_hooks, permission_hooks])
    """

    def __init__(
        self,
        broker: InteractionBroker,
        permission_manager: "PermissionManager",
        node_name: str,
        tool_registry: Optional[Dict[str, str]] = None,
    ):
        """Initialize the permission hooks.

        Args:
            broker: InteractionBroker for async user interactions
            permission_manager: PermissionManager for checking permissions
            node_name: Name of the current agentic node (e.g., "chat")
            tool_registry: Optional pre-populated tool_name -> category mapping
        """
        self.broker = broker
        self.permission_manager = permission_manager
        self.node_name = node_name
        # Tool registry: tool_name -> category
        # Populated by AgenticNode.setup_tools via register_tools()
        self.tool_registry: Dict[str, str] = tool_registry or {}

    def register_tools(self, category: str, tools: List[Any]) -> None:
        """Register tools with their category.

        Called by AgenticNode.setup_tools to populate the tool registry.

        Args:
            category: Tool category (e.g., "db_tools", "skills")
            tools: List of Tool objects with .name attribute
        """
        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            self.tool_registry[tool_name] = category
            logger.debug(f"Registered tool '{tool_name}' with category '{category}'")

    async def on_tool_start(self, context, agent, tool) -> None:
        """Intercept ALL tool calls for permission checking.

        This method is called before each tool execution. It:
        1. Determines the tool category and pattern name
        2. Checks permission against the PermissionManager
        3. For DENY: raises PermissionDeniedException
        4. For ASK: prompts user via InteractionBroker, handles response
        5. For ALLOW: continues without interruption

        Args:
            context: Tool context with arguments
            agent: The agent instance
            tool: The tool being called

        Raises:
            PermissionDeniedException: If permission is denied or user rejects
        """
        tool_name = getattr(tool, "name", str(tool))

        # Get tool category and pattern name for permission checking
        category, pattern_name = self._get_category_and_pattern(tool_name, context)

        logger.debug(f"Permission check for tool '{tool_name}': category='{category}', pattern='{pattern_name}'")

        # Check permission
        permission = self.permission_manager.check_permission(category, pattern_name, self.node_name)

        if permission == PermissionLevel.DENY:
            logger.warning(f"Tool '{tool_name}' denied by permission rules")
            raise PermissionDeniedException(
                f"Tool '{tool_name}' is not permitted in node '{self.node_name}'",
                tool_category=category,
                tool_name=pattern_name,
            )

        if permission == PermissionLevel.ASK:
            # Check multiple cache keys (tool_name and pattern_name might differ)
            cache_keys = [
                f"{category}.{pattern_name}",
                f"{category}.{tool_name}",
                f"{category}.*",  # Wildcard approval for category
            ]

            for cache_key in cache_keys:
                if self.permission_manager._session_approvals.get(cache_key):
                    logger.debug(f"Tool '{tool_name}' already approved for session (cache_key: {cache_key})")
                    return

            # Use lock to prevent multiple prompts at once (for parallel tool calls)
            async with _permission_prompt_lock:
                # Re-check cache after acquiring lock (another prompt may have approved it)
                for cache_key in cache_keys:
                    if self.permission_manager._session_approvals.get(cache_key):
                        logger.debug(f"Tool '{tool_name}' approved while waiting for lock")
                        return

                # Request user confirmation via InteractionBroker
                approved = await self._request_user_confirmation(category, pattern_name, context)

                if not approved:
                    logger.info(f"User rejected tool '{tool_name}'")
                    raise PermissionDeniedException(
                        f"User rejected execution of '{tool_name}'",
                        tool_category=category,
                        tool_name=pattern_name,
                    )

                logger.info(f"User approved tool '{tool_name}'")

    def _get_category_and_pattern(self, tool_name: str, context: Any) -> Tuple[str, str]:
        """Get tool category and pattern name for permission checking.

        This method determines how to classify a tool for permission rules.

        Returns:
            Tuple of (category, pattern_name)

        Examples:
            Native:  ("db_tools", "execute_sql")
            MCP:     ("mcp.filesystem", "read_file")
            Skills:  ("skills", "deep-analysis")  # skill_name from args
        """
        # 1. Skills: load_skill -> extract skill_name as pattern (check BEFORE registry)
        #    This allows permission rules like "skills.admin-*" to match specific skills
        if tool_name == "load_skill":
            args = self._parse_tool_args(context)
            skill_name = args.get("skill_name", "*")
            return ("skills", skill_name)

        # 2. MCP Tools: format "mcp__{server}__{tool}" -> ("mcp.{server}", "{tool}")
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__")  # ["mcp", "filesystem", "read_file"]
            if len(parts) >= 3:
                server = parts[1]
                method = "__".join(parts[2:])  # Handle multi-part tool names
                return (f"mcp.{server}", method)

        # 3. Check tool registry (Native Tools registered via register_tools())
        if tool_name in self.tool_registry:
            category = self.tool_registry[tool_name]
            return (category, tool_name)

        # 4. Default: unknown category
        logger.debug(f"Tool '{tool_name}' not in registry, using default category 'tools'")
        return ("tools", tool_name)

    def _parse_tool_args(self, context: Any) -> dict:
        """Parse tool arguments from context.

        Args:
            context: Tool context object with tool_arguments attribute

        Returns:
            Dictionary of tool arguments
        """
        try:
            args_str = getattr(context, "tool_arguments", "{}")
            if isinstance(args_str, str):
                return json.loads(args_str)
            elif isinstance(args_str, dict):
                return args_str
            return {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse tool arguments: {e}")
            return {}

    async def _request_user_confirmation(
        self,
        category: str,
        pattern_name: str,
        context: Any,
    ) -> bool:
        """Request user confirmation via InteractionBroker.

        This method uses the async InteractionBroker pattern to prompt
        the user for permission approval.

        Args:
            category: Tool category (e.g., "skills", "mcp.filesystem")
            pattern_name: Specific tool/skill name
            context: Tool context for additional info

        Returns:
            True if user approved, False otherwise
        """
        # Build permission request content (markdown format)
        args = self._parse_tool_args(context)

        content = f"### Permission Request\n\n**Tool:** `{category}.{pattern_name}`\n"

        # Show tool arguments if available (truncate long args)
        if args:
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 200:
                args_str = args_str[:197] + "..."
            content += f"\n**Args:** `{args_str}`\n"

        try:
            choice, callback = await self.broker.request(
                content=content,
                choices={
                    "y": "Allow (once)",
                    "a": "Always allow (session)",
                    "n": "Deny",
                },
                default_choice="n",
            )

            if choice == "a":
                # Approve for session - all future calls to this tool are auto-approved
                self.permission_manager.approve_for_session(category, pattern_name)
                await callback(f"**{category}.{pattern_name}** approved for session")
                return True
            elif choice == "y":
                # One-time approval - do NOT cache, will prompt again next time
                await callback("**Approved**")
                return True
            else:
                await callback("**Denied**")
                return False

        except InteractionCancelled:
            return False
        except Exception as e:
            logger.error(f"Error in permission confirmation: {e}")
            return False
