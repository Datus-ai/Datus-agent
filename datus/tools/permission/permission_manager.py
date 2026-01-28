# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Permission manager for unified permission control.

Provides permission checking and filtering for all tools, MCP servers, and skills
following Claude Code and OpenCode patterns.
"""

import fnmatch
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule

if TYPE_CHECKING:
    from datus.tools.func_tool.base import Tool

logger = logging.getLogger(__name__)


class PermissionManager:
    """Unified permission manager for all tools, MCP, and skills.

    Handles permission checking, tool filtering, and user confirmation for ASK permissions.

    The permission system:
    1. Evaluates rules in order (last match wins)
    2. DENY permissions hide tools from system prompt (LLM never sees them)
    3. ASK permissions prompt user for confirmation before execution
    4. Node-specific overrides layer on top of global config

    Example usage:
        manager = PermissionManager(
            global_config=agent_config.permissions_config,
            node_overrides={"chatbot": chatbot_permissions}
        )

        # Check permission before tool execution
        perm = manager.check_permission("db_tools", "execute_sql", "chatbot")
        if perm == PermissionLevel.DENY:
            return error
        elif perm == PermissionLevel.ASK:
            approved = await manager.request_confirmation(...)
    """

    def __init__(
        self,
        global_config: Optional[PermissionConfig] = None,
        node_overrides: Optional[Dict[str, PermissionConfig]] = None,
    ):
        """Initialize the permission manager.

        Args:
            global_config: Global permission configuration from agent.yml
            node_overrides: Per-node permission overrides (node_name -> config)
        """
        self.global_config = global_config or PermissionConfig()
        self.node_overrides = node_overrides or {}
        self._permission_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[bool]]] = None

        # Cache for session-approved permissions (tool_category.tool_name -> approved)
        self._session_approvals: Dict[str, bool] = {}

        logger.debug(f"PermissionManager initialized with {len(self.global_config.rules)} global rules")

    def set_permission_callback(self, callback: Callable[[str, str, Dict[str, Any]], Awaitable[bool]]) -> None:
        """Set callback for ASK permission user prompts.

        Args:
            callback: Async function(tool_category, tool_name, context) -> bool
        """
        self._permission_callback = callback

    def get_effective_config(self, node_name: str) -> PermissionConfig:
        """Get effective permission config for a node (global + overrides).

        Args:
            node_name: Name of the agentic node

        Returns:
            Merged PermissionConfig
        """
        node_override = self.node_overrides.get(node_name)

        # Convert dict to PermissionConfig if needed
        if node_override is not None and isinstance(node_override, dict):
            node_override = PermissionConfig.from_dict(node_override)

        return self.global_config.merge_with(node_override)

    def check_permission(
        self,
        tool_category: str,
        tool_name: str,
        node_name: str,
    ) -> PermissionLevel:
        """Check permission for a tool invocation.

        Evaluation order:
        1. Global rules (first to last)
        2. Node-specific override rules (first to last)

        Last matching rule wins. DENY takes precedence at same specificity.

        Args:
            tool_category: Category of the tool (db_tools, mcp, skills, etc.)
            tool_name: Name of the specific tool or skill
            node_name: Name of the current agentic node

        Returns:
            PermissionLevel (ALLOW, DENY, or ASK)
        """
        effective_config = self.get_effective_config(node_name)

        # Start with default permission
        result = effective_config.default_permission

        # Evaluate rules in order (last match wins)
        for rule in effective_config.rules:
            if self._rule_matches(rule, tool_category, tool_name):
                result = rule.permission
                logger.debug(
                    f"Permission rule matched: {rule.tool}.{rule.pattern} -> {rule.permission} "
                    f"for {tool_category}.{tool_name}"
                )

        logger.debug(f"Permission check: {tool_category}.{tool_name} @ {node_name} = {result}")
        return result

    def _rule_matches(self, rule: PermissionRule, tool_category: str, tool_name: str) -> bool:
        """Check if a permission rule matches the given tool.

        Supports glob patterns:
        - "*" matches everything
        - "db_tools" matches category exactly
        - "execute_*" matches tools starting with "execute_"
        - "dangerous-*" matches skills with that prefix

        Args:
            rule: Permission rule to check
            tool_category: Category of the tool
            tool_name: Name of the tool

        Returns:
            True if rule matches
        """
        # Check tool category match
        if rule.tool != "*" and not fnmatch.fnmatch(tool_category, rule.tool):
            return False

        # Check pattern match within category
        if rule.pattern != "*" and not fnmatch.fnmatch(tool_name, rule.pattern):
            return False

        return True

    def filter_available_tools(
        self,
        tools: List["Tool"],
        node_name: str,
        tool_category: Optional[str] = None,
    ) -> List["Tool"]:
        """Filter tools list, hiding DENY tools from system prompt.

        DENY tools are completely hidden - the LLM never knows they exist.
        ALLOW and ASK tools are included.

        Args:
            tools: List of available tools
            node_name: Name of the current agentic node
            tool_category: Optional explicit tool category (auto-detected if not provided)

        Returns:
            Filtered list of tools (DENY removed)
        """
        filtered = []
        for tool in tools:
            # Use provided category or determine from tool name
            category = tool_category if tool_category else self._get_tool_category(tool.name)

            permission = self.check_permission(category, tool.name, node_name)

            if permission != PermissionLevel.DENY:
                filtered.append(tool)
            else:
                logger.debug(f"Tool {tool.name} hidden due to DENY permission for node {node_name}")

        logger.debug(f"Filtered tools: {len(filtered)}/{len(tools)} visible for node {node_name}")
        return filtered

    def filter_available_skills(
        self,
        skills: List[Any],  # List[SkillMetadata]
        node_name: str,
    ) -> List[Any]:
        """Filter skills list, hiding DENY skills from system prompt.

        DENY skills are completely hidden from <available_skills>.

        Args:
            skills: List of SkillMetadata objects
            node_name: Name of the current agentic node

        Returns:
            Filtered list of skills (DENY removed)
        """
        filtered = []
        for skill in skills:
            permission = self.check_permission("skills", skill.name, node_name)

            if permission != PermissionLevel.DENY:
                filtered.append(skill)
            else:
                logger.debug(f"Skill {skill.name} hidden due to DENY permission for node {node_name}")

        logger.debug(f"Filtered skills: {len(filtered)}/{len(skills)} visible for node {node_name}")
        return filtered

    async def request_user_confirmation(
        self,
        tool_category: str,
        tool_name: str,
        context: Dict[str, Any],
    ) -> bool:
        """Request user confirmation for ASK permission.

        Args:
            tool_category: Category of the tool
            tool_name: Name of the tool
            context: Additional context (arguments, etc.)

        Returns:
            True if user approved, False otherwise
        """
        # Check session cache first
        cache_key = f"{tool_category}.{tool_name}"
        if cache_key in self._session_approvals:
            return self._session_approvals[cache_key]

        if not self._permission_callback:
            logger.warning(f"ASK permission for {cache_key} but no callback set, defaulting to deny")
            return False

        try:
            approved = await self._permission_callback(tool_category, tool_name, context)
            logger.info(f"User {'approved' if approved else 'rejected'} {cache_key}")
            return approved
        except Exception as e:
            logger.error(f"Permission callback failed for {cache_key}: {e}")
            return False

    def approve_for_session(self, tool_category: str, tool_name: str) -> None:
        """Mark a tool as approved for the rest of the session.

        Used when user selects "always" for an ASK permission.

        Args:
            tool_category: Category of the tool
            tool_name: Name of the tool
        """
        cache_key = f"{tool_category}.{tool_name}"
        self._session_approvals[cache_key] = True
        logger.info(f"Session approval granted for {cache_key}")

    def clear_session_approvals(self) -> None:
        """Clear all session approvals (e.g., on session end)."""
        self._session_approvals.clear()

    def _get_tool_category(self, tool_name: str) -> str:
        """Determine tool category from tool name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool category string
        """
        # Common tool name prefixes to categories
        if tool_name.startswith("db_") or tool_name in (
            "execute_sql",
            "list_tables",
            "describe_table",
            "get_table_schema",
            "get_sample_data",
        ):
            return "db_tools"
        elif tool_name == "load_skill" or tool_name.startswith("skill_"):
            return "skills"
        elif tool_name.startswith("search_") or tool_name in ("search_metrics", "search_tables", "search_documents"):
            return "context_search_tools"
        elif tool_name.startswith("fs_") or tool_name in (
            "read_file",
            "write_file",
            "list_files",
            "list_directory",
        ):
            return "filesystem_tools"
        elif tool_name.startswith("date_") or tool_name in ("parse_date", "get_current_date"):
            return "date_tools"
        else:
            # For MCP tools, the category might be in the format "server.tool"
            if "." in tool_name:
                return "mcp"
            return "tools"

    def get_permission_summary(self, node_name: str) -> Dict[str, Any]:
        """Get a summary of permissions for debugging.

        Args:
            node_name: Name of the agentic node

        Returns:
            Dictionary with permission summary
        """
        effective_config = self.get_effective_config(node_name)
        return {
            "node_name": node_name,
            "default_permission": effective_config.default_permission.value,
            "rule_count": len(effective_config.rules),
            "rules": [
                {"tool": r.tool, "pattern": r.pattern, "permission": r.permission.value} for r in effective_config.rules
            ],
            "session_approvals": list(self._session_approvals.keys()),
        }
