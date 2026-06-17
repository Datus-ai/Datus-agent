# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill function tool for loading skills on demand.

Provides the `load_skill` tool that allows the LLM to load full skill content
from the <available_skills> list in the system prompt.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.skill_tools.skill_manager import SkillManager

logger = logging.getLogger(__name__)


class SkillFuncTool:
    """Native tool for loading skills on demand.

    Provides the `load_skill` function tool that the LLM can call to retrieve
    full skill content. Integrates with the permission system for ASK prompts.

    Example usage:
        skill_tool = SkillFuncTool(
            manager=skill_manager,
            node_name="chatbot"
        )
        skill_tool.set_permission_callback(permission_callback)

        # Add to tools list
        tools = skill_tool.available_tools()

        # LLM calls load_skill(skill_name="sql-optimization")
    """

    permission_category: str = "skills"

    def __init__(
        self,
        manager: SkillManager,
        node_name: str,
        node_class: Optional[str] = None,
        authoring_mode: bool = False,
    ):
        """Initialize the skill function tool.

        Args:
            manager: SkillManager for skill operations.
            node_name: Agent node name (alias) of the current agentic node.
            node_class: Canonical class identifier (e.g. ``gen_dashboard`` for
                a subagent aliased as ``my_dashboard``). Passed alongside
                ``node_name`` when matching ``allowed_agents`` so class-level
                scoping applies to custom aliases. Defaults to ``node_name``.
            authoring_mode: When True, ``load_skill`` bypasses each skill's
                ``allowed_agents`` scope so that a skill-authoring workflow
                (e.g. the ``gen_skill`` subagent editing an existing skill)
                can fetch content by explicit name. Visibility through
                ``get_available_skills`` is unaffected and permissions still
                apply.
        """
        self.manager = manager
        self.node_name = node_name
        self.node_class = node_class or node_name
        self.authoring_mode = authoring_mode
        self._tool_context: Any = None
        self._permission_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[bool]]] = None

    def set_tool_context(self, ctx: Any) -> None:
        """Set tool context (called by framework before tool invocation).

        Args:
            ctx: Tool context from the agent framework
        """
        self._tool_context = ctx

    def set_permission_callback(self, callback: Callable[[str, str, Dict[str, Any]], Awaitable[bool]]) -> None:
        """Set callback for ASK permission prompts.

        Args:
            callback: Async function(tool_category, tool_name, context) -> bool
        """
        self._permission_callback = callback

    def load_skill(self, skill_name: str) -> FuncToolResult:
        """Load a skill by name from the available skills list.

        This tool should be called when you need detailed instructions from a skill.
        The skill content will be returned and can be used to guide your responses.

        Skills are listed in <available_skills> in the system prompt. Use this tool
        to retrieve the full content of a skill when you need its detailed instructions.

        Args:
            skill_name: Name of the skill to load (from <available_skills>)

        Returns:
            FuncToolResult with skill content on success, error on failure

        Example:
            load_skill(skill_name="sql-optimization")
            # Returns full SKILL.md content with detailed instructions
        """
        try:
            # Check permission first
            permission = self.manager.check_skill_permission(skill_name, self.node_name)

            if permission == PermissionLevel.DENY:
                logger.warning(f"Skill '{skill_name}' denied for node '{self.node_name}'")
                return FuncToolResult(
                    success=0,
                    error=f"Skill '{skill_name}' is not available",
                )

            if permission == PermissionLevel.ASK:
                # ASK permissions are handled by PermissionHooks in on_tool_start
                # which runs BEFORE this tool function executes.
                #
                # If we reach here, it means one of:
                # 1. Hooks prompted user and they approved (cached in session)
                # 2. Hooks are not configured (fallback behavior)
                #
                # In case 1: User already approved, proceed to load skill
                # In case 2: Without hooks, we can't prompt - proceed anyway
                #            (this maintains backward compatibility)
                #
                # If user denied via hooks, PermissionDeniedException was raised
                # and we never reach this code.
                logger.debug(f"Skill '{skill_name}' has ASK permission, proceeding (hooks handle prompts)")
                # Continue to load the skill below

            # Load the skill content
            success, message, content = self.manager.load_skill(
                skill_name=skill_name,
                node_name=self.node_name,
                check_permission=False,  # Already checked above
                check_scope=not self.authoring_mode,
                node_class=self.node_class,
            )

            if not success:
                return FuncToolResult(success=0, error=message)

            logger.info(f"Skill '{skill_name}' loaded successfully for node '{self.node_name}'")
            return FuncToolResult(
                success=1,
                result=content,
            )

        except Exception as e:
            logger.error(f"Failed to load skill '{skill_name}': {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to load skill: {str(e)}",
            )

    def available_tools(self) -> List[Tool]:
        """Return the list of tools provided by this class.

        Returns:
            List containing the load_skill tool
        """
        return [
            trans_to_function_tool(self.load_skill),
        ]
