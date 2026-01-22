# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Skill function tool for loading skills on demand.

Provides the `load_skill` tool that allows the LLM to load full skill content
from the <available_skills> list in the system prompt.
"""

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.skill_tools.skill_manager import SkillManager

if TYPE_CHECKING:
    from datus.tools.skill_tools.skill_bash_tool import SkillBashTool

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

    def __init__(
        self,
        manager: SkillManager,
        node_name: str,
    ):
        """Initialize the skill function tool.

        Args:
            manager: SkillManager for skill operations
            node_name: Name of the current agentic node
        """
        self.manager = manager
        self.node_name = node_name
        self._tool_context: Any = None
        self._permission_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[bool]]] = None

        # Track loaded skills for bash tool creation
        self._loaded_skills: Dict[str, "SkillBashTool"] = {}

    def set_tool_context(self, ctx: Any) -> None:
        """Set tool context (called by framework before tool invocation).

        Args:
            ctx: Tool context from the agent framework
        """
        self._tool_context = ctx

    def set_permission_callback(
        self, callback: Callable[[str, str, Dict[str, Any]], Awaitable[bool]]
    ) -> None:
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
                # For ASK, we need to handle this differently
                # The actual prompt happens at the AgenticNode level
                # Here we just note that approval is needed
                logger.info(f"Skill '{skill_name}' requires user approval")
                return FuncToolResult(
                    success=0,
                    error=f"Skill '{skill_name}' requires user approval before loading",
                )

            # Load the skill content
            success, message, content = self.manager.load_skill(
                skill_name=skill_name,
                node_name=self.node_name,
                check_permission=False,  # Already checked above
            )

            if not success:
                return FuncToolResult(success=0, error=message)

            # Create bash tool if skill has scripts
            skill = self.manager.get_skill(skill_name)
            if skill and skill.has_scripts():
                self._create_skill_bash_tool(skill)

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

    def _create_skill_bash_tool(self, skill) -> None:
        """Create a SkillBashTool for a skill with scripts.

        Args:
            skill: SkillMetadata with allowed_commands
        """
        if skill.name in self._loaded_skills:
            return

        try:
            from datus.tools.skill_tools.skill_bash_tool import SkillBashTool

            bash_tool = SkillBashTool(
                skill_metadata=skill,
                workspace_root=str(skill.location),
            )
            self._loaded_skills[skill.name] = bash_tool
            logger.debug(f"Created SkillBashTool for skill '{skill.name}'")
        except Exception as e:
            logger.error(f"Failed to create SkillBashTool for '{skill.name}': {e}")

    def get_skill_bash_tool(self, skill_name: str) -> Optional["SkillBashTool"]:
        """Get the bash tool for a loaded skill.

        Args:
            skill_name: Name of the skill

        Returns:
            SkillBashTool if skill has scripts and is loaded, None otherwise
        """
        return self._loaded_skills.get(skill_name)

    def get_all_skill_bash_tools(self) -> Dict[str, "SkillBashTool"]:
        """Get all created skill bash tools.

        Returns:
            Dictionary of skill_name -> SkillBashTool
        """
        return self._loaded_skills

    def available_tools(self) -> List[Tool]:
        """Return the list of tools provided by this class.

        Returns:
            List containing the load_skill tool
        """
        return [trans_to_function_tool(self.load_skill)]

    def get_loaded_skill_tools(self) -> List[Tool]:
        """Get tools from all loaded skills (including bash tools).

        Returns:
            List of tools from loaded skills with script capabilities
        """
        tools = []
        for bash_tool in self._loaded_skills.values():
            tools.extend(bash_tool.available_tools())
        return tools
