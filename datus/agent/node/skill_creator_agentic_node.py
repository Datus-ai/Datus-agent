# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SkillCreatorAgenticNode implementation for interactive skill creation and editing.

This module provides an AgenticNode that guides users through creating,
editing, and scaffolding Datus skills. It exposes filesystem (read+write),
database (optional), ask_user, and skill loading tools, running with a
higher max_turns budget for extended multi-step interactions.
"""

import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.workflow import Workflow
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.skill_creator_agentic_node_models import SkillCreatorNodeInput, SkillCreatorNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool, FilesystemFuncTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Read-only filesystem methods (use workspace root — can read anything)
READONLY_FILESYSTEM_METHODS = [
    "read_file",
    "read_multiple_files",
    "list_directory",
    "directory_tree",
    "search_files",
]

# Write filesystem methods (restricted to skills directory only)
WRITE_FILESYSTEM_METHODS = [
    "write_file",
    "edit_file",
    "create_directory",
    "move_file",
]


class SkillCreatorAgenticNode(AgenticNode):
    """
    Interactive skill creation and editing agentic node.

    Guides users through creating new skills, editing existing skills,
    and scaffolding skill directory structures. Exposes full filesystem
    tools (read+write), optional database tools, ask_user for interactive
    interview, and skill loading tools for edit mode.
    """

    def __init__(
        self,
        node_id: str = "skill_creator",
        description: str = "Skill creation node",
        node_type: str = "skill_creator",
        input_data: Optional[SkillCreatorNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
    ):
        self.configured_node_name = node_name

        # Default max_turns = 30, can be overridden by agent.yml
        self.max_turns = 30
        if (
            agent_config
            and hasattr(agent_config, "agentic_nodes")
            and (node_name or "create_skill") in (agent_config.agentic_nodes or {})
        ):
            agentic_node_config = agent_config.agentic_nodes.get(node_name or "create_skill", {})
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Initialize tool attributes before parent constructor
        self.db_func_tool: Optional[DBFuncTool] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self._write_filesystem_tool: Optional[FilesystemFuncTool] = None
        self.skill_func_tool_instance = None
        self.skill_validate_tool = None

        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},
        )

        # Setup tools
        self.setup_tools()
        logger.debug(f"SkillCreatorAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

    def get_node_name(self) -> str:
        return self.configured_node_name or "create_skill"

    def setup_tools(self):
        """Setup tools for skill creation: filesystem, db, ask_user, skill loading."""
        if not self.agent_config:
            return

        self.tools = []
        self._setup_full_filesystem_tools()
        self._setup_db_tools()
        self._setup_ask_user_tool()
        self._setup_skill_loading_tools()
        self._setup_validate_tool()

        logger.debug(f"Setup {len(self.tools)} skill creator tools: {[tool.name for tool in self.tools]}")

    def _resolve_skills_write_root(self) -> str:
        """Resolve the root path for write operations — restricted to the first skills directory."""
        if self.agent_config:
            skills_config = getattr(self.agent_config, "skills_config", None)
            if skills_config and hasattr(skills_config, "directories") and skills_config.directories:
                first_dir = skills_config.directories[0]
                expanded = os.path.expanduser(first_dir)
                resolved = str(Path(expanded).resolve())
                # Ensure the directory exists
                Path(resolved).mkdir(parents=True, exist_ok=True)
                return resolved
        # Fallback: ~/.datus/skills/
        fallback = os.path.expanduser("~/.datus/skills")
        Path(fallback).mkdir(parents=True, exist_ok=True)
        return fallback

    def _setup_full_filesystem_tools(self):
        """Setup filesystem tools: read from workspace root, write restricted to skills directory."""
        try:
            # Read tools — use home dir so we can browse both workspace and ~/.datus/skills
            read_root = os.path.expanduser("~")
            self.filesystem_func_tool = FilesystemFuncTool(root_path=read_root)
            for method_name in READONLY_FILESYSTEM_METHODS:
                if hasattr(self.filesystem_func_tool, method_name):
                    method = getattr(self.filesystem_func_tool, method_name)
                    self.tools.append(trans_to_function_tool(method))
            logger.debug(f"Setup read-only filesystem tools with root: {read_root}")

            # Write tools — restricted to skills directory only
            write_root = self._resolve_skills_write_root()
            self._write_filesystem_tool = FilesystemFuncTool(root_path=write_root)
            for method_name in WRITE_FILESYSTEM_METHODS:
                if hasattr(self._write_filesystem_tool, method_name):
                    method = getattr(self._write_filesystem_tool, method_name)
                    self.tools.append(trans_to_function_tool(method))
            logger.info(f"Setup write filesystem tools restricted to: {write_root}")
        except Exception as e:
            logger.warning(f"Failed to setup filesystem tools, continuing without: {e}")

    def _setup_db_tools(self):
        """Setup database tools (optional, for understanding schema when creating data-related skills)."""
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
            self.db_func_tool = DBFuncTool(
                conn,
                agent_config=self.agent_config,
            )
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as e:
            logger.warning(f"Failed to setup database tools, continuing without: {e}")

    def _setup_skill_loading_tools(self):
        """Setup skill loading tools for reading existing skills (edit mode)."""
        try:
            from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
            from datus.tools.skill_tools.skill_manager import SkillManager

            skill_manager = SkillManager(
                permission_manager=self.permission_manager,
            )
            self.skill_func_tool_instance = SkillFuncTool(
                manager=skill_manager,
                node_name=self.get_node_name(),
            )
            self.tools.extend(self.skill_func_tool_instance.available_tools())
            logger.debug(f"Setup skill loading tools with {skill_manager.get_skill_count()} skills")
        except Exception as e:
            logger.warning(f"Failed to setup skill loading tools, continuing without: {e}")

    def _setup_validate_tool(self):
        """Setup the validate_skill tool for checking SKILL.md correctness."""
        try:
            from datus.tools.func_tool.skill_validate_tool import SkillValidateTool

            self.skill_validate_tool = SkillValidateTool()
            self.tools.extend(self.skill_validate_tool.available_tools())
            logger.debug("Setup skill validate tool")
        except Exception as e:
            logger.warning(f"Failed to setup skill validate tool, continuing without: {e}")

    def _load_companion_skill_content(self) -> str:
        """Load the companion skill-creator SKILL.md content for injection into system prompt.

        Searches the SkillRegistry for the 'skill-creator' skill and returns its
        markdown body (without frontmatter). Returns empty string if not found.
        """
        if not self.skill_func_tool_instance or not self.skill_func_tool_instance.manager:
            return ""
        try:
            registry = self.skill_func_tool_instance.manager.registry
            skills = registry.list_skills()
            metadata = skills.get("skill-creator")
            if not metadata:
                return ""
            content = registry.load_skill_content("skill-creator")
            if not content:
                return ""
            # Strip YAML frontmatter — return only the markdown body
            import re

            match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
            return content.strip()
        except Exception as e:
            logger.debug(f"Could not load companion skill-creator content: {e}")
            return ""

    def _get_system_prompt(
        self, conversation_summary: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> str:
        """Get the system prompt for the skill creator node."""
        from datus.prompts.prompt_manager import prompt_manager
        from datus.utils.time_utils import get_default_current_date

        version = prompt_version or self.node_config.get("prompt_version")
        template_name = "skill_creator_system"

        # Gather existing skill names for context
        existing_skills = ""
        if self.skill_func_tool_instance and self.skill_func_tool_instance.manager:
            try:
                skill_names = list(self.skill_func_tool_instance.manager.registry.list_skills().keys())
                if skill_names:
                    existing_skills = ", ".join(sorted(skill_names))
            except Exception:
                pass

        # Gather configured skill directories
        skill_directories: List[str] = []
        if self.agent_config:
            skills_config = getattr(self.agent_config, "skills_config", None)
            if skills_config and hasattr(skills_config, "directories"):
                skill_directories = skills_config.directories
            else:
                skill_directories = ["~/.datus/skills", "./skills"]

        context = {
            "has_db_tools": bool(self.db_func_tool),
            "has_filesystem_tools": bool(self.filesystem_func_tool),
            "has_ask_user_tool": bool(self.ask_user_tool),
            "has_skill_tools": bool(self.skill_func_tool_instance),
            "skill_directories": skill_directories,
            "existing_skills": existing_skills,
            "workspace_root": self._resolve_workspace_root(),
            "conversation_summary": conversation_summary,
            "current_date": get_default_current_date(None),
        }

        try:
            base_prompt = prompt_manager.render_template(template_name=template_name, version=version, **context)

            # Auto-load the companion skill-creator SKILL.md for deep knowledge
            companion_content = self._load_companion_skill_content()
            if companion_content:
                base_prompt += "\n\n## Skill Creator Reference Guide\n\n" + companion_content

            return self._finalize_system_prompt(base_prompt)
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    def setup_input(self, workflow: Workflow) -> dict:
        """Setup skill creator input from workflow context."""
        if not self.input or not isinstance(self.input, SkillCreatorNodeInput):
            self.input = SkillCreatorNodeInput(
                user_message=workflow.task.task,
            )
        return {"success": True, "message": "Skill creator input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> Dict:
        """Skill creator does not update workflow context."""
        return {"success": True, "message": "Skill creator node does not update workflow context"}

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the skill creation workflow with streaming support.

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if not self.input:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": "Skill creator input not set. Call setup_input() first or set self.input directly."
                },
            )

        user_input = self.input

        # Create initial action
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=self.get_node_name(),
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        try:
            await self._auto_compact()

            session, conversation_summary = self._get_or_create_session()

            system_prompt = self._get_system_prompt(conversation_summary)

            response_content = ""
            tokens_used = 0
            last_successful_output = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=user_input.user_message,
                tools=self.tools,
                mcp_servers={},
                instruction=system_prompt,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=None,
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or stream_action.output.get("raw_output", "")
                            or response_content
                        )

            if not response_content and last_successful_output:
                response_content = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("text", "")
                    or last_successful_output.get("response", "")
                    or last_successful_output.get("raw_output", "")
                    or str(last_successful_output)
                )

            # Extract token usage
            for act in reversed(action_history_manager.get_actions()):
                if act.role == "assistant" and act.output and isinstance(act.output, dict):
                    usage_info = act.output.get("usage", {})
                    if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                        try:
                            tokens_used = int(usage_info.get("total_tokens", 0))
                        except (TypeError, ValueError):
                            tokens_used = 0
                        if tokens_used > 0:
                            break

            # Extract skill_name and skill_path from the response if available
            skill_name = None
            skill_path = None
            if last_successful_output:
                skill_name = last_successful_output.get("skill_name")
                skill_path = last_successful_output.get("skill_path")

            # Build result
            all_actions = action_history_manager.get_actions()
            tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]

            result = SkillCreatorNodeResult(
                success=True,
                response=response_content,
                skill_name=skill_name,
                skill_path=skill_path,
                tokens_used=int(tokens_used),
                action_history=[a.model_dump() for a in all_actions],
                execution_stats={
                    "total_actions": len(all_actions),
                    "tool_calls_count": len(tool_calls),
                    "tools_used": sorted({a.action_type for a in tool_calls}),
                    "total_tokens": int(tokens_used),
                },
            )

            self.actions.extend(all_actions)
            self.result = result

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{self.get_node_name()}_response",
                messages=f"{self.get_node_name()} skill creation completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise

        except Exception as e:
            from datus.utils.exceptions import DatusException

            if isinstance(e, DatusException):
                error_msg = f"[{e.code}] {e}"
                logger.error(f"{self.get_node_name()} execution error: {error_msg}")
            else:
                error_msg = str(e)
                logger.error(f"{self.get_node_name()} execution error: {error_msg}")

            error_result = SkillCreatorNodeResult(
                success=False,
                error=error_msg,
                response="Sorry, I encountered an error during skill creation.",
                tokens_used=0,
            )

            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {error_msg}",
            )

            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{self.get_node_name()}_error",
                messages=f"Error: {error_msg}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            self.result = error_result
            yield error_action
