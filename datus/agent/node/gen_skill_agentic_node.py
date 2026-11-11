# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SkillCreatorAgenticNode implementation for skill creation, editing, and review.

This module provides an AgenticNode that guides users through creating,
editing, reviewing, and scaffolding Datus skills. It exposes filesystem (read+write),
database (optional), ask_user, and skill loading tools, running with a
configurable max_turns budget for extended multi-step interactions.
"""

from typing import Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput, SkillCreatorNodeResult
from datus.tools.func_tool import DBFuncTool, FilesystemFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SkillCreatorAgenticNode(AgenticNode):
    """
    Skill creation, editing, and review agentic node.

    Guides users through creating new skills, editing and reviewing existing skills,
    and scaffolding skill directory structures. Exposes full filesystem
    tools (read+write), optional database tools, ask_user for interactive
    interview, and skill loading tools for edit mode.
    """

    # Canonical class identifier. ``get_node_class_name()`` returns this even
    # when a custom alias (e.g. ``my_skill_editor: { node_class: gen_skill }``)
    # is used, so skill ``allowed_agents: [gen_skill]`` still matches aliases.
    NODE_NAME = "gen_skill"
    result_class = SkillCreatorNodeResult

    # Authoring agent: its ``load_skill`` bypasses ``allowed_agents`` so the
    # optimize/edit workflow can read any skill by name.
    SKILL_AUTHORING_MODE = True

    # Optional Datus format reference; the system prompt owns the workflow.
    DEFAULT_SKILLS = "create-skill"

    def __init__(
        self,
        node_id: str = "skill_creator",
        description: str = "Skill creation node",
        node_type: str = "gen_skill",
        input_data: Optional[SkillCreatorNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        # Support custom node_name for alias subagents (e.g. my_skill_editor:
        # {node_class: gen_skill}); fall back to the canonical class name.
        self._configured_node_name = node_name or self.NODE_NAME
        self.execution_mode = execution_mode

        # Default max_turns = 100, can be overridden by agent.yml
        self.max_turns = 100
        config_key = self._configured_node_name
        if agent_config and hasattr(agent_config, "agentic_nodes") and config_key in (agent_config.agentic_nodes or {}):
            agentic_node_config = agent_config.agentic_nodes.get(config_key, {})
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 100)

        # Initialize tool attributes before parent constructor
        self.db_func_tool: Optional[DBFuncTool] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.skill_func_tool_instance = None
        self._session_search_tool = None
        self.skill_validate_tool = None

        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Setup tools
        self.setup_tools()
        logger.debug(f"SkillCreatorAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

    def get_node_name(self) -> str:
        return self._configured_node_name

    def _setup_skill_func_tools(self) -> None:
        """Ignore the retired guide in old configuration before resolving skills."""
        configured_skills = self.node_config.get("skills")
        if isinstance(configured_skills, str):
            patterns = [pattern.strip() for pattern in configured_skills.split(",") if pattern.strip()]
            if "optimize-skill" in patterns:
                logger.warning(
                    "Ignoring obsolete skill 'optimize-skill' in node '%s' skills configuration.",
                    self.get_node_name(),
                )
                # Normalize only this node's runtime view, leaving agent.yml
                # and the shared request configuration unchanged.
                self.node_config = dict(self.node_config)
                remaining = [pattern for pattern in patterns if pattern != "optimize-skill"]
                if remaining:
                    self.node_config["skills"] = ", ".join(remaining)
                else:
                    self.node_config.pop("skills", None)
        super()._setup_skill_func_tools()

    def setup_tools(self):
        """Setup tools for skill creation: filesystem, db, ask_user, skill loading."""
        if not self.agent_config:
            return

        self.tools = []
        self._setup_full_filesystem_tools()
        if not self.filesystem_func_tool:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": "Failed to setup filesystem tools — cannot create skills"},
            )
        self._setup_db_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        self._setup_skill_loading_tools()
        self._setup_validate_tool()
        if not self.skill_validate_tool:
            logger.warning("validate_skill tool unavailable — skill validation will be skipped")
        self._setup_session_search_tool()

        logger.debug(f"Setup {len(self.tools)} skill creator tools: {[tool.name for tool in self.tools]}")

    def _setup_full_filesystem_tools(self):
        """Setup a single filesystem tool rooted at the project.

        Visibility follows the zone classifier in ``classify_path``:
        ``.datus/skills/`` and ``~/.datus/skills/`` are WHITELIST (writable),
        the rest of the project tree is INTERNAL (also writable by this node),
        ``.datus/`` internals other than skills are HIDDEN (invisible), and
        anything outside the project root is EXTERNAL (the permission hook
        prompts; strict mode rejects). There is no per-kind write gate — the
        prompt is responsible for steering skill writes into the whitelist.
        """
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.info(f"Setup filesystem tools rooted at: {self.filesystem_func_tool.root_path}")
        except Exception as e:
            logger.warning(f"Failed to setup filesystem tools, continuing without: {e}")

    def _setup_db_tools(self):
        """Setup database tools (optional, for understanding schema when creating data-related skills)."""
        try:
            self.db_func_tool = DBFuncTool(
                # Same anchor the filesystem tools use, so a path the model got
                # from glob resolves identically in load_file_as_table.
                filesystem_root=self._resolve_workspace_root(),
                agent_config=self.agent_config,
                sub_agent_name=self.get_node_name(),
            )
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as e:
            logger.warning(f"Failed to setup database tools, continuing without: {e}")

    def _setup_skill_loading_tools(self):
        """Setup skill loading tools for reading existing skills (edit mode)."""
        try:
            from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
            from datus.tools.skill_tools.skill_manager import SkillManager

            # Share the base node's registry and loaded roots with filesystem
            # tools/hooks so loaded plugin bundles become readable immediately.
            if self.skill_manager is None:
                self.skill_manager = SkillManager(
                    config=getattr(self.agent_config, "skills_config", None),
                    permission_manager=self.permission_manager,
                    config_mutable=self._resolve_config_mutable(),
                    agent_config=self.agent_config,
                )
            if self.skill_func_tool is None:
                self.skill_func_tool = SkillFuncTool(
                    manager=self.skill_manager,
                    node_name=self.get_node_name(),
                    node_class=self.get_node_class_name(),
                    authoring_mode=self.SKILL_AUTHORING_MODE,
                )
            self.skill_func_tool_instance = self.skill_func_tool
            self._ensure_skill_tools_in_tools()
            logger.debug(f"Setup skill loading tools with {self.skill_manager.get_skill_count()} skills")
        except Exception as e:
            logger.warning(f"Failed to setup skill loading tools, continuing without: {e}")

    def _setup_validate_tool(self):
        """Setup the validate_skill tool for checking SKILL.md correctness."""
        try:
            from datus.tools.func_tool.skill_validate_tool import SkillValidateTool

            self.skill_validate_tool = SkillValidateTool(filesystem_tool=self.filesystem_func_tool)
            self.tools.extend(self.skill_validate_tool.available_tools())
            logger.debug("Setup skill validate tool")
        except Exception as e:
            logger.warning(f"Failed to setup skill validate tool, continuing without: {e}")

    def _setup_session_search_tool(self):
        """Setup session search tool for finding historical skill usage patterns."""
        try:
            from datus.tools.func_tool.session_search_tool import SessionSearchTool

            sessions_dir = None
            if self.agent_config:
                try:
                    from datus.utils.path_manager import get_path_manager

                    pm = get_path_manager(agent_config=self.agent_config)
                    sessions_dir = str(pm.sessions_dir)
                except Exception:
                    pass
            self._session_search_tool = SessionSearchTool(sessions_dir=sessions_dir)
            self.tools.extend(self._session_search_tool.available_tools())
            logger.debug(f"Setup session search tool with sessions_dir: {sessions_dir}")
        except Exception as e:
            logger.warning(f"Failed to setup session search tool, continuing without: {e}")

    def _get_system_prompt(self, prompt_version: Optional[str] = None) -> str:
        """Get the system prompt for the skill creator node."""
        from datus.prompts.prompt_manager import get_prompt_manager

        version = prompt_version or self.node_config.get("prompt_version")
        template_name = "skill_creator_system"
        self._ensure_lazy_tools_mounted()
        self._populate_tool_registry()
        tool_names = {tool.name for tool in self.tools}

        # Gather configured skill directories
        skill_directories: List[str] = []
        if self.agent_config:
            skills_config = getattr(self.agent_config, "skills_config", None)
            if skills_config and hasattr(skills_config, "directories"):
                skill_directories = skills_config.directories
            else:
                skill_directories = ["./.datus/skills", "~/.datus/skills"]

        from datus.utils.node_utils import build_datasource_prompt_context

        context = {
            "has_db_tools": any(self.tool_registry.get(name) == "db_tools" for name in tool_names),
            "has_validate_skill": "validate_skill" in tool_names,
            "skill_directories": skill_directories,
            "workspace_root": self._resolve_workspace_root(),
            **build_datasource_prompt_context(self.agent_config),
        }

        try:
            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, version=version, **context
            )

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

    def _build_success_result(self, ctx: StreamRunContext) -> SkillCreatorNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            response_content = (
                ctx.last_successful_output.get("content", "")
                or ctx.last_successful_output.get("text", "")
                or ctx.last_successful_output.get("response", "")
                or ctx.last_successful_output.get("raw_output", "")
                or str(ctx.last_successful_output)
            )
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        skill_name = None
        skill_path = None
        if ctx.last_successful_output:
            skill_name = ctx.last_successful_output.get("skill_name")
            skill_path = ctx.last_successful_output.get("skill_path")

        all_actions = ctx.action_history_manager.get_actions()
        tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]
        tokens_used = self._extract_total_tokens(all_actions)

        return SkillCreatorNodeResult(
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
