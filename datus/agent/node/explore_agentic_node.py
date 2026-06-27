# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ExploreAgenticNode implementation for read-only data exploration.

This module provides a lightweight AgenticNode focused on gathering context
(schema structure, data samples, metrics, reference SQL, knowledge) before
SQL generation. It exposes only read-only tools and runs with a low max_turns
budget for fast, focused exploration.
"""

from typing import Dict, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.explore_agentic_node_models import ExploreNodeInput, ExploreNodeResult
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.tools.func_tool.date_parsing_tools import DateParsingTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Read-only filesystem methods exposed to the explore agent
READONLY_FILESYSTEM_METHODS = [
    "read_file",
    "glob",
    "grep",
]


class ExploreAgenticNode(AgenticNode):
    """
    Read-only data exploration agentic node.

    Gathers context information (schema, data samples, metrics, knowledge)
    to support downstream SQL generation. Exposes only read-only tools
    and uses a low max_turns budget for fast exploration.
    """

    # Canonical class identifier. ``get_node_class_name()`` returns this even
    # when a custom alias (e.g. ``my_explorer: { node_class: explore }``) is
    # used, so skill ``allowed_agents: [explore]`` still matches aliases.
    NODE_NAME = "explore"
    result_class = ExploreNodeResult

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ExploreNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        self.configured_node_name = node_name
        # Surface to permission-hook code paths so workflow callers (API /
        # gateway) can short-circuit interactive ASK prompts that would
        # otherwise block on a missing broker listener.
        self.execution_mode = execution_mode

        # Default max_turns = 50, can be overridden by agent.yml
        self.max_turns = 50
        config_key = node_name or self.NODE_NAME
        if agent_config and hasattr(agent_config, "agentic_nodes") and config_key in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[config_key]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        # Initialize tool attributes before parent constructor
        self.db_func_tool: Optional[DBFuncTool] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.date_parsing_tools: Optional[DateParsingTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None

        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Setup read-only tools. When input_data is None (e.g. factory path),
        # scoped_tables are not yet available, so tools are set up without
        # scoping. execute_stream() will call setup_tools() again after input
        # is set to rebuild DB tools with the per-run scoped_tables allowlist.
        self.setup_tools()
        logger.debug(f"ExploreAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

    def get_node_name(self) -> str:
        return self.configured_node_name or "explore"

    def setup_tools(self):
        """Setup read-only tools for exploration."""
        if not self.agent_config:
            return

        self.tools = []
        self._setup_db_tools()
        self._setup_context_search_tools()
        self._setup_readonly_filesystem_tools()
        self._setup_date_parsing_tools()

        logger.debug(f"Setup {len(self.tools)} explore tools: {[tool.name for tool in self.tools]}")

    def _setup_db_tools(self):
        """Setup database tools (all are read-only)."""
        try:
            dynamic_scoped_tables = None
            if isinstance(self.input, ExploreNodeInput) and self.input.scoped_tables:
                dynamic_scoped_tables = self.input.scoped_tables
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.get_node_name(),
                scoped_tables=dynamic_scoped_tables,
                # Explore profiles the datasource and must stay read-only;
                # ``execute_sql`` is write-capable, so reject non-read SQL at the
                # tool layer (the permission gate alone would defer writes to the
                # normal ASK/ALLOW flow rather than hard-deny them).
                read_only=True,
            )
            if dynamic_scoped_tables:
                # A per-run scoped table allowlist indicates a tightly
                # bounded profiling task. Keep the DB tool surface narrow so
                # the model cannot drift into broader schema exploration.
                self.tools.extend(
                    [
                        self.db_func_tool.to_function_tool(self.db_func_tool.describe_table),
                        self.db_func_tool.to_function_tool(self.db_func_tool.execute_sql),
                    ]
                )
            else:
                self.tools.extend(self.db_func_tool.available_tools())
        except Exception as e:
            logger.warning(f"Failed to setup database tools, continuing without: {e}")

    def _setup_context_search_tools(self):
        """Setup context search tools."""
        try:
            self.context_search_tools = ContextSearchTools(
                self.agent_config,
                sub_agent_name=self.get_node_name(),
            )
            self.tools.extend(self.context_search_tools.available_tools())
        except Exception as e:
            logger.warning(f"Failed to setup context search tools, continuing without: {e}")

    def _setup_readonly_filesystem_tools(self):
        """Setup only read-only filesystem tools (no write/edit/create/move)."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            for method_name in READONLY_FILESYSTEM_METHODS:
                if hasattr(self.filesystem_func_tool, method_name):
                    method = getattr(self.filesystem_func_tool, method_name)
                    self.tools.append(trans_to_function_tool(method))
            logger.debug(f"Setup readonly filesystem tools with root path: {self.filesystem_func_tool.root_path}")
        except Exception as e:
            logger.warning(f"Failed to setup filesystem tools, continuing without: {e}")

    def _setup_date_parsing_tools(self):
        """Setup date parsing tools."""
        try:
            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            self.tools.extend(self.date_parsing_tools.available_tools())
        except Exception as e:
            logger.warning(f"Failed to setup date parsing tools, continuing without: {e}")

    def _get_system_prompt(self, prompt_version: Optional[str] = None) -> str:
        """Get the system prompt for the explore node."""
        from datus.prompts.prompt_manager import get_prompt_manager

        version = prompt_version or self.node_config.get("prompt_version")
        template_name = "explore_system"

        from datus.utils.node_utils import build_datasource_prompt_context

        context_search_tool_names = []
        if self.context_search_tools:
            context_search_tool_names = [tool.name for tool in self.context_search_tools.available_tools()]

        context = {
            "has_db_tools": bool(self.db_func_tool),
            "has_context_search_tools": bool(context_search_tool_names),
            "available_context_search_tools": context_search_tool_names,
            "has_filesystem_tools": bool(self.filesystem_func_tool),
            "has_date_parsing_tools": bool(self.date_parsing_tools),
            **build_datasource_prompt_context(self.agent_config),
            "workspace_root": self._resolve_workspace_root(),
            "scoped_tables": (
                self.input.scoped_tables
                if isinstance(self.input, ExploreNodeInput) and self.input.scoped_tables
                else []
            ),
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
        """Setup explore input from workflow context."""
        if not self.input or not isinstance(self.input, ExploreNodeInput):
            self.input = ExploreNodeInput(
                user_message=workflow.task.task,
                database=workflow.task.database_name,
            )
        return {"success": True, "message": "Explore input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> Dict:
        """Explore is read-only, no context updates needed."""
        return {"success": True, "message": "Explore node is read-only, no context updates"}

    def _build_success_result(self, ctx: StreamRunContext) -> ExploreNodeResult:
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

        all_actions = ctx.action_history_manager.get_actions()
        tokens_used = self._extract_total_tokens(all_actions)
        tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]

        return ExploreNodeResult(
            success=True,
            response=response_content,
            tokens_used=int(tokens_used),
            action_history=[a.model_dump() for a in all_actions],
            execution_stats={
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": sorted({a.action_type for a in tool_calls}),
                "total_tokens": int(tokens_used),
            },
        )
