# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of AgenticNode for general-purpose
chat interactions with markdown output, database/filesystem tool support,
skills, and permissions. This node is fully independent from GenSQLAgenticNode.
"""

from typing import ClassVar, Dict, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.gen_sql_agentic_node import prepare_template_context
from datus.agent.node.stream_run_context import StreamRunContext
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool, PlatformDocSearchTool
from datus.tools.func_tool.date_parsing_tools import DateParsingTools
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatAgenticNode(AgenticNode):
    """
    General-purpose chat agentic node with markdown output.

    This node provides flexible chat capabilities with:
    - Direct markdown response output (no JSON/SQL extraction)
    - Full tool support: database, filesystem, context search, date parsing
    - Skill discovery and execution with permission control
    - Permission hooks for tool access control
    - Plan mode (state and tooling provided by ``AgenticNode`` base class)
    - Session-based conversation management with MCP server integration
    """

    DEFAULT_SUBAGENTS = "*"
    result_class = ChatNodeResult

    # Subclasses whose contract is read-only (e.g. artifact ask agents) set this
    # to True so the unified write-capable ``execute_sql`` tool hard-rejects any
    # non-read statement at the tool layer. Defaults to False (full SQL access).
    _db_read_only: bool = False

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        scope: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the ChatAgenticNode.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (should be 'chat')
            input_data: Chat input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
            execution_mode: Execution mode - "interactive" (default) or "workflow"
        """
        self.execution_mode = execution_mode

        # Node name for config lookup and template resolution
        self.configured_node_name = "chat"

        # Max turns from config
        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and "chat" in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes["chat"]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        # Initialize tool attributes BEFORE calling parent constructor
        self.db_func_tool: Optional[DBFuncTool] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.degraded_capabilities: Dict[str, str] = {}
        self.date_parsing_tools: Optional[DateParsingTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self._platform_doc_tool: Optional[PlatformDocSearchTool] = None
        self.reference_template_tools: Optional[ReferenceTemplateTools] = None

        # Call parent constructor
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

        # Execution mode: "interactive" enables ask_user tool; "workflow"
        # disables it so the agent never pauses for user input.
        self.execution_mode = execution_mode

        # Setup tools
        self.setup_tools()
        logger.debug(f"ChatAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
        logger.debug(f"ChatAgenticNode initialized: {self.agent_config.current_datasource}")

    def get_node_name(self) -> str:
        """Get the configured node name."""
        return self.configured_node_name

    # ── Tool Setup ──────────────────────────────────────────────────────

    #: Whether ``agentic_nodes[<name>].tools`` narrows what this node builds.
    #:
    #: True for a plain chat node. The ask_* subclasses set it False: they run
    #: their own whitelist over ``self.tools`` *after* this base setup, and that
    #: design depends on every tool instance existing — the artifact-anchored
    #: filesystem instance in particular is infrastructure they use whether or
    #: not ``filesystem_tools`` is exposed to the LLM. Gating construction here
    #: would leave those attributes None and break the subclass, so the two
    #: mechanisms must not both act on the same field.
    HONOURS_CONFIGURED_TOOLS: ClassVar[bool] = True

    def _selected_tool_families(self) -> Optional[set]:
        """Tool families this node may mount, or ``None`` for "all of them".

        ``None`` is the default and means today's behaviour verbatim: a chat
        node with no ``tools`` key mounts every family, and the overwhelming
        majority of chat nodes have no such key. Narrowing that default would be
        a silent capability regression, so absence must never be read as "none".

        A configured value is a comma-separated list of the same family patterns
        the other agentic nodes accept (``db_tools``, ``db_tools.*``, and
        ``db_tools.execute_sql``-style entries all select the ``db_tools``
        family). Family granularity is what this node acts on; a per-tool suffix
        still enables its family rather than being dropped on the floor, since
        dropping it would quietly remove a tool the operator asked for.
        """
        if not self.HONOURS_CONFIGURED_TOOLS:
            return None
        raw = self.node_config.get("tools") if self.node_config else None
        if not raw:
            return None
        families = set()
        for pattern in str(raw).split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            families.add(pattern.split(".", 1)[0])
        return families or None

    @staticmethod
    def _family_enabled(selected: Optional[set], family: str) -> bool:
        return selected is None or family in selected

    def setup_tools(self):
        """Initialize tools, binding the DB tool to the active database (if any).

        Honours ``agentic_nodes[<name>].tools`` the way the five other agentic
        node classes do. Chat was the one class that ignored it, which mattered
        beyond tidiness: a host publishing a chat sub-agent strips write-capable
        tools from that list, and here the strip had no effect — the model was
        still shown tools it would then be denied at the permission layer.
        Nothing became writable (the permission profile and the database grants
        both still refused), but a layer of defence was missing and the denials
        surfaced as confusing mid-conversation errors.
        """
        selected = self._selected_tool_families()
        node_name = self.get_node_name()
        if self._family_enabled(selected, "db_tools"):
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=node_name,
                default_database=getattr(self, "active_database", "") or None,
                read_only=self._db_read_only,
                # Same anchor the filesystem tools use, so a path the model got back
                # from glob resolves identically inside load_file_as_table.
                filesystem_root=self._resolve_workspace_root(),
            )
        if self._family_enabled(selected, "context_search_tools"):
            self._setup_context_search_tools()
        if self._family_enabled(selected, "reference_template_tools"):
            self._setup_reference_template_tools()
        if self._family_enabled(selected, "date_parsing_tools"):
            self._setup_date_parsing_tools()
        if self._family_enabled(selected, "filesystem_tools"):
            self._setup_filesystem_tools()
        if self._family_enabled(selected, "memory_tools"):
            self._setup_memory_tools()
        # self.bash_tool was created in AgenticNode.__init__, so leaving it in
        # place is what mounts it — dropping the reference is the only way to
        # exclude it, and _rebuild_tools re-adds it from the attribute on every
        # rebuild otherwise.
        if self._family_enabled(selected, "bash_tools"):
            if self.bash_tool:
                self.tools.extend(self.bash_tool.available_tools())
        else:
            self.bash_tool = None
        if self._family_enabled(selected, "skills"):
            self._setup_skill_tools()
        if self._family_enabled(selected, "sub_agent_tools"):
            self._setup_sub_agent_task_tool()
        # Setup ask_user tool for clarification questions (interactive mode only)
        #
        # From here down is deliberately ungated: asking the user a question and
        # reporting a result to an orchestrator are how the node converses and
        # terminates, not data access an operator would want to scope away.
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        # No-op unless the request declared an orchestrator origin.
        self._setup_task_result_tool()
        self._rebuild_tools()
        if self._family_enabled(selected, "platform_doc_tools"):
            self._setup_platform_doc_tools()

        # Populate the shared tool_registry eagerly so consumers that inspect
        # categories before the first LLM turn (tests, ``apply_proxy_tools``'
        # FS-dependent exclusion, etc.) see a populated mapping. The actual
        # ``PermissionHooks`` is still built lazily by
        # ``_ensure_permission_hooks`` via ``_compose_hooks``.
        self._populate_tool_registry()

    def _setup_context_search_tools(self):
        """Setup context search tools without blocking DB/chat capabilities."""
        try:
            self.context_search_tools = ContextSearchTools(self.agent_config, sub_agent_name=self.get_node_name())
        except Exception as exc:
            self.context_search_tools = None
            warning = self._record_context_search_degraded(exc)
            logger.warning("Failed to setup context search tools, continuing without: %s", warning)

    def _setup_reference_template_tools(self):
        """Setup reference-template tools without blocking DB/chat capabilities."""
        try:
            self.reference_template_tools = ReferenceTemplateTools(
                self.agent_config,
                sub_agent_name=self.get_node_name(),
                db_func_tool=self.db_func_tool,
            )
        except Exception as exc:
            self.reference_template_tools = None
            message = (
                "Reference template tools are disabled because the embedding-backed "
                f"template store is unavailable. Details: {exc}"
            )
            self._record_degraded_capability("reference_template_tools", message)
            logger.warning("Failed to setup reference template tools, continuing without: %s", message)

    def _setup_date_parsing_tools(self):
        """Setup date parsing tools."""
        try:
            self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            self.tools.extend(self.date_parsing_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup date parsing tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug(f"Setup filesystem tools with root path: {self.filesystem_func_tool.root_path}")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_platform_doc_tools(self):
        """Setup platform documentation search tools."""
        try:
            self._platform_doc_tool = PlatformDocSearchTool(self.agent_config)
            self.tools.extend(self._platform_doc_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup platform_doc_search tools: {e}")

    def _setup_skill_tools(self):
        """Setup skill discovery and loading tools with permission control."""
        try:
            from datus.tools.permission.permission_config import PermissionConfig

            base_config = self.agent_config.permissions_config
            if base_config is not None:
                base_config = base_config.model_copy(deep=True)
            else:
                base_config = PermissionConfig()

            self.permission_manager = PermissionManager(
                global_config=base_config,
                node_overrides=self._get_node_permission_overrides(),
                active_profile=getattr(self.agent_config, "active_profile_name", None) or "normal",
                plugin_bash_rules=getattr(self.agent_config, "plugin_bash_rules", None),
                project_bash_allows=getattr(self.agent_config, "project_bash_allow", None),
                project_sql_allows=getattr(self.agent_config, "project_sql_allow", None),
            )
            self.permission_manager.set_permission_callback(self._handle_permission_ask)

            skills_config = getattr(self.agent_config, "skills_config", None) if self.agent_config else None
            self.skill_manager = SkillManager(
                config=skills_config,
                permission_manager=self.permission_manager,
                config_mutable=self._resolve_config_mutable(),
                agent_config=self.agent_config,
            )
            self.skill_func_tool = SkillFuncTool(
                manager=self.skill_manager,
                node_name="chat",
                node_class=self.get_node_class_name(),
                authoring_mode=self.SKILL_AUTHORING_MODE,
            )
            logger.debug(f"Setup skill tools: {self.skill_manager.get_skill_count()} skills discovered")
        except Exception as e:
            logger.error(f"Failed to setup skill tools: {e}")

    # ── Lazy re-injection gates ────────────────────────────────────────
    #
    # `AgenticNode._ensure_lazy_tools_mounted` re-adds bash, skills, memory and
    # web on every prompt build and on a snapshot-cache hit — AFTER setup_tools
    # already decided what this node may have. Without these gates the exclusion
    # holds only until the first prompt is built, and memory is the worst case:
    # the base *re-creates* the instance when it finds None, so clearing it in
    # setup_tools is not enough on its own. The ask_* subclass hit the same
    # bypass and gates it the same way.

    def _ensure_bash_tool_in_tools(self) -> None:
        if not self._family_enabled(self._selected_tool_families(), "bash_tools"):
            return
        super()._ensure_bash_tool_in_tools()

    def _ensure_skill_tools_in_tools(self) -> None:
        if not self._family_enabled(self._selected_tool_families(), "skills"):
            return
        super()._ensure_skill_tools_in_tools()

    def _ensure_memory_tool_in_tools(self) -> None:
        if not self._family_enabled(self._selected_tool_families(), "memory_tools"):
            return
        super()._ensure_memory_tool_in_tools()

    def _ensure_web_tools_in_tools(self) -> None:
        selected = self._selected_tool_families()
        if not self._family_enabled(selected, "web_tool"):
            # Scrub rather than just skip: the base resolves provider-native
            # builtins too, and a stale local web tool left in the list would
            # still be offered to the model.
            from datus.tools.func_tool.web_tool import WebTool

            web_names = set(WebTool.all_tools_name())
            self.tools = [t for t in (self.tools or []) if getattr(t, "name", None) not in web_names]
            self._builtin_web_tools = {"web_search": False, "web_fetch": False}
            self._web_tool = None
            return
        super()._ensure_web_tools_in_tools()

    def _rebuild_tools(self):
        """Rebuild the tools list with current tool instances including skills.

        Every configurable family is re-checked against ``tools`` here, not just
        tested for "did setup_tools leave an instance behind". Relying on the
        attribute being None makes the exclusion depend on nothing else ever
        populating it — an assumption that has already failed twice in this
        class, once because the base re-creates the memory tool when it finds
        None, and once because platform docs were mounted after the rebuild that
        was supposed to own them. Asking the config makes the exclusion a
        property of the node instead of an accident of ordering.
        """
        selected = self._selected_tool_families()

        def enabled(family: str) -> bool:
            return self._family_enabled(selected, family)

        self.tools = []
        if self.db_func_tool and enabled("db_tools"):
            self.tools.extend(self.db_func_tool.available_tools())
        if self.context_search_tools and enabled("context_search_tools"):
            self.tools.extend(self.context_search_tools.available_tools())
        if self.reference_template_tools and enabled("reference_template_tools"):
            self.tools.extend(self.reference_template_tools.available_tools())
        if self.date_parsing_tools and enabled("date_parsing_tools"):
            self.tools.extend(self.date_parsing_tools.available_tools())
        if self.filesystem_func_tool and enabled("filesystem_tools"):
            self.tools.extend(self.filesystem_func_tool.available_tools())
        if self.memory_func_tool and enabled("memory_tools"):
            self.tools.extend(self.memory_func_tool.available_tools())
        if self.bash_tool and enabled("bash_tools"):
            self.tools.extend(self.bash_tool.available_tools())
        if self.skill_func_tool and enabled("skills"):
            self.tools.extend(self.skill_func_tool.available_tools())
        if self.sub_agent_task_tool and enabled("sub_agent_tools"):
            self.tools.extend(self.sub_agent_task_tool.available_tools())
        if self.ask_user_tool:
            self.tools.extend(self.ask_user_tool.available_tools())
        # None unless the request declared an orchestrator origin. Must be re-added
        # here and not only in setup_tools: every rebuild (task-database switch,
        # skill reload) clears the list, and dropping this one silently strips the
        # dispatcher's only structured way to learn how the run ended.
        if getattr(self, "task_result_tool", None):
            self.tools.extend(self.task_result_tool.available_tools())
        # Rebuilt here rather than only in setup_tools, where it is mounted
        # *after* the initial rebuild and so was never re-added by a later one.
        # `_update_database_connection` rebuilds on a task-database switch, which
        # silently dropped platform docs from the LLM's surface mid-session.
        # Gated like the rest: a node that excluded the family must not get it
        # back on the next rebuild.
        if self._platform_doc_tool and enabled("platform_doc_tools"):
            self.tools.extend(self._platform_doc_tool.available_tools())
        # Plan-mode tools (confirm_plan + todo_*) for main agents; no-op for sub-agents.
        self._register_plan_mode_tools()
        # The rebuilt list holds fresh, unwrapped FunctionTool instances —
        # force plugin tool transformers to re-apply on the next hook
        # composition or the policy wrappers silently vanish mid-session
        # (e.g. after a task-database switch).
        self._tool_transformers_applied = False

    def _update_database_connection(self, database_name: str):
        """Rebuild the DB tool bound to ``database_name`` and remember it as the active database.

        ``default_database`` makes DBFuncTool connect to that database (selects the file for a
        glob datasource; sets the database in the URI for PG/server DBs). Remembering it on the
        node keeps subsequent ``setup_tools()`` rebuilds bound to the same database.
        """
        self.active_database = database_name or ""
        self.db_func_tool = DBFuncTool(
            agent_config=self.agent_config,
            sub_agent_name=self.get_node_name(),
            default_database=database_name,
            read_only=self._db_read_only,
            filesystem_root=self._resolve_workspace_root(),
        )
        self._rebuild_tools()

    # ── Permission Helpers ──────────────────────────────────────────────

    async def _handle_permission_ask(
        self,
        tool_category: str,
        tool_name: str,
        context: dict,
    ) -> bool:
        """Handle ASK permission by prompting user for confirmation."""
        try:
            from rich.console import Console
            from rich.prompt import Confirm

            console = Console()
            console.print(f"\n[yellow]Permission required:[/yellow] {tool_category}.{tool_name}")
            if context:
                console.print(f"[dim]Context: {context}[/dim]")

            approved = Confirm.ask(f"Allow {tool_name}?", default=False)

            if approved:
                always = Confirm.ask("Always allow this session?", default=False)
                if always and self.permission_manager:
                    self.permission_manager.approve_for_session(tool_category, tool_name)

            return approved
        except Exception as e:
            logger.error(f"Permission prompt failed: {e}")
            return False

    def _get_node_permission_overrides(self) -> dict:
        """Get node-specific permission overrides from agent config."""
        if not self.agent_config:
            return {}

        chat_config = self.agent_config.agentic_nodes.get("chat", {})
        if isinstance(chat_config, dict) and "permissions" in chat_config:
            return {"chat": chat_config["permissions"]}

        return {}

    # ── System Prompt ───────────────────────────────────────────────────

    def _exposed_tool_names(self) -> set:
        """Names of the tools currently exposed to the LLM (``self.tools``)."""
        return {tool.name for tool in (self.tools or [])}

    @staticmethod
    def _tool_group_exposed(tool_instance, exposed_names: set) -> bool:
        """True if ``tool_instance`` exists and at least one of its tools is exposed.

        Used to derive the prompt's ``has_*`` flags from the live tool surface
        so a pruned-but-still-instantiated group (see ``_get_system_prompt``)
        is not advertised to the model.
        """
        if not tool_instance:
            return False
        try:
            return any(tool.name in exposed_names for tool in tool_instance.available_tools())
        except Exception:
            return False

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
    ) -> str:
        """Get the system prompt using enhanced template context."""
        # Derive the ``has_*`` flags from the tools actually exposed to the LLM
        # (``self.tools``), not from the existence of the tool *instance*. A
        # subclass may build a tool instance for internal use yet prune its
        # tools from the LLM-facing list (e.g. ``BaseArtifactAskAgenticNode``
        # keeps ``db_func_tool`` for reference-template execution but drops
        # db_tools from ``self.tools`` when the subagent whitelist excludes
        # them). Keying the prompt flags off the instance would then advertise
        # tools the model cannot actually call, producing "Tool X not found".
        exposed = self._exposed_tool_names()
        context = prepare_template_context(
            node_config=self.node_config,
            has_db_tools=self._tool_group_exposed(self.db_func_tool, exposed),
            has_filesystem_tools=self._tool_group_exposed(self.filesystem_func_tool, exposed),
            has_mf_tools=False,
            has_context_search_tools=self._tool_group_exposed(self.context_search_tools, exposed),
            has_reference_template_tools=(
                self._tool_group_exposed(self.reference_template_tools, exposed)
                and bool(self.reference_template_tools and self.reference_template_tools.has_reference_templates)
            ),
            has_parsing_tools=self._tool_group_exposed(self.date_parsing_tools, exposed),
            has_platform_doc_tools=self._tool_group_exposed(self._platform_doc_tool, exposed),
            has_semantic_tools=self._tool_group_exposed(getattr(self, "semantic_tools", None), exposed),
            agent_config=self.agent_config,
            workspace_root=self._resolve_workspace_root(),
        )
        context["has_task_tool"] = bool(self.sub_agent_task_tool)
        context["has_ask_user_tool"] = "ask_user" in exposed
        # Web tools (web_search / web_fetch) are NOT advertised in the system
        # prompt: their own tool-schema descriptions document usage, and prompt
        # advertisement can't track per-tool availability (builtin-only vs
        # local-only, Tavily key absent) without drifting from the real tool set.
        # No per-turn values here: the current date lives in the shared
        # runtime-context block, the current datasource/dialect in the user
        # turn's <system_reminder>, and the permission profile is enforced by
        # the permission hooks at tool-call time rather than prompted.
        prompt_version = prompt_version or self.node_config.get("prompt_version")

        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=prompt_version, **context)
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError:
            logger.warning(f"Failed to render system prompt '{system_prompt_name}', using the default template instead")
            base_prompt = pm.render_template(template_name="chat_system", version=None, **context)
            return self._finalize_system_prompt(base_prompt)
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    # ── Workflow Integration ────────────────────────────────────────────

    def setup_input(self, workflow: Workflow) -> dict:
        """Setup chat input from workflow context."""
        task_database = workflow.task.database_name
        if task_database and self.db_func_tool and task_database != self.db_func_tool.connector.database_name:
            logger.debug(
                f"Updating database connection from '{self.db_func_tool.connector.database_name}' "
                f"to '{task_database}' based on workflow task"
            )
            self._update_database_connection(task_database)

        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        if not self.input:
            self.input = ChatNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
                reference_sql=None,
                plan_mode=plan_mode,
                auto_execute_plan=auto_execute_plan,
                prompt_version=self.node_config.get("prompt_version"),
            )
        else:
            self.input.user_message = workflow.task.task
            self.input.external_knowledge = workflow.task.external_knowledge
            self.input.catalog = workflow.task.catalog_name
            self.input.database = workflow.task.database_name
            self.input.db_schema = workflow.task.schema_name
            self.input.schemas = workflow.context.table_schemas
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": "Chat input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> dict:
        """Update workflow context with chat results. Chat node produces markdown, no SQL."""
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        return {"success": True, "message": "Updated chat context"}

    # ── execute_stream hooks ────────────────────────────────────────────

    def _build_success_result(self, ctx: StreamRunContext) -> ChatNodeResult:
        response_content = ctx.response_content

        # Fallback chain when the assistant's content channel was empty:
        # last_successful_output → last_tool_summary → scan for summary_report.
        if not response_content and ctx.last_successful_output:
            candidate = (
                ctx.last_successful_output.get("content", "")
                or ctx.last_successful_output.get("text", "")
                or ctx.last_successful_output.get("response", "")
                or ctx.last_successful_output.get("raw_output", "")
            )
            if isinstance(candidate, str) and candidate:
                response_content = candidate
            elif candidate and not isinstance(candidate, str):
                response_content = str(candidate)

        if not response_content and ctx.last_tool_summary:
            response_content = ctx.last_tool_summary

        if not response_content:
            for stream_action in reversed(ctx.action_history_manager.get_actions()):
                if stream_action.action_type == "summary_report" and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        candidate = (
                            stream_action.output.get("markdown", "")
                            or stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                        )
                        if candidate:
                            response_content = str(candidate) if not isinstance(candidate, str) else candidate
                            break

        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        all_actions = ctx.action_history_manager.get_actions()
        tokens_used = self._extract_total_tokens(all_actions)
        tool_calls = [
            action for action in all_actions if action.role == ActionRole.TOOL and action.status == ActionStatus.SUCCESS
        ]
        return ChatNodeResult(
            success=True,
            response=response_content,
            tokens_used=int(tokens_used),
            action_history=[action.model_dump() for action in all_actions],
            execution_stats={
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": list({a.action_type for a in tool_calls}),
                "total_tokens": int(tokens_used),
            },
        )
