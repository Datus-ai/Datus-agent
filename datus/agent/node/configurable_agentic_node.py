# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Common execution for user definitions and dependency-free private runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from agents import FunctionTool
from agents.tool_context import ToolContext

from datus.agent.agent_registry import resolve_configurable_agent
from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.configurable_scope import ConfigurableScope, ScopedAgentFilesystem, ScopedSemanticFilesystem
from datus.agent.node.configurable_tools import ConfigurableToolSet, build_mcp_servers
from datus.agent.node.stream_run_context import StreamRunContext
from datus.agent.semantic_authoring_draft import SemanticAuthoringDraft
from datus.agent.semantic_authoring_session import (
    AUTHORING_MUTATION_TOOL_IDS,
    AUTHORING_TOOL_NAMES,
    SemanticAuthoringSession,
    requests_semantic_authoring,
)
from datus.configuration.agent_config import AgentConfig
from datus.plugins.agent_tools import bind_plugin_tools
from datus.plugins.agents import RuntimePreparation
from datus.plugins.registry import collect_plugin_agents, resolve_agent_runtime_factory
from datus.plugins.semantic_authoring import SemanticAuthoringHost
from datus.schemas.action_history import ActionHistory
from datus.schemas.configurable_agentic_node_models import ConfigurableNodeInput, ConfigurableNodeResult
from datus.tools.db_tools.async_workers import DATABASE_CLEANUP
from datus.tools.func_tool.base import parse_tool_args, write_back_tool_args
from datus.tools.permission.permission_hooks import CompositeHooks
from datus.tools.registry.contracts import tool_definition
from datus.tools.registry.tool_registry import ToolRegistry
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
_RUNTIME_CLOSE_TIMEOUT = 5.0


async def _await_result(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class _ScopeHooks(CompositeHooks):
    """Run before permission hooks, which may themselves read SQL file arguments."""

    def __init__(self, node: ConfigurableAgenticNode, hooks: Any):
        super().__init__([hooks])
        self.node = node

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        identities = {item.name: tool_definition(item).tool_id for item in self.node._native_tools}
        identity = identities.get(tool.name)
        if identity is not None or tool.name == "load_skill":
            args, canonical, error = parse_tool_args(context.tool_arguments, tool_name=tool.name)
            write_back_tool_args(context, canonical)
            if error:
                context._datus_configurable_argument_error = error
                return
            if identity is not None:
                self.node.execution_scope.require_tool_scope(identity, args)
        await super().on_tool_start(context, agent, tool)


class AgentRuntimeHost:
    """A small capability facade; plugins receive neither node nor AgentConfig.

    Publisher code is trusted Python, not a sandbox. The facade is the supported
    contract for doing project I/O with the same permissions as model tool calls.
    """

    def __init__(
        self,
        invoke: Callable[[str, dict[str, Any]], Awaitable[Any]],
        scoped_context: Callable[[], dict[str, Any]],
        semantic_authoring: SemanticAuthoringHost,
    ):
        self._invoke = invoke
        self._scoped_context = scoped_context
        self.semantic_authoring = semantic_authoring

    @property
    def scoped_context(self) -> dict[str, Any]:
        return self._scoped_context()

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> Any:
        """Invoke an explicitly selected native tool through the host's permission gate."""
        return await self._invoke(tool_id, arguments)


class ConfigurableAgenticNode(AgenticNode):
    """One standard AgenticNode loop, with optional per-instance business state."""

    result_class = ConfigurableNodeResult

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str = "configurable",
        input_data: ConfigurableNodeInput | None = None,
        agent_config: AgentConfig | None = None,
        tools: list[FunctionTool] | None = None,
        node_name: str | None = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: str | None = None,
        is_subagent: bool = False,
        session_id: str | None = None,
    ) -> None:
        self.configured_node_name = node_name or node_id
        self.definition = resolve_configurable_agent(self.configured_node_name, agent_config)
        if self.definition is None:
            raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message="Configurable agent definition was not found.")
        if tools:
            raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message="Select configurable tools through AgentSpec.")
        self.spec = self.definition.spec
        self._uses_semantic_authoring = requests_semantic_authoring(self.spec.tools)
        self._semantic_authoring_session: SemanticAuthoringSession | None = None
        self._semantic_draft: SemanticAuthoringDraft | None = None
        self._semantic_draft_completed = False
        self.execution_scope = ConfigurableScope(self.spec, agent_config)
        self.execution_mode = execution_mode
        self.max_turns = 100
        self._runtime = None
        self._runtime_initialized = False
        self._runtime_closed = False
        self._prepared = RuntimePreparation()
        self._native_tools: list[FunctionTool] = []
        self._native_scope_identity: tuple[str, str] | None = None
        self._active_skills: list[str] = []
        self._turn_context: dict[str, Any] = {}
        self.db_func_tool = None
        self.context_search_tools = None
        self.semantic_tools = None
        self.filesystem_func_tool = None
        super().__init__(
            node_id,
            description,
            node_type,
            input_data,
            self.execution_scope.config,
            tools=[],
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )
        self._tool_set = ConfigurableToolSet(self)
        self._host = AgentRuntimeHost(
            self._call_host_tool,
            lambda: self.execution_scope.effective.model_dump(mode="json", exclude_none=True),
            SemanticAuthoringHost(self._call_semantic_authoring),
        )
        self.metadata["agent_name"] = self.get_node_name()

    def get_node_name(self) -> str:
        return self.configured_node_name

    def get_node_class_name(self) -> str:
        return "configurable"

    @property
    def defer_model_output(self) -> bool:
        plugin = self.definition.plugin
        return (
            self.spec.output.format == "json"
            or self._uses_semantic_authoring
            or bool(plugin and plugin.runtime_factory)
        )

    def _parse_node_config(self, agent_config: AgentConfig | None, node_name: str) -> dict[str, Any]:
        config = super()._parse_node_config(agent_config, node_name)
        config.update(system_prompt="configurable", tools="", mcp="", skills=",".join(self.spec.skills))
        config["workspace_root"] = str(self.execution_scope.root)
        return config

    def _setup_mcp_servers(self) -> dict[str, Any]:
        return {}

    def _setup_bash_tool(self) -> None:
        # There is no implicit business tool set on this node.
        return None

    def _setup_skill_manager(self) -> None:
        if not self.spec.skills and not self._uses_semantic_authoring:
            return
        from datus.tools.skill_tools.skill_config import SkillConfig
        from datus.tools.skill_tools.skill_manager import SkillManager

        config = self.agent_config.skills_config or SkillConfig.from_dict({}, agent_config=self.agent_config)
        config = config.model_copy(deep=True)
        config.directories = [
            str(Path(directory).expanduser())
            if Path(directory).expanduser().is_absolute()
            else str(self.execution_scope.root / directory)
            for directory in config.directories
        ]
        self.skill_manager = SkillManager(
            config=config,
            permission_manager=self.permission_manager,
            agent_config=self.agent_config,
            config_mutable=self._resolve_config_mutable(),
        )

    def _ensure_lazy_tools_mounted(self) -> None:
        # Every model-visible capability is assembled once in prepare_turn.
        return None

    def _system_prompt_template_name(self) -> str:
        return "configurable_system"

    def _get_required_skills(self) -> list[str]:
        return ["dosi-semantic-authoring"] if self._uses_semantic_authoring else []

    def _required_skill_agent(self, skill_name: str) -> str:
        if self._uses_semantic_authoring and skill_name == "dosi-semantic-authoring":
            return "semantic_modeling"
        return super()._required_skill_agent(skill_name)

    def _render_required_skill_content(self, skill_name: str, content: str) -> str:
        from datus.agent.node.semantic_authoring import prepare_authoring_references, render_required_authoring_skill

        if not self._uses_semantic_authoring:
            return super()._render_required_skill_content(skill_name, content)
        data_dir = self._resolve_session_data_dir()
        if not data_dir:
            raise DatusException(ErrorCode.STORAGE_FAILED, message="Session data directory is unavailable")
        reference_dir = prepare_authoring_references(Path(data_dir) / "semantic_authoring")
        skill = self.skill_manager.get_skill(skill_name)
        self.execution_scope.host_read_roots += (reference_dir, skill.location.resolve())
        return render_required_authoring_skill(
            skill_name,
            super()._render_required_skill_content(skill_name, content),
            include_osi_core=True,
            reference_dir=reference_dir,
        )

    def _get_database_skill_candidates(self) -> list[str]:
        if self._uses_semantic_authoring:
            return super()._get_database_skill_candidates()
        return []

    def _make_filesystem_tool(self, **kwargs: Any) -> ScopedAgentFilesystem:
        allowlist = self._resolve_filesystem_allowlist()
        return ScopedAgentFilesystem(
            execution_scope=self.execution_scope,
            root_path=self._resolve_workspace_root(),
            current_node=self.get_node_name(),
            strict=True,
            path_allowlist=allowlist,
            path_allowlist_provider=lambda: self._with_loaded_skill_roots(allowlist),
            **kwargs,
        )

    def _finalize_system_prompt(self, base_prompt: str, memory_node_name_override: str | None = None) -> str:
        # Business project files, memory and unrelated plugin context are not
        # read implicitly. Skill metadata is controlled by the selected names.
        prompt = self._inject_runtime_context(base_prompt)
        if self._uses_semantic_authoring:
            prompt = self._inject_required_skills(prompt)
        if self._active_skills:
            prompt += "\n\n" + self._get_available_skills_context()
        return self._inject_response_language(prompt)

    def _system_prompt_snapshot_meta(self, prompt_version: str | None) -> dict[str, str]:
        meta = super()._system_prompt_snapshot_meta(prompt_version)
        value = {
            "spec": self.spec.model_dump(mode="json"),
            "preparation": self._prepared.model_dump(exclude={"tool_bindings"}, mode="json"),
            "instructions_file": self._publisher_instructions(),
            "skills": [
                self.skill_manager.format_skill_content(
                    name,
                    self.skill_manager.registry.load_skill_content(name) or "",
                )
                for name in self._active_skills
            ],
        }
        if self._uses_semantic_authoring:
            from datus.agent.node.semantic_authoring import authoring_prompt_snapshot_meta

            meta.update(authoring_prompt_snapshot_meta(self.agent_config, "semantic_modeling"))
        meta["agent_definition_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return meta

    def _publisher_instructions(self) -> str:
        plugin = self.definition.plugin
        if plugin is None or plugin.instructions_file is None:
            return ""
        root = plugin.package_dir.resolve()
        path = (root / plugin.instructions_file).resolve()
        if not path.is_relative_to(root):
            raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message="Plugin instruction file escaped its package.")
        return path.read_text(encoding="utf-8")

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        if self._runtime_closed:
            raise RuntimeError("This agent instance has been closed")
        self.result = None
        plugin = self.definition.plugin
        if (
            self._runtime_initialized
            and plugin is not None
            and collect_plugin_agents(self.agent_config, plugin.spec.name).get(plugin.spec.name) != plugin
        ):
            raise ValueError(f"Plugin agent {plugin.spec.name!r} is no longer enabled with this definition")
        self._sync_plan_mode_state(ctx.user_input)
        self.execution_scope.host_read_roots = ()
        if self.skill_manager is not None:
            self.skill_manager.loaded_skill_roots.clear()
        if self._semantic_draft is not None:
            self._rollback_authoring()
            self._semantic_draft.close()
            self._semantic_draft = None
            await self._retire_database_tools()
            self._semantic_authoring_session = None
            self._native_scope_identity = None
        self._semantic_draft_completed = False
        if self._semantic_authoring_session is not None:
            self._semantic_authoring_session.begin_turn()
        self.definition.validate_runtime_config()
        self.execution_scope.prepare(None, ctx.user_input)
        await self._refresh_native_tools()
        self.tools = list(self._native_tools)
        self._tool_transformers_applied = False
        self.permission_hooks = None
        self._filter_permission_denied_tools()
        self._apply_semantic_plan_review()
        if not self._runtime_initialized:
            if plugin and plugin.runtime_factory:
                factory = resolve_agent_runtime_factory(plugin, self.agent_config)
                self._runtime = await _await_result(factory(deepcopy(self.definition.runtime_config or {}), self._host))
                if self._runtime is None:
                    raise TypeError("create_runtime must return a runtime object")
            self._runtime_initialized = True
        self._turn_context = {
            "agent_name": self.get_node_name(),
            "session_id": self.session_id,
            "input": ctx.user_input.model_dump(mode="json"),
            "scoped_context": self._host.scoped_context,
        }
        prepare = getattr(self._runtime, "prepare_turn", None)
        prepared = await _await_result(prepare(deepcopy(self._turn_context))) if prepare else None
        self._prepared = RuntimePreparation.model_validate(prepared or {})
        # Reject non-serializable context before opening a model request.
        json.dumps(self._prepared.context)
        previous_scope = self.execution_scope.effective
        self.execution_scope.prepare(self._prepared.scoped_context, ctx.user_input)
        if self._semantic_draft is not None and self.execution_scope.effective != previous_scope:
            session = self._semantic_authoring_session
            if session is not None and session.osi_target_state.selected is not None:
                raise ValueError("A publisher must narrow the scope before selecting a semantic target")
            self._semantic_draft.close()
            self._semantic_draft = SemanticAuthoringDraft(self.execution_scope)
        await self._refresh_native_tools()
        self._turn_context["scoped_context"] = self._host.scoped_context
        self._turn_context["context"] = dict(self._prepared.context)
        skills = self.spec.skills if self._prepared.skills is None else self._prepared.skills
        if not set(skills).issubset(self.spec.skills):
            raise ValueError("Prepared skills exceed the static declaration")
        available = (
            {
                item.name
                for item in self.skill_manager.get_available_skills(self.get_node_name(), node_class="configurable")
            }
            if self.skill_manager
            else set()
        )
        self._active_skills = [name for name in skills if name in available]
        if any(self.skill_manager is None or self.skill_manager.get_skill(name) is None for name in skills):
            raise ValueError("Selected skills must be installed in this project")
        self.node_config["skills"] = ",".join(self._active_skills)
        self.tools = list(self._native_tools)
        if self._active_skills and self.skill_func_tool:
            tool = self.skill_func_tool.available_tools()[0]
            invoke = tool.on_invoke_tool

            async def load_selected(context, arguments):
                args, canonical, error = parse_tool_args(arguments, tool_name="load_skill")
                write_back_tool_args(context, canonical)
                error = getattr(context, "_datus_configurable_argument_error", None) or error
                if error:
                    return {"success": 0, "error": error, "result": None}
                name = args.get("skill_name")
                if name not in self._active_skills:
                    return {"success": 0, "error": "Skill is outside this agent's selection", "result": None}
                result = await invoke(context, canonical)
                root = self.skill_manager.get_skill(name).location.resolve()
                if root in self.skill_manager.loaded_skill_roots and root not in self.execution_scope.host_read_roots:
                    self.execution_scope.host_read_roots += (root,)
                return result

            tool.on_invoke_tool = load_selected
            tool.__datus_tool_local_only__ = True
            self.tools.append(tool)
        plugin = self.definition.plugin
        self.tools.extend(bind_plugin_tools(plugin.tool_definitions if plugin else [], self._prepared.tool_bindings))
        if self.is_in_plan_mode():
            self._register_plan_mode_tools()
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Native and plugin tools must have distinct model-visible names")
        selections = self.spec.mcp if self._prepared.mcp is None else self._prepared.mcp
        declared = {item.connection_ref: set(item.tools) for item in self.spec.mcp}
        if any(not set(item.tools).issubset(declared.get(item.connection_ref, set())) for item in selections):
            raise ValueError("Prepared MCP tools exceed the static declaration")
        node_name = self.get_node_name()
        inherited = getattr(self.agent_config, "_inherit_project_mcp_for", ())
        if self._prepared.mcp is None and not self.spec.mcp and node_name in inherited:
            # Project connections keep their existing filters and call authorization.
            self.mcp_servers = {
                name: server
                for name in self.agent_config.services.mcp_servers
                if (server := self._setup_mcp_server_from_config(name)) is not None
            }
        else:
            self.mcp_servers = build_mcp_servers(
                self.agent_config, selections, permission_manager=self.permission_manager, node_name=node_name
            )
        self.permission_hooks = None
        self._tool_transformers_applied = False
        self._filter_permission_denied_tools()
        if self.proxy_tool_patterns:
            from datus.tools.proxy.proxy_tool import apply_proxy_tools

            apply_proxy_tools(self, self.proxy_tool_patterns, channel=getattr(self, "_proxy_tool_channel", None))
        self._apply_semantic_plan_review()

    def _apply_semantic_plan_review(self) -> None:
        from datus.tools.proxy.semantic_plan_review import SemanticPlanReviewContext, apply_semantic_plan_review

        context = getattr(self, "semantic_plan_review_context", None)
        if isinstance(context, SemanticPlanReviewContext):
            apply_semantic_plan_review(self, context=context)

    async def _refresh_native_tools(self) -> None:
        """Rebind data/RAG tools when the effective scope changes between turns."""
        self.node_config["scoped_context"] = self.execution_scope.effective.model_dump(exclude_none=True)
        identity = (
            self.execution_scope.effective.model_dump_json(),
            json.dumps(self.agent_config.runtime_db_context(), sort_keys=True),
        )
        if identity != self._native_scope_identity:
            self._native_scope_identity = None
            self._native_tools = []
            self.tools = []
            if self._semantic_authoring_session is not None:
                self._rollback_authoring()
            await self._retire_database_tools()
            self._semantic_authoring_session = None
            self.context_search_tools = None
            self.semantic_tools = None
            self.filesystem_func_tool = None
            self._tool_set = ConfigurableToolSet(self)
            self._native_tools = self._tool_set.select(self.spec.tools)
            self._native_scope_identity = identity

    async def _retire_database_tools(self) -> None:
        semantic_db = getattr(self._semantic_authoring_session, "db_func_tool", None)
        tools = {tool for tool in (self.db_func_tool, semantic_db) if tool is not None}
        self.db_func_tool = None
        jobs = [DATABASE_CLEANUP.submit(tool.close) for tool in tools]
        await asyncio.gather(*(self._wait_for_database_release(job) for job in jobs))

    def _populate_tool_registry(self) -> None:
        # Every mounted tool has its own verified contract. Discovery of
        # unrelated tool groups must not assign a fallback permission family.
        registry = ToolRegistry()
        for tool in self.tools or []:
            if isinstance(tool, FunctionTool):
                registry.register_tools(tool_definition(tool).category, [tool])
        self.tool_registry = registry

    def _ensure_permission_hooks(self) -> None:
        super()._ensure_permission_hooks()
        if self.permission_hooks is not None:
            self.permission_hooks.sql_file_allowed = self.execution_scope.allows_file

    async def _authorize_host_tool(self, identity: str, arguments: dict[str, Any]) -> tuple[FunctionTool, ToolContext]:
        self._ensure_tool_transformers()
        tool = next((item for item in self.tools if tool_definition(item).tool_id == identity), None)
        if tool is None or identity not in {tool_definition(item).tool_id for item in self._native_tools}:
            raise PermissionError(f"Host tool {identity!r} is not in this agent's permitted native selection")
        self.execution_scope.require_tool_scope(identity, arguments)
        self._ensure_permission_hooks()
        context = ToolContext(
            context=None,
            tool_name=tool.name,
            tool_call_id=f"runtime_{uuid4().hex}",
            tool_arguments=json.dumps(arguments),
        )
        if self.permission_hooks:
            await self.permission_hooks.on_tool_start(context, None, tool)
        return tool, context

    async def _call_host_tool(self, identity: str, arguments: dict[str, Any]) -> Any:
        tool, context = await self._authorize_host_tool(identity, arguments)
        value = await tool.on_invoke_tool(context, context.tool_arguments)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                pass
        return value

    def _get_semantic_authoring_session(self) -> SemanticAuthoringSession:
        if not self._uses_semantic_authoring:
            raise PermissionError("Select semantic authoring tools before using this host capability")
        if self._semantic_authoring_session is None:
            session = SemanticAuthoringSession(
                self._semantic_draft.config
                if self._semantic_draft is not None and not self._semantic_draft.published
                else self.agent_config,
                node_name=self.get_node_name(),
                path_allowed=self._allows_authoring_path,
            )
            self._semantic_authoring_session = session
            session.setup(
                filesystem_factory=self._make_semantic_filesystem,
                runtime_db_context=self._semantic_runtime_db_context,
                filesystem_root=str(self.execution_scope.root),
                database_config=self.agent_config,
            )
            # Plans are reviewed project artifacts, separate from business drafts.
            from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools

            session.semantic_plan_file_tool = SemanticPlanFileTools(
                self.agent_config.project_root, path_allowed=self.execution_scope.allows_file
            )
        return self._semantic_authoring_session

    def _make_semantic_filesystem(self, **kwargs: Any) -> Any:
        session = self._semantic_authoring_session
        return ScopedSemanticFilesystem(
            reference_scope=(
                self.execution_scope
                if self._semantic_draft is not None and not self._semantic_draft.published
                else None
            ),
            execution_scope=(
                self._semantic_draft.filesystem_scope
                if self._semantic_draft is not None and not self._semantic_draft.published
                else self.execution_scope
            ),
            root_path=str(self._authoring_root()),
            current_node=self.get_node_name(),
            strict=True,
            session_data_dir=self._resolve_session_data_dir(),
            osi_target_state=session.osi_target_state,
            generation_evidence=session.generation_evidence,
            path_allowlist=self._resolve_filesystem_allowlist(),
            path_allowlist_provider=lambda: self._with_loaded_skill_roots(self._resolve_filesystem_allowlist()),
            **kwargs,
        )

    def _authoring_root(self) -> Path:
        draft = self._semantic_draft
        return draft.root if draft is not None and not draft.published else self.execution_scope.root

    def _allows_authoring_path(self, path: Path) -> bool:
        draft = self._semantic_draft
        return self.execution_scope.allows_file(draft.live_path(path) if draft is not None else path)

    def _rollback_authoring(self) -> None:
        session = self._semantic_authoring_session
        if (
            session is None
            or session._reconciled_sha256
            or (self._semantic_draft is not None and self._semantic_draft.kb_sync_pending)
        ):
            return
        if self._semantic_draft is not None and self._semantic_draft.published:
            self._semantic_draft.rollback()
        else:
            session.rollback()

    async def _begin_semantic_draft(self) -> dict[str, Any]:
        if self.definition.plugin is None:
            raise PermissionError("Isolated business drafts require an installed publisher runtime")
        if self._semantic_draft is not None:
            raise RuntimeError("A semantic draft is already active for this turn")
        current = self._semantic_authoring_session
        if current is not None and current.osi_target_state.selected is not None:
            raise RuntimeError("Begin the draft before selecting or modifying a semantic target")
        await self._retire_database_tools()
        self._semantic_authoring_session = None
        self._semantic_draft = SemanticAuthoringDraft(self.execution_scope)
        self._native_scope_identity = None
        await self._refresh_native_tools()
        self.tools = list(self._native_tools)
        self.permission_hooks = None
        self._tool_transformers_applied = False
        self._filter_permission_denied_tools()
        self._apply_semantic_plan_review()
        from datus.prompts.prompt_manager import get_prompt_manager
        from datus.utils.node_utils import build_datasource_prompt_context

        self._ensure_interaction_channel()
        context = build_datasource_prompt_context(self.agent_config)
        context.update(
            native_tools=", ".join(tool.name for tool in self.tools),
            mcp_tools=", ".join(f"{item.connection_ref}.{name}" for item in self.spec.mcp for name in item.tools)
            or "None",
            authoring_scope="full",
            semantic_model_dir=f"subject/semantic_models/{self.agent_config.current_datasource}",
            knowledge_base_dir="subject",
            kind_subdir=f"subject/semantic_models/{self.agent_config.current_datasource}",
            current_datasource=self.agent_config.current_datasource,
            has_ask_user_tool=any(tool.name == "ask_user" for tool in self.tools),
        )
        instructions = get_prompt_manager(agent_config=self.agent_config).render_template(
            "semantic_modeling_system",
            agent_config=self.agent_config,
            **context,
        )
        return {**self._semantic_draft.snapshot(), "instructions": instructions}

    async def _publish_semantic_draft(self, arguments: dict[str, Any]) -> dict[str, Any]:
        draft = self._semantic_draft
        if draft is None:
            raise RuntimeError("Begin an isolated draft before publishing a bundle")
        session = self._get_semantic_authoring_session()
        snapshot = session.snapshot()
        receipt = draft.publish(
            arguments.get("files", {}), snapshot["semantic_model_file"], arguments["expected_revision"]
        )
        old_path = session.osi_target_state.selected_path
        live_path = str(draft.live_path(old_path))
        state, evidence = session.osi_target_state, session.generation_evidence
        # All evidence refers to identical published bytes, never to a new model.
        evidence.rebase_artifact(old_path, live_path)
        state.selected["absolute_path"] = live_path
        state.artifact_snapshot_path = live_path
        original = draft.baseline.get(snapshot["semantic_model_file"])
        state.artifact_snapshot_content = original if original is not None else b""
        state.artifact_snapshot_existed = original is not None
        await self._retire_database_tools()
        new_session = SemanticAuthoringSession(
            self.agent_config,
            node_name=self.get_node_name(),
            target_state=state,
            generation_evidence=evidence,
            path_allowed=self._allows_authoring_path,
        )
        self._semantic_authoring_session = new_session
        new_session.setup(
            filesystem_factory=self._make_semantic_filesystem,
            runtime_db_context=self._semantic_runtime_db_context,
            filesystem_root=str(self.execution_scope.root),
        )
        new_session._existing_models_checked = True
        native_ids = {tool_definition(tool).tool_id for tool in self._native_tools}
        contributed = [tool for tool in self.tools if tool_definition(tool).tool_id not in native_ids]
        self._tool_set = ConfigurableToolSet(self)
        self._native_tools = self._tool_set.select(self.spec.tools)
        self.tools = [*self._native_tools, *contributed]
        self.permission_hooks = None
        self._tool_transformers_applied = False
        self._filter_permission_denied_tools()
        self._apply_semantic_plan_review()
        return receipt

    async def _call_semantic_authoring(self, operation: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._runtime_closed or not self._uses_semantic_authoring:
            raise PermissionError("Semantic authoring is not available on this agent")
        write = operation in {"prepare_artifact", "reconcile_artifact", "complete", "publish"}
        native_ids = {tool_definition(tool).tool_id for tool in self._native_tools}
        selected = [tool_definition(tool).tool_id for tool in self.tools if tool_definition(tool).tool_id in native_ids]
        candidates = (
            [identity for identity in selected if identity in AUTHORING_MUTATION_TOOL_IDS]
            if write
            else [
                identity
                for identity in selected
                if identity.split(".", 1)[-1] in AUTHORING_TOOL_NAMES or identity == "semantic_tools.validate_semantic"
            ]
        )
        if not candidates:
            raise PermissionError(
                "Semantic authoring requires an explicitly selected permitted mutation tool"
                if write
                else "Semantic authoring requires an explicitly selected permitted semantic tool"
            )
        if operation == "begin_draft":
            await self._authorize_host_tool(candidates[0], {})
            return await self._begin_semantic_draft()
        session = self._get_semantic_authoring_session()
        if operation == "state":
            await self._authorize_host_tool(candidates[0], {})
            return {"selected": session.osi_target_state.selected is not None, "changed": session._target_was_mutated()}
        path = session.osi_target_state.selected_path
        if not path or not self._allows_authoring_path(Path(path)):
            raise PermissionError("Semantic model is outside scoped_context.project_files")
        live_path = str(self._semantic_draft.live_path(path)) if self._semantic_draft is not None else path
        await self._authorize_host_tool(candidates[0], {"path": live_path})
        with self.execution_scope.activate():
            if operation == "inspect_bundle":
                if self._semantic_draft is None:
                    raise RuntimeError("Begin an isolated draft before inspecting a bundle")
                args = arguments or {}
                return self._semantic_draft.inspect(
                    args.get("files", {}),
                    args["mapping_file"],
                    args.get("inspection", "validate"),
                    args.get("arguments"),
                )
            if operation == "publish":
                return await self._publish_semantic_draft(arguments or {})
            if (
                operation == "reconcile_artifact"
                and self._semantic_draft is not None
                and not self._semantic_draft.published
            ):
                raise RuntimeError("Publish the validated bundle before KB reconciliation")
            try:
                value = getattr(session, operation)()
            except Exception as exc:
                if operation == "reconcile_artifact" and self._semantic_draft is not None and session.kb_sync_attempted:
                    self._semantic_draft.kb_sync_pending = True
                    return {
                        "success": False,
                        "code": "kb_sync_failed",
                        "files_retained": True,
                        "error": str(exc),
                        "semantic_models": [session.osi_target_state.selected["semantic_model_file"]],
                    }
                raise
            if operation == "complete" and self._semantic_draft is not None:
                self._semantic_draft_completed = True
            return value

    @asynccontextmanager
    async def _authoring_guard(self):
        if self._uses_semantic_authoring:
            from datus.agent.node.semantic_authoring import semantic_authoring_guard

            async with semantic_authoring_guard(
                self.agent_config, datasources=self.execution_scope.declared_datasources
            ):
                yield
        else:
            yield

    async def execute_stream(self, action_history_manager=None):
        async with self._authoring_guard():
            completed = False
            try:
                async for action in super().execute_stream(action_history_manager):
                    yield action
                completed = self.result is not None and self.result.success
            finally:
                if self._semantic_authoring_session is not None and not completed:
                    self._rollback_authoring()

    def _compose_hooks(self, extra: Any = None) -> Any:
        return _ScopeHooks(self, super()._compose_hooks(extra))

    def _build_template_context(self, ctx: StreamRunContext) -> dict[str, str]:
        return {
            "instructions": "\n\n".join(filter(None, [self.spec.instructions, self._publisher_instructions()])),
            "runtime_instructions": self._prepared.instructions,
            "runtime_context": json.dumps(self._prepared.context, ensure_ascii=False),
            "scoped_context": json.dumps(self._host.scoped_context, ensure_ascii=False),
            "output_format": self.spec.output.format,
            "output_instructions": self.spec.output.instructions,
        }

    def _build_success_result(self, ctx: StreamRunContext) -> ConfigurableNodeResult:
        content = ctx.response_content
        response = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        data = json.loads(response) if self.spec.output.format == "json" else None
        return ConfigurableNodeResult(
            success=True, response=response, data=data, tokens_used=self._extract_total_tokens(ctx.get_run_actions())
        )

    async def _stream_post_build(
        self, ctx: StreamRunContext, result: ConfigurableNodeResult
    ) -> AsyncGenerator[ActionHistory, None]:
        finalize = getattr(self._runtime, "finalize", None)
        if finalize:
            updated = await _await_result(finalize(result.model_dump(mode="json"), deepcopy(self._turn_context)))
            if updated is not None:
                if not isinstance(updated, dict):
                    raise TypeError("finalize must return a result mapping or None")
                merged = ConfigurableNodeResult.model_validate({**result.model_dump(), **updated})
                # Keep the same object the base stream serializes after this hook.
                result.__dict__.update(merged.__dict__)
                result.__pydantic_extra__ = merged.__pydantic_extra__
        session = self._semantic_authoring_session
        if result.success and session is not None and session._target_was_mutated() and self._runtime is None:
            await self._call_semantic_authoring("reconcile_artifact")
            completed = await self._call_semantic_authoring("complete")
            result.__pydantic_extra__ = {
                **(result.__pydantic_extra__ or {}),
                "semantic_models": [completed["semantic_model_file"]],
                "business_validation": completed["business_validation"],
            }
        if (
            result.success
            and session is not None
            and session._target_was_mutated()
            and session.osi_target_state.artifact_snapshot_path
        ):
            raise RuntimeError("Complete semantic authoring through the host before returning success")
        if (
            result.success
            and self._semantic_draft is not None
            and self._semantic_draft.published
            and not self._semantic_draft_completed
        ):
            raise RuntimeError("Complete the published semantic bundle before returning success")
        if False:  # pragma: no cover - async generator protocol
            yield

    async def aclose(self) -> None:
        try:
            if not self._runtime_closed:
                self._runtime_closed = True
                close = getattr(self._runtime, "close", None)
                if close:
                    deadline = asyncio.timeout(_RUNTIME_CLOSE_TIMEOUT)
                    try:
                        async with deadline:
                            await _await_result(close())
                    except TimeoutError:
                        if not deadline.expired():
                            raise
                        logger.warning("Plugin runtime close timed out for %s", self.get_node_name())
        finally:
            try:
                if self._semantic_authoring_session is not None:
                    self._rollback_authoring()
                if self._semantic_draft is not None:
                    self._semantic_draft.close()
            finally:
                try:
                    await self._retire_database_tools()
                finally:
                    await super().aclose()
