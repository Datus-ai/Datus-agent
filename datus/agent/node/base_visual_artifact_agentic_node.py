# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared base for the two visual-artifact subagents.

``GenVisualReportAgenticNode`` (pre-baked JSON results, optional CLI HTML
compile) and ``GenVisualDashboardAgenticNode`` (Jinja2 SQL templates,
live runtime queries) share roughly 75 % of their code: tool setup,
prompt rendering, the LLM streaming loop, the result-action wiring, and
the post-run validation extraction. This module hosts the common
machinery so each concrete node only needs to declare:

* the artifact kind / directory / id regex (``ARTIFACT_*`` class vars),
* the filesystem-tool subclass that enforces artifact write protection,
* the artifact tools subclass that exposes ``start_new_*`` / ``save_*`` /
  ``validate_render``,
* the per-tool-call ``save_query[_template]`` action type that counts as
  a "query saved",
* the fallback prompt template name,
* the result :class:`pydantic.BaseModel` (``GenVisual*NodeResult``),
* (optionally) a post-validate hook — used by the report node to compile
  a standalone HTML and open it in the user's browser; dashboard mode
  doesn't have an equivalent path.

Anything that's *byte-identical* between the two node files lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Generic, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel

from datus.agent.node._visual_artifact_finalize import run_finalize_analysis
from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

if TYPE_CHECKING:
    from datus.agent.node.stream_run_context import StreamRunContext
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, FilesystemFuncTool
from datus.tools.func_tool._visual_artifact_helpers import (
    extract_artifact_result_field,
    extract_artifact_result_list,
)
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.loggings import get_logger
from datus.utils.message_utils import build_structured_content

logger = get_logger(__name__)


# Generic over the concrete pydantic input/result models so subclasses
# can type-narrow without bypassing the base contract.
InputT = TypeVar("InputT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


class BaseVisualArtifactAgenticNode(AgenticNode, Generic[InputT, ResultT]):
    """Common framework for visual-artifact subagents (report / dashboard).

    Subclasses declare the artifact-specific class variables below, plus
    a small set of hook methods that customize the parts that genuinely
    differ between report and dashboard mode.
    """

    # ── Artifact-specific class variables (subclass MUST override) ─────────

    #: ``"report"`` | ``"dashboard"`` — used for error messages and the
    #: fallback prompt-context key (``report_slug`` / ``dashboard_slug``).
    ARTIFACT_KIND: ClassVar[str] = ""

    #: Top-level workspace directory, e.g. ``"reports"`` or ``"dashboards"``.
    ARTIFACT_ROOT_DIR_NAME: ClassVar[str] = ""

    #: Concrete :class:`FilesystemFuncTool` subclass (e.g.
    #: ``ReportFilesystemFuncTool``) that locks out direct writes to
    #: protected artifact paths.
    FILESYSTEM_TOOL_CLS: ClassVar[Type[FilesystemFuncTool]] = FilesystemFuncTool

    #: ``"save_query"`` or ``"save_query_template"`` — the action type
    #: that ``execute_stream`` counts to populate the result's
    #: ``query_count`` / ``template_count`` field.
    QUERY_SAVE_ACTION_TYPE: ClassVar[str] = ""

    #: Template name used as the fallback when the prompt registry can't
    #: find ``<node_name>_system`` (e.g. when a custom subagent renames
    #: the node but didn't ship a template).
    FALLBACK_TEMPLATE_NAME: ClassVar[str] = ""

    #: Default tools when ``agent.yml`` doesn't override ``tools:``.
    #: ``context_search_tools.*`` expands at setup-time into whichever search
    #: helpers the active project actually supports — ``list_subject_tree`` is
    #: always present, while ``search_metrics`` / ``get_metrics`` /
    #: ``search_reference_sql`` / etc. only surface when the corresponding
    #: store has indexed content. Without the wildcard, the prompt advertises
    #: metric-discovery tools the LLM can't actually call, so the model
    #: silently falls back to deriving SQL from raw table DDL.
    DEFAULT_TOOLS: ClassVar[str] = "semantic_tools.*, db_tools.*, context_search_tools.*"

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[InputT] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
    ):
        self.execution_mode = execution_mode
        self.configured_node_name = node_name

        self.max_turns = 80
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            cfg = agent_config.agentic_nodes[node_name]
            if isinstance(cfg, dict):
                self.max_turns = cfg.get("max_turns", 80)

        # Tool attributes must exist before the parent constructor calls
        # ``_get_system_prompt`` indirectly via skill setup.
        self.db_func_tool: Optional[DBFuncTool] = None
        self.semantic_tools: Optional[SemanticTools] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        # ``artifact_tools`` is the bag of artifact-specific tools (e.g.
        # ``ReportArtifactTools`` / ``DashboardArtifactTools``); the
        # subclass instantiates it in ``_make_artifact_tools``.
        self.artifact_tools: Optional[Any] = None
        self._active_artifact_slug: Optional[str] = None
        # Captures the root cause when ``_setup_db_tools`` fails so
        # ``_prepare_artifacts`` can surface it instead of the generic
        # "db_tools not configured" message.
        self._db_tool_setup_error: Optional[BaseException] = None

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
        )

        self.setup_tools()

        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(
            "%s tools: %d - %s",
            type(self).__name__,
            len(self.tools),
            [t.name for t in self.tools],
        )

    # ── Tool setup ────────────────────────────────────────────────────────

    def setup_tools(self) -> None:
        if not self.agent_config:
            return

        self.tools = []
        config_value = self.node_config.get("tools") or self.DEFAULT_TOOLS
        for pattern in (p.strip() for p in config_value.split(",") if p.strip()):
            self._setup_tool_pattern(pattern)

        # Always provide the hardened filesystem tool — the node needs it
        # for authoring render/*.jsx (write_file / edit_file / delete_file)
        # and for general exploration.
        if not self.filesystem_func_tool:
            self._setup_filesystem_tools()

        self._setup_sub_agent_task_tool()
        if self.sub_agent_task_tool:
            self.tools.extend(self.sub_agent_task_tool.available_tools())

        logger.info("setup_tools done: %d tools - %s", len(self.tools), [t.name for t in self.tools])

    def _make_filesystem_tool(self, **kwargs):  # type: ignore[override]
        """Swap in the artifact-specific filesystem tool class.

        Identical resolution rules to :meth:`AgenticNode._make_filesystem_tool`,
        but constructs ``self.FILESYSTEM_TOOL_CLS`` so write-protection
        for the artifact directory is in effect.
        """
        from datus.configuration.inherited_memory_overrides import get_inherited_memory

        root_path = kwargs.pop("root_path", None) or self._resolve_workspace_root()
        datus_home = kwargs.pop("datus_home", None)
        if datus_home is None and self.agent_config is not None:
            path_manager = getattr(self.agent_config, "path_manager", None)
            if path_manager is not None:
                try:
                    datus_home = str(path_manager.datus_home)
                except Exception:
                    datus_home = None
        strict = kwargs.pop("strict", None)
        if strict is None:
            strict = self._resolve_filesystem_strict()
        current_node = kwargs.pop("current_node", None) or self.get_node_name()
        inherited_memory_node = kwargs.pop("inherited_memory_node", None)
        if inherited_memory_node is None:
            inherited_memory_node = get_inherited_memory(current_node)
        return self.FILESYSTEM_TOOL_CLS(
            root_path=root_path,
            current_node=current_node,
            datus_home=datus_home,
            strict=strict,
            inherited_memory_node=inherited_memory_node,
            **kwargs,
        )

    def _setup_tool_pattern(self, pattern: str) -> None:
        try:
            if pattern.endswith(".*"):
                base = pattern[:-2]
                if base == "semantic_tools":
                    self._setup_semantic_tools()
                elif base == "db_tools":
                    self._setup_db_tools()
                elif base == "context_search_tools":
                    self._setup_context_search_tools()
                elif base == "filesystem_tools":
                    self._setup_filesystem_tools()
                else:
                    logger.warning("Unknown tool type: %s", base)
                return

            if pattern == "semantic_tools":
                self._setup_semantic_tools()
            elif pattern == "db_tools":
                self._setup_db_tools()
            elif pattern == "context_search_tools":
                self._setup_context_search_tools()
            elif pattern == "filesystem_tools":
                self._setup_filesystem_tools()
            elif "." in pattern:
                tool_type, method_name = pattern.split(".", 1)
                self._setup_specific_tool_method(tool_type, method_name)
            else:
                logger.warning("Unknown tool pattern: %s", pattern)
        except Exception as exc:
            logger.error("Failed to setup tool pattern %r: %s", pattern, exc)

    def _setup_db_tools(self) -> None:
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
            )
            self.tools.extend(self.db_func_tool.available_tools())
        except Exception as exc:
            logger.error("Failed to setup db tools: %s", exc, exc_info=True)
            self._db_tool_setup_error = exc

    def _setup_semantic_tools(self) -> None:
        try:
            adapter_type = self.node_config.get("adapter_type", "metricflow")
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
                adapter_type=adapter_type,
            )
            self.tools.extend(self.semantic_tools.available_tools())
        except Exception as exc:
            logger.error("Failed to setup semantic tools: %s", exc)

    def _setup_context_search_tools(self) -> None:
        try:
            self.context_search_tools = ContextSearchTools(
                self.agent_config, sub_agent_name=self.node_config.get("system_prompt")
            )
            self.tools.extend(self.context_search_tools.available_tools())
        except Exception as exc:
            logger.error("Failed to setup context search tools: %s", exc)

    def _setup_filesystem_tools(self) -> None:
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
        except Exception as exc:
            logger.error("Failed to setup filesystem tools: %s", exc)

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Register tool buckets so category-scoped rules and the
        ``_FS_DEPENDENT_NODES`` exclusion in ``apply_proxy_tools`` apply.

        Without this mapping, every tool falls back to the catch-all
        ``tools`` category. That breaks two things for the visual artifact
        nodes:

        * ``filesystem_tools.*`` rules (and the zone-based fs policy in
          ``PermissionHooks._handle_filesystem_zone``) never match, so the
          INTERNAL / EXTERNAL gating that every other fs-using node gets
          would silently skip visual reports/dashboards.
        * ``apply_proxy_tools`` cannot recognise their filesystem tools as
          excluded — it looks up the tool's category in the registry — so
          a parent agent's ``write_file`` / ``edit_file`` proxy patterns
          end up wrapping the sub-agent's filesystem tools, round-tripping
          every chunk to the browser instead of writing ``render/*.jsx``
          server-side.
        """
        mapping = super()._tool_category_map()
        if self.db_func_tool:
            mapping["db_tools"] = list(self.db_func_tool.available_tools())
        semantic_bucket: List[Any] = []
        if self.semantic_tools:
            semantic_bucket.extend(self.semantic_tools.available_tools())
        # Artifact-specific helpers (``start_new_*`` / ``bind_existing_*`` /
        # ``save_query*`` / ``validate_render``) are subagent-internal state
        # mutations; lump them into the ``semantic_tools`` bucket so the
        # ``semantic_tools.*`` ALLOW rule (normal profile) covers them.
        # Without this they fall through to ``tools.<name>`` and the default
        # ASK gate would block at the broker.
        if self.artifact_tools and hasattr(self.artifact_tools, "available_tools"):
            try:
                semantic_bucket.extend(self.artifact_tools.available_tools())
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("artifact_tools.available_tools() failed: %s", exc)
        if semantic_bucket:
            mapping["semantic_tools"] = semantic_bucket
        if self.context_search_tools:
            mapping["context_search_tools"] = list(self.context_search_tools.available_tools())
        if self.filesystem_func_tool:
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping

    def _setup_specific_tool_method(self, tool_type: str, method_name: str) -> None:
        try:
            if tool_type == "semantic_tools":
                if not self.semantic_tools:
                    self.semantic_tools = SemanticTools(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                        adapter_type=self.node_config.get("adapter_type", "metricflow"),
                    )
                tool_instance = self.semantic_tools
            elif tool_type == "db_tools":
                if not self.db_func_tool:
                    self.db_func_tool = DBFuncTool(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                    )
                tool_instance = self.db_func_tool
            elif tool_type == "context_search_tools":
                if not self.context_search_tools:
                    self.context_search_tools = ContextSearchTools(
                        self.agent_config, sub_agent_name=self.node_config.get("system_prompt")
                    )
                tool_instance = self.context_search_tools
            elif tool_type == "filesystem_tools":
                if not self.filesystem_func_tool:
                    self.filesystem_func_tool = self._make_filesystem_tool()
                tool_instance = self.filesystem_func_tool
            else:
                logger.warning("Unknown tool type: %s", tool_type)
                return

            if hasattr(tool_instance, method_name):
                from datus.tools.func_tool import trans_to_function_tool

                self.tools.append(trans_to_function_tool(getattr(tool_instance, method_name)))
            else:
                logger.warning("Method %r not found in %s", method_name, tool_type)
        except Exception as exc:
            logger.error("Failed to setup %s.%s: %s", tool_type, method_name, exc)

    # ── Prompt + message wiring ───────────────────────────────────────────

    def _artifact_slug_prompt_key(self) -> str:
        """Prompt-context key for the active artifact slug.

        Default: ``"<kind>_slug"`` (i.e. ``report_slug`` or
        ``dashboard_slug``) — matches what the system prompt templates
        expect.
        """
        return f"{self.ARTIFACT_KIND}_slug"

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        context: Dict[str, Any] = {
            "has_semantic_tools": bool(self.semantic_tools),
            "has_db_tools": bool(self.db_func_tool),
            "has_context_search_tools": bool(self.context_search_tools),
            "has_ask_user_tool": self.ask_user_tool is not None,
            "has_task_tool": bool(self.sub_agent_task_tool),
            "agent_config": self.agent_config,
            "conversation_summary": conversation_summary,
            self._artifact_slug_prompt_key(): self._active_artifact_slug,
            "rules": self.node_config.get("rules", []),
            "agent_description": self.node_config.get("agent_description", ""),
        }

        if self.agent_config:
            from datus.utils.node_utils import build_datasource_prompt_context

            context.update(build_datasource_prompt_context(self.agent_config))
            context["db_name"] = context.get("datasource")

        from datus.utils.time_utils import get_default_current_date

        context["current_date"] = get_default_current_date(None)

        version = None if prompt_version in (None, "") else str(prompt_version)
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=version, **context)
        except FileNotFoundError:
            logger.warning(
                "Template %r missing, falling back to %s",
                system_prompt_name,
                self.FALLBACK_TEMPLATE_NAME,
            )
            base_prompt = pm.render_template(template_name=self.FALLBACK_TEMPLATE_NAME, version=version, **context)

        return self._finalize_system_prompt(base_prompt)

    def _build_enhanced_message(self, user_input: InputT) -> str:
        parts: List[str] = []
        catalog = getattr(user_input, "catalog", None)
        database = getattr(user_input, "database", None)
        db_schema = getattr(user_input, "db_schema", None)
        user_message = getattr(user_input, "user_message", "")

        if catalog:
            parts.append(f"Catalog: {catalog}")
        if database:
            parts.append(f"Database context: {database}")
        if db_schema:
            parts.append(f"Schema: {db_schema}")

        if parts:
            return build_structured_content("\n".join(parts), user_message)
        return user_message

    # ── Finalize stage ────────────────────────────────────────────────────

    def _run_finalize(self, artifact_slug: str, actions: List[ActionHistory]) -> Dict[str, Any]:
        """Run the finalize analysis stage for the just-completed artifact.

        Wraps :func:`run_finalize_analysis` so the orchestrator can stay
        UI-agnostic (no node-specific paths in the helper module).
        Failures are logged and returned as a dict; never re-raised.
        """
        try:
            project_root = Path(self.agent_config.project_root).resolve()
            artifact_dir = project_root / self.ARTIFACT_ROOT_DIR_NAME / artifact_slug
            queries_dir = artifact_dir / "queries"
            analysis_dir = artifact_dir / "analysis"
            return run_finalize_analysis(
                model=self.model,
                artifact_kind=self.ARTIFACT_KIND,
                artifact_dir=artifact_dir,
                queries_dir=queries_dir,
                analysis_dir=analysis_dir,
                actions=actions,
                # ``db_func_tool`` is wired by ``setup_tools`` early in
                # the run. Passing it through enables the finalize-time
                # ``key_tables_schema.json`` bake (describe_table snapshot
                # per manifest.key_tables entry). The bake is best-effort
                # so finalize still runs when this is None (e.g. node was
                # configured without db tools — unusual but supported).
                db_func_tool=self.db_func_tool,
            )
        except Exception as exc:
            logger.warning("Finalize stage crashed for %s/%s: %s", self.ARTIFACT_KIND, artifact_slug, exc)
            return {"ok": False, "warnings": [], "error": f"finalize crashed: {exc}"}

    # ── Artifact tools wiring ─────────────────────────────────────────────

    def _make_artifact_tools(self, user_input: InputT) -> Any:
        """Build the artifact-specific tools instance (subclass implements).

        The tools instance must expose ``available_tools()`` returning the
        list of bound function-tools to add to ``self.tools``; the active
        artifact slug (after ``start_new_*`` / ``bind_existing_*``) must
        live on an attribute the subclass knows how to read (it is fetched
        via :meth:`_read_artifact_slug_from_tools`).

        ``user_input`` is passed through so subclasses can forward the
        raw ``user_message`` to the tools instance — the analysis-artifact
        layer needs it verbatim to seed ``analysis/intent.md`` on
        ``start_new_*`` / ``bind_existing_*``.
        """
        raise NotImplementedError

    def _read_artifact_slug_from_tools(self) -> Optional[str]:
        """Return the active artifact slug off the artifact tools instance.

        Subclasses know the attribute name (``report_slug`` /
        ``dashboard_slug``). Default looks for either, in priority order,
        so a minimal subclass can rely on naming convention.
        """
        if self.artifact_tools is None:
            return None
        for attr in (f"{self.ARTIFACT_KIND}_slug", "artifact_slug"):
            value = getattr(self.artifact_tools, attr, None)
            if value:
                return value
        return None

    def _read_artifact_manifest(self, artifact_slug: Optional[str]) -> Dict[str, Any]:
        """Return the parsed manifest.json for ``artifact_slug`` or an empty dict.

        Used to surface ``name`` / ``description`` / ``created_at`` in the
        node result so the SSE artifact card can be rendered without a
        follow-up call to ``/api/v1/report/detail``. Missing or unreadable
        manifests are treated as soft-failures because the run may have
        crashed before binding (mode is then ``None`` too).
        """
        if not artifact_slug:
            return {}
        project_root = getattr(self.agent_config, "project_root", None) if self.agent_config else None
        if not project_root:
            return {}
        manifest_path = Path(project_root) / self.ARTIFACT_ROOT_DIR_NAME / artifact_slug / "manifest.json"
        if not manifest_path.is_file():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("Failed to read %s manifest %s: %s", self.ARTIFACT_KIND, manifest_path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _prepare_artifacts(self, user_input: InputT) -> None:
        """Wire artifact tools into ``self.tools`` and reset the active slug.

        The LLM decides between ``start_new_<kind>`` (create) and
        ``bind_existing_<kind>`` (edit) at execution time; we just make
        both tools available. ``_active_artifact_slug`` stays ``None``
        until the LLM commits to one or the other.
        """
        if not self.agent_config or not getattr(self.agent_config, "project_root", None):
            raise ValueError(f"agent_config.project_root is required for gen_visual_{self.ARTIFACT_KIND}")
        if not self.db_func_tool:
            # save_query[_template] needs a connector; fail loud rather
            # than silently produce a no-op tool.
            root_cause = self._db_tool_setup_error
            if root_cause is not None:
                raise ValueError(
                    f"gen_visual_{self.ARTIFACT_KIND} requires db_tools to be configured "
                    "(DEFAULT_TOOLS includes db_tools.*); DBFuncTool initialization failed: "
                    f"{type(root_cause).__name__}: {root_cause}"
                ) from root_cause
            raise ValueError(
                f"gen_visual_{self.ARTIFACT_KIND} requires db_tools to be configured "
                "(DEFAULT_TOOLS includes db_tools.*)."
            )

        self._active_artifact_slug = None
        self.artifact_tools = self._make_artifact_tools(user_input)
        # Repeated ``execute_stream`` calls on the same node instance
        # would otherwise stack stale tool wrappers bound to the previous
        # artifact tools instance, which could resolve calls against an
        # outdated artifact id. Replace any prior registration by name
        # before extending with the freshly-built tools.
        new_tools = self.artifact_tools.available_tools()
        replaced_names = {getattr(t, "name", None) for t in new_tools}
        self.tools = [t for t in self.tools if getattr(t, "name", None) not in replaced_names]
        self.tools.extend(new_tools)

    # ── Result construction (subclass overrides) ──────────────────────────

    def _finalize_artifact_success(
        self,
        *,
        user_input: InputT,
        response_content: str,
        artifact_slug: Optional[str],
        app_jsx_rel_path: Optional[str],
        render_file_count: int,
        query_actions: List[ActionHistory],
        tokens_used: int,
        all_actions: List[ActionHistory],
        tool_calls: List[ActionHistory],
    ) -> ResultT:
        """Construct the per-run result object (typed per artifact kind).

        Renamed from ``_build_success_result`` so it does not clash with the
        ``AgenticNode._build_success_result(ctx)`` template hook. Subclasses
        (gen_visual_dashboard, gen_visual_report) implement this and let the
        ``_build_success_result(ctx)`` adapter below assemble the kwargs.
        """
        raise NotImplementedError

    def _finalize_artifact_error(self, exc: BaseException) -> ResultT:
        """Construct the result returned when ``execute_stream`` raises.

        Renamed from ``_build_error_result`` so it does not clash with the
        ``AgenticNode._build_error_result(exc, ctx)`` adapter.
        """
        raise NotImplementedError

    def _post_validate_hook(self, artifact_slug: str, result: ResultT) -> None:
        """Run artifact-specific work after a successful ``validate_render``.

        The report subagent uses this to compile a standalone HTML and
        optionally open it in the browser; dashboard mode has nothing to
        do here. Default is a no-op.
        """
        return None

    def _missing_binding_error(self) -> str:
        kind = self.ARTIFACT_KIND
        return (
            f"Run finished without binding a {kind}. The LLM must call either "
            f"start_new_{kind}(...) or bind_existing_{kind}(...) before producing "
            "the artifact."
        )

    def _incomplete_render_error(self) -> str:
        return (
            "validate_render never returned success — the "
            f"{self.ARTIFACT_KIND} artifact is incomplete. "
            "The LLM must write_file the render/*.jsx components and then "
            "call validate_render() to finalize."
        )

    def _final_summary_message(self, artifact_slug: Optional[str], app_jsx_rel_path: Optional[str]) -> str:
        if app_jsx_rel_path:
            return (
                f"Visual {self.ARTIFACT_KIND} generated: {self.ARTIFACT_ROOT_DIR_NAME}/{artifact_slug}/render/app.jsx"
            )
        return f"Visual {self.ARTIFACT_KIND} run finished without a validated render/ tree."

    # ── Helpers re-exposed as static methods (legacy API preserved) ──────

    @staticmethod
    def _extract_artifact_result_field(action: ActionHistory, field: str) -> Optional[str]:
        return extract_artifact_result_field(action, field)

    @staticmethod
    def _extract_artifact_result_list(action: ActionHistory, field: str) -> Optional[List[Any]]:
        return extract_artifact_result_list(action, field)

    @staticmethod
    def _find_artifact_tool_call(actions: List[ActionHistory], tool_name: str) -> Optional[ActionHistory]:
        for a in reversed(actions):
            if a.action_type == tool_name:
                return a
        return None

    # ── Template hooks ─────────────────────────────────────────────────────

    async def _before_stream(self, ctx: "StreamRunContext") -> None:
        # Bind artifact tools for this run (regenerates artifact id every call).
        self._prepare_artifacts(ctx.user_input)

    def _build_success_result(self, ctx: "StreamRunContext") -> ResultT:
        all_actions = ctx.action_history_manager.get_actions()
        tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]
        tokens_used = self._extract_total_tokens(all_actions)

        query_actions = [a for a in tool_calls if a.action_type == self.QUERY_SAVE_ACTION_TYPE]
        app_jsx_rel_path: Optional[str] = None
        render_file_count = 0
        for tc in reversed(tool_calls):
            if tc.action_type != "validate_render":
                continue
            candidate = extract_artifact_result_field(tc, "app_jsx_path")
            if candidate:
                app_jsx_rel_path = candidate
                render_files = extract_artifact_result_list(tc, "render_files")
                render_file_count = len(render_files) if render_files else 0
                break

        # The LLM picked the active artifact slug by calling
        # start_new_<kind> / bind_existing_<kind>; the tools instance owns it.
        picked = self._read_artifact_slug_from_tools()
        if picked:
            self._active_artifact_slug = picked

        response_content = ctx.response_content
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        result = self._finalize_artifact_success(
            user_input=ctx.user_input,
            response_content=response_content,
            artifact_slug=self._active_artifact_slug,
            app_jsx_rel_path=app_jsx_rel_path,
            render_file_count=render_file_count,
            query_actions=query_actions,
            tokens_used=tokens_used,
            all_actions=all_actions,
            tool_calls=tool_calls,
        )

        if app_jsx_rel_path is None:
            error_msg = (
                self._missing_binding_error() if self._active_artifact_slug is None else self._incomplete_render_error()
            )
            if hasattr(result, "error"):
                result.error = error_msg  # type: ignore[attr-defined]
            # When the render is incomplete, success must reflect that so the
            # template's final action emits SUCCESS=False status.
            if hasattr(result, "success"):
                result.success = False  # type: ignore[attr-defined]
        elif self._active_artifact_slug:
            # Finalize stage: produce insights / suggested_questions /
            # subject_refs (present iff non-empty) under analysis/. Runs
            # before _post_validate_hook so the HTML compile step (CLI
            # mode) can pick up the freshly-written analysis files if
            # it ever wants to. Finalize failures are surfaced via
            # ``result.finalize_warnings`` but never block the main
            # artifact — the SQL+render bundle is already on disk.
            finalize_summary = self._run_finalize(self._active_artifact_slug, all_actions)
            if hasattr(result, "finalize_warnings") and finalize_summary.get("warnings"):
                result.finalize_warnings = finalize_summary["warnings"]  # type: ignore[attr-defined]
            if hasattr(result, "finalize_error") and finalize_summary.get("error"):
                result.finalize_error = finalize_summary["error"]  # type: ignore[attr-defined]
            self._post_validate_hook(self._active_artifact_slug, result)

        # Stash artifact summary so ``_final_summary_for_action`` (used by the
        # base template's final action emit) can render the right message.
        ctx.extras["artifact_app_jsx_rel_path"] = app_jsx_rel_path
        ctx.extras["artifact_slug"] = self._active_artifact_slug
        return result

    def _build_error_result(self, exc: BaseException, ctx: "StreamRunContext") -> ResultT:
        # Visual-artifact subclasses pre-existed the unified base error helper
        # and own a richer error shape (manifest snapshot, artifact mode).
        # Refresh the bound artifact slug from the tools so manifest/artifact
        # metadata is not lost when the LLM bound an artifact before the
        # run failed (success path does the same refresh in _build_success_result).
        picked = self._read_artifact_slug_from_tools()
        if picked:
            self._active_artifact_slug = picked
        return self._finalize_artifact_error(exc)
