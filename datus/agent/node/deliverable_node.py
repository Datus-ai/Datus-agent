# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared base class for deliverable-producing subagents — ``gen_table`` and
``gen_job`` today; ``gen_dashboard`` and ``scheduler`` join in follow-up chunks.

Centralizes ~85 % of the boilerplate that used to be copy-pasted across nodes:
tool / filesystem / prompt setup, the stream loop, session handling, and
ValidationHook wiring (with retry loop).

Subclasses provide four class-level constants (:attr:`NODE_NAME`,
:attr:`DEFAULT_SKILLS`, :attr:`PROMPT_TEMPLATE`, :attr:`ACTION_TYPE`) and one
hook method :meth:`_setup_domain_tools` — everything else is inherited.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.semantic_agentic_node_models import SemanticNodeResult
from datus.tools.func_tool import DBFuncTool, FilesystemFuncTool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.message_utils import build_structured_content
from datus.validation import ValidationHook
from datus.validation.report import build_retry_prompt

logger = get_logger(__name__)


class ValidationHookRetryPolicy:
    """:class:`~datus.agent.node.retry_policy.RetryPolicy` driven by ``ValidationHook``.

    Used exclusively by :class:`DeliverableAgenticNode` and its subclasses
    (gen_dashboard, scheduler, gen_table, gen_job). After each stream
    completes, the policy inspects ``hook.final_report`` and reschedules
    with a context-aware retry prompt when a blocking failure is recorded.

    Lives in this module — not in a shared ``policies/`` package — because
    it is tied to ``ValidationHook``'s state machine and would not be reused
    by any other node.
    """

    def __init__(self, hook: ValidationHook, max_attempts: int = 3, node_name: str = "deliverable"):
        self.hook = hook
        self.max_attempts = max(1, max_attempts)
        self.node_name = node_name
        self._blocking_report: Optional[dict] = None

    def reset(self, ctx: StreamRunContext) -> None:
        # Drop the prior attempt's blocking report so a recovered retry does
        # not inherit a stale ``success=False`` decision.
        self._blocking_report = None
        self.hook.reset_session()

    def should_retry(self, ctx: StreamRunContext) -> bool:
        report = self.hook.final_report
        if report is None or not report.has_blocking_failure():
            return False
        self._blocking_report = report.model_dump(by_alias=True, exclude_none=True)
        logger.info(
            "Validation blocked attempt %d/%d for %s: %s",
            ctx.attempt,
            self.max_attempts,
            self.node_name,
            [c.name for c in report.checks if not c.passed],
        )
        return True

    def next_prompt(self, ctx: StreamRunContext) -> Optional[str]:
        report = self.hook.final_report
        if report is None:
            return None
        return build_retry_prompt(report, list(self.hook.session_targets))

    def on_retry_actions(self, ctx: StreamRunContext) -> Iterable[ActionHistory]:
        # Pre-refactor Deliverable did not surface a user-visible action
        # between retry attempts — keep that behaviour.
        return ()

    def finalise(self, ctx: StreamRunContext) -> None:
        # Blocking failure (when retries exhausted) takes precedence over
        # the vanilla on_end report. Stash both decisions for the success
        # builder to translate into ``NodeResult.success`` + ``validation_report``.
        report = self.hook.final_report
        on_end_report: Optional[dict] = None
        if report is not None:
            on_end_report = report.model_dump(by_alias=True, exclude_none=True)
        ctx.extras["validation_report"] = self._blocking_report if self._blocking_report is not None else on_end_report
        ctx.extras["blocked"] = self._blocking_report is not None


class DeliverableAgenticNode(AgenticNode):
    """Base class for subagents that produce validation-worthy deliverables
    (tables, transfers, dashboards, charts, datasets, scheduler jobs, ...).

    Subclasses must set the four class constants below and implement
    :meth:`_setup_domain_tools`. Everything else (including the validation retry
    loop) is provided by this class.
    """

    # ── subclass-provided class constants ─────────────────────────────

    #: Name used by ``get_node_name()`` and by the skill system's
    #: ``allowed_agents`` scoping.
    NODE_NAME: ClassVar[str] = ""

    #: Comma-separated skill pattern string that becomes ``DEFAULT_SKILLS`` in
    #: the shared AgenticNode plumbing.
    DEFAULT_SKILLS: ClassVar[Optional[str]] = None

    #: Name of the Jinja template file (sans version suffix) to load from
    #: ``datus/prompts/prompt_templates/``. For most subclasses this is
    #: ``f"{NODE_NAME}_system"``.
    PROMPT_TEMPLATE: ClassVar[str] = ""

    #: ActionHistory action_type emitted for the final assistant action.
    ACTION_TYPE: ClassVar[str] = ""

    #: Associated :class:`NodeType` — used for the base class constructor.
    NODE_TYPE: ClassVar[str] = ""

    #: Default max_turns cap; subclasses override when the flow is deeper.
    DEFAULT_MAX_TURNS: ClassVar[int] = 50

    # Default ``result_class`` — gen_table / gen_job use SemanticNodeResult;
    # gen_dashboard / scheduler override with their specialised models.
    result_class = SemanticNodeResult

    # ── constructor ───────────────────────────────────────────────────

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        self.execution_mode = execution_mode
        # ``node_name`` supports custom aliases (``my_table: {node_class: gen_table}``).
        self._configured_node_name = node_name or self.NODE_NAME

        self.max_turns = self.DEFAULT_MAX_TURNS
        config_key = self._configured_node_name
        if agent_config and hasattr(agent_config, "agentic_nodes") and config_key in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[config_key]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", self.DEFAULT_MAX_TURNS)

        super().__init__(
            node_id=node_id or f"{self.NODE_NAME}_node",
            description=f"Deliverable-producing node: {self.NODE_NAME}",
            node_type=self.NODE_TYPE,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        self.db_func_tool: Optional[DBFuncTool] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.ask_user_tool = None

        # Hook is wired AFTER tools are set up so it captures the configured
        # model / db tool references.
        self._validation_hook: Optional[ValidationHook] = None

        self.setup_tools()
        self._setup_validation_hook()

    # ── inheritance hooks ─────────────────────────────────────────────

    def get_node_name(self) -> str:
        return self._configured_node_name

    def setup_tools(self):
        if not self.agent_config:
            return
        self.tools = []
        self._setup_domain_tools()
        self._setup_filesystem_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        logger.debug("Setup %d tools for %s: %s", len(self.tools), self.NODE_NAME, [t.name for t in self.tools])

    def _setup_domain_tools(self) -> None:
        """Subclass-specific tool registration.

        gen_table registers only ``execute_ddl``; gen_job additionally registers
        ``execute_write``, ``transfer_query_result``, and the MigrationTargetMixin
        wrappers.
        """
        raise NotImplementedError("_setup_domain_tools must be implemented by subclasses")

    def _setup_filesystem_tools(self) -> None:
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()
            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug("Setup filesystem tools with root path: %s", self.filesystem_func_tool.root_path)
        except Exception as e:
            logger.error("Failed to setup filesystem tools: %s", e)

    def _setup_validation_hook(self) -> None:
        """Attach :class:`ValidationHook` so post-mutation validation runs."""
        if self.agent_config is None:
            return
        try:
            registry = self.skill_manager.registry if self.skill_manager else None
            if registry is None:
                # Create a default registry so the hook can still dispatch
                # validators declared in the standard skill directories.
                from datus.tools.skill_tools.skill_config import SkillConfig
                from datus.tools.skill_tools.skill_registry import SkillRegistry

                registry = SkillRegistry(config=SkillConfig())
            validation_cfg = getattr(self.agent_config, "validation_config", None)
            enabled = bool(getattr(validation_cfg, "skill_validators_enabled", True)) if validation_cfg else True

            self._validation_hook = ValidationHook(
                node_name=self.get_node_name(),
                node_class=self.get_node_class_name(),
                registry=registry,
                model=self.model,
                db_func_tool=self.db_func_tool,
                bi_tool=getattr(self, "bi_func_tool", None),
                scheduler_tool=getattr(self, "scheduler_func_tool", None),
                skill_validators_enabled=enabled,
            )

            if enabled:
                node = self.get_node_name()
                klass = self.get_node_class_name()
                has_any = bool(registry.get_validators(node, node_class=klass))
                if not has_any:
                    logger.warning(
                        "No validator skills discovered for '%s'. Run `datus configure` (shell) "
                        "to deploy bundled skills (table-validation, transfer-reconciliation) "
                        "into ~/.datus/skills, or author project-level validators under "
                        "./.datus/skills.",
                        node,
                    )
        except Exception as e:
            logger.error("Failed to setup ValidationHook: %s", e)
            self._validation_hook = None

    def _prepare_template_context(self, user_input: Any) -> dict:
        # Session-stable values only — this render is frozen into the
        # per-session system-prompt snapshot. The current datasource and its
        # dialect are deliberately absent: they arrive per turn in the user
        # message (see ``_build_enhanced_message``).
        context = {
            "native_tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "None",
            "mcp_tools": ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None",
            "has_ask_user_tool": self.ask_user_tool is not None,
        }
        logger.debug("Prepared template context: %s", context)
        return context

    def _get_system_prompt(
        self,
        template_context: Optional[dict] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        version = prompt_version or self.node_config.get("prompt_version")
        # Template resolution order:
        # 1. ``node_config.system_prompt`` — user-specified override in agent.yml
        # 2. Alias-aware: ``{node_name}_system`` where node_name may be the
        #    alias (``my_dashboard`` → ``my_dashboard_system``)
        # 3. Class-level :attr:`PROMPT_TEMPLATE` fallback when set
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system" if system_prompt_name else self.PROMPT_TEMPLATE
        try:
            template_vars = {
                "agent_config": self.agent_config,
            }
            if template_context:
                template_vars.update(template_context)
            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, version=version, **template_vars
            )
            return self._finalize_system_prompt(base_prompt)
        except FileNotFoundError as e:
            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": version},
            ) from e
        except Exception as e:
            logger.error("Template loading error for '%s': %s", template_name, e)
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    # ── template hooks ───────────────────────────────────────────────

    def _build_template_context(self, ctx: StreamRunContext) -> Optional[dict]:
        # ``_build_template_context`` runs after session setup, so the parent
        # session is available — wire it into the validation hook here so
        # Layer-B validators can fork its tool-event history.
        if self._validation_hook is not None:
            self._validation_hook.set_parent_session(ctx.session)
        return self._prepare_template_context(ctx.user_input)

    def _compose_run_hooks(self, ctx: StreamRunContext) -> Any:
        # Pre-refactor every Deliverable run wired its ``_validation_hook``
        # into the model's hooks list — keep that behaviour so validators
        # (and the retry policy that consumes their report) actually fire.
        return self._compose_hooks(self._validation_hook)

    def _get_retry_policy(self):
        if self._validation_hook is None:
            from datus.agent.node.retry_policy import NoRetryPolicy

            return NoRetryPolicy()
        validation_cfg = getattr(self.agent_config, "validation_config", None)
        max_retries = int(getattr(validation_cfg, "max_retries", 3)) if validation_cfg else 3
        return ValidationHookRetryPolicy(
            hook=self._validation_hook,
            max_attempts=max_retries,
            node_name=self.get_node_name(),
        )

    def _build_success_result(self, ctx: StreamRunContext) -> Any:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            if isinstance(raw_output, dict) or raw_output:
                response_content = raw_output
            else:
                response_content = str(ctx.last_successful_output)

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        # ``ctx.extras`` populated by ``_run_stream_loop`` above.
        validation_report = ctx.extras.get("validation_report")
        blocked = bool(ctx.extras.get("blocked"))
        success = not blocked

        return self._make_success_result(
            success=success,
            response_content=response_content,
            tokens_used=int(tokens_used),
            validation_report=validation_report,
            last_successful_output=ctx.last_successful_output,
            blocked=blocked,
        )

    def _build_error_result(self, exc: BaseException, ctx: StreamRunContext) -> Any:
        # Deliverable subclasses (gen_dashboard / scheduler) override
        # ``_make_error_result`` to return their typed NodeResult.
        return self._make_error_result(
            error=self._format_execution_error(exc),
            action_history_manager=ctx.action_history_manager,
        )

    # ── result construction hooks ─────────────────────────────────────
    #
    # Subclasses with their own *NodeResult schema (gen_dashboard →
    # GenDashboardNodeResult, scheduler → SchedulerNodeResult) override these
    # two hooks. The default returns a :class:`SemanticNodeResult` so gen_table
    # and gen_job keep working unchanged.

    def _make_success_result(
        self,
        *,
        success: bool,
        response_content: Any,
        tokens_used: int,
        validation_report: Optional[dict],
        last_successful_output: Optional[dict],
        blocked: bool,
    ) -> Any:
        """Build the ``NodeResult`` returned after the stream completes."""
        return SemanticNodeResult(
            success=success,
            response=response_content if response_content else ("Validation failed" if blocked else ""),
            semantic_models=[],
            tokens_used=tokens_used,
            error=None if success else ("Validation blocked the run" if blocked else None),
            validation_report=validation_report,
        )

    def _make_error_result(
        self,
        *,
        error: str,
        action_history_manager: ActionHistoryManager,
    ) -> Any:
        """Build the ``NodeResult`` returned when the stream raises."""
        return SemanticNodeResult(
            success=False,
            error=error,
            response="Sorry, I encountered an error while processing your request.",
            tokens_used=0,
        )

    def _build_enhanced_message(self, user_input: Any) -> str:
        """Enrich the user message with catalog / database / schema context.

        Uses ``getattr`` so subclasses with narrower Input schemas (e.g.
        ``GenDashboardNodeInput`` omits ``catalog`` / ``db_schema``) still work.
        """
        from datus.utils.node_utils import resolve_database_name_for_prompt

        enhanced_parts = []
        # Per-turn datasource/dialect line — the frozen system prompt never
        # carries the current selection. The reminder already merges
        # catalog/database/schema, superseding the legacy Context line; nodes
        # without a DB tool (gen_dashboard, scheduler) get an empty reminder
        # and keep the legacy fallback below.
        datasource_reminder = self._build_datasource_reminder(user_input)
        if datasource_reminder:
            enhanced_parts.append(datasource_reminder)
        else:
            catalog = getattr(user_input, "catalog", None)
            db_schema = getattr(user_input, "db_schema", None)
            database_raw = getattr(user_input, "database", None) or ""
            effective_db = resolve_database_name_for_prompt(
                self.db_func_tool.connector if self.db_func_tool else None,
                database_raw,
            )
            if catalog or effective_db or db_schema:
                context_parts = []
                if catalog:
                    context_parts.append(f"catalog: {catalog}")
                if effective_db:
                    context_parts.append(f"database: {effective_db}")
                if db_schema:
                    context_parts.append(f"schema: {db_schema}")
                enhanced_parts.append(f"Context: {', '.join(context_parts)}")

        if enhanced_parts:
            enhanced_context = "\n\n".join(enhanced_parts)
            return build_structured_content(enhanced_context, user_input.user_message)
        return user_input.user_message
