# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared base class for table-producing subagents (``gen_table``, ``gen_job``).

Centralizes the ~85 % of boilerplate that used to be copy-pasted between the
two nodes: tool/filesystem/prompt setup, the stream loop, session handling,
and ValidationHook wiring (with retry loop).

Subclasses provide four class-level constants (:attr:`NODE_NAME`,
:attr:`DEFAULT_SKILLS`, :attr:`PROMPT_TEMPLATE`, :attr:`ACTION_TYPE`) and one
hook method :meth:`_setup_db_tools` — everything else is inherited.
"""

from __future__ import annotations

from typing import AsyncGenerator, ClassVar, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool import DBFuncTool, FilesystemFuncTool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content
from datus.validation import ValidationBlockingException, ValidationHook
from datus.validation.report import build_retry_prompt

logger = get_logger(__name__)


class TableDeliverableAgenticNode(AgenticNode):
    """Base class for subagents whose deliverable is a physical table.

    Subclasses must set the four class constants below and implement
    :meth:`_setup_db_tools`. Everything else (including the validation retry
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
    DEFAULT_MAX_TURNS: ClassVar[int] = 20

    # ── constructor ───────────────────────────────────────────────────

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        is_subagent: bool = False,
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
            description=f"Table-deliverable node: {self.NODE_NAME}",
            node_type=self.NODE_TYPE,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            is_subagent=is_subagent,
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
        self._setup_db_tools()
        self._setup_filesystem_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        logger.debug("Setup %d tools for %s: %s", len(self.tools), self.NODE_NAME, [t.name for t in self.tools])

    def _setup_db_tools(self) -> None:
        """Subclass-specific tool registration.

        gen_table registers only ``execute_ddl``; gen_job additionally registers
        ``execute_write``, ``transfer_query_result``, and the MigrationTargetMixin
        wrappers.
        """
        raise NotImplementedError("_setup_db_tools must be implemented by subclasses")

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
                skill_validators_enabled=enabled,
            )

            if enabled:
                node = self.get_node_name()
                klass = self.get_node_class_name()
                has_any = bool(
                    registry.get_validators(node, "on_tool_end", node_class=klass)
                    or registry.get_validators(node, "on_end", node_class=klass)
                )
                if not has_any:
                    logger.warning(
                        "No validator skills discovered for '%s'. Run `datus configure` (shell) "
                        "to deploy bundled skills (table-validation, migration-reconciliation) "
                        "into ~/.datus/skills, or author project-level validators under "
                        "./.datus/skills.",
                        node,
                    )
        except Exception as e:
            logger.error("Failed to setup ValidationHook: %s", e)
            self._validation_hook = None

    def _prepare_template_context(self, user_input: SemanticNodeInput) -> dict:
        from datus.utils.node_utils import build_datasource_prompt_context

        context = {
            "native_tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "None",
            "mcp_tools": ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None",
            "has_ask_user_tool": self.ask_user_tool is not None,
        }
        context.update(build_datasource_prompt_context(self.agent_config))
        logger.debug("Prepared template context: %s", context)
        return context

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        version = self.node_config.get("prompt_version")
        template_name = self.PROMPT_TEMPLATE or f"{self.NODE_NAME}_system"
        try:
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
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

    # ── main execution with validation retry ──────────────────────────

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()
        if self.input is None:
            raise DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message_args={"field_name": "input"})

        user_input = self.input

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
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                await self._auto_compact()
                session, conversation_summary = self._get_or_create_session()

            template_context = self._prepare_template_context(user_input)
            system_instruction = self._get_system_prompt(conversation_summary, template_context)

            enhanced_message = self._build_enhanced_message(user_input)

            # Retry loop around ValidationBlockingException. Capped at
            # ``agent.validation.max_retries`` (default 3). See design doc §5.7.
            validation_cfg = getattr(self.agent_config, "validation_config", None)
            max_retries = int(getattr(validation_cfg, "max_retries", 3)) if validation_cfg else 3
            if max_retries < 1:
                max_retries = 1
            if self._validation_hook is not None:
                self._validation_hook.reset_session()
                # Expose the parent session so Layer B validators can fork its
                # tool-event history (drop user/assistant text, keep tool calls
                # and results). See :func:`run_llm_validator`'s
                # ``parent_session`` parameter.
                self._validation_hook.set_parent_session(session)

            current_prompt = enhanced_message
            response_content = ""
            last_successful_output = None
            completed = False
            tokens_used = 0
            last_validation_report: Optional[dict] = None

            for attempt in range(1, max_retries + 1):
                # Reset per-attempt so only the final attempt's outcome sticks.
                # Without this a blocked attempt 1 would poison a recovered
                # attempt 2's NodeResult.success.
                last_validation_report = None
                try:
                    async for stream_action in self.model.generate_with_tools_stream(
                        prompt=current_prompt,
                        tools=self.tools,
                        mcp_servers=self.mcp_servers,
                        instruction=system_instruction,
                        max_turns=user_input.max_turns if user_input.max_turns else self.max_turns,
                        session=session,
                        action_history_manager=action_history_manager,
                        hooks=self._compose_hooks(self._validation_hook),
                        agent_name=self.get_node_name(),
                        interrupt_controller=self.interrupt_controller,
                    ):
                        yield stream_action
                        if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                            if isinstance(stream_action.output, dict):
                                last_successful_output = stream_action.output
                                raw_output = stream_action.output.get("raw_output", "")
                                if isinstance(raw_output, dict):
                                    response_content = raw_output
                                elif raw_output:
                                    response_content = raw_output

                    # Stream ended normally. Drive retry from on_end's accumulated
                    # report if it recorded a blocking failure — on_end records
                    # to final_report instead of raising (raising there would
                    # skip downstream hook chain for the completed run).
                    hook = self._validation_hook
                    if hook is not None and hook.final_report is not None and hook.final_report.has_blocking_failure():
                        last_validation_report = hook.final_report.model_dump(by_alias=True, exclude_none=True)
                        if attempt >= max_retries:
                            logger.warning(
                                "on_end validation blocked after %d attempts for %s: %s",
                                attempt,
                                self.get_node_name(),
                                [c.name for c in hook.final_report.checks if not c.passed],
                            )
                            break
                        current_prompt = build_retry_prompt(hook.final_report, list(hook.session_targets))
                        hook.reset_session()
                        logger.info(
                            "on_end validation blocked attempt %d/%d for %s, retrying with failure context",
                            attempt,
                            max_retries,
                            self.get_node_name(),
                        )
                        continue

                    completed = True
                    break
                except ValidationBlockingException as exc:
                    # Escape-hatch path: a skill declared trigger=[on_tool_end]
                    # and raised mid-stream. Feed the report back so the agent
                    # can fix and retry.
                    last_validation_report = exc.report.model_dump(by_alias=True, exclude_none=True)
                    if attempt >= max_retries:
                        logger.warning(
                            "Validation blocked after %d attempts for %s: %s",
                            attempt,
                            self.get_node_name(),
                            [c.name for c in exc.report.checks if not c.passed],
                        )
                        break
                    hook = self._validation_hook
                    session_snapshot = list(hook.session_targets) if hook is not None else []
                    current_prompt = build_retry_prompt(exc.report, session_snapshot)
                    if hook is not None:
                        hook.reset_session()
                    logger.info(
                        "Validation blocked attempt %d/%d for %s, retrying with failure context",
                        attempt,
                        max_retries,
                        self.get_node_name(),
                    )

            if not response_content and last_successful_output:
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = raw_output
                elif raw_output:
                    response_content = raw_output
                else:
                    response_content = str(last_successful_output)

            if self.execution_mode == "interactive":
                final_actions = action_history_manager.get_actions()
                for a in reversed(final_actions):
                    if a.role == "assistant" and a.output and isinstance(a.output, dict):
                        usage_info = a.output.get("usage", {})
                        if isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                            tokens_used = usage_info.get("total_tokens", 0)
                            if tokens_used > 0:
                                break

            # Merge on_end validation report (if any) with the blocking report
            # from a retry-exhausted attempt.
            final_validation: Optional[dict] = None
            if self._validation_hook is not None and self._validation_hook.final_report is not None:
                final_validation = self._validation_hook.final_report.model_dump(by_alias=True, exclude_none=True)
            if last_validation_report is not None:
                # Blocking failure takes precedence — surface that instead
                # of (or alongside) the on_end report.
                final_validation = last_validation_report

            success = completed and last_validation_report is None
            result = SemanticNodeResult(
                success=success,
                response=response_content
                if response_content
                else (last_validation_report and "Validation failed") or "",
                semantic_models=[],
                tokens_used=int(tokens_used),
                error=None if success else "Validation blocked the run" if last_validation_report else None,
                validation_report=final_validation,
            )
            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=self.ACTION_TYPE or f"{self.NODE_NAME}_response",
                messages=f"{self.get_node_name()} interaction completed {'successfully' if success else 'with validation failures'}",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS if success else ActionStatus.FAILED,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise
        except Exception as e:
            logger.error("%s execution error: %s", self.get_node_name(), e)
            error_result = SemanticNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )
            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{self.get_node_name()} interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

    def _build_enhanced_message(self, user_input: SemanticNodeInput) -> str:
        """Enrich the user message with catalog / database / schema context."""
        from datus.utils.node_utils import resolve_database_name_for_prompt

        enhanced_parts = []
        effective_db = resolve_database_name_for_prompt(
            self.db_func_tool.connector if self.db_func_tool else None,
            user_input.database or "",
        )
        if user_input.catalog or effective_db or user_input.db_schema:
            context_parts = []
            if user_input.catalog:
                context_parts.append(f"catalog: {user_input.catalog}")
            if effective_db:
                context_parts.append(f"database: {effective_db}")
            if user_input.db_schema:
                context_parts.append(f"schema: {user_input.db_schema}")
            enhanced_parts.append(f"Context: {', '.join(context_parts)}")

        if enhanced_parts:
            enhanced_context = "\n\n".join(enhanced_parts)
            return build_structured_content(
                [
                    MessagePart(type="enhanced", content=enhanced_context),
                    MessagePart(type="user", content=user_input.user_message),
                ]
            )
        return user_input.user_message
