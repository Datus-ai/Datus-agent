# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SchedulerAgenticNode — job scheduling subagent.

Most plumbing (session management, validation retry loop, prompt rendering)
lives in :class:`DeliverableAgenticNode`. This subclass only declares the
scheduler-specific pieces: scheduler tool registration, workflow/validation
skill injection, and the domain result shape.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional

from datus.agent.node.deliverable_node import DeliverableAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.scheduler_agentic_node_models import SchedulerNodeResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchedulerAgenticNode(DeliverableAgenticNode):
    """Job scheduler subagent (Airflow, DolphinScheduler)."""

    NODE_NAME: ClassVar[str] = "scheduler"
    NODE_TYPE: ClassVar[str] = NodeType.TYPE_SCHEDULER
    ACTION_TYPE: ClassVar[str] = "scheduler_response"
    PROMPT_TEMPLATE: ClassVar[str] = "scheduler_system"
    DEFAULT_MAX_TURNS: ClassVar[int] = 30
    # Scheduler workflow + validation skills are injected dynamically based
    # on the configured platform. Leave ``DEFAULT_SKILLS`` empty so the
    # base-class fallback does not over-expose unrelated platforms.
    DEFAULT_SKILLS: ClassVar[Optional[str]] = None

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        scope: Optional[str] = None,
        is_subagent: bool = False,
    ):
        # Populated by ``_setup_domain_tools`` on success. Initialized here so
        # the attribute exists when the base ``setup_tools`` chain runs during
        # ``super().__init__``. Exposed to :class:`ValidationHook` via the
        # ``scheduler_func_tool`` attribute name that the hook looks for.
        self.scheduler_tools = None
        super().__init__(
            agent_config=agent_config,
            execution_mode=execution_mode,
            node_id=node_id,
            node_name=node_name,
            scope=scope,
            is_subagent=is_subagent,
        )
        # Scheduler service override resolved after base init so
        # ``self.node_config`` exists.
        self.scheduler_service = self.node_config.get("scheduler_service")

    # ── expose scheduler_tools as the canonical attribute for ValidationHook ─

    @property
    def scheduler_func_tool(self):
        """ValidationHook reads ``scheduler_func_tool`` to dispatch Layer A
        scheduler_job checks. Alias for :attr:`scheduler_tools`.
        """
        return self.scheduler_tools

    # ── tool setup ────────────────────────────────────────────────────

    def setup_tools(self):
        """Setup scheduler, filesystem, and optional ask_user tools."""
        if not self.agent_config:
            return
        self.tools = []
        self._setup_domain_tools()
        self._setup_filesystem_tools()
        self._setup_scheduler_skills()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        logger.debug("Setup %d tools for %s", len(self.tools), self.NODE_NAME)

    def _setup_domain_tools(self) -> None:
        """Register :class:`SchedulerTools` if a scheduler service is configured."""
        if not getattr(self.agent_config, "scheduler_services", None):
            return
        try:
            from datus.tools.func_tool.scheduler_tools import SchedulerTools

            # Resolve service config early to fail fast on mis-configuration.
            scheduler_service = getattr(self, "scheduler_service", None) or (
                self.node_config.get("scheduler_service") if self.node_config else None
            )
            self.agent_config.get_scheduler_config(scheduler_service)
            self.scheduler_tools = SchedulerTools(
                self.agent_config,
                scheduler_service=scheduler_service,
            )
            self.tools.extend(self.scheduler_tools.available_tools())
            logger.info("Scheduler tools initialized")
        except ImportError as e:
            logger.warning("Scheduler adapter package not installed: %s", e)
        except Exception as e:
            logger.error("Failed to setup scheduler tools: %s", e)

    def _setup_scheduler_skills(self):
        """Inject only the platform-specific scheduler workflow skill.

        Platform-specific workflow skills (e.g. ``airflow-workflow``) assume
        the scheduler tools are present; we only inject them when
        ``_setup_domain_tools`` succeeded. Scheduler validation is handled by
        :class:`ValidationHook` through the validator registry.
        """
        if not self.scheduler_tools:
            return
        scheduler_service = getattr(self, "scheduler_service", None) or (
            self.node_config.get("scheduler_service") if self.node_config else None
        )
        scheduler_config = self.agent_config.get_scheduler_config(scheduler_service)
        platform = scheduler_config.get("type", "airflow")

        skills_to_inject = [f"{platform}-workflow"]
        self.node_config["skills"] = self._merge_skill_patterns(
            self.node_config.get("skills"),
            skills_to_inject,
        )
        self._setup_skill_func_tools()

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Register scheduler tools so ``scheduler_tools.delete_job`` DENY fires."""
        mapping = super()._tool_category_map()
        if self.scheduler_tools:
            mapping["scheduler_tools"] = list(self.scheduler_tools.available_tools())
        if getattr(self, "filesystem_func_tool", None):
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping

    # ── template context ──────────────────────────────────────────────

    def _prepare_template_context(self, user_input: Any = None) -> dict:
        """Scheduler-specific template context."""
        return {
            "native_tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "None",
            "has_ask_user_tool": self.ask_user_tool is not None,
        }

    # ── result construction ────────────────────────────────────────────

    def _make_success_result(
        self,
        *,
        success: bool,
        response_content: Any,
        tokens_used: int,
        validation_report: Optional[dict],
        last_successful_output: Optional[dict],
        blocked: bool,
    ) -> SchedulerNodeResult:
        structured_result = None
        if last_successful_output and isinstance(last_successful_output, dict):
            result_data = last_successful_output.get("result")
            if isinstance(result_data, dict):
                structured_result = result_data

        response_str = ""
        if response_content:
            response_str = response_content if isinstance(response_content, str) else str(response_content)
        elif blocked:
            response_str = "Validation failed"

        return SchedulerNodeResult(
            success=success,
            response=response_str,
            scheduler_result=structured_result,
            tokens_used=tokens_used,
            error=None if success else ("Validation blocked the run" if blocked else None),
            validation_report=validation_report,
        )

    def _make_error_result(
        self,
        *,
        error: str,
        action_history_manager: ActionHistoryManager,
    ) -> SchedulerNodeResult:
        partial = self._collect_submitted_jobs(action_history_manager)
        return SchedulerNodeResult(
            success=False,
            error=error,
            response="Sorry, I encountered an error while processing your request.",
            scheduler_result=partial if partial else None,
            tokens_used=0,
        )

    # ── partial-job collection (used on error) ────────────────────────

    @staticmethod
    def _collect_submitted_jobs(action_history_manager: ActionHistoryManager) -> dict:
        """Scan action history for jobs submitted before a failure."""
        jobs: list = []
        job_tools = {"submit_sql_job", "submit_sparksql_job"}

        for action in action_history_manager.get_actions():
            if action.status != ActionStatus.SUCCESS:
                continue
            if action.action_type not in job_tools:
                continue
            output = action.output
            if not isinstance(output, dict):
                continue
            result = output.get("result")
            if not isinstance(result, dict):
                continue
            job_id = result.get("job_id")
            if job_id:
                jobs.append(
                    {
                        "job_id": job_id,
                        "job_name": result.get("job_name", ""),
                        "status": result.get("status", ""),
                    }
                )
        return {"submitted_jobs": jobs} if jobs else {}
