# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
:class:`ValidationHook` — AgentHooks integration that fires after mutating tool
calls and at agent run end to enforce both Layer A (built-in invariants) and
Layer B (LLM-mode validator skills) for table-producing subagents.

Wire-up:

    hook = ValidationHook(
        node_name=self.get_node_name(),
        registry=skill_manager.registry,
        model=self.model,
        db_func_tool=self.db_func_tool,
        skill_validators_enabled=agent_config.validation_config.skill_validators_enabled,
    )
    # Chain with other hooks
    self.hooks = CompositeHooks([permission_hooks, hook, generation_hooks])

Blocking failures raise :class:`ValidationBlockingException`, which the owning
node's ``execute_stream`` catches to drive the retry loop. Callers must also
call ``hook.reset_session()`` at the start of each agent run and between
retries so ``_session_targets`` does not leak across runs (see design doc §5.7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from agents.lifecycle import AgentHooks

from datus.utils.loggings import get_logger
from datus.validation.builtin_checks import run_session_builtin_checks
from datus.validation.exceptions import ValidationBlockingException
from datus.validation.llm_runner import run_llm_validator
from datus.validation.report import (
    DeliverableTarget,
    SessionTarget,
    TableTarget,
    TransferTarget,
    ValidationReport,
    skill_matches_target,
)

if TYPE_CHECKING:
    from datus.tools.func_tool.database import DBFuncTool
    from datus.tools.skill_tools.skill_registry import SkillRegistry

logger = get_logger(__name__)


class ValidationHook(AgentHooks):
    """Drives post-mutation validation for table-producing subagents.

    One instance per agent-run-owning node. State (``_session_targets``,
    ``_final_report``) is per-run — callers must call :meth:`reset_session`
    at the start of each agent run and between retries.
    """

    def __init__(
        self,
        node_name: str,
        registry: "SkillRegistry",
        model: Any,
        db_func_tool: Optional["DBFuncTool"] = None,
        skill_validators_enabled: bool = True,
        node_class: Optional[str] = None,
    ):
        self.node_name = node_name
        self.node_class = node_class or node_name
        self.registry = registry
        self.model = model
        self.db_func_tool = db_func_tool
        self.skill_validators_enabled = skill_validators_enabled

        # Per-run state — reset() must be called at run boundaries.
        self._session_targets: List[DeliverableTarget] = []
        self._final_report: Optional[ValidationReport] = None
        # Parent agent's SDK session — supplied by the owning node via
        # :meth:`set_parent_session` before each stream attempt so Layer B
        # validators can fork tool-event history. ``None`` when the node
        # runs session-less (e.g. workflow mode) — validators fall back to
        # cold-start then.
        self._parent_session: Optional[Any] = None

    # ── public API ─────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Clear per-run state. Call at ``execute_stream`` entry AND between retries."""
        self._session_targets = []
        self._final_report = None

    def set_parent_session(self, session: Optional[Any]) -> None:
        """Update the parent-agent session reference used for Layer B forks.

        The node owns the session lifecycle; this is a weak reference for
        read-only item extraction in :func:`run_llm_validator`. Set to
        ``None`` when no session is active.
        """
        self._parent_session = session

    @property
    def final_report(self) -> Optional[ValidationReport]:
        """Report produced by the most recent ``on_end`` invocation."""
        return self._final_report

    @property
    def session_targets(self) -> List[DeliverableTarget]:
        """Read-only accessor for tests / callers."""
        return list(self._session_targets)

    # ── AgentHooks implementation ──────────────────────────────────────

    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Fires after every tool call. Appends the tool's self-reported
        ``deliverable_target`` to the session. Layer A + default Layer B
        validators run at ``on_end``; this method is the escape hatch for
        validator skills that explicitly declare ``trigger: [on_tool_end]``.
        """
        target = self._extract_target(result)
        if target is None:
            return
        self._session_targets.append(target)

        if self.skill_validators_enabled:
            combined = ValidationReport(target=target, checks=[])
            await self._run_layer_b(
                trigger="on_tool_end",
                target=target,
                combined=combined,
                precheck_context=None,
            )
            if combined.has_blocking_failure():
                raise ValidationBlockingException(combined)

    async def on_end(self, context, agent, output) -> None:
        """Fires when the agent run completes. Runs Layer A + Layer B against
        the accumulated session.
        """
        if not self._session_targets:
            self._final_report = ValidationReport(target=None, checks=[])
            return

        session = SessionTarget(targets=list(self._session_targets))
        combined = ValidationReport(target=session, checks=[])

        try:
            a_report = await run_session_builtin_checks(session, db_func_tool=self.db_func_tool)
            combined.checks.extend(a_report.checks)
            combined.warnings.extend(a_report.warnings)
        except Exception as e:
            logger.exception("Layer A session builtin_checks raised")
            combined.add_warning({"type": "builtin_checks_error", "error": str(e)})

        if combined.has_blocking_failure():
            # Record but don't raise here — on_end raising doesn't drive a retry
            # (the agent has already decided it's done). execute_stream reads
            # self.final_report after the run and acts on blocking failures
            # there.
            self._final_report = combined
            return

        if self.skill_validators_enabled:
            await self._run_layer_b(
                trigger="on_end",
                target=session,
                combined=combined,
                precheck_context=combined.model_copy(deep=True),
            )

        self._final_report = combined

    # ── helpers ────────────────────────────────────────────────────────

    def _extract_target(self, tool_result: Any) -> Optional[DeliverableTarget]:
        """Pull ``deliverable_target`` out of ``FuncToolResult.result``.

        Tolerates either a pydantic ``FuncToolResult`` instance or a plain dict
        (since the SDK may wrap tool return values).
        """
        payload = None
        if hasattr(tool_result, "result"):
            payload = getattr(tool_result, "result", None)
        elif isinstance(tool_result, dict):
            payload = tool_result.get("result")
        if not isinstance(payload, dict):
            return None
        target_dict = payload.get("deliverable_target")
        if not isinstance(target_dict, dict):
            return None

        target_type = target_dict.get("type")
        try:
            if target_type == "table":
                return TableTarget.model_validate(target_dict)
            if target_type == "transfer":
                return TransferTarget.model_validate(target_dict)
        except Exception as e:
            logger.warning("Failed to validate deliverable_target payload: %s", e)
            return None
        return None

    async def _run_layer_b(
        self,
        trigger: str,
        target: DeliverableTarget,
        combined: ValidationReport,
        precheck_context: Optional[ValidationReport],
    ) -> None:
        """Discover matching validator skills and merge their reports into ``combined``."""
        try:
            skills = self.registry.get_validators(
                node_name=self.node_name,
                trigger=trigger,
                node_class=self.node_class,
            )
        except Exception as e:
            logger.exception("Failed to query validator skills from registry")
            combined.add_warning({"type": "registry_error", "error": str(e)})
            return

        for skill in skills:
            if not skill_matches_target(skill.targets, target):
                continue
            if skill.severity == "off":
                continue
            try:
                per_skill = await run_llm_validator(
                    skill=skill,
                    registry=self.registry,
                    target=target,
                    model=self.model,
                    db_func_tool=self.db_func_tool,
                    precheck_context=precheck_context,
                    parent_session=self._parent_session,
                )
            except Exception as e:
                logger.exception("Validator skill '%s' raised unexpectedly", skill.name)
                combined.add_warning({"type": "validator_runner_error", "skill_name": skill.name, "error": str(e)})
                continue
            combined.merge(
                per_skill,
                source=f"skill:{skill.name}",
                severity_override=None if skill.severity == "blocking" else skill.severity,
            )
