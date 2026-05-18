# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Retry strategy driven by ``ValidationHook`` blocking failures.

Used by :class:`DeliverableAgenticNode` and its subclasses (gen_dashboard,
scheduler, gen_table, gen_job). After each stream completes, the policy
inspects the hook's ``final_report`` and reschedules with a context-aware
retry prompt when a blocking failure is recorded.

Per-run state (``_blocking_report``) is captured in :meth:`should_retry`
so :meth:`finalise` can stash it into ``ctx.extras`` for the node's
``_build_success_result`` hook to merge into the final NodeResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from datus.utils.loggings import get_logger
from datus.validation.report import build_retry_prompt

if TYPE_CHECKING:
    from datus.agent.node.stream_run_context import StreamRunContext
    from datus.schemas.action_history import ActionHistory
    from datus.validation.hook import ValidationHook

logger = get_logger(__name__)


class ValidationHookRetryPolicy:
    """Re-prompt the model when ``ValidationHook.final_report`` blocks."""

    def __init__(self, hook: "ValidationHook", max_attempts: int = 3, node_name: str = "deliverable"):
        if max_attempts < 1:
            max_attempts = 1
        self.hook = hook
        self.max_attempts = max_attempts
        self.node_name = node_name
        self._blocking_report: Optional[dict] = None

    def reset(self, ctx: "StreamRunContext") -> None:
        # Drop the last attempt's blocking report so a recovered retry does
        # not inherit a stale ``success=False`` decision.
        self._blocking_report = None

    def should_retry(self, ctx: "StreamRunContext") -> bool:
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

    def next_prompt(self, ctx: "StreamRunContext") -> Optional[str]:
        report = self.hook.final_report
        if report is None:
            return None
        try:
            prompt = build_retry_prompt(report, list(self.hook.session_targets))
        finally:
            # Reset the hook so the next attempt records a fresh report.
            self.hook.reset_session()
        return prompt

    def on_retry_actions(self, ctx: "StreamRunContext") -> Iterable["ActionHistory"]:
        # DeliverableAgenticNode does not surface a per-retry user-visible
        # action today — keep parity with the pre-refactor behaviour.
        return ()

    def finalise(self, ctx: "StreamRunContext") -> None:
        # Surface the (possibly blocking) report so the success builder can
        # decide between SUCCESS and "blocked" outcomes.
        report = self.hook.final_report
        on_end_report: Optional[dict] = None
        if report is not None:
            on_end_report = report.model_dump(by_alias=True, exclude_none=True)
        # Blocking failure (when retries exhaust) takes precedence over the
        # vanilla on_end report.
        final = self._blocking_report if self._blocking_report is not None else on_end_report
        ctx.extras["validation_report"] = final
        ctx.extras["blocked"] = self._blocking_report is not None
