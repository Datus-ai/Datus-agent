# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic runtime validation for delivered scheduler jobs."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable
from datus.validation.report import (
    CheckResult,
    SchedulerJobTarget,
    SessionTarget,
    ValidationReport,
    describe_target,
)

logger = get_logger(__name__)

SCHEDULER_RUNTIME_TIMEOUT_SECONDS = 300
SCHEDULER_RUNTIME_POLL_INTERVAL_SECONDS = 5
_SCHEDULER_RUN_SUCCESS = {"success", "succeeded"}
_SCHEDULER_RUN_FAILED = {"failed", "error", "killed", "upstream_failed"}


@optional_traceable(name="scheduler_runtime_validation", run_type="chain")
async def run_scheduler_runtime_validation(
    session: SessionTarget,
    scheduler_tool: Optional[Any],
    *,
    timeout_seconds: float = SCHEDULER_RUNTIME_TIMEOUT_SECONDS,
    poll_interval_seconds: float = SCHEDULER_RUNTIME_POLL_INTERVAL_SECONDS,
) -> ValidationReport:
    """Trigger delivered scheduler jobs once and require the triggered run to pass.

    This is a deterministic ValidationHook step, not an LLM validator skill.
    The hook calls it after Layer A proves the job exists and before Layer B
    scheduler-validation receives the merged precheck context.
    """
    report = ValidationReport(target=session, checks=[])
    scheduler_targets = [target for target in session.targets if isinstance(target, SchedulerJobTarget)]
    if not scheduler_targets:
        return report
    if scheduler_tool is None:
        logger.info("Scheduler runtime validation skipped: no scheduler tool available")
        return report

    for target in scheduler_targets:
        check = await asyncio.to_thread(
            _trigger_and_poll_job,
            target,
            scheduler_tool,
            timeout_seconds,
            poll_interval_seconds,
        )
        observed = dict(check.observed) if check.observed else {}
        observed["_target"] = describe_target(target)
        check.observed = observed
        report.checks.append(check)

    return report


def _trigger_and_poll_job(
    target: SchedulerJobTarget,
    scheduler_tool: Any,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> CheckResult:
    if not hasattr(scheduler_tool, "trigger_scheduler_job") or not hasattr(scheduler_tool, "list_job_runs"):
        return _runtime_check(
            passed=False,
            observed={"job_id": target.job_id},
            expected={"trigger_scheduler_job": True, "list_job_runs": True},
            error="scheduler tool does not expose trigger/list run APIs",
        )

    try:
        trigger_result = scheduler_tool.trigger_scheduler_job(job_id=target.job_id)
    except Exception as exc:
        return _runtime_check(
            passed=False,
            observed={"job_id": target.job_id},
            error=f"trigger_scheduler_job raised: {exc}",
        )

    if not getattr(trigger_result, "success", False):
        return _runtime_check(
            passed=False,
            observed={"job_id": target.job_id},
            error=getattr(trigger_result, "error", "trigger_scheduler_job failed"),
        )

    trigger_payload = getattr(trigger_result, "result", None) or {}
    run_id = trigger_payload.get("run_id") if isinstance(trigger_payload, dict) else None
    initial_status = _normalize_status(trigger_payload.get("status") if isinstance(trigger_payload, dict) else None)
    observed: dict[str, Any] = {
        "job_id": target.job_id,
        "run_id": run_id,
        "initial_status": initial_status,
        "final_status": initial_status,
        "polls": 0,
    }
    if not run_id:
        observed["trigger_result"] = trigger_payload
        return _runtime_check(
            passed=False,
            observed=observed,
            expected={"run_id": "non-empty"},
            error="trigger_scheduler_job did not return a run_id",
        )

    if initial_status in _SCHEDULER_RUN_SUCCESS:
        return _runtime_check(passed=True, observed=observed)
    if initial_status in _SCHEDULER_RUN_FAILED:
        _attach_run_log(target, scheduler_tool, run_id, observed)
        return _runtime_check(passed=False, observed=observed)

    deadline = time.monotonic() + timeout_seconds
    final_status = initial_status
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if poll_interval_seconds > 0:
            time.sleep(min(poll_interval_seconds, remaining))
        observed["polls"] += 1

        try:
            runs_result = scheduler_tool.list_job_runs(job_id=target.job_id, limit=10)
        except Exception as exc:
            return _runtime_check(
                passed=False,
                observed=observed,
                error=f"list_job_runs raised while polling {run_id}: {exc}",
            )

        if not getattr(runs_result, "success", False):
            return _runtime_check(
                passed=False,
                observed=observed,
                error=getattr(runs_result, "error", f"list_job_runs failed while polling {run_id}"),
            )

        run = _find_run(getattr(runs_result, "result", None), run_id)
        if run is None:
            observed["last_poll_missing_run"] = True
            continue

        final_status = _normalize_status(run.get("status"))
        observed["final_status"] = final_status
        if final_status in _SCHEDULER_RUN_SUCCESS:
            return _runtime_check(passed=True, observed=observed)
        if final_status in _SCHEDULER_RUN_FAILED:
            _attach_run_log(target, scheduler_tool, run_id, observed)
            return _runtime_check(passed=False, observed=observed)

    observed["final_status"] = final_status
    observed["timed_out"] = True
    _attach_run_log(target, scheduler_tool, run_id, observed)
    return _runtime_check(
        passed=False,
        observed=observed,
        error=f"scheduler run {run_id} did not finish within {timeout_seconds:g}s",
    )


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _find_run(result_payload: Any, run_id: str) -> Optional[dict[str, Any]]:
    if not isinstance(result_payload, dict):
        return None
    items = result_payload.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("run_id") == run_id:
            return item
    return None


def _attach_run_log(
    target: SchedulerJobTarget,
    scheduler_tool: Any,
    run_id: str,
    observed: dict[str, Any],
) -> None:
    if not hasattr(scheduler_tool, "get_run_log"):
        return
    try:
        log_result = scheduler_tool.get_run_log(job_id=target.job_id, run_id=run_id)
    except Exception as exc:
        observed["log_error"] = f"get_run_log raised: {exc}"
        return
    if not getattr(log_result, "success", False):
        observed["log_error"] = getattr(log_result, "error", "get_run_log failed")
        return
    payload = getattr(log_result, "result", None) or {}
    log_text = payload.get("log") if isinstance(payload, dict) else None
    if log_text:
        observed["log_excerpt"] = str(log_text)[-2000:]


def _runtime_check(
    *,
    passed: bool,
    observed: dict[str, Any],
    expected: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> CheckResult:
    final_status = observed.get("final_status")
    return CheckResult(
        name="scheduler_job_trigger_run",
        passed=passed,
        severity="blocking",
        source="builtin",
        observed=observed,
        expected=expected or {"final_status": "success"},
        error=error if error is not None else (None if passed else f"scheduler run status={final_status}"),
    )
