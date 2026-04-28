# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for deterministic scheduler runtime validation."""

from __future__ import annotations

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.validation.report import SchedulerJobTarget, SessionTarget
from datus.validation.scheduler_runtime import run_scheduler_runtime_validation


class FakeSchedulerTool:
    def __init__(
        self,
        *,
        trigger_success: bool = True,
        trigger_status: str = "queued",
        poll_statuses: list[str] | None = None,
        run_id: str = "run-1",
    ):
        self.trigger_success = trigger_success
        self.trigger_status = trigger_status
        self.poll_statuses = list(poll_statuses or ["success"])
        self.run_id = run_id
        self.calls: list[tuple] = []

    def trigger_scheduler_job(self, job_id):
        self.calls.append(("trigger_scheduler_job", job_id))
        if not self.trigger_success:
            return FuncToolResult(success=0, error="trigger failed")
        return FuncToolResult(result={"job_id": job_id, "run_id": self.run_id, "status": self.trigger_status})

    def list_job_runs(self, job_id, limit=10):
        self.calls.append(("list_job_runs", job_id, limit))
        status = self.poll_statuses.pop(0) if self.poll_statuses else "running"
        return FuncToolResult(
            result={
                "items": [{"job_id": job_id, "run_id": self.run_id, "status": status}],
                "total": 1,
                "has_more": False,
            }
        )

    def get_run_log(self, job_id, run_id):
        self.calls.append(("get_run_log", job_id, run_id))
        return FuncToolResult(result={"job_id": job_id, "run_id": run_id, "log": "task failed\nstack"})


def _session() -> SessionTarget:
    return SessionTarget(targets=[SchedulerJobTarget(platform="airflow", job_id="j-1")])


@pytest.mark.asyncio
async def test_triggered_run_success_passes():
    tool = FakeSchedulerTool(trigger_status="queued", poll_statuses=["running", "success"])

    report = await run_scheduler_runtime_validation(_session(), tool, timeout_seconds=1, poll_interval_seconds=0)

    check = report.checks[0]
    assert check.name == "scheduler_job_trigger_run"
    assert check.passed is True
    assert check.observed["final_status"] == "success"
    assert check.observed["_target"] == "scheduler_job airflow:j-1"
    assert ("trigger_scheduler_job", "j-1") in tool.calls


@pytest.mark.asyncio
async def test_trigger_failure_blocks():
    tool = FakeSchedulerTool(trigger_success=False)

    report = await run_scheduler_runtime_validation(_session(), tool, timeout_seconds=1, poll_interval_seconds=0)

    check = report.checks[0]
    assert check.passed is False
    assert check.severity == "blocking"
    assert "trigger failed" in (check.error or "")


@pytest.mark.asyncio
async def test_failed_run_blocks_with_log_excerpt():
    tool = FakeSchedulerTool(trigger_status="queued", poll_statuses=["failed"])

    report = await run_scheduler_runtime_validation(_session(), tool, timeout_seconds=1, poll_interval_seconds=0)

    check = report.checks[0]
    assert check.passed is False
    assert check.observed["final_status"] == "failed"
    assert check.observed["log_excerpt"] == "task failed\nstack"
    assert ("get_run_log", "j-1", "run-1") in tool.calls


@pytest.mark.asyncio
async def test_timeout_blocks_and_fetches_log_when_available():
    tool = FakeSchedulerTool(trigger_status="queued", poll_statuses=["running"])

    report = await run_scheduler_runtime_validation(_session(), tool, timeout_seconds=0, poll_interval_seconds=0)

    check = report.checks[0]
    assert check.passed is False
    assert check.observed["timed_out"] is True
    assert "did not finish within 0s" in (check.error or "")
    assert check.observed["log_excerpt"] == "task failed\nstack"


@pytest.mark.asyncio
async def test_missing_scheduler_tool_skips():
    report = await run_scheduler_runtime_validation(_session(), None, timeout_seconds=1, poll_interval_seconds=0)

    assert report.checks == []
