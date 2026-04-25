# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.validation.builtin_checks`` — Layer A dispatches.

Focus on the new BI / scheduler target dispatches added in Chunk 4. The
existing table / transfer paths are covered via ``test_hook.py`` at the
``on_end`` integration level.
"""

from __future__ import annotations

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.validation.builtin_checks import run_builtin_checks, run_session_builtin_checks
from datus.validation.report import (
    ChartTarget,
    DashboardTarget,
    DatasetTarget,
    SchedulerJobTarget,
    SessionTarget,
)


class FakeBITool:
    """Mock BIFuncTool with configurable get_* behavior."""

    def __init__(
        self,
        dashboard_found=True,
        chart_found=True,
        dataset_found=True,
        chart_dashboard_id=None,
    ):
        self.dashboard_found = dashboard_found
        self.chart_found = chart_found
        self.dataset_found = dataset_found
        self.chart_dashboard_id = chart_dashboard_id
        self.calls: list = []

    def get_dashboard(self, dashboard_id):
        self.calls.append(("get_dashboard", dashboard_id))
        if self.dashboard_found:
            return FuncToolResult(result={"id": dashboard_id, "title": "t"})
        return FuncToolResult(success=0, error="dashboard not found")

    def get_chart(self, chart_id, dashboard_id=None):
        self.calls.append(("get_chart", chart_id, dashboard_id))
        if self.chart_found:
            payload = {"id": chart_id}
            if self.chart_dashboard_id is not None:
                payload["dashboard_id"] = self.chart_dashboard_id
            return FuncToolResult(result=payload)
        return FuncToolResult(success=0, error="chart not found")

    def get_dataset(self, dataset_id, dashboard_id=None):
        self.calls.append(("get_dataset", dataset_id, dashboard_id))
        if self.dataset_found:
            return FuncToolResult(result={"id": dataset_id})
        return FuncToolResult(success=0, error="dataset not found")


class EmptyPayloadBITool(FakeBITool):
    """BI adapter returning an empty dict for an existing resource."""

    def get_dashboard(self, dashboard_id):
        self.calls.append(("get_dashboard", dashboard_id))
        return FuncToolResult(result={})


class FakeSchedulerTool:
    """Mock SchedulerTools with configurable get_scheduler_job behavior."""

    def __init__(self, found=True, status="active"):
        self.found = found
        self.status = status
        self.calls: list = []

    def get_scheduler_job(self, job_id):
        self.calls.append(("get_scheduler_job", job_id))
        if self.found:
            return FuncToolResult(result={"found": True, "job_id": job_id, "status": self.status})
        return FuncToolResult(result={"found": False, "job_id": job_id})


class TestCheckDashboard:
    @pytest.mark.asyncio
    async def test_found_passes(self):
        tool = FakeBITool(dashboard_found=True)
        target = DashboardTarget(platform="superset", dashboard_id="42")
        report = await run_builtin_checks(target, bi_tool=tool)
        assert any(c.name == "dashboard_exists" and c.passed for c in report.checks)
        assert ("get_dashboard", "42") in tool.calls

    @pytest.mark.asyncio
    async def test_empty_payload_still_counts_as_found_when_successful(self):
        tool = EmptyPayloadBITool()
        target = DashboardTarget(platform="superset", dashboard_id="42")
        report = await run_builtin_checks(target, bi_tool=tool)
        assert any(c.name == "dashboard_exists" and c.passed for c in report.checks)

    @pytest.mark.asyncio
    async def test_not_found_blocks(self):
        tool = FakeBITool(dashboard_found=False)
        target = DashboardTarget(platform="superset", dashboard_id="42")
        report = await run_builtin_checks(target, bi_tool=tool)
        failed = [c for c in report.checks if c.name == "dashboard_exists" and not c.passed]
        assert failed and failed[0].severity == "blocking"

    @pytest.mark.asyncio
    async def test_missing_bi_tool_skips(self):
        target = DashboardTarget(platform="superset", dashboard_id="42")
        report = await run_builtin_checks(target, bi_tool=None)
        assert report.checks == []


class TestCheckChart:
    @pytest.mark.asyncio
    async def test_found_passes(self):
        tool = FakeBITool(chart_found=True)
        target = ChartTarget(platform="grafana", chart_id="c1")
        report = await run_builtin_checks(target, bi_tool=tool)
        assert any(c.name == "chart_exists" and c.passed for c in report.checks)

    @pytest.mark.asyncio
    async def test_not_found_blocks(self):
        tool = FakeBITool(chart_found=False)
        target = ChartTarget(platform="grafana", chart_id="c1")
        report = await run_builtin_checks(target, bi_tool=tool)
        failed = [c for c in report.checks if c.name == "chart_exists" and not c.passed]
        assert failed and failed[0].severity == "blocking"

    @pytest.mark.asyncio
    async def test_dashboard_link_parity_advisory_when_mismatched(self):
        """When a ChartTarget declares dashboard_id, Layer A warns (advisory) if
        the stored chart.dashboard_id doesn't match."""
        tool = FakeBITool(chart_found=True, chart_dashboard_id="99")
        target = ChartTarget(platform="superset", chart_id="c1", dashboard_id="42")
        report = await run_builtin_checks(target, bi_tool=tool)
        mismatch = [c for c in report.checks if c.name == "chart_dashboard_link"]
        assert mismatch
        assert mismatch[0].severity == "advisory"
        assert mismatch[0].passed is False


class TestCheckDataset:
    @pytest.mark.asyncio
    async def test_found_passes(self):
        tool = FakeBITool(dataset_found=True)
        target = DatasetTarget(platform="superset", dataset_id="d1")
        report = await run_builtin_checks(target, bi_tool=tool)
        assert any(c.name == "dataset_exists" and c.passed for c in report.checks)
        assert ("get_dataset", "d1", None) in tool.calls

    @pytest.mark.asyncio
    async def test_not_found_blocks(self):
        tool = FakeBITool(dataset_found=False)
        target = DatasetTarget(platform="superset", dataset_id="d1")
        report = await run_builtin_checks(target, bi_tool=tool)
        failed = [c for c in report.checks if c.name == "dataset_exists" and not c.passed]
        assert failed and failed[0].severity == "blocking"


class TestCheckSchedulerJob:
    @pytest.mark.asyncio
    async def test_found_active_passes(self):
        tool = FakeSchedulerTool(found=True, status="active")
        target = SchedulerJobTarget(platform="airflow", job_id="j-1")
        report = await run_builtin_checks(target, scheduler_tool=tool)
        exists = [c for c in report.checks if c.name == "scheduler_job_exists"]
        status = [c for c in report.checks if c.name == "scheduler_job_status"]
        assert exists and exists[0].passed
        assert status and status[0].passed

    @pytest.mark.asyncio
    async def test_not_found_blocks(self):
        tool = FakeSchedulerTool(found=False)
        target = SchedulerJobTarget(platform="airflow", job_id="j-1")
        report = await run_builtin_checks(target, scheduler_tool=tool)
        failed = [c for c in report.checks if c.name == "scheduler_job_exists" and not c.passed]
        assert failed and failed[0].severity == "blocking"

    @pytest.mark.asyncio
    async def test_failed_status_blocks(self):
        tool = FakeSchedulerTool(found=True, status="failed")
        target = SchedulerJobTarget(platform="airflow", job_id="j-1")
        report = await run_builtin_checks(target, scheduler_tool=tool)
        status = [c for c in report.checks if c.name == "scheduler_job_status" and not c.passed]
        assert status and status[0].severity == "blocking"

    @pytest.mark.asyncio
    async def test_pending_status_advisory(self):
        """Pending is neither definitely good nor definitely bad — advisory."""
        tool = FakeSchedulerTool(found=True, status="pending")
        target = SchedulerJobTarget(platform="airflow", job_id="j-1")
        report = await run_builtin_checks(target, scheduler_tool=tool)
        status = [c for c in report.checks if c.name == "scheduler_job_status"]
        assert status
        # pending is allowed (passed=True) but with an advisory note in observed
        assert status[0].passed is True

    @pytest.mark.asyncio
    async def test_missing_scheduler_tool_skips(self):
        target = SchedulerJobTarget(platform="airflow", job_id="j-1")
        report = await run_builtin_checks(target, scheduler_tool=None)
        assert report.checks == []


class TestSessionTargetWithMixedTypes:
    @pytest.mark.asyncio
    async def test_session_dispatches_per_target_type(self):
        """SessionTarget recursion routes each inner target to the right tool."""
        bi = FakeBITool(dashboard_found=True, chart_found=True, dataset_found=True)
        sched = FakeSchedulerTool(found=True, status="active")
        session = SessionTarget(
            targets=[
                DashboardTarget(platform="superset", dashboard_id="42"),
                ChartTarget(platform="superset", chart_id="c1"),
                SchedulerJobTarget(platform="airflow", job_id="j-1"),
            ]
        )
        report = await run_session_builtin_checks(session, bi_tool=bi, scheduler_tool=sched)
        names = [c.name for c in report.checks]
        assert "dashboard_exists" in names
        assert "chart_exists" in names
        assert "scheduler_job_exists" in names
        # Checks must be tagged with their originating inner target
        assert all(c.observed and c.observed.get("_target") for c in report.checks)
