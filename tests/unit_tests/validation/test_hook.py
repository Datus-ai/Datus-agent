# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.validation.hook.ValidationHook`."""

from __future__ import annotations

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_registry import SkillRegistry
from datus.validation import ChartTarget, DBRef, TableTarget, TransferTarget, ValidationHook


class FakeDBFuncTool:
    """Mock DBFuncTool with configurable describe/count behavior."""

    def __init__(self, exists=True, rows=5, skip_count=False):
        self.exists = exists
        self.rows = rows
        self.read_query_called = False
        self._skip_count = skip_count

    def describe_table(self, table_name, catalog="", database="", schema_name="", datasource=""):
        # ``datasource`` must be accepted so _run_describe_table's routing
        # keyword doesn't fall through to an unexpected-kwarg error.
        self.describe_datasource_arg = datasource
        self.describe_database_arg = database
        if self.exists:
            return FuncToolResult(result={"columns": [{"name": "id", "type": "int"}]})
        return FuncToolResult(success=0, error="not found")

    def read_query(self, sql, database="", datasource=""):
        self.read_query_called = True
        return FuncToolResult(result={"rows": [{"c": self.rows}]})

    def _get_connector(self, db):
        class C:
            pass

        c = C()
        c.skip_expensive_count_check = self._skip_count
        return c


class FakeToolResult:
    def __init__(self, payload):
        self.result = payload


def _make_hook(db_func_tool, skill_validators_enabled=False):
    reg = SkillRegistry(config=SkillConfig(directories=["/nonexistent-for-test"]))
    return ValidationHook(
        node_name="gen_table",
        registry=reg,
        model=None,
        db_func_tool=db_func_tool,
        skill_validators_enabled=skill_validators_enabled,
    )


class TestOnToolEnd:
    """on_tool_end only collects targets — Layer A runs at on_end."""

    @pytest.mark.asyncio
    async def test_happy_path_table_target(self):
        hook = _make_hook(FakeDBFuncTool(exists=True, rows=3))
        hook.reset_session()
        tgt = TableTarget(database="db1", db_schema="public", table="users").model_dump(
            by_alias=True, exclude_none=True
        )
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert len(hook.session_targets) == 1

    @pytest.mark.asyncio
    async def test_dashboard_scoped_chart_replaces_standalone_chart_target(self):
        hook = _make_hook(FakeDBFuncTool(exists=True, rows=3))
        hook.reset_session()

        standalone = ChartTarget(platform="superset", chart_id="5").model_dump(exclude_none=True)
        scoped = ChartTarget(platform="superset", chart_id="5", dashboard_id="42").model_dump(exclude_none=True)

        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": standalone}))
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": scoped}))

        assert len(hook.session_targets) == 1
        assert isinstance(hook.session_targets[0], ChartTarget)
        assert hook.session_targets[0].dashboard_id == "42"

    @pytest.mark.asyncio
    async def test_same_chart_on_different_dashboards_keeps_both_targets(self):
        hook = _make_hook(FakeDBFuncTool(exists=True, rows=3))
        hook.reset_session()

        first = ChartTarget(platform="superset", chart_id="5", dashboard_id="42").model_dump(exclude_none=True)
        second = ChartTarget(platform="superset", chart_id="5", dashboard_id="43").model_dump(exclude_none=True)

        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": first}))
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": second}))

        assert len(hook.session_targets) == 2
        assert {t.dashboard_id for t in hook.session_targets if isinstance(t, ChartTarget)} == {"42", "43"}

    @pytest.mark.asyncio
    async def test_on_tool_end_never_runs_layer_a(self):
        """Even when the table would fail Layer A (describe returns empty),
        on_tool_end must not raise and must not touch the DB. A-class runs
        exclusively at on_end in the unified model."""

        class RecordingFakeDB(FakeDBFuncTool):
            def __init__(self):
                super().__init__(exists=False)
                self.describe_called = False

            def describe_table(self, *args, **kwargs):
                self.describe_called = True
                return super().describe_table(*args, **kwargs)

        f = RecordingFakeDB()
        hook = _make_hook(f)
        hook.reset_session()
        # Three targets, any of which would fail A if A ran here
        for table in ("a", "b", "c"):
            tgt = TableTarget(database="db1", table=table).model_dump(by_alias=True, exclude_none=True)
            # Must not raise
            await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert len(hook.session_targets) == 3
        assert f.describe_called is False, "on_tool_end must not invoke describe_table"
        assert f.read_query_called is False

    @pytest.mark.asyncio
    async def test_empty_ctas_does_not_block(self):
        """Empty table after CREATE TABLE is a legitimate pattern — on_tool_end
        must not raise and must not issue any DB query."""
        f = FakeDBFuncTool(exists=True, rows=0)
        hook = _make_hook(f)
        hook.reset_session()
        tgt = TableTarget(database="db1", table="empty").model_dump(by_alias=True, exclude_none=True)
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert f.read_query_called is False

    @pytest.mark.asyncio
    async def test_non_mutating_tool_result_skipped(self):
        """Tool result without deliverable_target → hook does nothing."""
        hook = _make_hook(FakeDBFuncTool())
        hook.reset_session()
        await hook.on_tool_end(None, None, None, FakeToolResult({"message": "just a read"}))
        assert hook.session_targets == []

    @pytest.mark.asyncio
    async def test_malformed_target_ignored(self):
        """Malformed deliverable_target payload must not raise."""
        hook = _make_hook(FakeDBFuncTool())
        hook.reset_session()
        # Missing required fields
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": {"type": "table"}}))
        assert hook.session_targets == []


class TestOnEnd:
    @pytest.mark.asyncio
    async def test_on_end_with_no_targets_emits_empty_report(self):
        hook = _make_hook(FakeDBFuncTool(exists=True))
        hook.reset_session()
        await hook.on_end(None, None, None)
        assert hook.final_report is not None
        assert hook.final_report.checks == []

    @pytest.mark.asyncio
    async def test_on_end_aggregates_session(self):
        hook = _make_hook(FakeDBFuncTool(exists=True, rows=5))
        hook.reset_session()
        for table in ("t1", "t2"):
            tgt = TableTarget(database="d", table=table, rows_affected=10).model_dump(by_alias=True, exclude_none=True)
            await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        await hook.on_end(None, None, None)
        assert hook.final_report is not None
        # Each target contributes at least an existence check
        assert len(hook.final_report.checks) >= 2

    @pytest.mark.asyncio
    async def test_on_end_blocking_failure_recorded_not_raised(self):
        """on_end does not raise — it records to final_report for execute_stream."""
        hook = _make_hook(FakeDBFuncTool(exists=False))
        hook.reset_session()
        hook._session_targets.append(TableTarget(database="d", table="missing"))
        await hook.on_end(None, None, None)
        # Did not raise
        assert hook.final_report is not None
        assert hook.final_report.has_blocking_failure()

    @pytest.mark.asyncio
    async def test_on_end_missing_table_recorded_not_raised(self):
        """End-to-end: on_tool_end appends, on_end runs Layer A, failure ends
        up in final_report."""
        hook = _make_hook(FakeDBFuncTool(exists=False))
        hook.reset_session()
        tgt = TableTarget(database="db1", table="missing").model_dump(by_alias=True, exclude_none=True)
        # on_tool_end must NOT raise now — it only collects
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        # on_end runs Layer A and records the failure
        await hook.on_end(None, None, None)
        assert hook.final_report is not None
        assert hook.final_report.has_blocking_failure()
        assert any(c.name == "table_exists" and not c.passed for c in hook.final_report.checks)

    @pytest.mark.asyncio
    async def test_on_end_transfer_parity_mismatch_recorded(self):
        """Transfer row-count mismatch surfaces at on_end (not on_tool_end)."""
        hook = _make_hook(FakeDBFuncTool(exists=True))
        hook.reset_session()
        tgt = TransferTarget(
            source=DBRef(name="pg"),
            target=TableTarget(database="ch", table="f"),
            source_row_count=100,
            transferred_row_count=50,
        ).model_dump(by_alias=True, exclude_none=True)
        # on_tool_end collects without raising
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert len(hook.session_targets) == 1
        # on_end finds the parity mismatch
        await hook.on_end(None, None, None)
        assert hook.final_report is not None
        assert hook.final_report.has_blocking_failure()
        assert any(c.name == "transfer_row_count_parity" and not c.passed for c in hook.final_report.checks)

    @pytest.mark.asyncio
    async def test_on_end_routes_describe_via_datasource(self):
        """Cross-datasource writes must validate against the same connector
        the tool wrote through. Layer A forwards ``target.datasource`` as the
        ``datasource`` kwarg of ``describe_table`` — without it the default
        connector would be used and the table would look missing (P1-2)."""
        f = FakeDBFuncTool(exists=True)
        hook = _make_hook(f)
        hook.reset_session()
        tgt = TableTarget(datasource="ch_prod", database="ch_prod", table="rev").model_dump(
            by_alias=True, exclude_none=True
        )
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        await hook.on_end(None, None, None)
        assert f.describe_datasource_arg == "ch_prod"


class TestRunLayerB:
    """``_run_layer_b`` is the enabled-validator path. Happy / error paths
    are exercised here with a monkeypatched ``run_llm_validator`` so we
    don't need a real LLM or registry."""

    class _FakeSkillEntry:
        def __init__(self, name="v1", severity="blocking", targets=None):
            self.name = name
            self.severity = severity
            self.targets = targets or []
            self.content = "body"

    class _FakeRegistry:
        def __init__(self, skills=None, raise_on_get=None):
            self._skills = skills or []
            self._raise = raise_on_get

        def get_validators(self, node_name=None, node_class=None):
            if self._raise:
                raise self._raise
            return list(self._skills)

        def load_skill_content(self, name):
            return "body"

    def _make_hook_with_registry(self, registry, skill_validators_enabled=True):
        from datus.validation import ValidationHook

        return ValidationHook(
            node_name="gen_table",
            registry=registry,
            model=None,
            db_func_tool=FakeDBFuncTool(),
            skill_validators_enabled=skill_validators_enabled,
        )

    @pytest.mark.asyncio
    async def test_run_llm_validator_output_merged(self, monkeypatch):
        """``run_llm_validator`` result is merged with the right source tag."""
        from datus.validation.report import CheckResult, ValidationReport

        async def fake_runner(**kwargs):
            return ValidationReport(
                target=kwargs["target"],
                checks=[CheckResult(name="v_check", passed=False, severity="blocking", source="skill:v1")],
            )

        monkeypatch.setattr("datus.validation.hook.run_llm_validator", fake_runner)
        hook = self._make_hook_with_registry(self._FakeRegistry(skills=[self._FakeSkillEntry()]))
        hook.reset_session()
        hook._session_targets.append(TableTarget(database="d", table="t"))
        await hook.on_end(None, None, None)
        assert hook.final_report.has_blocking_failure()
        assert any(c.name == "v_check" for c in hook.final_report.checks)

    @pytest.mark.asyncio
    async def test_skill_severity_off_skipped(self, monkeypatch):
        """A validator skill declared ``severity: off`` must not invoke the
        runner — short-circuit before any LLM cost."""
        called = {"n": 0}

        async def fake_runner(**kwargs):
            called["n"] += 1
            from datus.validation.report import ValidationReport

            return ValidationReport(target=kwargs["target"], checks=[])

        monkeypatch.setattr("datus.validation.hook.run_llm_validator", fake_runner)
        hook = self._make_hook_with_registry(self._FakeRegistry(skills=[self._FakeSkillEntry(severity="off")]))
        hook.reset_session()
        hook._session_targets.append(TableTarget(database="d", table="t"))
        await hook.on_end(None, None, None)
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_registry_get_validators_error_recorded_as_warning(self, monkeypatch):
        """Registry failure during get_validators → warning, not exception."""

        async def _should_not_be_called(**kwargs):
            raise AssertionError("runner should not run if registry failed")

        monkeypatch.setattr("datus.validation.hook.run_llm_validator", _should_not_be_called)
        hook = self._make_hook_with_registry(self._FakeRegistry(raise_on_get=RuntimeError("boom")))
        hook.reset_session()
        hook._session_targets.append(TableTarget(database="d", table="t"))
        await hook.on_end(None, None, None)
        assert any(w.get("type") == "registry_error" for w in hook.final_report.warnings)

    @pytest.mark.asyncio
    async def test_runner_exception_recorded_as_warning(self, monkeypatch):
        """Runner crash → warning entry, hook continues with other skills."""

        async def crash_runner(**kwargs):
            raise RuntimeError("upstream boom")

        monkeypatch.setattr("datus.validation.hook.run_llm_validator", crash_runner)
        hook = self._make_hook_with_registry(self._FakeRegistry(skills=[self._FakeSkillEntry()]))
        hook.reset_session()
        hook._session_targets.append(TableTarget(database="d", table="t"))
        await hook.on_end(None, None, None)
        assert any(w.get("type") == "validator_runner_error" for w in hook.final_report.warnings)

    @pytest.mark.asyncio
    async def test_target_filter_skips_non_matching_skills(self, monkeypatch):
        """Skill with ``targets`` filter only runs when target matches."""
        from datus.validation.report import TargetFilter

        async def fake_runner(**kwargs):
            from datus.validation.report import CheckResult, ValidationReport

            return ValidationReport(
                target=kwargs["target"],
                checks=[CheckResult(name="ran", passed=True, severity="advisory", source="skill:v1")],
            )

        monkeypatch.setattr("datus.validation.hook.run_llm_validator", fake_runner)
        skill = self._FakeSkillEntry(targets=[TargetFilter(type="transfer")])  # only transfer
        hook = self._make_hook_with_registry(self._FakeRegistry(skills=[skill]))
        hook.reset_session()
        hook._session_targets.append(TableTarget(database="d", table="t"))  # not a transfer
        await hook.on_end(None, None, None)
        # No ran check — target didn't match
        assert not any(c.name == "ran" for c in hook.final_report.checks)

    @pytest.mark.asyncio
    async def test_parent_session_forwarded_to_runner(self, monkeypatch):
        """Hook must pass its stored parent session into ``run_llm_validator``."""
        captured = {}

        async def fake_runner(**kwargs):
            captured.update(kwargs)
            from datus.validation.report import ValidationReport

            return ValidationReport(target=kwargs["target"], checks=[])

        monkeypatch.setattr("datus.validation.hook.run_llm_validator", fake_runner)
        hook = self._make_hook_with_registry(self._FakeRegistry(skills=[self._FakeSkillEntry()]))
        hook.reset_session()
        hook.set_parent_session("PARENT_SESSION_SENTINEL")
        hook._session_targets.append(TableTarget(database="d", table="t"))
        await hook.on_end(None, None, None)
        assert captured.get("parent_session") == "PARENT_SESSION_SENTINEL"


class TestResetSession:
    @pytest.mark.asyncio
    async def test_reset_clears_session_and_final_report(self):
        hook = _make_hook(FakeDBFuncTool(exists=True))
        tgt = TableTarget(database="d", table="t", rows_affected=1).model_dump(by_alias=True, exclude_none=True)
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert len(hook.session_targets) == 1
        await hook.on_end(None, None, None)
        assert hook.final_report is not None
        hook.reset_session()
        assert hook.session_targets == []
        assert hook.final_report is None


class TestSkillValidatorToggle:
    @pytest.mark.asyncio
    async def test_disabled_skips_layer_b_completely(self):
        """With skill_validators_enabled=False, Layer A still runs at end of
        stream but the validator registry is never queried."""
        hook = _make_hook(FakeDBFuncTool(exists=True), skill_validators_enabled=False)
        hook.reset_session()
        tgt = TableTarget(database="d", table="t", rows_affected=1).model_dump(by_alias=True, exclude_none=True)

        # Instrument registry to detect any call
        queried = []

        original = hook.registry.get_validators

        def spy(*args, **kwargs):
            queried.append((args, kwargs))
            return original(*args, **kwargs)

        hook.registry.get_validators = spy  # type: ignore[assignment]
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert queried == [], "get_validators should NOT be called when skill_validators_enabled is False"


class TestLayerAForBITargets:
    """ValidationHook must thread ``bi_tool`` through to Layer A so BI targets
    get ``dashboard_exists`` / ``chart_exists`` / ``dataset_exists`` checks."""

    class FakeBITool:
        def __init__(self, found=True):
            self.found = found

        def get_dashboard(self, dashboard_id):
            if self.found:
                return FuncToolResult(result={"id": dashboard_id})
            return FuncToolResult(success=0, error="not found")

        def get_chart(self, chart_id, dashboard_id=None):
            return FuncToolResult(result={"id": chart_id}) if self.found else FuncToolResult(success=0, error="nf")

        def get_dataset(self, dataset_id, dashboard_id=None):
            return FuncToolResult(result={"id": dataset_id}) if self.found else FuncToolResult(success=0, error="nf")

    def _make_bi_hook(self, bi_tool, skill_validators_enabled=False):
        reg = SkillRegistry(config=SkillConfig(directories=["/nonexistent-for-test"]))
        return ValidationHook(
            node_name="gen_dashboard",
            registry=reg,
            model=None,
            db_func_tool=None,
            bi_tool=bi_tool,
            skill_validators_enabled=skill_validators_enabled,
        )

    @pytest.mark.asyncio
    async def test_dashboard_target_dispatches_via_bi_tool(self):
        from datus.validation import DashboardTarget  # noqa: F401 — discovers module

        bi = self.FakeBITool(found=True)
        hook = self._make_bi_hook(bi)
        hook.reset_session()
        payload = {
            "deliverable_target": {
                "type": "dashboard",
                "platform": "superset",
                "dashboard_id": "42",
            }
        }
        await hook.on_tool_end(None, None, None, FakeToolResult(payload))
        await hook.on_end(None, None, None)
        names = [c.name for c in hook.final_report.checks]
        assert "dashboard_exists" in names

    @pytest.mark.asyncio
    async def test_dashboard_missing_is_blocking(self):
        bi = self.FakeBITool(found=False)
        hook = self._make_bi_hook(bi)
        hook.reset_session()
        payload = {
            "deliverable_target": {
                "type": "dashboard",
                "platform": "superset",
                "dashboard_id": "42",
            }
        }
        await hook.on_tool_end(None, None, None, FakeToolResult(payload))
        await hook.on_end(None, None, None)
        assert hook.final_report.has_blocking_failure()


class TestLayerAForSchedulerTargets:
    """ValidationHook must thread ``scheduler_tool`` through to Layer A."""

    class FakeSchedulerTool:
        def __init__(self, found=True, status="active"):
            self.found = found
            self.status = status

        def get_scheduler_job(self, job_id):
            if self.found:
                return FuncToolResult(result={"found": True, "job_id": job_id, "status": self.status})
            return FuncToolResult(result={"found": False, "job_id": job_id})

    def _make_sched_hook(self, sched_tool, skill_validators_enabled=False):
        reg = SkillRegistry(config=SkillConfig(directories=["/nonexistent-for-test"]))
        return ValidationHook(
            node_name="scheduler",
            registry=reg,
            model=None,
            db_func_tool=None,
            scheduler_tool=sched_tool,
            skill_validators_enabled=skill_validators_enabled,
        )

    @pytest.mark.asyncio
    async def test_scheduler_job_target_dispatches(self):
        sched = self.FakeSchedulerTool(found=True, status="active")
        hook = self._make_sched_hook(sched)
        hook.reset_session()
        payload = {
            "deliverable_target": {
                "type": "scheduler_job",
                "platform": "airflow",
                "job_id": "j-1",
            }
        }
        await hook.on_tool_end(None, None, None, FakeToolResult(payload))
        await hook.on_end(None, None, None)
        names = [c.name for c in hook.final_report.checks]
        assert "scheduler_job_exists" in names

    @pytest.mark.asyncio
    async def test_scheduler_failed_status_is_blocking(self):
        sched = self.FakeSchedulerTool(found=True, status="failed")
        hook = self._make_sched_hook(sched)
        hook.reset_session()
        payload = {
            "deliverable_target": {
                "type": "scheduler_job",
                "platform": "airflow",
                "job_id": "j-1",
            }
        }
        await hook.on_tool_end(None, None, None, FakeToolResult(payload))
        await hook.on_end(None, None, None)
        assert hook.final_report.has_blocking_failure()
