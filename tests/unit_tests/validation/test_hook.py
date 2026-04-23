# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.validation.hook.ValidationHook`."""

from __future__ import annotations

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_registry import SkillRegistry
from datus.validation import DBRef, TableTarget, TransferTarget, ValidationBlockingException, ValidationHook


class FakeDBFuncTool:
    """Mock DBFuncTool with configurable describe/count behavior."""

    def __init__(self, exists=True, rows=5, skip_count=False):
        self.exists = exists
        self.rows = rows
        self.read_query_called = False
        self._skip_count = skip_count

    def describe_table(self, table_name, catalog="", database="", schema_name=""):
        if self.exists:
            return FuncToolResult(result={"columns": [{"name": "id", "type": "int"}]})
        return FuncToolResult(success=0, error="not found")

    def read_query(self, sql, database=""):
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
    async def test_missing_table_raises_blocking(self):
        hook = _make_hook(FakeDBFuncTool(exists=False))
        hook.reset_session()
        tgt = TableTarget(database="db1", table="missing").model_dump(by_alias=True, exclude_none=True)
        with pytest.raises(ValidationBlockingException) as exc:
            await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert exc.value.report.has_blocking_failure()
        assert any(c.name == "table_exists" and not c.passed for c in exc.value.report.checks)

    @pytest.mark.asyncio
    async def test_empty_ctas_does_not_block(self):
        """Empty table after CREATE TABLE is a legitimate pattern — A class
        must NOT block it. Row-count expectations belong to validator skills."""
        f = FakeDBFuncTool(exists=True, rows=0)
        hook = _make_hook(f)
        hook.reset_session()
        tgt = TableTarget(database="db1", table="empty").model_dump(by_alias=True, exclude_none=True)
        # Must not raise
        await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        # And no COUNT(*) should have been issued — we dropped that check
        assert f.read_query_called is False

    @pytest.mark.asyncio
    async def test_transfer_row_count_mismatch_blocks(self):
        hook = _make_hook(FakeDBFuncTool(exists=True))
        hook.reset_session()
        tgt = TransferTarget(
            source=DBRef(name="pg"),
            target=TableTarget(database="ch", table="f"),
            source_row_count=100,
            transferred_row_count=50,
        ).model_dump(by_alias=True, exclude_none=True)
        with pytest.raises(ValidationBlockingException) as exc:
            await hook.on_tool_end(None, None, None, FakeToolResult({"deliverable_target": tgt}))
        assert any(c.name == "transfer_row_count_parity" and not c.passed for c in exc.value.report.checks)

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
        # The on_tool_end itself would raise — suppress to simulate a flow where
        # builtin checks pass per-tool but fail at session aggregation. Here we
        # skip on_tool_end and directly append to session.
        hook._session_targets.append(TableTarget(database="d", table="missing"))
        await hook.on_end(None, None, None)
        # Did not raise
        assert hook.final_report is not None
        assert hook.final_report.has_blocking_failure()


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
        """With skill_validators_enabled=False the registry is never queried."""
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
