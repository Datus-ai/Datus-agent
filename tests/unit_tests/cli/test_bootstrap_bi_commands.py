# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_bi_commands`.

We mock the picker, the four streams, and the sub-agent persistence
stream so the test covers the orchestration logic only: header
messages, semantic-model gating, ScopedContext assembly, and adapter
cleanup on every exit path.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bootstrap_bi_commands import BootstrapBiCommands
from datus.cli.bootstrap_bi_picker import BootstrapBiPlan, DashboardCliOptions
from datus.cli.bootstrap_bi_streams import BiBuildState
from datus.schemas.action_history import ActionHistory, ActionStatus
from datus.schemas.agent_models import ScopedContext

# ─────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120, log_path=False)


@pytest.fixture()
def agent_config() -> SimpleNamespace:
    return SimpleNamespace(
        db_type="duckdb",
        current_datasource="local",
        agentic_nodes={},
        path_manager=SimpleNamespace(),
        resolve_semantic_adapter=lambda x: x,
        current_db_config=lambda *_a, **_k: SimpleNamespace(catalog="cat", database="db", schema="sch"),
    )


def _plan(**overrides) -> BootstrapBiPlan:
    """Build a BootstrapBiPlan stub. Overrides update the dataclass fields."""
    base = dict(
        options=DashboardCliOptions(
            platform="superset",
            dashboard_url="http://x/d/1",
            api_base_url="http://x",
            auth_params=None,
            dialect="duckdb",
        ),
        adapter=MagicMock(),
        dashboard=SimpleNamespace(id=1, name="Sales", description="quarterly"),
        dashboard_id=1,
        chart_selections_ref=[MagicMock()],
        chart_selections_metrics=[MagicMock()],
        assembled=SimpleNamespace(
            tables=["orders"],
            reference_sqls=[MagicMock()],
            metric_sqls=[MagicMock()],
        ),
        pool_size=3,
    )
    base.update(overrides)
    return BootstrapBiPlan(**base)


async def _empty_stream(*_a, **_k) -> AsyncGenerator[ActionHistory, None]:
    return
    yield  # pragma: no cover


async def _streams_no_yield(*_a, **_k):
    """Stream that yields nothing — caller still iterates fine."""
    if False:
        yield  # pragma: no cover


# ─────────────────────────────────────────────────────────────────
# cmd happy path / cancel paths
# ─────────────────────────────────────────────────────────────────


def test_cmd_aborts_when_picker_returns_none(agent_config, console) -> None:
    cmd = BootstrapBiCommands(agent_config, console)
    with patch("datus.cli.bootstrap_bi_commands.BootstrapBiPicker") as picker_cls:
        picker_cls.return_value.run.return_value = None
        cmd.cmd()
    output = console.file.getvalue()
    assert "Cancelled" in output


def test_cmd_aborts_when_picker_raises(agent_config, console) -> None:
    cmd = BootstrapBiCommands(agent_config, console)
    with patch("datus.cli.bootstrap_bi_commands.BootstrapBiPicker") as picker_cls:
        picker_cls.return_value.run.side_effect = ValueError("no service")
        cmd.cmd()
    assert "no service" in console.file.getvalue()


def test_cmd_aborts_on_keyboard_interrupt_in_picker(agent_config, console) -> None:
    cmd = BootstrapBiCommands(agent_config, console)
    with patch("datus.cli.bootstrap_bi_commands.BootstrapBiPicker") as picker_cls:
        picker_cls.return_value.run.side_effect = KeyboardInterrupt
        cmd.cmd()
    assert "Cancelled" in console.file.getvalue()


def test_cmd_closes_adapter_after_full_run(agent_config, console) -> None:
    plan = _plan()
    cmd = BootstrapBiCommands(agent_config, console)

    with (
        patch("datus.cli.bootstrap_bi_commands.BootstrapBiPicker") as picker_cls,
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metrics", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=["cat.db.sch.orders"]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        picker_cls.return_value.run.return_value = plan
        cmd.cmd()

    plan.adapter.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# _run_plan orchestration (drive directly via asyncio)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_plan_skips_metrics_when_semantic_validation_fails(agent_config, console) -> None:
    plan = _plan()
    cmd = BootstrapBiCommands(agent_config, console)

    metric_calls: list = []

    async def _meta_stream(*_a, **_k):
        if False:
            yield  # pragma: no cover

    async def _ref_stream(*_a, **_k):
        if False:
            yield  # pragma: no cover

    async def _sem_stream(*_, state, **_k):
        # The semantic stream stays in its default state.semantic_ok=False.
        return
        yield  # pragma: no cover

    async def _metrics_stream(*_a, **_k):
        metric_calls.append(True)
        if False:
            yield  # pragma: no cover

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_meta_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_ref_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_sem_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metrics", side_effect=_metrics_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=["t"]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    assert metric_calls == []  # metrics stream never invoked
    assert any(a.status == ActionStatus.FAILED.value and "Skipping metrics" in a.messages for a in actions)


@pytest.mark.asyncio
async def test_run_plan_runs_metrics_when_semantic_ok_set(agent_config, console) -> None:
    plan = _plan()
    cmd = BootstrapBiCommands(agent_config, console)

    metric_calls: list = []

    async def _set_ok_stream(*_, state, **_k):
        state.semantic_ok = True
        if False:
            yield  # pragma: no cover

    async def _metrics_stream(*_, state, **_k):
        metric_calls.append(state.semantic_ok)
        if False:
            yield  # pragma: no cover

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_set_ok_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metrics", side_effect=_metrics_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=["t"]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    assert metric_calls == [True]


@pytest.mark.asyncio
async def test_run_plan_aborts_when_no_charts_selected(agent_config, console) -> None:
    plan = _plan(chart_selections_ref=[], chart_selections_metrics=[])
    cmd = BootstrapBiCommands(agent_config, console)

    actions: list[ActionHistory] = []
    with patch("datus.cli.bootstrap_bi_commands.qualify_table_names") as q:
        await cmd._run_plan(plan, actions)
        q.assert_not_called()  # never made it past the selection check

    assert any(a.status == ActionStatus.FAILED.value and "No charts selected" in a.messages for a in actions)


@pytest.mark.asyncio
async def test_run_plan_aborts_when_datasource_missing(console) -> None:
    cfg = SimpleNamespace(current_datasource="", db_type="duckdb")
    plan = _plan()
    cmd = BootstrapBiCommands(cfg, console)

    actions: list[ActionHistory] = []
    await cmd._run_plan(plan, actions)
    assert any(a.status == ActionStatus.FAILED.value and "datasource" in a.messages.lower() for a in actions)


@pytest.mark.asyncio
async def test_run_plan_skips_save_when_scoped_context_empty(agent_config, console) -> None:
    plan = _plan()
    cmd = BootstrapBiCommands(agent_config, console)
    save_calls: list = []

    async def _sem_stream(*_, state, **_k):
        state.semantic_ok = True
        if False:
            yield  # pragma: no cover

    async def _save_stream(*_a, **_k):
        save_calls.append(True)
        if False:
            yield  # pragma: no cover

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_sem_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metrics", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=_save_stream),
        # qualify_table_names returns empty so ScopedContext is also empty.
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=[]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    assert save_calls == []
    assert any("No scoped context" in a.messages for a in actions)


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


def test_build_scoped_context_returns_none_when_state_empty() -> None:
    out = BootstrapBiCommands._build_scoped_context(BiBuildState())
    assert out is None


def test_build_scoped_context_joins_lists_with_commas() -> None:
    state = BiBuildState(table_names=["a", "b"], ref_sqls=["s1"], metrics=["m1", "m2"])
    sc = BootstrapBiCommands._build_scoped_context(state)
    assert isinstance(sc, ScopedContext)
    assert sc.tables == "a,b"
    assert sc.sqls == "s1"
    assert sc.metrics == "m1,m2"


def test_resolve_default_table_context_uses_db_config_fallback(agent_config, console) -> None:
    cmd = BootstrapBiCommands(agent_config, console)
    cmd.cli = None  # no cli_context
    assert cmd._resolve_default_table_context() == ("cat", "db", "sch")
