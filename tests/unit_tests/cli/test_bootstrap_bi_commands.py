# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_bi_commands`.

The public slash command is a skill shortcut. Legacy ``_run_plan`` coverage is
kept below while that internal migration surface remains available.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bootstrap_bi_commands import _DASHBOARD_BOOTSTRAP_PROMPT, BootstrapBiCommands
from datus.cli.bootstrap_bi_picker import BootstrapBiPlan, DashboardCliOptions
from datus.cli.bootstrap_bi_streams import BiBuildState
from datus.cli.skill_command_utils import render_skill_prompt
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
        resolve_semantic_adapter=lambda _x: "dosi",
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
# skill-backed slash command
# ─────────────────────────────────────────────────────────────────


def _cli(agent_config, console, *, plan_mode: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        agent_config=agent_config,
        console=console,
        chat_commands=MagicMock(),
        plan_mode_active=plan_mode,
        configuration_manager=None,
    )


def test_cmd_delegates_to_dashboard_bootstrap_skill(agent_config, console) -> None:
    cli = _cli(agent_config, console)
    cmd = BootstrapBiCommands(cli)

    cmd.cmd()

    cli.chat_commands.execute_chat_command.assert_called_once_with(
        render_skill_prompt(_DASHBOARD_BOOTSTRAP_PROMPT, ""),
        plan_mode=False,
        subagent_name=None,
    )
    message = cli.chat_commands.execute_chat_command.call_args.args[0]
    assert 'load_skill(skill_name="dashboard-bootstrap")' in message
    assert "legacy bootstrap picker" in message
    assert "Let the skill decide" in message
    assert "or create dashboard-specific subagents" not in message


def test_cmd_forwards_user_context_and_plan_mode(agent_config, console) -> None:
    cli = _cli(agent_config, console, plan_mode=True)
    cmd = BootstrapBiCommands(cli)

    cmd.cmd("Superset profile prod, dashboard 42, run automatically")

    cli.chat_commands.execute_chat_command.assert_called_once_with(
        render_skill_prompt(
            _DASHBOARD_BOOTSTRAP_PROMPT,
            "Superset profile prod, dashboard 42, run automatically",
        ),
        plan_mode=True,
        subagent_name=None,
    )


def test_cmd_without_cli_reports_chat_requirement(agent_config, console) -> None:
    cmd = BootstrapBiCommands(agent_config, console)

    cmd.cmd()

    assert "relies on the chat pipeline" in console.file.getvalue()


# ─────────────────────────────────────────────────────────────────
# _run_plan orchestration (drive directly via asyncio)
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_plan_reports_unified_semantic_failure(agent_config, console) -> None:
    plan = _plan()
    cmd = BootstrapBiCommands(agent_config, console)

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

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_meta_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_ref_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_sem_stream),
        patch(
            "datus.cli.bootstrap_bi_commands.stream_bi_save_subagents",
            side_effect=_streams_no_yield,
        ) as save_subagents,
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=["t"]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    assert any(
        a.status == ActionStatus.FAILED.value and "Unified semantic modeling failed" in a.messages for a in actions
    )
    save_subagents.assert_not_called()
    assert not any("Sub-Agent build successful" in a.messages for a in actions)


@pytest.mark.asyncio
async def test_run_plan_uses_metrics_collected_by_semantic_modeling(agent_config, console) -> None:
    plan = _plan()
    cmd = BootstrapBiCommands(agent_config, console)

    async def _set_ok_stream(*_, state, **_k):
        state.semantic_ok = True
        state.metrics.append("sales.orders.total_orders")
        if False:
            yield  # pragma: no cover

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_set_ok_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=_streams_no_yield),
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=["t"]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    assert not any("Unified semantic modeling failed" in a.messages for a in actions)


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
