# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_bi_subagents`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from datus.cli.bootstrap_bi_subagents import (
    build_sql_comment_lines,
    build_sub_agent_name,
    clean_comment_text,
    dedupe_values,
    ensure_file_name,
    normalize_identifier,
    parse_subject_path_for_metrics,
    qualify_table_names,
    stream_bi_save_subagents,
    write_chart_sql_files,
    write_metrics_csv,
)
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.tools.bi_tools.dashboard_assembler import SelectedSqlCandidate

# ─────────────────────────────────────────────────────────────────
# normalize_identifier / build_sub_agent_name
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sales Overview", "sales_overview"),
        ("  multi   whitespace  ", "multi_whitespace"),
        ("销量看板", "销量看板"),
        ("MixedCJK与English", "mixedcjk_与_english"),
        ("", "fb"),
        ("@@@***", "fb"),
    ],
)
def test_normalize_identifier_handles_ascii_and_cjk(raw: str, expected: str) -> None:
    assert normalize_identifier(raw, fallback="fb") == expected


def test_normalize_identifier_max_words_truncates() -> None:
    assert normalize_identifier("one two three four", max_words=2) == "one_two"


def test_build_sub_agent_name_combines_platform_and_dashboard() -> None:
    assert build_sub_agent_name("Superset", "Quarterly Sales Overview") == "superset_quarterly_sales_overview"


def test_build_sub_agent_name_falls_back_when_input_blank() -> None:
    name = build_sub_agent_name("", "")
    assert name == "bi_dashboard"
    assert name[0].isalpha()


def test_build_sub_agent_name_prefixes_when_starts_with_digit() -> None:
    # Digit-only platform produces a leading-digit token; helper prepends ``dashboard_``.
    assert build_sub_agent_name("123", "456").startswith("dashboard_")


# ─────────────────────────────────────────────────────────────────
# parse_subject_path_for_metrics
# ─────────────────────────────────────────────────────────────────


def test_parse_subject_path_for_metrics_extracts_dotted_path() -> None:
    tags = ["unrelated", "subject_tree:superset/sales/q1"]
    assert parse_subject_path_for_metrics(tags) == "superset.sales.q1"


def test_parse_subject_path_for_metrics_returns_none_when_missing() -> None:
    assert parse_subject_path_for_metrics([]) is None
    assert parse_subject_path_for_metrics(["other:tag"]) is None


# ─────────────────────────────────────────────────────────────────
# dedupe_values / clean_comment_text
# ─────────────────────────────────────────────────────────────────


def test_dedupe_values_strips_and_preserves_order() -> None:
    assert dedupe_values([" a", "b", "a", "", "c", " b "]) == ["a", "b", "c"]


def test_clean_comment_text_collapses_whitespace_and_newlines() -> None:
    assert clean_comment_text("foo\n\tbar  baz") == "foo bar baz"


# ─────────────────────────────────────────────────────────────────
# qualify_table_names
# ─────────────────────────────────────────────────────────────────


def test_qualify_table_names_fills_missing_qualifiers() -> None:
    agent_config = SimpleNamespace(db_type="duckdb")
    out = qualify_table_names(
        ["orders", "shop.orders"],
        agent_config,
        catalog="",
        database="prod",
        schema="public",
    )
    # duckdb dialect uses database.schema.table; ``orders`` gets prefixed,
    # ``shop.orders`` keeps its own schema and only gets the database filled in.
    assert out == ["prod.public.orders", "prod.shop.orders"]


def test_qualify_table_names_skips_blank_entries() -> None:
    agent_config = SimpleNamespace(db_type="duckdb")
    assert qualify_table_names(["", "  "], agent_config, catalog="", database="d", schema="s") == []


# ─────────────────────────────────────────────────────────────────
# ensure_file_name / write_chart_sql_files / write_metrics_csv
# ─────────────────────────────────────────────────────────────────


def _agent_config_with_path(tmp_path: Path) -> SimpleNamespace:
    """Build a minimal AgentConfig stub exposing the path_manager API used by these helpers."""
    pm = SimpleNamespace(dashboard_path=lambda: tmp_path)
    return SimpleNamespace(path_manager=pm)


def test_ensure_file_name_creates_platform_subdir(tmp_path: Path) -> None:
    cfg = _agent_config_with_path(tmp_path)
    out = ensure_file_name(cfg, "superset", "Sales Overview")
    assert out.parent == tmp_path / "superset"
    assert out.parent.exists()
    assert out.suffix == ".sql"
    assert "superset_sales_overview" in out.name


def test_write_chart_sql_files_groups_by_chart_and_writes_comments(tmp_path: Path) -> None:
    cfg = _agent_config_with_path(tmp_path)
    sqls = [
        SelectedSqlCandidate(chart_id=1, chart_name="A", description="desc", sql="select 1"),
        SelectedSqlCandidate(chart_id=1, chart_name="A", description=None, sql="select 2;"),
        SelectedSqlCandidate(chart_id=2, chart_name="B", description=None, sql="select 3"),
    ]
    out = write_chart_sql_files(sqls, platform="superset", dashboard_name="Sales", agent_config=cfg)
    text = out.read_text(encoding="utf-8")
    # Comment + trailing semicolon normalization.
    assert "-- Dashboard=Sales;" in text
    assert "select 1;" in text
    assert "select 2;" in text  # already had trailing semicolon, preserved
    assert "select 3;" in text  # trailing semicolon added


def test_write_chart_sql_files_returns_none_when_empty(tmp_path: Path) -> None:
    cfg = _agent_config_with_path(tmp_path)
    assert write_chart_sql_files([], platform="x", dashboard_name="y", agent_config=cfg) is None


def test_write_metrics_csv_creates_question_sql_columns(tmp_path: Path) -> None:
    cfg = _agent_config_with_path(tmp_path)
    sqls = [
        SelectedSqlCandidate(chart_id=10, chart_name="orders", description="day", sql="select count(*)"),
        SelectedSqlCandidate(chart_id=11, chart_name="users", description=None, sql="select 1"),
    ]
    out = write_metrics_csv(sqls, platform="superset", dashboard_name="Sales", agent_config=cfg)
    assert out.suffix == ".csv"
    df = pd.read_csv(out)
    assert list(df.columns) == ["question", "sql"]
    assert "Dashboard=Sales" in df.iloc[0]["question"]
    assert "Description=day" in df.iloc[0]["question"]


def test_write_metrics_csv_is_idempotent(tmp_path: Path) -> None:
    cfg = _agent_config_with_path(tmp_path)
    out = ensure_file_name(cfg, "superset", "Sales", suffix=".csv")
    out.write_text("preserved", encoding="utf-8")
    sqls = [SelectedSqlCandidate(chart_id=1, chart_name="x", description=None, sql="select 1")]
    again = write_metrics_csv(sqls, platform="superset", dashboard_name="Sales", agent_config=cfg)
    # Existing file is untouched (semantic_model and metrics share the same file).
    assert again.read_text(encoding="utf-8") == "preserved"


def test_build_sql_comment_lines_skips_description_when_blank() -> None:
    item = SelectedSqlCandidate(chart_id=1, chart_name="A", description=None, sql="select 1")
    lines = build_sql_comment_lines(item, "Dashboard")
    assert "-- Dashboard=Dashboard;" in lines
    assert all("Description=" not in line for line in lines)


# ─────────────────────────────────────────────────────────────────
# stream_bi_save_subagents
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_bi_save_subagents_persists_two_configs() -> None:
    saved: list[SubAgentConfig] = []

    manager = MagicMock()
    manager.save_agent.side_effect = lambda cfg, previous_name=None: saved.append(cfg) or {}
    manager.list_agents.return_value = {
        "chat": {},
        "superset_sales": {},
        "superset_sales_attribution": {},
    }

    cli = SimpleNamespace(available_subagents={"existing"})
    agent_config = SimpleNamespace(agentic_nodes={})

    actions = []
    async for a in stream_bi_save_subagents(
        agent_config,
        sub_agent_name="superset_sales",
        description="Sales overview",
        scoped_context=ScopedContext(tables="t"),
        sub_agent_manager=manager,
        cli_ref=cli,
    ):
        actions.append(a)

    # outer + 2 persist_yaml + subagent_complete + terminal == 5 actions
    assert len(actions) == 5
    persist_actions = [a for a in actions if a.action_type == "persist_yaml"]
    assert len(persist_actions) == 2
    assert all(a.status == ActionStatus.SUCCESS.value for a in persist_actions)
    assert {cfg.system_prompt for cfg in saved} == {"superset_sales", "superset_sales_attribution"}

    # Terminal SUCCESS, agentic_nodes refreshed, cli set updated.
    assert actions[-1].status == ActionStatus.SUCCESS.value
    assert "superset_sales" in agent_config.agentic_nodes
    assert "chat" not in cli.available_subagents
    assert "superset_sales" in cli.available_subagents


@pytest.mark.asyncio
async def test_stream_bi_save_subagents_marks_failure_when_persist_raises() -> None:
    def _raise(_cfg, previous_name=None):
        raise RuntimeError("disk full")

    manager = MagicMock()
    manager.save_agent.side_effect = _raise
    manager.list_agents.return_value = {}

    actions = []
    async for a in stream_bi_save_subagents(
        SimpleNamespace(agentic_nodes={}),
        sub_agent_name="super_agent",
        description="d",
        scoped_context=ScopedContext(),
        sub_agent_manager=manager,
        cli_ref=None,
    ):
        actions.append(a)

    persist_actions = [a for a in actions if a.action_type == "persist_yaml"]
    assert len(persist_actions) == 2
    assert all(a.status == ActionStatus.FAILED.value for a in persist_actions)
    assert all("disk full" in a.messages for a in persist_actions)
    assert actions[-1].status == ActionStatus.FAILED.value


@pytest.mark.asyncio
async def test_stream_bi_save_subagents_tolerates_list_agents_failure() -> None:
    manager = MagicMock()
    manager.save_agent.return_value = {}
    manager.list_agents.side_effect = RuntimeError("config locked")

    cli = SimpleNamespace(available_subagents={"existing"})
    agent_config = SimpleNamespace(agentic_nodes={"prior": {}})

    actions = [
        a
        async for a in stream_bi_save_subagents(
            agent_config,
            sub_agent_name="superset_x",
            description="d",
            scoped_context=ScopedContext(),
            sub_agent_manager=manager,
            cli_ref=cli,
        )
    ]
    # Persist actions still SUCCESS; list_agents failure is swallowed.
    assert all(a.status == ActionStatus.SUCCESS.value for a in actions if a.action_type == "persist_yaml")
    assert agent_config.agentic_nodes == {"prior": {}}  # unchanged
    assert cli.available_subagents == {"existing"}  # unchanged
    assert actions[-1].status == ActionStatus.SUCCESS.value


# Smoke: assistant role on outer task message.
@pytest.mark.asyncio
async def test_stream_bi_save_subagents_outer_role_is_tool() -> None:
    manager = MagicMock()
    manager.save_agent.return_value = {}
    manager.list_agents.return_value = {}
    actions = [
        a
        async for a in stream_bi_save_subagents(
            SimpleNamespace(agentic_nodes={}),
            sub_agent_name="x",
            description="d",
            scoped_context=ScopedContext(),
            sub_agent_manager=manager,
            cli_ref=None,
        )
    ]
    outer = actions[0]
    assert outer.action_type == "task"
    assert outer.role == ActionRole.TOOL.value
    assert outer.input["type"] == "save_subagents"
