# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Nightly integration tests for GenTableAgenticNode (issue #923).

Expands nightly coverage for the data-product ``gen_table`` subagent.

Two tiers:
* A deterministic ``@pytest.mark.nightly`` init test that builds the node and
  asserts the expected DDL/DB tool surface — no LLM call.
* A real-LLM ``@pytest.mark.product_e2e`` test that drives ``execute_stream``
  end to end. The LLM authors and executes a ``CREATE TABLE`` against an
  ISOLATED SQLite copy so the shared benchmark database is never mutated and
  concurrent nightly agents cannot race on the same file.

Isolation strategy: copy the shared ``california_schools.sqlite`` to a tmp dir,
repoint every datasource whose URI is that shared file (``bird_school`` plus the
glob-discovered ``california_schools``) to the copy, then clear the DBManager
CLI cache so no connector bound to the original path survives. This guarantees
the DDL lands on the copy regardless of which datasource name the LLM routes to.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.tools.db_tools import db_manager as _db_manager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _table_names(db_path: Path) -> set[str]:
    """Return the set of user table names in a SQLite file."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier so LLM-chosen table names are interpolated safely."""
    return '"' + name.replace('"', '""') + '"'


def _isolate_shared_sqlite(nightly_agent_config, tmp_dir: Path) -> Path:
    """Copy the shared benchmark SQLite file and repoint every datasource at it.

    The loaded acceptance config exposes the same physical
    ``california_schools.sqlite`` under several datasource names (``bird_school``
    plus the glob-discovered ``california_schools``). The LLM may route a DDL to
    any of them by name, so isolating only the selected datasource is not enough
    — a stray ``datasource='california_schools'`` would still hit the shared
    file and corrupt data shared with other nightly agents.

    This repoints *all* datasources whose URI is the shared file to a per-test
    writable copy, then clears the DBManager CLI cache so no connector bound to
    the original path survives. Returns the path to the writable copy.
    """
    base = nightly_agent_config.services.datasources["bird_school"]
    shared_uri = base.uri
    src = shared_uri.replace("sqlite:///", "")
    assert os.path.exists(src), f"source sqlite db not found: {src}"

    dst = tmp_dir / "california_schools.sqlite"
    shutil.copy2(src, dst)
    dst_uri = f"sqlite:///{dst}"

    repointed = 0
    for cfg in nightly_agent_config.services.datasources.values():
        if cfg.uri == shared_uri:
            cfg.uri = dst_uri
            repointed += 1
    assert repointed >= 1, "expected at least one datasource pointing at the shared sqlite file"

    # Drop any DBManager cached against the original path so connectors rebind
    # to the copy on next use.
    _db_manager._cli_cache.clear()

    nightly_agent_config.current_datasource = "bird_school"
    return dst


@pytest.fixture
def isolated_table_config(nightly_agent_config, tmp_path):
    """nightly_agent_config repointed at a per-test writable SQLite copy.

    Yields ``(config, db_path)`` so tests can both drive the node and
    introspect the isolated database after the run.
    """
    db_path = _isolate_shared_sqlite(nightly_agent_config, tmp_path)
    yield nightly_agent_config, db_path
    # Evict the copy-bound DBManager so the next test rebuilds against its own
    # path instead of reusing a connector pinned to this (now-deleted) tmp file.
    _db_manager._cli_cache.clear()


@pytest.mark.nightly
class TestGenTableAgenticInit:
    """Deterministic node-construction coverage (no LLM)."""

    def test_node_initialization(self, nightly_agent_config):
        """Node initializes with the expected DDL + DB tool surface."""
        node = GenTableAgenticNode(
            agent_config=nightly_agent_config,
            execution_mode="workflow",
        )

        assert node.get_node_name() == "gen_table", f"unexpected node name: {node.get_node_name()}"
        assert node.NODE_NAME == "gen_table", f"unexpected NODE_NAME: {node.NODE_NAME}"
        assert node.ACTION_TYPE == "gen_table_response", f"unexpected ACTION_TYPE: {node.ACTION_TYPE}"
        assert node.execution_mode == "workflow", f"unexpected execution mode: {node.execution_mode}"

        tool_names = [tool.name for tool in node.tools]
        assert "execute_ddl" in tool_names, f"missing execute_ddl, got: {tool_names}"
        assert "list_tables" in tool_names, f"missing list_tables, got: {tool_names}"
        assert "describe_table" in tool_names, f"missing describe_table, got: {tool_names}"
        assert "read_query" in tool_names, f"missing read_query, got: {tool_names}"

        logger.info("gen_table node initialized with %d tools: %s", len(node.tools), tool_names)


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestGenTableAgenticRealLLM:
    """Real-LLM smoke for the gen_table DDL path on an isolated SQLite copy."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_execute_stream_creates_table(self, isolated_table_config):
        """LLM authors + executes CREATE TABLE; last action is SUCCESS."""
        config, db_path = isolated_table_config
        tables_before = _table_names(db_path)

        node = GenTableAgenticNode(
            agent_config=config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(
            user_message=(
                "Create a new summary table named nightly_school_counts that stores, "
                "per county, the number of schools. Read it from the schools table "
                "(group by the County column). Use a CREATE TABLE ... AS SELECT statement "
                "and execute it with the execute_ddl tool. This is an explicit, "
                "non-destructive create request."
            ),
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info("Action: role=%s status=%s type=%s", action.role, action.status, action.action_type)

        assert len(actions) >= 2, f"expected at least 2 actions, got {len(actions)}"
        assert actions[0].role == ActionRole.USER, f"first action role was {actions[0].role}"
        assert actions[0].status == ActionStatus.PROCESSING, f"first action status was {actions[0].status}"

        last = actions[-1]
        assert last.status == ActionStatus.SUCCESS, f"last action should be SUCCESS, got {last.status}: {last.output}"
        assert last.action_type == "gen_table_response", (
            f"last action_type should be gen_table_response, got {last.action_type}"
        )

        # The DDL must have landed in the ISOLATED copy: exactly the new table(s)
        # the run created appear, and at least one is non-empty/queryable. The
        # assertion is name-agnostic so it stays robust to the LLM choosing a
        # slightly different identifier while still proving a real CREATE TABLE ran.
        tables_after = _table_names(db_path)
        new_tables = tables_after - tables_before
        assert new_tables, f"no new table created in isolated db; before={tables_before}, after={tables_after}"

        created = sorted(new_tables)[0]
        query_result = node.db_func_tool.read_query(f"SELECT COUNT(*) AS n FROM {_quote_ident(created)}")
        assert query_result.success == 1, f"created table {created!r} not queryable: {query_result}"
