# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for the post-deliverable ValidationHook (``integration/validation.md``).

ValidationHook is the guardrail that runs *after* a deliverable-producing
subagent finishes mutating: it collects each mutating tool's
``deliverable_target`` (``on_tool_end``) and, when the run ends (``on_end``),
runs Layer-A built-in checks (e.g. "the table actually exists") plus Layer-B
validator skills against the accumulated session. Before this suite the hook had
only unit coverage — the gen_* subagents had nightly e2e tests but none asserted
that the hook actually collected a target and validated it.

This drives ``gen_table`` (a ``DeliverableNode``) end to end with a real LLM
against an ISOLATED SQLite copy, then inspects the hook's own public state
(``session_targets`` / ``final_report``) to prove the deliverable was collected
and validated — i.e. the guardrail ran, not just the generation.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from datus.agent.node.deliverable_node import ValidationHookRetryPolicy
from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.tools.db_tools import db_manager as _db_manager
from datus.utils.loggings import get_logger
from datus.validation import ValidationHook
from datus.validation.report import ValidationReport

logger = get_logger(__name__)


def _table_names(db_path: Path) -> set[str]:
    con = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    finally:
        con.close()


def _isolate_shared_sqlite(nightly_agent_config, tmp_dir: Path) -> Path:
    """Copy the shared benchmark SQLite file and repoint every datasource at the
    writable copy so the gen_table DDL never mutates the shared fixture (and
    concurrent nightly agents cannot race on it)."""
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

    _db_manager._cli_cache.clear()
    nightly_agent_config.current_datasource = "bird_school"
    return dst


@pytest.fixture
def isolated_table_config(nightly_agent_config, tmp_path):
    db_path = _isolate_shared_sqlite(nightly_agent_config, tmp_path)
    yield nightly_agent_config, db_path
    _db_manager._cli_cache.clear()


@pytest.mark.nightly
class TestValidationHookWiring:
    """Deterministic (no-LLM) wiring checks for the ValidationHook on a DeliverableNode."""

    def test_validation_hook_attached_and_drives_retry(self, nightly_agent_config):
        node = GenTableAgenticNode(agent_config=nightly_agent_config, execution_mode="workflow")

        # The hook is wired in DeliverableNode.__init__ regardless of execution
        # mode, and the node's retry policy is driven by it.
        assert isinstance(node._validation_hook, ValidationHook), "DeliverableNode should attach a ValidationHook"
        assert node._validation_hook.node_name == "gen_table"
        assert isinstance(node._get_retry_policy(), ValidationHookRetryPolicy), (
            "gen_table should use the ValidationHook-driven retry policy"
        )
        # Fresh node: no targets collected and no report yet.
        assert node._validation_hook.session_targets == []
        assert node._validation_hook.final_report is None


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestValidationHookRealLLM:
    """Real-LLM e2e: the hook collects the created-table target and validates it."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_created_table_is_collected_and_validated(self, isolated_table_config):
        config, db_path = isolated_table_config
        tables_before = _table_names(db_path)

        node = GenTableAgenticNode(agent_config=config, execution_mode="workflow")
        node.input = SemanticNodeInput(
            user_message=(
                "Create a new summary table named nightly_validation_counts that stores, "
                "per county, the number of schools. Read it from the schools table "
                "(group by the County column) using a CREATE TABLE ... AS SELECT statement "
                "executed with the execute_ddl tool. This is an explicit, non-destructive create."
            ),
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info("Action: role=%s status=%s type=%s", action.role, action.status, action.action_type)

        # Generation succeeded and a real table landed in the isolated copy.
        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
        after_tables = _table_names(db_path)
        new_tables = after_tables - tables_before
        # The prompt explicitly names the table, so assert that name was created
        # (case-insensitive — identifier casing can vary), not just "any" table.
        assert "nightly_validation_counts" in {t.lower() for t in after_tables}, (
            f"expected table 'nightly_validation_counts'; new_tables={new_tables}, after={after_tables}"
        )

        hook = node._validation_hook
        assert isinstance(hook, ValidationHook)

        # on_tool_end must have collected the execute_ddl deliverable target —
        # this is the signal that the guardrail observed the mutation.
        assert hook.session_targets, (
            "ValidationHook collected no deliverable_target; the execute_ddl path "
            "should report a table target via on_tool_end"
        )

        # on_end must have produced a report with at least one executed check.
        report = hook.final_report
        assert isinstance(report, ValidationReport), "ValidationHook.final_report should be populated after the run"
        assert report.checks, f"validation report has no checks: {report}"

        # The created table really exists, so the built-in table-existence check
        # must pass and the run must not carry a blocking validation failure.
        assert any(c.passed for c in report.checks), (
            f"expected at least one passing validation check, got: {[(c.name, c.passed) for c in report.checks]}"
        )
        assert not report.has_blocking_failure(), (
            f"validation reported a blocking failure for a table that was created: "
            f"{[(c.name, c.passed, c.error) for c in report.checks]}"
        )
