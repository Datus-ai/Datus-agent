# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_streams` helpers.

Focus: ``_run_helper_with_actions`` must forward native ActionHistory
emissions (depth=0 from ``action_callback``) alongside translated
BatchEvent markers, in arrival order. This is the load-bearing change
that exposes inner AgenticNode actions through the ``task(...)``
subagent group rather than collapsing them into BatchEvent counters.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.bootstrap_streams import _run_helper_with_actions, merge_streams
from datus.schemas.action_history import (
    ActionHistory,
    ActionRole,
    ActionStatus,
)
from datus.schemas.batch_events import BatchEvent, BatchStage


def _make_action(text: str, status: ActionStatus = ActionStatus.SUCCESS) -> ActionHistory:
    return ActionHistory.create_action(
        role=ActionRole.TOOL,
        action_type="probe",
        messages=text,
        input_data={"function_name": "probe"},
        status=status,
    )


@pytest.mark.asyncio
async def test_stream_metadata_uses_metadata_rag_factory() -> None:
    from datus.cli.bootstrap_streams import stream_metadata

    async def _ok(*_args, **_kwargs):
        return None

    agent_config = SimpleNamespace(current_datasource="", datasource_configs={"local": {}})
    metadata_store = MagicMock()
    with (
        patch("datus.storage.schema_metadata.create_metadata_rag", return_value=metadata_store) as create_metadata_rag,
        patch("datus.storage.schema_metadata.local_init.init_local_schema_async", side_effect=_ok),
        patch("datus.tools.db_tools.db_manager.db_manager_instance"),
    ):
        actions = [action async for action in stream_metadata(agent_config, datasource="local")]

    create_metadata_rag.assert_called_once_with(agent_config)
    assert agent_config.current_datasource == "local"
    assert "finished" in actions[-1].messages.lower()


@pytest.mark.asyncio
async def test_run_helper_forwards_native_actions_in_arrival_order() -> None:
    """Native ActionHistory entries from ``on_action`` reach the consumer
    intact and in the order the helper produced them."""

    async def helper(_emit, on_action):
        on_action(_make_action("step 1"))
        await asyncio.sleep(0)
        on_action(_make_action("step 2"))
        on_action(_make_action("step 3"))

    out: List[ActionHistory] = []
    async for act in _run_helper_with_actions(helper, function_name="gen_test"):
        out.append(act)

    assert [a.messages for a in out] == ["step 1", "step 2", "step 3"]


@pytest.mark.asyncio
async def test_run_helper_interleaves_events_and_actions() -> None:
    """BatchEvent markers and native actions interleave in submission order."""

    async def helper(emit, on_action):
        emit(BatchEvent(biz_name="bm", stage=BatchStage.TASK_VALIDATED, total_items=3))
        on_action(_make_action("native-1"))
        emit(BatchEvent(biz_name="bm", stage=BatchStage.TASK_COMPLETED, completed_items=3, total_items=3))
        on_action(_make_action("native-2"))

    out: List[ActionHistory] = []
    async for act in _run_helper_with_actions(helper, function_name="gen_test"):
        out.append(act)

    # 4 entries: validated, native-1, completed, native-2 (in submission order)
    assert len(out) == 4
    assert out[0].messages == "validated 3 item(s)"
    assert out[1].messages == "native-1"
    assert out[2].messages == "completed 3/3"
    assert out[3].messages == "native-2"


@pytest.mark.asyncio
async def test_run_helper_propagates_exception() -> None:
    """If the helper raises, the exception surfaces after the queue drains."""

    async def helper(_emit, on_action):
        on_action(_make_action("partial"))
        raise RuntimeError("boom")

    out: List[ActionHistory] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for act in _run_helper_with_actions(helper, function_name="gen_test"):
            out.append(act)

    # Pre-error action still made it through
    assert [a.messages for a in out] == ["partial"]


@pytest.mark.asyncio
async def test_run_helper_failed_action_marks_status() -> None:
    """A FAILED native action is forwarded as-is (the wrapper does not edit status)."""

    async def helper(_emit, on_action):
        on_action(_make_action("ouch", status=ActionStatus.FAILED))

    out = [a async for a in _run_helper_with_actions(helper, function_name="gen_test")]
    assert len(out) == 1
    assert out[0].status == ActionStatus.FAILED.value


@pytest.mark.asyncio
async def test_merge_streams_interleaves_two_producers() -> None:
    """``merge_streams`` is the load-bearing fan-in for per-item subagent
    groups in ``stream_reference_sql``. Both producers' actions reach the
    consumer; arrival order is the only guarantee (no per-stream
    ordering across queues)."""

    async def producer(label: str, count: int):
        for i in range(count):
            await asyncio.sleep(0)
            yield _make_action(f"{label}-{i}")

    out = [a async for a in merge_streams(producer("A", 2), producer("B", 2))]

    assert len(out) == 4
    assert {a.messages for a in out} == {"A-0", "A-1", "B-0", "B-1"}


@pytest.mark.asyncio
async def test_merge_streams_empty_fanin() -> None:
    out = [a async for a in merge_streams()]
    assert out == []


# ============================================================
# stream_reference_sql — overwrite truncates the storage
# ============================================================


@pytest.mark.asyncio
async def test_stream_reference_sql_overwrite_calls_truncate(monkeypatch) -> None:
    """build_mode='overwrite' must call storage.truncate() before any item processing."""
    from unittest.mock import MagicMock

    from datus.cli.bootstrap_streams import stream_reference_sql

    fake_storage = MagicMock()
    monkeypatch.setattr(
        "datus.storage.reference_sql.store.ReferenceSqlRAG",
        MagicMock(return_value=fake_storage),
    )
    # Empty file scan → function yields "no new items" and returns after the truncate branch
    monkeypatch.setattr(
        "datus.storage.reference_sql.sql_file_processor.process_sql_files",
        MagicMock(return_value=([], [])),
    )

    mock_config = MagicMock()
    mock_config.project_name = "unit-test-project"
    mock_config.current_datasource = "ds1"

    actions = [
        a
        async for a in stream_reference_sql(
            mock_config,
            datasource="ds1",
            sql_dir="/tmp/some_dir",
            build_mode="overwrite",
        )
    ]

    fake_storage.truncate.assert_called_once_with()
    # User-facing progress hint must be emitted
    assert any("cleared existing entries (overwrite mode)" in a.messages for a in actions)


@pytest.mark.asyncio
async def test_stream_reference_sql_incremental_does_not_truncate(monkeypatch) -> None:
    """build_mode='incremental' must NOT call storage.truncate()."""
    from unittest.mock import MagicMock

    from datus.cli.bootstrap_streams import stream_reference_sql

    fake_storage = MagicMock()
    monkeypatch.setattr(
        "datus.storage.reference_sql.store.ReferenceSqlRAG",
        MagicMock(return_value=fake_storage),
    )
    monkeypatch.setattr(
        "datus.storage.reference_sql.sql_file_processor.process_sql_files",
        MagicMock(return_value=([], [])),
    )
    monkeypatch.setattr(
        "datus.storage.reference_sql.init_utils.exists_reference_sql",
        MagicMock(return_value=set()),
    )

    mock_config = MagicMock()
    mock_config.project_name = "unit-test-project"
    mock_config.current_datasource = "ds1"

    _ = [
        a
        async for a in stream_reference_sql(
            mock_config,
            datasource="ds1",
            sql_dir="/tmp/some_dir",
            build_mode="incremental",
        )
    ]

    fake_storage.truncate.assert_not_called()


@pytest.mark.asyncio
async def test_stream_reference_sql_syncs_provenance_sidecar_for_successful_items(monkeypatch, tmp_path) -> None:
    """The /bootstrap reference_sql stream should sync provenance only after subagent success."""
    from unittest.mock import MagicMock

    from datus.cli.bootstrap_streams import stream_reference_sql
    from datus.storage.knowledge_provenance import REFERENCE_SQL_ARTIFACT_TYPE, KnowledgeProvenanceStore
    from datus.storage.reference_sql.init_utils import gen_reference_sql_id

    success_sql = "SELECT 1 AS ok"
    failed_sql = "SELECT 2 AS bad"
    items = [
        {
            "sql": success_sql,
            "filepath": "success.sql",
            "source_context_ids": ["refsql:task:1"],
            "source_metadata": {"task_id": "1"},
        },
        {
            "sql": failed_sql,
            "filepath": "failed.sql",
            "source_context_ids": ["refsql:task:2"],
            "source_metadata": {"task_id": "2"},
        },
    ]

    fake_storage = MagicMock()
    monkeypatch.setattr(
        "datus.storage.reference_sql.store.ReferenceSqlRAG",
        MagicMock(return_value=fake_storage),
    )
    monkeypatch.setattr(
        "datus.storage.reference_sql.sql_file_processor.process_sql_files",
        MagicMock(return_value=(items, [])),
    )
    monkeypatch.setattr(
        "datus.storage.reference_sql.init_utils.exists_reference_sql",
        MagicMock(return_value=set()),
    )

    class FakeSqlSummaryAgenticNode:
        def __init__(self, **_kwargs):
            self.input = None

        async def execute_stream(self, _manager):
            if self.input and self.input.sql_query == failed_sql:
                yield _make_action("failed", status=ActionStatus.FAILED)
            else:
                yield _make_action("ok")

    monkeypatch.setattr(
        "datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode",
        FakeSqlSummaryAgenticNode,
    )

    config = SimpleNamespace(
        project_name="unit-test-project",
        current_datasource="ds1",
        knowledge_base={"provenance": {"enabled": True}},
        path_manager=SimpleNamespace(project_data_dir=tmp_path),
    )

    actions = [
        action
        async for action in stream_reference_sql(
            config,
            datasource="ds1",
            sql_dir="/tmp/sql",
            build_mode="incremental",
        )
    ]

    success_id = gen_reference_sql_id(success_sql)
    failed_id = gen_reference_sql_id(failed_sql)
    provenance = KnowledgeProvenanceStore(config).find_by_artifact_ids(
        REFERENCE_SQL_ARTIFACT_TYPE,
        [success_id, failed_id],
    )

    assert provenance[success_id]["source_context_ids"] == ["refsql:task:1"]
    assert failed_id not in provenance
    assert any("Synced 1 reference SQL provenance row(s)." in action.messages for action in actions)
    assert any("Indexed 1 reference SQL item(s)." in action.messages for action in actions)
