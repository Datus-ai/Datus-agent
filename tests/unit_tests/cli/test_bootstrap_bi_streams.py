# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_bi_streams`.

Each ``stream_bi_*`` is exercised with the upstream async helper
(``init_local_schema_async`` / ``init_success_story_*_async``) mocked. We
collect the yielded :class:`ActionHistory` entries and assert both the
shape (entry / exit messages, success/failure status) and the side
effects on :class:`BiBuildState`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from datus.cli.bootstrap_bi_streams import (
    BiBuildState,
    _collect_metrics_from_semantic_models,
    _collect_ref_sqls_from_summary_files,
    _validate_semantic_model_sync,
    stream_bi_metadata,
    stream_bi_metrics,
    stream_bi_reference_sql,
    stream_bi_semantic_model,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.tools.bi_tools.dashboard_assembler import SelectedSqlCandidate

# ─────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def agent_config(tmp_path: Path) -> SimpleNamespace:
    """Minimal AgentConfig stub satisfying the path/storage APIs the streams touch."""
    return SimpleNamespace(
        db_type="duckdb",
        current_datasource="local",
        datasource_configs={},
        agentic_nodes={},
        path_manager=SimpleNamespace(
            dashboard_path=lambda: tmp_path,
            sql_summary_path=lambda: tmp_path / "summaries",
            subject_dir=tmp_path / "subject",
        ),
        resolve_semantic_adapter=lambda x: x,
    )


@pytest.fixture()
def state() -> BiBuildState:
    return BiBuildState()


def _candidate(chart_id: int = 1, sql: str = "select 1") -> SelectedSqlCandidate:
    return SelectedSqlCandidate(chart_id=chart_id, chart_name=f"chart-{chart_id}", description=None, sql=sql)


async def _consume(gen: AsyncGenerator) -> list[ActionHistory]:
    return [a async for a in gen]


# ─────────────────────────────────────────────────────────────────
# stream_bi_metadata
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_bi_metadata_skips_when_no_tables(agent_config) -> None:
    actions = await _consume(stream_bi_metadata(agent_config, table_names=[]))
    assert len(actions) == 1
    assert actions[0].status == ActionStatus.FAILED.value
    assert "skipping" in actions[0].messages.lower()


@pytest.mark.asyncio
async def test_stream_bi_metadata_yields_entry_exit_messages(agent_config) -> None:
    async def _ok(*_a, **_k):
        return None

    with (
        patch("datus.storage.schema_metadata.local_init.init_local_schema_async", side_effect=_ok),
        patch("datus.storage.schema_metadata.create_metadata_rag", return_value=MagicMock()),
        patch("datus.tools.db_tools.db_manager.db_manager_instance"),
    ):
        actions = await _consume(stream_bi_metadata(agent_config, table_names=["t1", "t2"]))

    # entry message + exit message at minimum (no BatchEvents emitted by mock)
    assert len(actions) >= 2
    assert "Crawling metadata for 2 table(s)" in actions[0].messages
    assert "finished" in actions[-1].messages.lower()


@pytest.mark.asyncio
async def test_stream_bi_metadata_yields_failed_on_helper_exception(agent_config) -> None:
    async def _boom(*_a, **_k):
        raise RuntimeError("boom")

    with (
        patch("datus.storage.schema_metadata.local_init.init_local_schema_async", side_effect=_boom),
        patch("datus.storage.schema_metadata.create_metadata_rag", return_value=MagicMock()),
        patch("datus.tools.db_tools.db_manager.db_manager_instance"),
    ):
        actions = await _consume(stream_bi_metadata(agent_config, table_names=["t1"]))

    assert any(a.status == ActionStatus.FAILED.value and "boom" in a.messages for a in actions)


# ─────────────────────────────────────────────────────────────────
# stream_bi_reference_sql
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_bi_reference_sql_skips_when_no_candidates(agent_config, state) -> None:
    actions = await _consume(
        stream_bi_reference_sql(
            agent_config,
            reference_sqls=[],
            platform="superset",
            dashboard_name="Sales",
            pool_size=2,
            state=state,
        )
    )
    assert len(actions) == 1
    assert "No reference SQL" in actions[0].messages
    assert state.ref_sqls == []


@pytest.mark.asyncio
async def test_stream_bi_reference_sql_collects_summaries_into_state(agent_config, state, tmp_path: Path) -> None:
    # Materialize a fake summary YAML the post-processor can read.
    summaries_root = tmp_path / "summaries"
    summaries_root.mkdir()
    yaml_path = summaries_root / "orders_q1.yml"
    yaml_path.write_text(
        yaml.safe_dump({"name": "orders_q1", "subject_tree": "superset/sales"}),
        encoding="utf-8",
    )

    async def _fake_inner_stream(*_a, **_k):
        # Simulate the per-item bootstrap stream emitting a sql_summary_response
        # whose output points at the YAML we just wrote.
        yield ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="sql_summary_response",
            messages="done",
            input_data={},
            output_data={"success": True, "sql_summary_file": "orders_q1.yml"},
            status=ActionStatus.SUCCESS,
        )

    with patch("datus.cli.bootstrap_streams.stream_reference_sql", side_effect=_fake_inner_stream):
        actions = await _consume(
            stream_bi_reference_sql(
                agent_config,
                reference_sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                pool_size=2,
                state=state,
            )
        )

    # Wrote the SQL file, delegated, then yielded a "Collected N" message.
    assert any("Wrote 1 chart SQL" in a.messages for a in actions)
    assert any("Collected 1 reference" in a.messages for a in actions)
    assert state.ref_sqls == ["superset.sales.orders_q1"]


@pytest.mark.asyncio
async def test_stream_bi_reference_sql_marks_failure_when_no_summaries(agent_config, state) -> None:
    async def _empty_stream(*_a, **_k):
        return
        yield  # pragma: no cover

    with patch("datus.cli.bootstrap_streams.stream_reference_sql", side_effect=_empty_stream):
        actions = await _consume(
            stream_bi_reference_sql(
                agent_config,
                reference_sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                pool_size=2,
                state=state,
            )
        )

    assert any(a.status == ActionStatus.FAILED.value and "No SQL summaries" in a.messages for a in actions)
    assert state.ref_sqls == []


def test_collect_ref_sqls_dedupes_and_quotes(tmp_path: Path, agent_config) -> None:
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "a.yml").write_text(yaml.safe_dump({"name": "a", "subject_tree": "x/y"}), encoding="utf-8")
    (summaries / "b.yml").write_text(yaml.safe_dump({"name": "a", "subject_tree": "x/y"}), encoding="utf-8")
    out = _collect_ref_sqls_from_summary_files(["a.yml", "b.yml"], agent_config)
    assert out == ["x.y.a"]  # duplicate dropped


def test_collect_ref_sqls_skips_missing_file(tmp_path: Path, agent_config) -> None:
    out = _collect_ref_sqls_from_summary_files(["missing.yml"], agent_config)
    assert out == []


# ─────────────────────────────────────────────────────────────────
# stream_bi_semantic_model
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_bi_semantic_model_skips_when_no_sqls(agent_config, state) -> None:
    actions = await _consume(
        stream_bi_semantic_model(
            agent_config,
            sqls=[],
            platform="superset",
            dashboard_name="Sales",
            state=state,
        )
    )
    assert actions[0].status == ActionStatus.FAILED.value
    assert state.semantic_ok is False


@pytest.mark.asyncio
async def test_stream_bi_semantic_model_sets_semantic_ok_on_validation_success(agent_config, state) -> None:
    async def _ok_async(*_a, **_k):
        return True, ""

    validation_scopes: list[str] = []

    def _validate(_agent_config, *, scope: str = "all"):
        validation_scopes.append(scope)
        return True, None

    with (
        patch(
            "datus.storage.semantic_model.semantic_model_init.init_success_story_semantic_model_async",
            side_effect=_ok_async,
        ),
        patch(
            "datus.cli.bootstrap_bi_streams._validate_semantic_model_sync",
            side_effect=_validate,
        ),
    ):
        actions = await _consume(
            stream_bi_semantic_model(
                agent_config,
                sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                state=state,
            )
        )

    assert state.semantic_ok is True
    assert validation_scopes == ["semantic_model"]
    assert any("validated" in a.messages.lower() for a in actions)


@pytest.mark.asyncio
async def test_stream_bi_semantic_model_keeps_semantic_ok_false_on_validation_failure(agent_config, state) -> None:
    async def _ok_async(*_a, **_k):
        return True, ""

    with (
        patch(
            "datus.storage.semantic_model.semantic_model_init.init_success_story_semantic_model_async",
            side_effect=_ok_async,
        ),
        patch(
            "datus.cli.bootstrap_bi_streams._validate_semantic_model_sync",
            return_value=(False, "adapter missing"),
        ),
    ):
        actions = await _consume(
            stream_bi_semantic_model(
                agent_config,
                sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                state=state,
            )
        )

    assert state.semantic_ok is False
    assert any(a.status == ActionStatus.FAILED.value and "validation failed" in a.messages.lower() for a in actions)


# ─────────────────────────────────────────────────────────────────
# stream_bi_metrics
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_bi_metrics_skips_when_no_sqls(agent_config, state) -> None:
    actions = await _consume(
        stream_bi_metrics(
            agent_config,
            sqls=[],
            platform="superset",
            dashboard_name="Sales",
            state=state,
        )
    )
    assert actions[0].status == ActionStatus.FAILED.value
    assert state.metrics == []


@pytest.mark.asyncio
async def test_stream_bi_metrics_collects_metric_identifiers(agent_config, state, tmp_path: Path) -> None:
    semantic_dir = tmp_path / "subject" / "semantic_models" / "metrics"
    semantic_dir.mkdir(parents=True)
    yaml_file = semantic_dir / "orders.yml"
    yaml_file.write_text(
        yaml.safe_dump(
            {
                "metric": {
                    "name": "total_orders",
                    "locked_metadata": {"tags": ["subject_tree:superset/sales"]},
                }
            }
        ),
        encoding="utf-8",
    )

    async def _ok_metrics(*_a, **_k):
        return True, "", {"semantic_models": ["semantic_models/metrics/orders.yml"]}

    validation_scopes: list[str] = []

    def _validate(_agent_config, *, scope: str = "all"):
        validation_scopes.append(scope)
        return True, None

    with (
        patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            side_effect=_ok_metrics,
        ),
        patch(
            "datus.cli.bootstrap_bi_streams._validate_semantic_model_sync",
            side_effect=_validate,
        ),
        patch(
            "datus.cli.generation_hooks.resolve_kb_sandbox_path",
            return_value=str(yaml_file),
        ),
    ):
        actions = await _consume(
            stream_bi_metrics(
                agent_config,
                sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                state=state,
            )
        )

    assert state.metrics == ["superset.sales.total_orders"]
    assert validation_scopes == ["all"]
    assert any("Collected 1 metric" in a.messages for a in actions)


def test_validate_semantic_model_sync_passes_requested_scope(agent_config, monkeypatch) -> None:
    validation_scopes: list[str] = []

    class FakeSemanticTools:
        def __init__(self, *_a, **_k):
            self.adapter = object()

        def validate_semantic(self, scope: str = "all"):
            validation_scopes.append(scope)
            return SimpleNamespace(success=1, error=None)

    monkeypatch.setattr("datus.tools.func_tool.semantic_tools.SemanticTools", FakeSemanticTools)

    ok, err = _validate_semantic_model_sync(agent_config, scope="semantic_model")

    assert ok is True
    assert err is None
    assert validation_scopes == ["semantic_model"]


@pytest.mark.asyncio
async def test_stream_bi_metrics_does_not_collect_when_helper_fails(agent_config, state) -> None:
    async def _bad_metrics(*_a, **_k):
        return False, "model error", None

    with patch(
        "datus.storage.metric.metric_init.init_success_story_metrics_async",
        side_effect=_bad_metrics,
    ):
        await _consume(
            stream_bi_metrics(
                agent_config,
                sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                state=state,
            )
        )

    assert state.metrics == []


@pytest.mark.asyncio
async def test_stream_bi_metrics_aborts_collection_on_post_validation_failure(agent_config, state) -> None:
    async def _ok_metrics(*_a, **_k):
        return True, "", {"semantic_models": ["x.yml"]}

    with (
        patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            side_effect=_ok_metrics,
        ),
        patch(
            "datus.cli.bootstrap_bi_streams._validate_semantic_model_sync",
            return_value=(False, "post-fail"),
        ),
    ):
        actions = await _consume(
            stream_bi_metrics(
                agent_config,
                sqls=[_candidate()],
                platform="superset",
                dashboard_name="Sales",
                state=state,
            )
        )

    assert state.metrics == []
    assert any("Metrics validation failed" in a.messages for a in actions)


def test_collect_metrics_skips_files_outside_sandbox(agent_config) -> None:
    with patch("datus.cli.generation_hooks.resolve_kb_sandbox_path", return_value=None):
        out = _collect_metrics_from_semantic_models(["sketchy.yml"], agent_config)
    assert out == []


def test_collect_metrics_skips_when_yaml_missing_subject(tmp_path: Path, agent_config) -> None:
    yaml_file = tmp_path / "metric.yml"
    yaml_file.write_text(yaml.safe_dump({"metric": {"name": "x", "locked_metadata": {"tags": []}}}), encoding="utf-8")
    with patch("datus.cli.generation_hooks.resolve_kb_sandbox_path", return_value=str(yaml_file)):
        out = _collect_metrics_from_semantic_models(["metric.yml"], agent_config)
    assert out == []
