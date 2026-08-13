# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for the unified Dosi bootstrap orchestration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
from datus.storage.semantic_model.semantic_modeling_init import (
    SEMANTIC_MODELING_RESPONSE_ACTION_TYPE,
    _run_semantic_modeling_batch,
    init_success_story_semantic_modeling_async,
)


def _query(index: int) -> SourceQueryEvidence:
    return SourceQueryEvidence(
        source_sql_name=f"sql_{index}",
        question=f"Question {index}",
        sql=f"SELECT {index}",
    )


def _config() -> MagicMock:
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "dosi"
    config.current_db_config.return_value = SimpleNamespace(catalog="", database="db", schema="")
    config.runtime_db_context.return_value = {}
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("build_mode", ["overwrite", "incremental"])
async def test_batches_complete_semantic_modeling_calls_in_groups_of_five(build_mode):
    config = _config()
    queries = [_query(index) for index in range(1, 13)]
    run_batch = AsyncMock(
        side_effect=[
            (True, "", {"success": True, "status": "generated", "semantic_models": ["models/domain.yml"]}),
            (True, "", {"success": True, "status": "generated", "semantic_models": ["models/domain.yml"]}),
            (True, "", {"success": True, "status": "generated", "semantic_models": ["models/domain.yml"]}),
        ]
    )
    semantic_rag = MagicMock()
    semantic_rag.get_size.return_value = 3
    metric_rag = MagicMock()
    metric_rag.get_metrics_size.return_value = 8

    with (
        patch(
            "datus.storage.semantic_model.semantic_modeling_init._load_source_queries",
            return_value=(queries, ""),
        ),
        patch("datus.storage.semantic_model.semantic_modeling_init._prepare_storage"),
        patch("datus.storage.semantic_model.semantic_modeling_init._run_semantic_modeling_batch", run_batch),
        patch("datus.storage.semantic_model.semantic_modeling_init.SemanticModelRAG", return_value=semantic_rag),
        patch("datus.storage.semantic_model.semantic_modeling_init.MetricRAG", return_value=metric_rag),
    ):
        success, error, result = await init_success_story_semantic_modeling_async(
            config,
            "stories.csv",
            build_mode=build_mode,
            batch_size=5,
        )

    assert success is True
    assert error == ""
    assert [len(call.args[1]) for call in run_batch.await_args_list] == [5, 5, 2]
    assert run_batch.await_args_list[0].kwargs["target_hint"] == ""
    assert run_batch.await_args_list[1].kwargs["target_hint"] == "models/domain.yml"
    assert result["batches_completed"] == 3
    assert result["sql_entries_covered"] == 12


@pytest.mark.asyncio
async def test_batch_accepts_no_semantic_change_as_idempotent_success():
    config = _config()
    node = MagicMock()

    async def execute_stream(_history):
        yield SimpleNamespace(
            action_type=SEMANTIC_MODELING_RESPONSE_ACTION_TYPE,
            status="success",
            messages="No semantic change is required.",
            output={
                "success": True,
                "status": "skipped",
                "skip_reason": "no_semantic_change",
                "semantic_models": [],
            },
        )

    node.execute_stream = execute_stream
    event_helper = MagicMock()
    with patch(
        "datus.storage.semantic_model.semantic_modeling_init.SemanticModelingAgenticNode",
        return_value=node,
    ):
        success, error, result = await _run_semantic_modeling_batch(
            config,
            [_query(1)],
            subject_tree=None,
            target_hint="",
            event_helper=event_helper,
            batch_index=0,
            action_callback=None,
        )

    assert success is True
    assert error == ""
    assert result["status"] == "skipped"
    event_helper.item_completed.assert_called_once_with("batch-1", sql_count=1)


@pytest.mark.asyncio
async def test_stops_after_first_failed_batch():
    config = _config()
    queries = [_query(index) for index in range(1, 13)]
    run_batch = AsyncMock(
        side_effect=[
            (True, "", {"success": True, "status": "generated", "semantic_models": ["models/domain.yml"]}),
            (False, "no structured result", None),
        ]
    )

    with (
        patch(
            "datus.storage.semantic_model.semantic_modeling_init._load_source_queries",
            return_value=(queries, ""),
        ),
        patch("datus.storage.semantic_model.semantic_modeling_init._prepare_storage"),
        patch("datus.storage.semantic_model.semantic_modeling_init._run_semantic_modeling_batch", run_batch),
    ):
        success, error, result = await init_success_story_semantic_modeling_async(
            config,
            "stories.csv",
            batch_size=5,
        )

    assert success is False
    assert "batch 2/3 failed" in error
    assert run_batch.await_count == 2
    assert result["batches_completed"] == 1
    assert result["sql_entries_covered"] == 5


@pytest.mark.asyncio
async def test_rejects_non_dosi_adapter_before_loading_csv():
    config = _config()
    config.resolve_semantic_adapter.return_value = "metricflow"

    with patch("datus.storage.semantic_model.semantic_modeling_init._load_source_queries") as load:
        success, error, result = await init_success_story_semantic_modeling_async(config, "stories.csv")

    assert success is False
    assert "semantic_adapter=dosi" in error
    assert result is None
    load.assert_not_called()


def test_storage_setup_preserves_existing_artifact_projections():
    from datus.storage.semantic_model.semantic_modeling_init import _prepare_storage

    config = _config()

    _prepare_storage(config)

    assert config.check_init_storage_config.call_args_list == [call("semantic_model"), call("metric")]
