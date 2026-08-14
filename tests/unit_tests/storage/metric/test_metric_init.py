# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for metric bootstrap compatibility routing and YAML import."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.storage.metric.metric_init import (
    init_semantic_yaml_metrics,
    init_success_story_metrics_async,
)


@pytest.mark.asyncio
async def test_success_story_metrics_routes_to_full_semantic_modeling():
    config = MagicMock()
    unified = AsyncMock(return_value=(True, "", {"metrics_count": 3}))

    with patch(
        "datus.storage.semantic_model.semantic_modeling_init.init_success_story_semantic_modeling_async",
        unified,
    ):
        result = await init_success_story_metrics_async(
            config,
            "stories.csv",
            ["Sales"],
            build_mode="incremental",
            batch_size=3,
        )

    assert result == (True, "", {"metrics_count": 3})
    unified.assert_awaited_once_with(
        config,
        "stories.csv",
        ["Sales"],
        None,
        build_mode="incremental",
        action_callback=None,
        batch_size=3,
        authoring_scope="full",
    )


def test_semantic_yaml_metrics_reports_missing_file(tmp_path):
    config = MagicMock()

    success, error = init_semantic_yaml_metrics(str(tmp_path / "missing.yml"), config)

    assert success is False
    assert "not found" in error


def test_dosi_yaml_import_reconciles_metric_projection(tmp_path):
    yaml_path = tmp_path / "orders.yml"
    yaml_path.write_text("semantic_model: []\n", encoding="utf-8")
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "dosi"
    tools = MagicMock()
    tools.sync_osi_to_db.return_value = {"success": True, "message": "imported"}

    with patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=tools):
        result = init_semantic_yaml_metrics(str(yaml_path), config)

    assert result == (True, "imported")
    tools.sync_osi_to_db.assert_called_once_with(
        str(yaml_path),
        include_semantic_objects=False,
        include_metrics=True,
    )


def test_metricflow_yaml_import_keeps_non_llm_projection_path(tmp_path):
    yaml_path = tmp_path / "metrics.yml"
    yaml_path.write_text("metric: []\n", encoding="utf-8")
    config = MagicMock()
    config.resolve_semantic_adapter.return_value = "metricflow"

    with patch(
        "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
        return_value=(True, ""),
    ) as process:
        result = init_semantic_yaml_metrics(str(yaml_path), config)

    assert result == (True, "")
    process.assert_called_once_with(str(yaml_path), config, include_semantic_objects=False)
