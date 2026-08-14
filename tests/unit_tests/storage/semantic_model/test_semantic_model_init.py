# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for semantic bootstrap compatibility routing, YAML import, and profile parsing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.storage.semantic_model.semantic_model_init import (
    _load_success_story_profile_entries,
    init_semantic_yaml_semantic_model,
    init_success_story_semantic_model_async,
    process_semantic_yaml_file,
    refresh_semantic_yaml_profile_descriptions,
)


@pytest.mark.asyncio
async def test_success_story_semantic_model_routes_to_datasets_only_semantic_modeling():
    config = MagicMock()
    unified = AsyncMock(return_value=(True, "", {"semantic_object_count": 3}))

    with patch(
        "datus.storage.semantic_model.semantic_modeling_init.init_success_story_semantic_modeling_async",
        unified,
    ):
        result = await init_success_story_semantic_model_async(
            config,
            "stories.csv",
            build_mode="incremental",
        )

    assert result == (True, "")
    unified.assert_awaited_once_with(
        config,
        "stories.csv",
        emit=None,
        build_mode="incremental",
        action_callback=None,
        authoring_scope="datasets",
    )


def test_semantic_yaml_import_reports_missing_file(tmp_path):
    success, error = init_semantic_yaml_semantic_model(str(tmp_path / "missing.yml"), MagicMock())

    assert success is False
    assert "not found" in error


def test_semantic_yaml_import_uses_non_llm_projection_path(tmp_path):
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("semantic_models: []\n", encoding="utf-8")
    config = MagicMock()

    with patch(
        "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
        return_value={"success": True, "message": "imported"},
    ) as sync:
        result = init_semantic_yaml_semantic_model(str(yaml_path), config)

    assert result == (True, "")
    sync.assert_called_once_with(
        str(yaml_path),
        config,
        include_semantic_objects=True,
        include_metrics=False,
    )


def test_process_semantic_yaml_surfaces_projection_failure(tmp_path):
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("semantic_models: []\n", encoding="utf-8")

    with patch(
        "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
        return_value={"success": False, "error": "invalid YAML"},
    ):
        success, error = process_semantic_yaml_file(str(yaml_path), MagicMock())

    assert success is False
    assert "invalid YAML" in error


def test_profile_parser_keeps_question_and_sql_rows(tmp_path):
    csv_path = tmp_path / "stories.csv"
    csv_path.write_text(
        "question,sql,source_context_id\nHow many orders?,SELECT COUNT(*) FROM orders,orders_count\n",
        encoding="utf-8",
    )

    entries, error = _load_success_story_profile_entries(str(csv_path))

    assert error == ""
    assert entries == [
        {
            "name": "orders_count",
            "question": "How many orders?",
            "sql": "SELECT COUNT(*) FROM orders",
        }
    ]


def test_profile_description_refresh_preserves_yaml_and_syncs_projection(tmp_path):
    semantic_dir = tmp_path / "semantic_models"
    semantic_dir.mkdir()
    yaml_path = semantic_dir / "semantic.yml"
    yaml_path.write_text(
        "data_source:\n  name: orders\n  description: Orders\n",
        encoding="utf-8",
    )
    config = MagicMock()
    config.path_manager.subject_dir = str(tmp_path)

    with (
        patch(
            "datus.storage.semantic_model.profile_description.refresh_metricflow_yaml_descriptions",
            return_value=1,
        ),
        patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as sync,
    ):
        result = refresh_semantic_yaml_profile_descriptions(
            str(yaml_path),
            {"tables": []},
            authoring_format="metricflow",
            agent_config=config,
            sync_to_storage=True,
        )

    assert result == (True, "", 1)
    sync.assert_called_once_with(
        str(yaml_path),
        config,
        include_semantic_objects=True,
        include_metrics=True,
    )
