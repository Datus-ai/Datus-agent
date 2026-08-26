# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for semantic bootstrap compatibility routing, YAML import, and profile parsing."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.storage.semantic_model.semantic_model_init import (
    METRICFLOW_YAML_UNSUPPORTED_MESSAGE,
    _load_success_story_profile_entries,
    init_semantic_yaml_semantic_model,
    init_success_story_semantic_model_async,
    refresh_semantic_yaml_profile_descriptions,
    reject_non_dosi_semantic_yaml,
    semantic_yaml_files,
    sync_semantic_yaml_tree,
)


def _config(adapter: str = "dosi") -> MagicMock:
    config = MagicMock()
    config.resolve_semantic_adapter = MagicMock(return_value=adapter)
    return config


@pytest.mark.asyncio
async def test_success_story_semantic_model_routes_to_datasets_only_semantic_modeling():
    config = MagicMock()
    unified = AsyncMock(return_value=(True, "", {"semantic_dataset_count": 3}))

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
    success, error = init_semantic_yaml_semantic_model(str(tmp_path / "missing.yml"), _config())

    assert success is False
    assert "not found" in error


def test_semantic_yaml_import_syncs_osi_documents(tmp_path):
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("semantic_model:\n  - name: orders\n    datasets: []\n", encoding="utf-8")
    config = _config("dosi")

    with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
        tools_cls.return_value.sync_osi_to_db.return_value = {"success": True, "message": "imported"}
        result = init_semantic_yaml_semantic_model(str(yaml_path), config)

    assert result == (True, "")
    tools_cls.return_value.sync_osi_to_db.assert_called_once_with(
        str(yaml_path),
        include_semantic_objects=True,
        include_metrics=False,
    )


def test_semantic_yaml_import_surfaces_sync_failure(tmp_path):
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("semantic_model:\n  - name: orders\n    datasets: []\n", encoding="utf-8")

    with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
        tools_cls.return_value.sync_osi_to_db.return_value = {"success": False, "error": "invalid YAML"}
        success, error = init_semantic_yaml_semantic_model(str(yaml_path), _config("dosi"))

    assert success is False
    assert "invalid YAML" in error


def test_semantic_yaml_import_rejected_for_non_dosi_project(tmp_path):
    """Contract: non-Dosi projects are query-only — the YAML import entry
    must fail before any KB sync is attempted."""
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("semantic_model:\n  - name: orders\n    datasets: []\n", encoding="utf-8")

    with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
        success, error = init_semantic_yaml_semantic_model(str(yaml_path), _config("metricflow"))

    assert success is False
    assert "query-only" in error
    tools_cls.return_value.sync_osi_to_db.assert_not_called()


def test_semantic_yaml_import_rejects_metricflow_documents(tmp_path):
    """Contract: MetricFlow ``data_source:``/``metric:`` YAML can no longer be
    imported, even in a Dosi project — the error must be explicit."""
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("data_source:\n  name: orders\n  sql_table: public.orders\n", encoding="utf-8")

    with patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls:
        success, error = init_semantic_yaml_semantic_model(str(yaml_path), _config("dosi"))

    assert success is False
    assert error == METRICFLOW_YAML_UNSUPPORTED_MESSAGE
    tools_cls.return_value.sync_osi_to_db.assert_not_called()


def test_reject_helper_accepts_osi_documents_in_dosi_project(tmp_path):
    yaml_path = tmp_path / "semantic.yml"
    yaml_path.write_text("semantic_model:\n  - name: orders\n    datasets: []\n", encoding="utf-8")

    assert reject_non_dosi_semantic_yaml(str(yaml_path), _config("dosi")) is None


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
        "semantic_model:\n  - name: orders\n    datasets:\n      - name: orders\n        source: public.orders\n",
        encoding="utf-8",
    )
    config = _config("dosi")
    config.path_manager.subject_dir = str(tmp_path)

    with (
        patch(
            "datus.storage.semantic_model.profile_description.refresh_osi_yaml_descriptions",
            return_value=1,
        ),
        patch("datus.tools.func_tool.generation_tools.GenerationTools") as tools_cls,
    ):
        tools_cls.return_value.sync_osi_to_db.return_value = {"success": True}
        result = refresh_semantic_yaml_profile_descriptions(
            str(yaml_path),
            {"tables": []},
            authoring_format="osi",
            agent_config=config,
            sync_to_storage=True,
        )

    assert result == (True, "", 1)
    tools_cls.return_value.sync_osi_to_db.assert_called_once_with(
        str(yaml_path),
        include_semantic_objects=True,
        include_metrics=False,
    )


def test_profile_description_refresh_rejects_metricflow_yaml(tmp_path):
    """Contract: profile refresh no longer patches MetricFlow YAML."""
    semantic_dir = tmp_path / "semantic_models"
    semantic_dir.mkdir()
    yaml_path = semantic_dir / "semantic.yml"
    yaml_path.write_text("data_source:\n  name: orders\n  description: Orders\n", encoding="utf-8")
    config = _config("dosi")
    config.path_manager.subject_dir = str(tmp_path)

    result = refresh_semantic_yaml_profile_descriptions(
        str(yaml_path),
        {"tables": []},
        agent_config=config,
        sync_to_storage=True,
    )

    assert result == (False, METRICFLOW_YAML_UNSUPPORTED_MESSAGE, 0)


def test_profile_description_refresh_rejects_non_dosi_project(tmp_path):
    """Contract: profile refresh is an authoring mutation — an OSI-shaped YAML
    in a non-Dosi project must be rejected by the adapter gate, not slip
    through document-shape inference."""
    semantic_dir = tmp_path / "semantic_models"
    semantic_dir.mkdir()
    yaml_path = semantic_dir / "semantic.yml"
    yaml_path.write_text(
        "semantic_model:\n  - name: orders\n    datasets:\n      - name: orders\n        source: public.orders\n",
        encoding="utf-8",
    )
    config = _config("metricflow")
    config.path_manager.subject_dir = str(tmp_path)

    result = refresh_semantic_yaml_profile_descriptions(
        str(yaml_path),
        {"tables": []},
        authoring_format="osi",
        agent_config=config,
        sync_to_storage=True,
    )

    assert result[0] is False
    assert "query-only" in result[1]


def _write_model(path, name: str) -> None:
    path.write_text(
        f"version: 0.2.0.dev0\nsemantic_model:\n  - name: {name}\n"
        f"    datasets:\n      - name: {name}\n        source: public.{name}\n",
        encoding="utf-8",
    )


def test_sync_semantic_yaml_tree_projects_every_file_under_the_directory(tmp_path):
    _write_model(tmp_path / "sales.yml", "sales")
    _write_model(tmp_path / "fulfillment.yaml", "fulfillment")
    config = _config("dosi")
    tools = MagicMock()
    tools.sync_osi_to_db.return_value = {"success": True}

    with patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=tools):
        successful, message, synced = sync_semantic_yaml_tree(config, str(tmp_path))

    assert successful is True
    assert synced == 2
    assert "2 semantic YAML file(s)" in message
    synced_paths = sorted(call.args[0] for call in tools.sync_osi_to_db.call_args_list)
    assert [Path(path).name for path in synced_paths] == ["fulfillment.yaml", "sales.yml"]


def test_sync_semantic_yaml_tree_skips_the_metrics_fragment_directory(tmp_path):
    _write_model(tmp_path / "sales.yml", "sales")
    fragments = tmp_path / "metrics"
    fragments.mkdir()
    _write_model(fragments / "revenue.yml", "revenue")
    config = _config("dosi")
    tools = MagicMock()
    tools.sync_osi_to_db.return_value = {"success": True}

    with patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=tools):
        _, _, synced = sync_semantic_yaml_tree(config, str(tmp_path))

    assert synced == 1


def test_sync_semantic_yaml_tree_reports_failures_without_stopping(tmp_path):
    """One unreadable model must not cost the others their refresh."""
    _write_model(tmp_path / "sales.yml", "sales")
    _write_model(tmp_path / "broken.yml", "broken")
    config = _config("dosi")
    tools = MagicMock()
    tools.sync_osi_to_db.side_effect = [{"success": False, "error": "bad model"}, {"success": True}]

    with patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=tools):
        successful, message, synced = sync_semantic_yaml_tree(config, str(tmp_path))

    assert successful is False
    assert synced == 1
    assert "broken.yml: bad model" in message
    assert tools.sync_osi_to_db.call_count == 2


def test_sync_semantic_yaml_tree_rejects_a_missing_path(tmp_path):
    successful, message, synced = sync_semantic_yaml_tree(_config("dosi"), str(tmp_path / "absent"))

    assert successful is False
    assert synced == 0
    assert "not found" in message


def test_semantic_yaml_files_skips_the_metrics_fragment_directory(tmp_path):
    """The staleness hint reuses this, so both must agree on what counts."""
    _write_model(tmp_path / "sales.yml", "sales")
    fragments = tmp_path / "metrics"
    fragments.mkdir()
    _write_model(fragments / "revenue.yml", "revenue")

    found = semantic_yaml_files(tmp_path)

    assert [path.name for path in found] == ["sales.yml"]


def test_sync_semantic_yaml_tree_drops_rows_whose_file_is_gone(tmp_path, real_agent_config):
    """A sync is a reconcile, not an append. Deleting or renaming a model file
    has to remove what it projected: the rows outlive the file otherwise, and
    describe_table keeps offering a model whose yaml_path no longer opens."""
    from datus.storage.semantic_dataset.store import (
        KIND_DATASET,
        SemanticDatasetRAG,
        dataset_row_id,
    )

    rag = SemanticDatasetRAG(real_agent_config)
    removed = tmp_path / "finance.yml"
    kept = tmp_path / "ops.yml"
    for path, model in ((removed, "finance"), (kept, "ops")):
        rag.upsert_batch(
            [
                {
                    "id": dataset_row_id(model, "orders"),
                    "kind": KIND_DATASET,
                    "semantic_model_name": model,
                    "dataset_name": "orders",
                    "name": "orders",
                    "source_table": "orders",
                    "search_text": model,
                    "yaml_path": str(path),
                }
            ]
        )
    kept.write_text("version: 0.2.0.dev0\nsemantic_model: []\n")  # removed.yml intentionally absent

    tools = MagicMock()
    tools.sync_osi_to_db.return_value = {"success": True}
    with patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=tools):
        successful, _, _ = sync_semantic_yaml_tree(real_agent_config, str(tmp_path))

    assert successful is True
    surviving = {row["semantic_model_name"] for row in rag.storage.table.search_all(limit=100).to_pylist()}
    assert surviving == {"ops"}


def test_sync_semantic_yaml_tree_prunes_when_the_last_file_is_deleted(tmp_path, real_agent_config):
    """Deleting every model is the strongest form of the case pruning exists
    for, and the empty-directory branch used to return before reconciling."""
    from datus.storage.semantic_dataset.store import KIND_DATASET, SemanticDatasetRAG, dataset_row_id

    rag = SemanticDatasetRAG(real_agent_config)
    rag.upsert_batch(
        [
            {
                "id": dataset_row_id("finance", "orders"),
                "kind": KIND_DATASET,
                "semantic_model_name": "finance",
                "dataset_name": "orders",
                "name": "orders",
                "source_table": "orders",
                "search_text": "finance",
                "yaml_path": str(tmp_path / "finance.yml"),
            }
        ]
    )

    successful, message, synced = sync_semantic_yaml_tree(real_agent_config, str(tmp_path))

    assert successful is True and synced == 0
    assert "pruned 1 deleted artifact(s)" in message
    assert rag.storage.table.search_all(limit=100).to_pylist() == []
