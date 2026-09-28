# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for reconciling the KB and subject tree with semantic YAML on disk."""

import copy
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from datus.storage.metric.store import MetricRAG
from datus.storage.registry import get_subject_tree_store
from datus.storage.semantic_dataset.store import KIND_DATASET, SemanticDatasetRAG, dataset_row_id
from datus.storage.semantic_model.reconcile import reconcile_semantic_artifacts
from datus.storage.semantic_model.sync_state import file_digest, load_digests, record_digests

DATASOURCE = "california_schools"
OTHER = "warehouse"


def _model(name: str, metrics: dict[str, list[str]]) -> str:
    """An OSI model file declaring ``metrics`` (name -> subject_path)."""
    return yaml.safe_dump(
        {
            "version": "0.2.0.dev0",
            "semantic_model": [
                {
                    "name": name,
                    "datasets": [{"name": name, "source": f"public.{name}"}],
                    "metrics": [
                        {
                            "name": metric,
                            "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
                            "custom_extensions": [
                                {"vendor_name": "DATUS", "data": json.dumps({"subject_path": subject_path})}
                            ],
                        }
                        for metric, subject_path in metrics.items()
                    ],
                }
            ],
        }
    )


def _fake_sync(agent_config):
    """Stand-in for ``sync_osi_to_db``: projects the file's metrics into the real KB."""

    def sync(yaml_path, **_kwargs):
        rag = MetricRAG(agent_config)
        doc = yaml.safe_load(Path(yaml_path).read_text())
        rows = []
        for metric in doc["semantic_model"][0].get("metrics") or []:
            hints = json.loads(metric["custom_extensions"][0]["data"])
            rows.append(
                {
                    "subject_path": hints["subject_path"],
                    "id": f"metric:{metric['name']}",
                    "name": metric["name"],
                    "semantic_model_name": doc["semantic_model"][0]["name"],
                    "description": metric["name"],
                    "metric_type": "simple",
                    "measure_expr": "COUNT(*)",
                    "base_measures": [],
                    "dimensions": [],
                    "entities": [],
                    "catalog_name": "",
                    "database_name": "",
                    "schema_name": "",
                    "sql": "SELECT COUNT(*)",
                    "yaml_path": yaml_path,
                }
            )
        if rows:
            rag.upsert_batch(rows)
        rag.delete_artifact_rows_except(yaml_path, [row["id"] for row in rows])
        model = doc["semantic_model"][0]["name"]
        datasets = SemanticDatasetRAG(agent_config)
        datasets.upsert_batch(
            [
                {
                    "id": dataset_row_id(model, model),
                    "kind": KIND_DATASET,
                    "semantic_model_name": model,
                    "dataset_name": model,
                    "name": model,
                    "source_table": model,
                    "search_text": model,
                    "yaml_path": yaml_path,
                }
            ]
        )
        record_digests(agent_config, agent_config.current_datasource, {yaml_path: file_digest(Path(yaml_path))})
        return {"success": True}

    return sync


@pytest.fixture
def project(real_agent_config):
    """A Dosi project with a second datasource bound next to the default one."""
    real_agent_config.resolve_semantic_adapter = lambda *_: "dosi"
    datasources = real_agent_config.services.datasources
    datasources[OTHER] = copy.copy(datasources[DATASOURCE])
    root = Path(real_agent_config.project_root) / "subject" / "semantic_models"
    (root / DATASOURCE).mkdir(parents=True)
    (root / OTHER).mkdir(parents=True)
    return real_agent_config, root


def _reconcile(agent_config, paths=()):
    return _reconcile_with(agent_config, _fake_sync, paths)


def _reconcile_with(agent_config, make_sync, paths=()):
    """Reconcile with ``make_sync(config)`` standing in for each datasource's ``sync_osi_to_db``."""

    class _Tools:
        def __init__(self, agent_config, **_kwargs):
            self.sync_osi_to_db = make_sync(agent_config)

    with patch("datus.tools.func_tool.generation_tools.GenerationTools", _Tools):
        return reconcile_semantic_artifacts(agent_config, paths)


def _metric_names(agent_config, datasource=DATASOURCE) -> set[str]:
    from datus_storage_base.conditions import and_

    rag = MetricRAG(agent_config, datasource_id=datasource)
    rows = rag.storage._search_all(where=and_(*rag._sub_agent_conditions())).to_pylist()
    return {row["name"] for row in rows}


def _state_file(agent_config) -> Path:
    return Path(agent_config.path_manager.project_data_dir) / "semantic_sync_state.json"


def _node(agent_config, path, datasource=DATASOURCE):
    store = get_subject_tree_store(project=agent_config.project_name, datasource_id=datasource)
    return store.get_node_by_path(path)


def _seed(agent_config, root):
    """Two datasources, each with a model file; ``orders.yml`` holds two metrics."""
    orders = root / DATASOURCE / "orders.yml"
    orders.write_text(_model("orders", {"order_count": ["sales", "orders"], "order_total": ["sales", "orders"]}))
    users = root / DATASOURCE / "users.yml"
    users.write_text(_model("users", {"user_count": ["sales", "users"]}))
    stock = root / OTHER / "stock.yml"
    stock.write_text(_model("stock", {"stock_level": ["ops"]}))
    result = _reconcile(agent_config, ["subject/semantic_models"])
    assert not result.failures
    assert _metric_names(agent_config) == {"order_count", "order_total", "user_count"}
    assert _metric_names(agent_config, OTHER) == {"stock_level"}
    return orders, users, stock


def test_new_files_are_projected_per_datasource(project):
    agent_config, root = project
    _seed(agent_config, root)

    assert _node(agent_config, ["sales", "orders"])["name"] == "orders"
    assert _node(agent_config, ["ops"], OTHER)["name"] == "ops"
    # The other datasource's metric landed in its own scope, not the active one.
    assert _node(agent_config, ["ops"]) is None


def test_deleting_a_model_file_drops_its_metrics_and_emptied_directory(project):
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)

    orders.unlink()
    result = _reconcile(agent_config)

    assert result.pruned_files == [str(orders)]
    assert _metric_names(agent_config) == {"user_count"}
    assert _node(agent_config, ["sales", "orders"]) is None
    # ``sales`` still holds ``users``.
    assert _node(agent_config, ["sales", "users"])["name"] == "users"
    assert result.removed_subject_paths == [["sales", "orders"]]


def test_deleting_a_datasource_directory_prunes_only_that_datasource(project):
    agent_config, root = project
    _seed(agent_config, root)

    shutil.rmtree(root / DATASOURCE)
    result = _reconcile(agent_config)

    assert not result.failures
    assert _metric_names(agent_config) == set()
    assert _node(agent_config, ["sales"]) is None
    assert _metric_names(agent_config, OTHER) == {"stock_level"}


@pytest.mark.parametrize("deleted", ["subject/semantic_models", "subject"])
def test_deleting_an_ancestor_directory_prunes_every_datasource(project, deleted):
    agent_config, root = project
    _seed(agent_config, root)

    shutil.rmtree(Path(agent_config.project_root) / deleted)
    _reconcile(agent_config)

    assert _metric_names(agent_config) == set()
    assert _metric_names(agent_config, OTHER) == set()
    assert _node(agent_config, ["sales"]) is None
    assert _node(agent_config, ["ops"], OTHER) is None


def test_editing_a_file_removes_dropped_metrics_and_moves_the_rest(project):
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)

    orders.write_text(_model("orders", {"order_count": ["finance"]}))
    result = _reconcile(agent_config, ["subject/semantic_models/california_schools/orders.yml"])

    assert result.synced_files == [str(orders.resolve())]
    assert _metric_names(agent_config) == {"order_count", "user_count"}
    assert _node(agent_config, ["finance"])["name"] == "finance"
    assert _node(agent_config, ["sales", "orders"]) is None


def test_renaming_a_file_keeps_its_metrics(project):
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)

    renamed = orders.with_name("orders_v2.yml")
    orders.rename(renamed)
    result = _reconcile(agent_config, ["subject/semantic_models/california_schools/orders_v2.yml"])

    assert _metric_names(agent_config) == {"order_count", "order_total", "user_count"}
    assert _node(agent_config, ["sales", "orders"])["name"] == "orders"
    assert result.removed_subject_paths == []


def test_directories_that_were_already_empty_are_kept(project):
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)
    store = get_subject_tree_store(project=agent_config.project_name, datasource_id=DATASOURCE)
    store.find_or_create_path(["sales", "orders", "drafts"])
    store.find_or_create_path(["handmade"])

    orders.unlink()
    _reconcile(agent_config)

    # ``orders`` lost its metrics but still holds a child someone created.
    assert _node(agent_config, ["sales", "orders", "drafts"])["name"] == "drafts"
    assert _node(agent_config, ["handmade"])["name"] == "handmade"


def test_a_file_under_an_unbound_datasource_is_reported(project):
    agent_config, root = project
    stray = root / "unbound" / "model.yml"
    stray.parent.mkdir()
    stray.write_text(_model("model", {"m": ["x"]}))

    result = _reconcile(agent_config, ["subject/semantic_models/unbound"])

    assert "not a datasource of this project" in result.failures[str(stray.parent.resolve())]
    assert result.synced_files == []


def test_a_file_edited_behind_the_save_path_is_re_projected(project):
    """An agent ``edit_file`` or a shell ``sed`` names no path; the digest notices."""
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)

    orders.write_text(_model("orders", {"order_count": ["sales", "orders"], "refund_count": ["sales", "orders"]}))
    result = _reconcile(agent_config)

    assert result.synced_files == [str(orders.resolve())]
    assert _metric_names(agent_config) == {"order_count", "refund_count", "user_count"}


def test_an_untouched_file_is_not_re_projected(project):
    agent_config, root = project
    _seed(agent_config, root)

    result = _reconcile(agent_config)

    assert result.synced_files == []


def test_the_first_run_records_a_baseline_for_a_matching_projection(project):
    """Without digests, a projection that still matches its file is trusted, not re-embedded."""
    agent_config, root = project
    orders = root / DATASOURCE / "orders.yml"
    orders.write_text(_model("orders", {"order_count": ["sales"]}))
    _fake_sync(agent_config)(str(orders.resolve()))
    _state_file(agent_config).unlink()

    result = _reconcile(agent_config)

    assert result.synced_files == []
    assert load_digests(agent_config, DATASOURCE) == {str(orders.resolve()): file_digest(orders)}


def test_the_first_run_re_projects_a_file_the_kb_does_not_match(project):
    """Lost state must not bless a stale KB: a declared metric missing from it is synced."""
    agent_config, root = project
    orders = root / DATASOURCE / "orders.yml"
    orders.write_text(_model("orders", {"order_count": ["sales"]}))
    _fake_sync(agent_config)(str(orders.resolve()))
    _state_file(agent_config).unlink()
    orders.write_text(_model("orders", {"order_count": ["sales"], "refund_count": ["sales"]}))

    result = _reconcile(agent_config)

    assert result.synced_files == [str(orders.resolve())]
    assert _metric_names(agent_config) == {"order_count", "refund_count"}


def test_a_prune_that_fails_is_reported(project, monkeypatch):
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)
    orders.unlink()

    def boom(self, yaml_path):
        raise RuntimeError("storage down")

    monkeypatch.setattr(MetricRAG, "delete_artifact_rows", boom)
    result = _reconcile(agent_config)

    assert result.failures[str(orders.resolve())] == "storage down"


def test_a_pruned_file_is_forgotten(project):
    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)

    orders.unlink()
    _reconcile(agent_config)

    assert str(orders.resolve()) not in load_digests(agent_config, DATASOURCE)


def test_rows_recorded_outside_the_models_root_are_left_alone(project, tmp_path):
    """A relocated mount must not read as every file deleted."""
    agent_config, root = project
    elsewhere = tmp_path / "old_mount" / "orders.yml"
    elsewhere.parent.mkdir()
    elsewhere.write_text(_model("orders", {"order_count": ["sales"]}))
    _fake_sync(agent_config)(str(elsewhere))
    elsewhere.unlink()

    _reconcile(agent_config)

    assert _metric_names(agent_config) == {"order_count"}


def test_a_sync_that_lands_while_waiting_for_the_lock_is_not_repeated(project, monkeypatch):
    """Every open browser reconciles on the same fs:changed event. One that
    finds the file already projected once it holds the lock must skip it."""
    from contextlib import contextmanager

    from datus.storage.semantic_model import artifact_file

    agent_config, root = project
    orders, _, _ = _seed(agent_config, root)
    orders.write_text(_model("orders", {"order_count": ["sales", "orders"]}))

    calls = []

    def counting_sync(config):
        sync = _fake_sync(config)
        return lambda yaml_path, **kw: calls.append(yaml_path) or sync(yaml_path, **kw)

    real_lock = artifact_file.semantic_artifact_lock
    raced = []

    @contextmanager
    def lock_after_a_competing_reconcile(path):
        # The competing request wins the race: it projects the file first.
        if not raced:
            raced.append(path)
            _reconcile_with(agent_config, counting_sync)
        with real_lock(path):
            yield

    monkeypatch.setattr(artifact_file, "semantic_artifact_lock", lock_after_a_competing_reconcile)
    _reconcile_with(agent_config, counting_sync)

    assert calls == [str(orders.resolve())]
