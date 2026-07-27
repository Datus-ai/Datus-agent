from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.agent.node import semantic_authoring
from datus.tools.func_tool.osi_target_tools import (
    OsiSemanticModelTargetState,
    OsiSemanticModelTargetTools,
)


@pytest.fixture(autouse=True)
def _stub_osi_schema_validation(monkeypatch):
    monkeypatch.setattr(semantic_authoring, "validate_osi_core_document", lambda document: None)


def _config(tmp_path: Path):
    model_root = tmp_path / "subject" / "semantic_models"
    model_dir = model_root / "warehouse"
    return SimpleNamespace(
        current_datasource="warehouse",
        path_manager=SimpleNamespace(
            project_root=tmp_path,
            semantic_model_path=lambda datasource: model_root / datasource,
        ),
        project_root=tmp_path,
        resolve_semantic_adapter=lambda _: "osi",
    ), model_dir


def _write_model(
    path: Path,
    *,
    name: str,
    dataset: str = "orders",
    source: str = "analytics.orders",
    description: str = "Order analytics",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version: 0.2.0.dev0\n"
        "semantic_model:\n"
        f"  - name: {name}\n"
        f"    description: {description}\n"
        "    datasets:\n"
        f"      - name: {dataset}\n"
        f"        source: {source}\n"
        f"        description: {description} dataset\n",
        encoding="utf-8",
    )


def test_list_scans_live_yaml_inventory_without_mutating_binding(tmp_path):
    config, model_dir = _config(tmp_path)
    _write_model(model_dir / "orders.yml", name="orders_model")
    _write_model(
        model_dir / "domains" / "players.yaml",
        name="players_model",
        dataset="players",
        source="games.players",
        description="Player behavior",
    )
    (model_dir / "broken.yml").write_text("semantic_model: [\n", encoding="utf-8")
    (model_dir / "metrics").mkdir(parents=True)
    (model_dir / "metrics" / "legacy.yml").write_text("metric:\n  name: legacy\n", encoding="utf-8")
    other_dir = tmp_path / "subject" / "semantic_models" / "other"
    _write_model(other_dir / "ignored.yml", name="ignored")

    state = OsiSemanticModelTargetState()
    tools = OsiSemanticModelTargetTools(config, target_state=state)
    result = tools.list_existing_osi_semantic_models()

    assert result.success
    assert result.result["status"] == "partial"
    assert result.result["count"] == 2
    assert {item["semantic_model_name"] for item in result.result["semantic_models"]} == {
        "orders_model",
        "players_model",
    }
    players = next(item for item in result.result["semantic_models"] if item["semantic_model_name"] == "players_model")
    assert players["datasets"] == [
        {
            "name": "players",
            "source": "games.players",
            "description": "Player behavior dataset",
        }
    ]
    assert result.result["issues"][0]["semantic_model_file"].endswith("/broken.yml")
    assert state.selected is None
    assert state.last_error_code == ""


@pytest.mark.parametrize("selector_kind", ["name", "relative", "absolute"])
def test_bind_exact_selector_records_canonical_target_and_revision(tmp_path, selector_kind):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    _write_model(target, name="orders_model")
    tools = OsiSemanticModelTargetTools(config)

    kwargs = {
        "name": {"semantic_model_name": "orders_model"},
        "relative": {"semantic_model_file": "subject/semantic_models/warehouse/orders.yml"},
        "absolute": {"semantic_model_file": str(target)},
    }[selector_kind]
    result = tools.bind_osi_semantic_model_target(**kwargs)

    assert result.success
    assert result.result["status"] == "bound"
    assert tools.target_state.bound["absolute_path"] == str(target.resolve())
    assert tools.target_state.artifact_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "selector",
    [
        "../other/model.yml",
        "subject/semantic_models/other/model.yml",
        "subject/semantic_models/warehouse/missing.yml",
    ],
)
def test_bind_rejects_invalid_selector(tmp_path, selector):
    config, _ = _config(tmp_path)
    tools = OsiSemanticModelTargetTools(config)

    result = tools.bind_osi_semantic_model_target(semantic_model_file=selector)

    assert not result.success
    assert result.result["code"] == "semantic_model_target_invalid"


def test_bind_rejects_malformed_and_multi_model_yaml(tmp_path):
    config, model_dir = _config(tmp_path)
    model_dir.mkdir(parents=True)
    (model_dir / "broken.yml").write_text("semantic_model: [\n", encoding="utf-8")
    (model_dir / "multi.yml").write_text(
        "semantic_model:\n  - name: first\n    datasets: []\n  - name: second\n    datasets: []\n",
        encoding="utf-8",
    )
    tools = OsiSemanticModelTargetTools(config)

    broken = tools.bind_osi_semantic_model_target(semantic_model_file="broken.yml")
    multi = tools.bind_osi_semantic_model_target(semantic_model_file="multi.yml")

    assert not broken.success
    assert broken.result["code"] == "semantic_model_target_invalid"
    assert not multi.success
    assert multi.result["code"] == "semantic_model_target_invalid"
    assert semantic_authoring.inspect_osi_semantic_model_inventory(config)["recoverable_models"] == []


def test_inventory_excludes_core_schema_invalid_yaml(tmp_path, monkeypatch):
    config, model_dir = _config(tmp_path)
    target = model_dir / "invalid.yml"
    _write_model(target, name="invalid_model")
    monkeypatch.setattr(
        semantic_authoring,
        "validate_osi_core_document",
        lambda document: "version does not match the OSI core schema",
    )
    tools = OsiSemanticModelTargetTools(config)

    raw_inventory = semantic_authoring.inspect_osi_semantic_model_inventory(config)
    inventory = tools.list_existing_osi_semantic_models()
    bound = tools.bind_osi_semantic_model_target(semantic_model_file=str(target))

    assert raw_inventory["recoverable_models"][0]["semantic_model_name"] == "invalid_model"
    assert inventory.result["status"] == "invalid"
    assert inventory.result["semantic_models"] == []
    assert inventory.result["issues"][0]["code"] == "invalid_osi_core_schema"
    assert not bound.success
    assert bound.result["code"] == "semantic_model_target_invalid"


def test_semantic_plan_can_recover_unique_core_schema_invalid_model(tmp_path, monkeypatch):
    config, model_dir = _config(tmp_path)
    target = model_dir / "legacy.yml"
    _write_model(target, name="orders_model")
    monkeypatch.setattr(
        semantic_authoring,
        "validate_osi_core_document",
        lambda document: "datasets do not match the OSI core schema",
    )
    inspect_inventory = semantic_authoring.inspect_osi_semantic_model_inventory
    calls = 0

    def inspect_once(agent_config):
        nonlocal calls
        calls += 1
        return inspect_inventory(agent_config)

    monkeypatch.setattr(semantic_authoring, "inspect_osi_semantic_model_inventory", inspect_once)
    tools = OsiSemanticModelTargetTools(config)

    result = tools.plan_osi_semantic_model_target(semantic_model_name="orders_model")

    assert result.success
    assert result.result["repair_required"] is True
    assert tools.target_state.planned["absolute_path"] == str(target.resolve())
    assert tools.target_state.artifact_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert calls == 1


def test_planned_existing_target_rejects_external_revision_change(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    _write_model(target, name="orders_model")
    tools = OsiSemanticModelTargetTools(config)
    assert tools.plan_osi_semantic_model_target(semantic_model_name="orders_model").success

    target.write_text(target.read_text(encoding="utf-8") + "# external edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after planning"):
        tools.target_state.require_planned_path(target)


def test_planned_new_target_rejects_path_created_after_planning(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    tools = OsiSemanticModelTargetTools(config)
    assert tools.plan_osi_semantic_model_target(semantic_model_name="orders").success

    _write_model(target, name="orders")

    with pytest.raises(ValueError, match="created after planning"):
        tools.target_state.require_planned_path(target)


def test_bind_name_requires_unique_live_match(tmp_path):
    config, model_dir = _config(tmp_path)
    _write_model(model_dir / "one.yml", name="shared")
    _write_model(model_dir / "two.yaml", name="shared")
    tools = OsiSemanticModelTargetTools(config)

    inventory = tools.list_existing_osi_semantic_models()
    assert inventory.result["status"] == "invalid"
    assert inventory.result["semantic_models"] == []

    result = tools.bind_osi_semantic_model_target(semantic_model_name="shared")

    assert not result.success
    assert result.result["code"] == "semantic_model_target_invalid"
    assert {issue["semantic_model_file"] for issue in inventory.result["issues"]} == {
        "subject/semantic_models/warehouse/one.yml",
        "subject/semantic_models/warehouse/two.yaml",
    }


def test_bind_exact_path_rejects_duplicate_model_name(tmp_path):
    config, model_dir = _config(tmp_path)
    first = model_dir / "one.yml"
    _write_model(first, name="shared")
    _write_model(model_dir / "two.yaml", name="shared")
    tools = OsiSemanticModelTargetTools(config)

    result = tools.bind_osi_semantic_model_target(semantic_model_file=str(first))

    assert not result.success
    assert result.result["code"] == "semantic_model_target_invalid"
    assert "not safe" in result.error


def test_failed_rebind_clears_previous_unwritten_target(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    _write_model(target, name="orders_model")
    tools = OsiSemanticModelTargetTools(config)
    assert tools.bind_osi_semantic_model_target(semantic_model_file=str(target)).success

    result = tools.bind_osi_semantic_model_target(semantic_model_file="missing.yml")

    assert not result.success
    assert tools.target_state.bound is None
    with pytest.raises(ValueError, match="Bind an existing OSI semantic model"):
        tools.target_state.require_bound_path(target)


def test_failed_rebind_after_write_keeps_authored_target_poisoned(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    _write_model(target, name="orders_model")
    tools = OsiSemanticModelTargetTools(config)
    assert tools.bind_osi_semantic_model_target(semantic_model_file=str(target)).success
    tools.target_state.authored_metric_names = ["order_count"]

    result = tools.bind_osi_semantic_model_target(semantic_model_file="missing.yml")

    assert not result.success
    assert tools.target_state.bound is not None
    assert tools.target_state.last_error_code == "semantic_model_target_invalid"


def test_authored_target_cannot_rebind_to_a_different_revision_or_model(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    _write_model(target, name="orders_model")
    tools = OsiSemanticModelTargetTools(config)
    assert tools.bind_osi_semantic_model_target(semantic_model_file=str(target)).success
    tools.target_state.authored_metric_names = ["order_count"]

    _write_model(target, name="replacement_model")
    result = tools.bind_osi_semantic_model_target(semantic_model_file=str(target))

    assert not result.success
    assert "cannot change after authoring started" in result.error
    assert tools.target_state.bound["semantic_model_name"] == "orders_model"
    assert tools.target_state.last_error_code == "semantic_model_target_invalid"


def test_failed_replan_clears_previous_plan(tmp_path):
    config, _ = _config(tmp_path)
    tools = OsiSemanticModelTargetTools(config)
    assert tools.plan_osi_semantic_model_target(semantic_model_name="orders").success

    result = tools.plan_osi_semantic_model_target()

    assert not result.success
    assert tools.target_state.planned is None


def test_plan_only_blocks_duplicate_names_relevant_to_the_target(tmp_path):
    config, model_dir = _config(tmp_path)
    _write_model(model_dir / "one.yml", name="shared")
    _write_model(model_dir / "two.yml", name="shared")
    tools = OsiSemanticModelTargetTools(config)

    unrelated = tools.plan_osi_semantic_model_target(semantic_model_name="new_model")
    duplicate = tools.plan_osi_semantic_model_target(semantic_model_name="shared")

    assert unrelated.success
    assert not duplicate.success
    assert len(duplicate.result["candidates"]) == 2
    assert tools.target_state.planned is None


@pytest.mark.parametrize("invalid_sources", [{"analytics.two"}, {"analytics.one", "analytics.two"}])
def test_duplicate_names_span_valid_and_recoverable_models(tmp_path, monkeypatch, invalid_sources):
    config, model_dir = _config(tmp_path)
    _write_model(model_dir / "one.yml", name="shared", source="analytics.one")
    _write_model(model_dir / "two.yml", name="shared", source="analytics.two")

    def validate(document):
        source = document["semantic_model"][0]["datasets"][0]["source"]
        return "invalid core schema" if source in invalid_sources else None

    monkeypatch.setattr(semantic_authoring, "validate_osi_core_document", validate)
    inventory = semantic_authoring.inspect_osi_semantic_model_inventory(config)

    assert inventory["models"] == []
    assert inventory["recoverable_models"] == []
    duplicate_paths = {
        issue["semantic_model_file"]
        for issue in inventory["issues"]
        if issue["code"] == "duplicate_semantic_model_name"
    }
    assert duplicate_paths == {
        "subject/semantic_models/warehouse/one.yml",
        "subject/semantic_models/warehouse/two.yml",
    }


def test_query_backed_parse_warning_does_not_make_exact_target_unbindable(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "query.yml"
    target.parent.mkdir(parents=True)
    target.write_text(
        "version: 0.2.0.dev0\n"
        "semantic_model:\n"
        "  - name: query_model\n"
        "    datasets:\n"
        "      - name: query_dataset\n"
        "        source: SELECT * FROM\n"
        "        custom_extensions:\n"
        "          - vendor_name: DATUS\n"
        '            data: \'{"source_type":"query"}\'\n',
        encoding="utf-8",
    )
    tools = OsiSemanticModelTargetTools(config)

    inventory = tools.list_existing_osi_semantic_models()
    bound = tools.bind_osi_semantic_model_target(semantic_model_file=str(target))

    assert inventory.result["status"] == "found"
    assert inventory.result["issues"] == []
    assert inventory.result["discovery_warnings"][0]["code"] == "query_backed_dataset_tables_unknown"
    candidate = inventory.result["semantic_models"][0]
    assert "table_references" not in candidate
    assert "source" not in candidate["datasets"][0]
    assert bound.success


def test_bound_revision_must_match_live_file(tmp_path):
    config, model_dir = _config(tmp_path)
    target = model_dir / "orders.yml"
    _write_model(target, name="orders_model")
    tools = OsiSemanticModelTargetTools(config)
    assert tools.bind_osi_semantic_model_target(semantic_model_file=str(target)).success

    target.write_text(target.read_text(encoding="utf-8") + "\n# external edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after selection"):
        tools.target_state.require_current_revision(target)
