"""Metric source authoring and OSI sync share the flat document contract."""

import copy
import json

import pytest
import yaml

from datus.tools.semantic_tools.metric_author import MetricAuthor
from datus.tools.semantic_tools.osi_document import load_osi_document


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_synced_metrics_can_be_read_and_mutated_without_rewrapping(tmp_path, suffix):
    model = {
        "name": "shop",
        "datasets": [
            {
                "name": "customers",
                "source": "crm.customers",
                "fields": [{"name": "id", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "id"}]}}],
            }
        ],
        "metrics": [
            {
                "name": "customer_count",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(customers.id)"}]},
            }
        ],
    }
    document = model
    document["version"] = "0.2.0.dev0"
    original = copy.deepcopy(document)
    path = tmp_path / f"model.{suffix}"
    path.write_text(json.dumps(document) if suffix == "json" else yaml.safe_dump(document))
    published = load_osi_document(str(path), "shop")
    author = MetricAuthor(str(tmp_path))
    for metric in published.metrics:
        source = author.read(metric.name)
        assert source.semantic_model == model["name"]
        assert yaml.safe_load(source.text) == model["metrics"][0]
    assert yaml.safe_load(path.read_text()) == original

    updated = {**model["metrics"][0], "description": "Updated"}
    assert author.validate(yaml.safe_dump(updated)).valid
    author.write("customer_count", yaml.safe_dump(updated))
    assert yaml.safe_load(author.read("customer_count").text) == updated
    created = {**updated, "name": "new_count"}
    assert author.validate(yaml.safe_dump(created)).valid
    author.write("new_count", yaml.safe_dump(created), create=True)
    assert yaml.safe_load(author.read("new_count").text) == created
    author.delete("customer_count")
    author.delete("new_count")
    expected = copy.deepcopy(original)
    expected["metrics"] = []
    assert yaml.safe_load(path.read_text()) == expected


def test_multi_root_authoring_edits_legacy_files_in_place_and_creates_in_primary(tmp_path):
    primary, legacy = tmp_path / "stable-id", tmp_path / "old"
    primary.mkdir()
    legacy.mkdir()

    def model(name, metric):
        return {"version": "0.2.0.dev0", "name": name, "datasets": [], "metrics": [{"name": metric}]}

    old_file, new_file = legacy / "shop.yml", primary / "shop.yml"
    old_file.write_text(yaml.safe_dump(model("old", "old_metric")))
    new_file.write_text(yaml.safe_dump(model("current", "current_metric")))
    author = MetricAuthor(str(primary), legacy_semantic_models_paths=[str(legacy)], load_model=lambda _: None)
    assert author.read("old_metric").file_path == str(old_file)
    author.write("old_metric", yaml.safe_dump({"name": "old_metric", "description": "Edited"}))
    assert yaml.safe_load(old_file.read_text())["metrics"][0]["description"] == "Edited"
    assert yaml.safe_load(new_file.read_text())["metrics"] == [{"name": "current_metric"}]
    author.write("new_metric", yaml.safe_dump({"name": "new_metric"}), create=True)
    assert author.read("new_metric").file_path == str(new_file)
    author.delete("old_metric")
    assert yaml.safe_load(old_file.read_text())["metrics"] == []
