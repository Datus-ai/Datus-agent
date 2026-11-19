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
