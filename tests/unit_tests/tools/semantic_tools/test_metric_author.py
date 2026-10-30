"""Metric source authoring accepts the same document envelopes as OSI sync."""

import copy
import json

import pytest
import yaml

from datus.tools.semantic_tools.metric_author import MetricAuthor
from datus.tools.semantic_tools.osi_document import load_osi_document


@pytest.mark.parametrize("shape", ["flat", "mapping", "list"])
@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_synced_metrics_can_be_read_and_mutated_without_rewrapping(tmp_path, shape, suffix):
    model = {
        "name": "shop",
        "datasets": [{"name": "customers", "source": "crm.customers"}],
        "metrics": [
            {
                "name": "customer_count",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(customers.id)"}]},
            }
        ],
    }
    document = model if shape == "flat" else {"semantic_model": model if shape == "mapping" else [model]}
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
    expected_model = (
        expected
        if shape == "flat"
        else expected["semantic_model"]
        if shape == "mapping"
        else expected["semantic_model"][0]
    )
    expected_model["metrics"] = []
    assert yaml.safe_load(path.read_text()) == expected
