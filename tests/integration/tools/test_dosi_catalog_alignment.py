# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Every key the bundled Dosi engine publishes on a metric or dataset
description is either shown to the agent by ``get_metric`` / ``get_dataset`` or
withheld on purpose.

The descriptions grow on the engine side; this layer renders keys by name.
When the two drift, the agent silently loses what the engine already answers —
that is how ``params`` and ``required_dimensions`` once went missing from
``get_metric``. A key the engine adds fails here until it is classified below."""

from __future__ import annotations

from pathlib import Path

import pytest

from datus.tools.func_tool.semantic_tools import render_dataset_detail, render_metric_detail
from datus.tools.semantic_tools.dosi.engine import load_binding

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

MODELS = Path(__file__).parents[2] / "data" / "semantic_models"
MODEL_PATHS = [MODELS / "window_filters" / "model.yaml", MODELS / "attribution" / "model.yaml"]

# Metric keys get_metric renders.
METRIC_SHOWN = {
    "description",
    "kind",
    "expr",
    "datasets",
    "dimension_datasets",
    "window",
    "compares_across",
    "params",
    "required_dimensions",
    "requires_time_axis",
    "time_granularities",
}
# Metric keys withheld, or shown in another form.
METRIC_WITHHELD = {
    # The caller named it.
    "name",
    # Compiled measure names: unusable as tool arguments.
    "measures",
    "explicit_measure",
    # `time` says what a query can do with the axis; the column is in get_dataset.
    "time_dimension",
    # The same declaration as `params`, as a JSON schema.
    "param_schema",
    # Attribution reports its own strategy when it runs.
    "attribution_strategy",
    "attributable",
    "attribution",
    # `window` in full; these restate two of its keys.
    "window_family",
    "window_function",
    # `expr` states the derivation in one line.
    "derive",
    "derive_family",
    "derive_base",
    "derive_expr",
    "subset_of",
    "derive_members",
    "leaf_measures",
    # `dimension_datasets` applies it.
    "conformed_dimensions",
}
# Dataset keys get_dataset renders.
DATASET_SHOWN = {"description", "source", "primary_time_dimension", "primary_key", "fields", "relationships"}
DATASET_WITHHELD = {"name"}
FIELD_SHOWN = {
    "name",
    "description",
    "expr",
    "reads",
    "datasets",
    "time_axis",
    "is_time",
    "time_format",
    "is_dimension",
}


@pytest.mark.parametrize("model_path", MODEL_PATHS, ids=lambda p: p.parent.name)
def test_every_metric_key_is_classified_and_rendered(model_path):
    engine = load_binding().Engine(model_path=str(model_path), mode="datus")

    for row in engine.metrics():
        detail = engine.metric(row["name"])
        unclassified = set(detail) - METRIC_SHOWN - METRIC_WITHHELD
        assert not unclassified, f"classify the engine's new metric keys {sorted(unclassified)}"
        text = render_metric_detail(detail)
        expected_expr = [f"expr: {' '.join(detail['expr'].split())}"] if detail.get("expr") else []
        assert [line for line in text.splitlines() if line.startswith("expr: ")] == expected_expr, row["name"]
        for dataset in detail.get("datasets") or []:
            assert dataset in text, row["name"]
        for param in detail.get("params") or []:
            assert f"- {param['name']} " in text, row["name"]
        for dimension in detail.get("required_dimensions") or []:
            assert dimension in text, row["name"]
        assert ("requires_time_axis: true" in text) == bool(detail.get("requires_time_axis")), row["name"]


@pytest.mark.parametrize("model_path", MODEL_PATHS, ids=lambda p: p.parent.name)
def test_every_dataset_key_is_classified_and_rendered(model_path):
    engine = load_binding().Engine(model_path=str(model_path), mode="datus")

    for row in engine.datasets():
        detail = engine.dataset(row["name"])
        unclassified = set(detail) - DATASET_SHOWN - DATASET_WITHHELD
        assert not unclassified, f"classify the engine's new dataset keys {sorted(unclassified)}"
        text = render_dataset_detail(detail)
        for field in detail["fields"]:
            unclassified = set(field) - FIELD_SHOWN
            assert not unclassified, f"classify the engine's new field keys {sorted(unclassified)}"
            assert field["name"] in text, (row["name"], field["name"])
        for relationship in detail.get("relationships") or []:
            assert f"{relationship['cardinality']} {relationship['to']}" in text, row["name"]
