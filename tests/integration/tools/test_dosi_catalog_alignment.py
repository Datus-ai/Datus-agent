# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Every key the bundled Dosi engine publishes on a metric or dimension row is
either shown to the agent by ``get_metric`` or withheld on purpose.

The catalog grows on the engine side; this layer copies keys by name. When the
two drift, the agent silently loses what the engine already answers — that is
how ``params`` and ``required_dimensions`` went missing from ``get_metric``. A
key the engine adds fails here until it is classified below."""

from __future__ import annotations

from pathlib import Path

import pytest

from datus.tools.func_tool.semantic_tools import _metric_detail_row, _normalize_dimension_rows
from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.engine import load_binding
from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

MODELS = Path(__file__).parents[2] / "data" / "semantic_models"
MODEL_PATHS = [MODELS / "window_filters" / "model.yaml", MODELS / "attribution" / "model.yaml"]

# Metric keys get_metric shows under the engine's own name and value.
METRIC_SHOWN = {
    "name",
    "description",
    "kind",
    "datasets",
    "params",
    "window_family",
    "window_function",
    "derive_family",
    "derive_base",
    "derive_expr",
    "derive_predicate",
    "subset_of",
    "required_dimensions",
    "parts",
    "time_axis",
}
# Metric keys withheld, or shown in another form.
METRIC_WITHHELD = {
    # Compiled measure names: unusable as tool arguments.
    "measures",
    # Replaced by the effective time axis the runtime resolves.
    "time_dimension",
    # The same declaration as `params`, as a JSON schema.
    "param_schema",
    # Attribution reports its own strategy when it runs.
    "attribution_strategy",
    "attributable",
    "attribution",
    # Flattened structure that misleads read alone (see DosiRuntime.list_metrics).
    "derive_members",
    "leaf_measures",
    # get_dimensions applies it natively.
    "conformed_dimensions",
}
# Dimension keys get_metric shows under the engine's own name and value.
DIMENSION_SHOWN = {"name", "description", "parts", "pairs", "params"}
# Dimension keys withheld, or shown in another form.
DIMENSION_WITHHELD = {
    # Shown as `type`, `recommended`, `recommendation_source` and the probed
    # `time_granularities`.
    "is_time",
    "is_dimension",
    "source",
    "time_granularity",
    # Storage encoding the engine applies itself.
    "time_format",
    "time_storage",
    # Superseded by the `definition` of the field's parts entry.
    "lod_mode",
    "lod_dims",
    "lod_base",
}


def _shown(value):
    return value not in (None, [], {}, "")


def _without_slots(params):
    return [{k: v for k, v in p.items() if k != "slots"} for p in params]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_path", MODEL_PATHS, ids=lambda p: p.parent.name)
async def test_every_published_metric_key_is_classified_and_shown_as_published(model_path):
    engine = load_binding().Engine(model_path=str(model_path), mode="datus")
    runtime = DosiRuntime(DosiConfig(semantic_model_path=str(model_path), dialect="duckdb"))
    metrics = {m.name: m for m in await runtime.list_metrics(limit=10_000)}

    for row in engine.metrics():
        unclassified = set(row) - METRIC_SHOWN - METRIC_WITHHELD
        assert not unclassified, f"classify the engine's new metric keys {sorted(unclassified)}"
        published = {
            key: _without_slots(value) if key == "params" else value
            for key, value in row.items()
            if key in METRIC_SHOWN and _shown(value)
        }
        detail = _metric_detail_row(metrics[row["name"]])
        assert {key: detail.get(key) for key in published} == published, row["name"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_path", MODEL_PATHS, ids=lambda p: p.parent.name)
async def test_every_published_dimension_key_is_classified_and_shown_as_published(model_path):
    engine = load_binding().Engine(model_path=str(model_path), mode="datus")
    runtime = DosiRuntime(DosiConfig(semantic_model_path=str(model_path), dialect="duckdb"))
    metric = engine.metrics()[0]["name"]

    native = engine.dimensions(metric)
    shown = {row["name"]: row for row in _normalize_dimension_rows(await runtime.get_dimensions(metric))}
    for row in native:
        unclassified = set(row) - DIMENSION_SHOWN - DIMENSION_WITHHELD
        assert not unclassified, f"classify the engine's new dimension keys {sorted(unclassified)}"
        published = {key: value for key, value in row.items() if key in DIMENSION_SHOWN and _shown(value)}
        assert {key: shown[row["name"]].get(key) for key in published} == published, row["name"]
