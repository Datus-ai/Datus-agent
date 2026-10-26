# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""What a query of one metric has to carry, read from the bundled Dosi engine.

A metric's requirements are only as good as the probes behind them: a probe
that leaves out a dimension the engine requires is refused for that reason, and
the metric then reads as having no queryable grain at all. These run the real
engine, so a requirement it adds or a refusal it renames shows up here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

BASE_MODEL = Path(__file__).parents[2] / "data" / "semantic_models" / "window_filters" / "model.yaml"

# A period-over-period window, and a compose over it that has no window of its own.
EXTRA_METRICS = """
      - name: score_prev_day
        description: Total score one day earlier
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: SUM(scores.score)
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v": "1.2", "window": {"type": "pop", "offset": "1 day"}}'

      - name: score_change
        description: Day-over-day change of the total score
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: SUM(scores.score) - SUM(scores.score)
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v": "1.4", "derive": {"type": "compose", "expr": "score_total - score_prev_day"}}'
"""


@pytest.fixture
def runtime(tmp_path):
    model = tmp_path / "model.yaml"
    model.write_text(BASE_MODEL.read_text().rstrip("\n") + "\n" + EXTRA_METRICS)
    return DosiRuntime(DosiConfig(semantic_model_path=str(model), dialect="duckdb"))


@pytest.fixture
def probes(monkeypatch):
    """The group_by of every compile probe the runtime makes."""
    seen = []
    real = DosiRuntime._probe_compile

    def probe(engine, binding, metric_name, group_by, **kwargs):
        seen.append((metric_name, [item["field"] for item in group_by]))
        return real(engine, binding, metric_name, group_by, **kwargs)

    monkeypatch.setattr(DosiRuntime, "_probe_compile", staticmethod(probe))
    return seen


@pytest.mark.asyncio
async def test_listing_the_catalog_compiles_nothing(runtime, probes):
    metrics = {metric.name: metric for metric in await runtime.list_metrics(limit=100)}

    assert probes == []
    assert metrics["item_rank"].metadata["required_dimensions"] == ["scores.item"]
    assert metrics["score_total"].metadata["required_dimensions"] == []
    assert all("requires_time_axis" not in metric.metadata for metric in metrics.values())


@pytest.mark.asyncio
async def test_a_required_dimension_rides_along_in_every_grain_probe(runtime, probes):
    """Probed without scores.item, item_rank is refused for the missing
    exclusion and every grain of its time field reads as unsupported."""
    dimensions = {d.name: d for d in await runtime.get_dimensions("item_rank")}

    item = dimensions["scores.item"]
    assert item.required is True
    assert item.required_reason == "metric_definition"
    time_rows = [d for d in dimensions.values() if d.type == "time"]
    assert time_rows
    assert all(d.time_granularities for d in time_rows)
    assert not any(d.time_axis_required for d in dimensions.values())
    assert probes
    assert all("scores.item" in group_by for _, group_by in probes)


@pytest.mark.parametrize("metric", ["score_prev_day", "score_change"])
@pytest.mark.asyncio
async def test_a_window_needs_its_time_axis_through_a_compose_too(runtime, metric):
    dimensions = await runtime.get_dimensions(metric)

    (axis,) = [d for d in dimensions if d.time_axis_required]
    assert axis.required is True
    assert axis.required_reason == "time_axis"
    assert axis.time_granularities
