# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime behavior against the fake binding: query construction, slicing,
dry-run shape, engine lifecycle, and connections wiring."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from datus.tools.semantic_tools.models import AttributionRequest, AttributionWindow
from tests.unit_tests.tools.semantic_tools.dosi_fakes import (
    DIMENSION_ROWS,
    METRIC_ROWS,
    FakeEngine,
    QueryError,
    build_fake_module,
)


@pytest.fixture(autouse=True)
def fake_binding(monkeypatch):
    """Scope the engine test double to this module's tests."""
    FakeEngine.instances.clear()
    binding = build_fake_module()
    monkeypatch.setitem(sys.modules, "dosi_engine", binding)
    yield binding
    FakeEngine.instances.clear()


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("version: '0.2.0.dev0'\nsemantic_model: []\n")
    return path


@pytest.fixture
def make_runtime(model_file):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    def make(**overrides):
        return DosiRuntime(DosiConfig(semantic_model_path=str(model_file), **overrides))

    return make


def _write_model(path: Path, model_name: str, metric_names: list[str]) -> None:
    metrics = "\n".join(f"      - name: {name}" for name in metric_names)
    path.write_text(
        f"version: '0.2.0.dev0'\nsemantic_model:\n  - name: {model_name}\n    datasets: []\n    metrics:\n{metrics}\n"
    )


def _install_file_catalog(monkeypatch, metrics_by_stem: dict[str, list[str]]) -> None:
    def metrics(engine):
        stem = Path(engine.model_path).stem
        return [
            {
                "name": name,
                "kind": "aggregate",
                "datasets": [stem],
                "measures": [name],
                "time_dimension": f"{stem}.event_date",
                "description": f"Metric from {stem}",
            }
            for name in metrics_by_stem[stem]
        ]

    def datasets(engine):
        stem = Path(engine.model_path).stem
        return [
            {
                "name": stem,
                "source": f"main.{stem}",
                "primary_key": ["id"],
                "fields": 2,
                "time_dimensions": ["event_date"],
                "primary_time_dimension": "event_date",
            }
        ]

    def dimensions(engine, metric=None):
        stem = Path(engine.model_path).stem
        return [
            {"name": f"{stem}.status", "is_time": False},
            {
                "name": f"{stem}.event_date",
                "is_time": True,
                "time_granularity": "day",
            },
        ]

    monkeypatch.setattr(FakeEngine, "metrics", metrics)
    monkeypatch.setattr(FakeEngine, "datasets", datasets)
    monkeypatch.setattr(FakeEngine, "dimensions", dimensions)


@pytest.mark.asyncio
async def test_attribute_delegates_to_native_engine(make_runtime):
    runtime = make_runtime(connection="warehouse", timeout_seconds=42)
    request = AttributionRequest(
        metric="revenue",
        dimensions=[" orders.status ", "orders.status"],
        baseline=AttributionWindow(start="2026-01-01", end="2026-01-08"),
        current=AttributionWindow(start="2026-01-08", end="2026-01-15"),
        where_sql="orders.status IS NOT NULL",
        top_n_dimensions=2,
        top_n_values=5,
        path=["Finance"],
    )

    result = await runtime.attribute(request)

    assert result.implementation == "dosi"
    assert result.strategy == "term_wise"
    assert result.total_change.delta == 50
    call = FakeEngine.instances[-1].attribute_calls[-1]
    assert call["connection"] == "warehouse"
    assert call["timeout_secs"] == 42.0
    assert call["request"]["metric"] == "revenue"
    assert call["request"]["dimensions"] == ["orders.status"]
    assert call["request"]["baseline"] == {
        "start": "2026-01-01",
        "end": "2026-01-08",
    }
    assert "path" not in call["request"]


@pytest.mark.asyncio
async def test_attribute_rejects_missing_dimensions_with_structured_error(
    make_runtime,
):
    runtime = make_runtime()
    request = AttributionRequest(
        metric="revenue",
        dimensions=["", "  "],
        baseline=AttributionWindow(start="2026-01-01", end="2026-01-08"),
        current=AttributionWindow(start="2026-01-08", end="2026-01-15"),
    )

    with pytest.raises(SemanticValidationException) as exc_info:
        await runtime.attribute(request)

    assert exc_info.value.payload.code == "dimensions_required"
    assert exc_info.value.payload.metrics == ["revenue"]
    assert "get_dimensions" in exc_info.value.payload.message


@pytest.mark.asyncio
async def test_list_metrics_maps_rows_and_slices(make_runtime):
    runtime = make_runtime()
    metrics = await runtime.list_metrics()
    assert [m.name for m in metrics] == [
        "order_count",
        "revenue",
        "running_revenue",
    ]
    assert metrics[0].type == "aggregate"
    assert metrics[0].measures == ["order_count"]
    assert metrics[0].metadata == {
        "datasets": ["orders"],
        "base_kind": "aggregate",
        "datus_ext_version": "1.2",
        "time_dimension": "orders.order_date",
    }
    assert metrics[0].dimensions == []

    assert [m.name for m in await runtime.list_metrics(limit=1)] == ["order_count"]
    assert [m.name for m in await runtime.list_metrics(limit=5, offset=1)] == [
        "revenue",
        "running_revenue",
    ]


@pytest.mark.asyncio
async def test_list_metrics_exposes_parameter_schema(make_runtime, monkeypatch):
    rows = [
        {
            **METRIC_ROWS[0],
            "params": {"status": "paid"},
            "param_schema": {
                "status": {
                    "type": "string",
                    "required": False,
                    "default": "paid",
                }
            },
        }
    ]
    monkeypatch.setattr(FakeEngine, "metrics", lambda self: rows)

    (metric,) = await make_runtime().list_metrics()

    assert metric.metadata["params"] == {"status": "paid"}
    assert metric.metadata["param_schema"]["status"]["type"] == "string"


@pytest.mark.asyncio
async def test_list_metrics_exposes_native_window_metadata(make_runtime, model_file):
    model_file.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: orders_model
    datasets: []
    metrics:
      - name: running_revenue
        expression:
          dialects: [{dialect: ANSI_SQL, expression: 'SUM(orders.amount)'}]
        custom_extensions:
          - vendor_name: DATUS
            data: >-
              {"v":"1.2",
              "window":{"type":"cumulative","function":"sum"},
              "subject_path":["sales","revenue"],"unit":"USD"}
""".lstrip()
    )
    metrics = {metric.name: metric for metric in await make_runtime().list_metrics()}
    running = metrics["running_revenue"]
    assert running.type == "window"
    assert running.path == ["sales", "revenue"]
    assert running.unit == "USD"
    assert running.metadata["requires_time_axis"] is True
    assert running.metadata["window"] == {
        "type": "cumulative",
        "function": "sum",
    }


@pytest.mark.asyncio
async def test_window_axis_probe_uses_configured_target(make_runtime):
    runtime = make_runtime(dialect="starrocks", connection="warehouse")

    await runtime.get_dimensions("running_revenue")

    calls = FakeEngine.instances[-1].compile_calls
    assert calls
    assert {call["dialect"] for call in calls} == {"starrocks"}
    assert {call["connection"] for call in calls} == {"warehouse"}


@pytest.mark.asyncio
async def test_list_metrics_exposes_native_window_family_and_axis(make_runtime, model_file, monkeypatch):
    model_file.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: orders_model
    datasets: []
    metrics:
      - name: order_rank
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.3","window":{"rank":{"function":"rank"}}}'
      - name: first_revenue
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.3","window":{"value":{"function":"first_value"}}}'
      - name: revenue_range
        custom_extensions:
          - vendor_name: DATUS
            data: >-
              {"v":"1.3","window":{"frame":{"function":"count",
              "preceding":"unbounded","units":"range",
              "order":{"by":"value","direction":"asc"}}}}
""".lstrip()
    )
    rows = [
        {
            "name": "order_rank",
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": ["revenue"],
            "window_family": "rank",
            "window_function": "rank",
        },
        {
            "name": "first_revenue",
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": ["revenue"],
            "window_family": "value",
            "window_function": "first_value",
        },
        {
            "name": "revenue_range",
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": ["revenue"],
            "window_family": "frame",
            "window_function": "count",
        },
    ]
    monkeypatch.setattr(FakeEngine, "metrics", lambda self: [dict(row) for row in rows])
    monkeypatch.setattr(FakeEngine, "time_axis_metrics", {"first_revenue"})

    metrics = {metric.name: metric for metric in await make_runtime().list_metrics()}
    assert metrics["order_rank"].metadata["window_family"] == "rank"
    assert metrics["order_rank"].metadata["window_function"] == "rank"
    assert metrics["order_rank"].metadata["requires_time_axis"] is False
    assert metrics["first_revenue"].metadata["requires_time_axis"] is True
    assert metrics["revenue_range"].metadata["requires_time_axis"] is False


@pytest.mark.asyncio
async def test_list_metrics_exposes_derive_discriminators(make_runtime, monkeypatch):
    rows = [
        {
            "name": "new_revenue",
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": ["revenue"],
            "derive_family": "filter",
            "derive_base": "revenue",
            "subset_of": "revenue",
            "attribution": "exact",
        },
        {
            "name": "net_revenue",
            "kind": "aggregate",
            "datasets": ["orders", "refunds"],
            "measures": ["revenue", "total_refunds"],
            "derive_family": "compose",
            "derive_expr": "revenue - total_refunds",
            "derive_members": [
                {"metric": "revenue", "coefficient": 1.0},
                {"metric": "total_refunds", "coefficient": -1.0},
            ],
            "leaf_measures": [
                {
                    "measure": "orders_amount_sum",
                    "dataset": "orders",
                    "coefficient": 1.0,
                }
            ],
            "conformed_dimensions": ["stores.region"],
            "attribution": "exact",
        },
        {
            "name": "revenue",
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": ["revenue"],
            "derive_family": None,
            "derive_base": None,
            "subset_of": None,
        },
        {
            "name": "falsy_derive_scalar",
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": ["revenue"],
            "derive_family": "filter",
            "derive_base": "revenue",
            "subset_of": "",
        },
    ]
    monkeypatch.setattr(FakeEngine, "metrics", lambda self: [dict(r) for r in rows])

    metrics = {metric.name: metric for metric in await make_runtime().list_metrics()}

    assert metrics["new_revenue"].metadata["derive_family"] == "filter"
    assert metrics["new_revenue"].metadata["derive_base"] == "revenue"
    assert metrics["new_revenue"].metadata["subset_of"] == "revenue"
    assert metrics["net_revenue"].metadata["derive_family"] == "compose"
    # The authored formula: the only field that says how a composite is
    # assembled. `derive_members` folds coefficients across inlined levels and
    # keeps none for a non-linear member, so it cannot answer that.
    assert metrics["net_revenue"].metadata["derive_expr"] == "revenue - total_refunds"
    assert metrics["falsy_derive_scalar"].metadata["subset_of"] == ""
    # The structural columns stay off the catalog listing.
    for heavy in (
        "derive_members",
        "leaf_measures",
        "conformed_dimensions",
        "attribution",
    ):
        assert heavy not in metrics["net_revenue"].metadata
    # NULL derive columns on a plain metric add no metadata keys. `derive_expr`
    # is absent for every non-compose metric, plain or derived: a filter states
    # a predicate and a window a partition rule, neither of which is a formula.
    for key in ("derive_family", "derive_base", "subset_of", "derive_expr"):
        assert key not in metrics["revenue"].metadata
    assert "derive_expr" not in metrics["new_revenue"].metadata


@pytest.mark.asyncio
async def test_get_dimensions_uses_native_metric_catalog(make_runtime):
    runtime = make_runtime()
    dimensions = await runtime.get_dimensions("revenue")
    assert [dimension.name for dimension in dimensions] == [
        "metric_time",
        "orders.order_date",
        "orders.status",
        "customers.region",
        "orders.amount",
        "orders.order_id",
    ]
    dims = {d.name: d for d in dimensions}
    assert FakeEngine.instances[-1].dimension_calls[-1] == "revenue"
    assert dims["metric_time"].type == "time"
    assert dims["metric_time"].is_primary_time is True
    assert dims["metric_time"].recommended is True
    assert dims["metric_time"].recommendation_source == "inferred:time"
    assert dims["orders.order_date"].type == "time"
    assert dims["orders.order_date"].is_primary_time is False
    assert dims["orders.order_date"].time_granularities == [
        "day",
        "week",
        "month",
        "quarter",
        "year",
    ]
    assert dims["orders.status"].type is None
    assert dims["orders.status"].is_primary_time is False
    assert dims["orders.amount"].recommended is False
    assert dims["orders.amount"].recommendation_source == "inferred:measure"
    assert dims["orders.order_id"].recommended is False
    assert dims["orders.order_id"].recommendation_source == "inferred:primary_key"

    # Native Dosi owns non-time membership and recommendation. Runtime probes
    # only enrich time rows with supported grains.
    probed = {
        item["field"] for call in FakeEngine.instances[-1].compile_calls for item in call["query"].get("group_by") or []
    }
    assert probed <= {"metric_time", "orders.order_date"}


@pytest.mark.asyncio
async def test_get_dimensions_keeps_grains_for_other_time_dimensions(make_runtime, monkeypatch):
    original_dimensions = FakeEngine.dimensions

    def dimensions(self, metric=None):
        return original_dimensions(self, metric) + [
            {
                "name": "customers.signup_date",
                "is_time": True,
                "is_dimension": True,
                "source": "inferred:time",
                "time_granularity": "month",
                "description": "Signup month",
            }
        ]

    monkeypatch.setattr(FakeEngine, "dimensions", dimensions)

    dims = {d.name: d for d in await make_runtime().get_dimensions("revenue")}

    assert dims["customers.signup_date"].time_granularities == [
        "month",
        "quarter",
        "year",
    ]


@pytest.mark.asyncio
async def test_unknown_native_grain_is_left_to_planner_probing(make_runtime, monkeypatch):
    original_dimensions = FakeEngine.dimensions

    def dimensions(self, metric=None):
        rows = original_dimensions(self, metric)
        for row in rows:
            if row["name"] == "orders.order_date":
                row["time_granularity"] = "hour"
        return rows

    monkeypatch.setattr(FakeEngine, "dimensions", dimensions)

    dims = {d.name: d for d in await make_runtime().get_dimensions("revenue")}

    assert dims["orders.order_date"].time_granularities == [
        "day",
        "week",
        "month",
        "quarter",
        "year",
    ]


@pytest.mark.asyncio
async def test_missing_axis_grains_returns_structured_error(make_runtime, monkeypatch):
    import datus.tools.semantic_tools.dosi.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "queryable_grains", lambda _grain: [])

    with pytest.raises(SemanticValidationException) as exc:
        await make_runtime().get_dimensions("running_revenue")

    assert exc.value.payload.code == "no_primary_time_dimension"
    assert exc.value.payload.metrics == ["running_revenue"]


@pytest.mark.asyncio
async def test_get_dimensions_trusts_native_membership_without_non_time_probes(make_runtime, monkeypatch):
    native_rows = [
        {
            "name": "orders.status",
            "is_time": False,
            "is_dimension": True,
            "source": "declared",
        },
        {
            "name": "orders.amount",
            "is_time": False,
            "is_dimension": False,
            "source": "inferred:measure",
        },
    ]

    def dimensions(self, metric=None):
        self.dimension_calls.append(metric)
        return [dict(row) for row in native_rows]

    monkeypatch.setattr(FakeEngine, "dimensions", dimensions)

    result = await make_runtime().get_dimensions("revenue")

    assert [(d.name, d.recommended, d.recommendation_source) for d in result] == [
        ("orders.status", True, "declared"),
        ("orders.amount", False, "inferred:measure"),
    ]
    engine = FakeEngine.instances[-1]
    assert engine.dimension_calls[-1] == "revenue"
    assert [call["query"].get("group_by") for call in engine.compile_calls] == [[]]


@pytest.mark.asyncio
async def test_get_dimensions_preserves_native_empty_conformed_set(make_runtime, monkeypatch):
    def dimensions(self, metric=None):
        self.dimension_calls.append(metric)
        return [] if metric == "revenue" else [dict(row) for row in DIMENSION_ROWS]

    monkeypatch.setattr(FakeEngine, "dimensions", dimensions)

    assert await make_runtime().get_dimensions("revenue") == []


@pytest.mark.asyncio
async def test_window_dimension_discovery_includes_required_time_axis(make_runtime):
    dimensions = {dimension.name: dimension for dimension in await make_runtime().get_dimensions("running_revenue")}

    assert "orders.status" in dimensions
    assert "customers.region" in dimensions
    assert dimensions["metric_time"].is_primary_time is True
    assert dimensions["metric_time"].time_granularities == [
        "day",
        "week",
        "month",
        "quarter",
        "year",
    ]


@pytest.mark.asyncio
async def test_window_dimension_discovery_falls_back_to_physical_time_axis(make_runtime, monkeypatch):
    original_dimensions = FakeEngine.dimensions

    def dimensions(self, metric=None):
        return [row for row in original_dimensions(self, metric) if row["name"] != "metric_time"]

    monkeypatch.setattr(FakeEngine, "dimensions", dimensions)

    result = {dimension.name: dimension for dimension in await make_runtime().get_dimensions("running_revenue")}

    assert "metric_time" not in result
    assert result["orders.order_date"].is_primary_time is True
    assert result["orders.order_date"].time_granularities == [
        "day",
        "week",
        "month",
        "quarter",
        "year",
    ]


@pytest.mark.asyncio
async def test_window_dimension_discovery_probes_each_grain(make_runtime, monkeypatch):
    original_compile = FakeEngine.compile

    def compile_query(self, query, dialect=None, connection=None, pretty=False):
        metric_time = next(
            (item for item in query.get("group_by") or [] if item.get("field") == "metric_time"),
            None,
        )
        if metric_time and metric_time.get("grain") in {"quarter", "year"}:
            raise QueryError(
                "reset is finer than the query grain",
                code="window_reset_too_fine",
            )
        return original_compile(
            self,
            query,
            dialect=dialect,
            connection=connection,
            pretty=pretty,
        )

    monkeypatch.setattr(FakeEngine, "compile", compile_query)

    dimensions = {dimension.name: dimension for dimension in await make_runtime().get_dimensions("running_revenue")}

    assert dimensions["metric_time"].time_granularities == [
        "day",
        "week",
        "month",
    ]


@pytest.mark.asyncio
async def test_get_dimensions_unknown_metric_is_structured(make_runtime):
    runtime = make_runtime()
    with pytest.raises(SemanticValidationException) as exc:
        await runtime.get_dimensions("revenues")
    payload = exc.value.payload
    assert payload.code == "unknown_metric"
    assert payload.metrics == ["revenues"]
    assert "order_count" in payload.message


@pytest.mark.asyncio
async def test_query_metrics_builds_metric_query(make_runtime):
    runtime = make_runtime()
    await runtime.query_metrics(
        metrics=["revenue"],
        dimensions=["orders.status", "orders.order_date"],
        time_start="2025-01-01",
        time_end="2025-02-01",
        time_granularity="month",
        where="status <> 'void'",
        limit=10,
        order_by=["-revenue", "status"],
    )
    engine = FakeEngine.instances[-1]
    (call,) = engine.execute_calls
    assert call["query"] == {
        "metrics": ["revenue"],
        "group_by": [
            {"field": "orders.status"},
            {"field": "orders.order_date", "grain": "month"},
        ],
        "where_sql": "status <> 'void'",
        "time_range": {"start": "2025-01-01", "end": "2025-02-01"},
        "order_by": [
            {"key": "revenue", "desc": True},
            {"key": "status", "desc": False},
        ],
        "limit": 10,
    }
    assert call["timeout_secs"] == 30.0
    assert call["connection"] is None


@pytest.mark.asyncio
async def test_query_metrics_bare_time_dimension_gets_grain(make_runtime):
    runtime = make_runtime()
    await runtime.query_metrics(metrics=["revenue"], dimensions=["order_date"], time_granularity="day")
    engine = FakeEngine.instances[-1]
    assert engine.execute_calls[0]["query"]["group_by"] == [{"field": "order_date", "grain": "day"}]


@pytest.mark.asyncio
async def test_query_metrics_uses_native_metric_time_input(make_runtime):
    runtime = make_runtime()
    await runtime.query_metrics(
        metrics=["running_revenue"],
        dimensions=["metric_time"],
        time_granularity="month",
    )
    engine = FakeEngine.instances[-1]
    assert engine.execute_calls[0]["query"]["group_by"] == [{"field": "metric_time", "grain": "month"}]


@pytest.mark.asyncio
async def test_bare_ambiguous_dimension_is_forwarded_to_dosi(make_runtime):
    runtime = make_runtime()

    await runtime.query_metrics(metrics=["revenue"], dimensions=["status"])

    engine = FakeEngine.instances[-1]
    assert engine.execute_calls[0]["query"]["group_by"] == [{"field": "status"}]


@pytest.mark.asyncio
async def test_time_range_without_time_grouping_binds_metric_time_dimension(
    make_runtime,
):
    """A time filter with no time grouping resolves the metric's time dimension.

    The engine otherwise rejects the query with time_range_needs_dimension —
    but "total for September" style asks are the most common Datus shape.
    """
    runtime = make_runtime()
    await runtime.query_metrics(metrics=["revenue"], time_start="2025-09-01", time_end="2025-10-01")
    engine = FakeEngine.instances[-1]
    assert engine.execute_calls[0]["query"]["time_range"] == {
        "start": "2025-09-01",
        "end": "2025-10-01",
        "dimension": "metric_time",
    }


def test_time_range_with_ambiguous_time_dimensions_delegates_to_metric_time(
    make_runtime,
):
    runtime = make_runtime()
    query = runtime._build_query(
        [
            {"name": "orders.order_date", "is_time": True},
            {"name": "orders.ship_date", "is_time": True},
        ],
        [{"name": "revenue", "datasets": ["orders"]}],
        metrics=["revenue"],
        dimensions=[],
        time_start="2025-09-01",
        time_end="2025-10-01",
        time_granularity=None,
        where=None,
        limit=None,
        order_by=None,
    )
    assert query["time_range"]["dimension"] == "metric_time"


def test_time_range_with_no_reachable_time_dimension_uses_metric_time(make_runtime):
    """The engine reports no_primary_time_dimension from its compiled IR."""
    runtime = make_runtime()
    query = runtime._build_query(
        [{"name": "orders.status", "is_time": False}],
        [{"name": "revenue", "datasets": ["orders"]}],
        metrics=["revenue"],
        dimensions=[],
        time_start="2025-09-01",
        time_end="2025-10-01",
        time_granularity=None,
        where=None,
        limit=None,
        order_by=None,
    )
    assert query["time_range"] == {
        "start": "2025-09-01",
        "end": "2025-10-01",
        "dimension": "metric_time",
    }


@pytest.mark.asyncio
async def test_time_granularity_without_time_dimension_is_structured(make_runtime):
    runtime = make_runtime()
    with pytest.raises(SemanticValidationException) as exc:
        await runtime.query_metrics(metrics=["revenue"], dimensions=["orders.status"], time_granularity="day")
    payload = exc.value.payload
    assert payload.code == "time_grain_required"
    assert payload.required_dimensions[0] == "metric_time"
    assert payload.suggested_retry == {
        "metrics": ["revenue"],
        "dimensions": ["orders.status", "metric_time"],
        "time_granularity": "day",
    }


@pytest.mark.asyncio
async def test_dry_run_returns_sql_contract(make_runtime):
    runtime = make_runtime(db_config={"type": "postgresql"})
    result = await runtime.query_metrics(metrics=["revenue"], dry_run=True)
    assert result.columns == ["sql"]
    assert result.data == [{"sql": "SELECT 1 AS compiled"}]
    assert result.metadata["dry_run"] is True
    assert result.metadata["sql"] == "SELECT 1 AS compiled"
    assert result.metadata["outputs"] == [{"name": "revenue", "type": "metric"}]
    engine = FakeEngine.instances[-1]
    (call,) = engine.compile_calls
    # postgresql (Datus vocabulary) normalized to postgres (engine dialect)
    assert call["dialect"] == "postgres"
    assert call["pretty"] is True
    assert not engine.execute_calls


@pytest.mark.asyncio
async def test_query_metrics_preserves_positional_dry_run(make_runtime):
    runtime = make_runtime(db_config={"type": "postgresql"})

    result = await runtime.query_metrics(
        ["revenue"],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        True,
    )

    assert result.metadata["dry_run"] is True
    engine = FakeEngine.instances[-1]
    assert engine.compile_calls
    assert not engine.execute_calls


@pytest.mark.asyncio
async def test_execute_result_maps_to_query_result(make_runtime):
    runtime = make_runtime()
    result = await runtime.query_metrics(metrics=["order_count"], dimensions=["orders.status"])
    assert result.columns == ["status", "order_count"]
    assert result.data == [{"status": "paid", "order_count": 2}]
    assert result.metadata["row_count"] == 1
    assert "sql" in result.metadata
    assert result.metadata["outputs"] == [{"name": "order_count", "type": "metric"}]


@pytest.mark.asyncio
async def test_query_metrics_passes_scalar_and_list_params(make_runtime):
    runtime = make_runtime()

    await runtime.query_metrics(
        metrics=["revenue"],
        params={"region": ["APAC", "EMEA"], "threshold": 100},
        dry_run=True,
    )

    query = FakeEngine.instances[-1].compile_calls[0]["query"]
    assert query["params"] == {
        "region": ["APAC", "EMEA"],
        "threshold": 100,
    }


@pytest.mark.asyncio
async def test_engine_rebuilds_on_model_mtime_change(make_runtime, model_file):
    runtime = make_runtime()
    await runtime.list_metrics()
    await runtime.list_metrics()
    assert len(FakeEngine.instances) == 1

    model_file.write_text("version: '0.2.0.dev0'\nsemantic_model: []\n# touched\n")
    os.utime(model_file, (0, 0))  # force a different mtime regardless of clock
    await runtime.list_metrics()
    assert len(FakeEngine.instances) == 2


@pytest.mark.asyncio
async def test_db_config_passed_as_runtime_datasources(make_runtime):
    runtime = make_runtime(
        db_config={"type": "postgresql", "host": "db.local", "port": 5432},
        datasource="warehouse",
    )
    await runtime.query_metrics(metrics=["revenue"])
    engine = FakeEngine.instances[-1]
    assert engine.execute_calls[0]["connection"] == "warehouse"
    assert engine.connections == {
        "warehouse": {
            "type": "postgres",
            "host": "db.local",
            "port": 5432,
            "default": True,
        }
    }


@pytest.mark.parametrize(
    ("service_name", "database", "expected_database"),
    [
        ("FREEPDB1", None, "FREEPDB1"),
        (None, "DATABASEPDB", "DATABASEPDB"),
        ("FREEPDB1", "DATABASEPDB", "FREEPDB1"),
    ],
)
@pytest.mark.asyncio
async def test_oracle_service_name_is_mapped_for_native_executor(
    make_runtime, service_name, database, expected_database
):
    db_config = {
        "type": "oracle",
        "host": "db.local",
        "port": 1521,
    }
    if service_name:
        db_config["service_name"] = service_name
    if database:
        db_config["database"] = database
    runtime = make_runtime(db_config=db_config, datasource="oracle_hr")

    await runtime.query_metrics(metrics=["revenue"])

    engine = FakeEngine.instances[-1]
    datasource = engine.connections["oracle_hr"]
    if service_name:
        assert datasource["service_name"] == service_name
    else:
        assert "service_name" not in datasource
    assert datasource["database"] == expected_database


@pytest.mark.asyncio
async def test_explicit_connection_selects_inline_profile(make_runtime):
    runtime = make_runtime(
        db_config={"type": "mysql"},
        connection="prod",
    )
    await runtime.query_metrics(metrics=["revenue"])
    engine = FakeEngine.instances[-1]
    assert engine.connections == {"prod": {"type": "mysql", "default": True}}
    assert engine.execute_calls[0]["connection"] == "prod"


def test_list_semantic_models_maps_datasets(make_runtime):
    runtime = make_runtime()
    models = runtime.list_semantic_models()
    assert [m.name for m in models] == ["orders", "customers"]
    assert models[0].table_name == "main.orders"
    assert models[0].extra["primary_key"] == ["order_id"]

    assert runtime.get_semantic_model("orders").name == "orders"
    assert runtime.get_semantic_model("main.customers").name == "customers"
    assert runtime.get_semantic_model("nope") is None


@pytest.mark.asyncio
async def test_validate_semantic_maps_issues(make_runtime, fake_binding):
    def failing_validate(text):
        return {
            "valid": False,
            "issues": [
                {
                    "severity": "warning",
                    "code": "missing_sql_dialect",
                    "location": "semantic_model[0]",
                    "message": "field has no SQL dialect",
                }
            ],
            "compile_errors": [
                {
                    "code": "unknown_column",
                    "location": "metrics[0]",
                    "message": "no such column",
                    "hint": "did you mean amount?",
                }
            ],
        }

    fake_binding.validate = failing_validate
    runtime = make_runtime()
    result = await runtime.validate_semantic()
    assert result.valid is False
    assert len(result.issues) == 2
    assert result.issues[0].severity == "warning"
    assert result.issues[0].location == "semantic_model[0]"
    assert "unknown_column" in result.issues[1].message
    assert "did you mean amount?" in result.issues[1].message


@pytest.mark.asyncio
async def test_validate_semantic_ok(make_runtime):
    runtime = make_runtime()
    result = await runtime.validate_semantic()
    assert result.valid is True
    assert result.issues == []


@pytest.mark.asyncio
async def test_validate_semantic_returns_same_compile_evidence(make_runtime, model_file):
    runtime = make_runtime()

    result = await runtime.validate_semantic(metric_names=["revenue"])

    assert result.valid is True
    assert [row["name"] for row in result.metadata["compiled_metrics"]] == ["revenue"]
    assert result.metadata["compiled_metric_digests"] == {"revenue": "sha256:revenue"}
    assert result.metadata["contract_digest"] == "sha256:" + "a" * 64
    assert result.metadata["metric_names"] == ["revenue"]
    assert list(result.metadata["artifact_sha256"]) == [str(model_file.resolve())]


@pytest.mark.asyncio
async def test_validate_semantic_rejects_missing_requested_metric(make_runtime):
    result = await make_runtime().validate_semantic(metric_names=["missing"])

    assert result.valid is False
    assert result.metadata == {}
    assert "compiled_metric_not_found" in result.issues[-1].message


@pytest.mark.asyncio
async def test_validate_semantic_returns_structured_invalid_yaml_issue(make_runtime, model_file):
    model_file.write_text("semantic_model: [\n")

    result = await make_runtime().validate_semantic()

    assert result.valid is False
    assert len(result.issues) == 1
    assert "invalid_yaml" in result.issues[0].message


@pytest.mark.asyncio
async def test_validate_semantic_rejects_legacy_window_hints(make_runtime, model_file):
    model_file.write_text(
        """
version: 0.2.0.dev0
semantic_model:
  - name: orders_model
    datasets: []
    metrics:
      - name: legacy_revenue
        expression:
          dialects: [{dialect: ANSI_SQL, expression: 'SUM(orders.amount)'}]
        custom_extensions:
          - vendor_name: DATUS
            data: '{"grain_to_date":"year","window_aggregation":"sum"}'
""".lstrip()
    )

    result = await make_runtime().validate_semantic()

    assert result.valid is False
    assert len(result.issues) == 1
    assert "legacy_window_hint" in result.issues[0].message


@pytest.mark.asyncio
async def test_validate_semantic_supports_targeted_single_model(make_runtime, model_file):
    model_file.write_text("version: '0.2.0.dev0'\nsemantic_model:\n  - name: activity_management\n")
    runtime = make_runtime()

    matched = await runtime.validate_semantic(
        scope="semantic_model",
        semantic_model_name="activity_management",
    )
    missing = await runtime.validate_semantic(
        scope="semantic_model",
        semantic_model_name="missing_model",
    )

    assert matched.valid is True
    assert matched.issues == []
    assert missing.valid is False
    assert len(missing.issues) == 1
    assert "semantic_model_not_found" in missing.issues[0].message


@pytest.mark.asyncio
async def test_validate_semantic_preserves_positional_model_name(make_runtime, model_file):
    model_file.write_text("version: '0.2.0.dev0'\nsemantic_model:\n  - name: activity_management\n")

    result = await make_runtime().validate_semantic(
        "semantic_model",
        "activity_management",
        metric_names=[],
    )

    assert result.valid is True


@pytest.mark.asyncio
async def test_semantic_models_path_directory_single_file(tmp_path):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    (tmp_path / "model.yaml").write_text("version: '0.2.0.dev0'\nsemantic_model: []\n")
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))
    await runtime.list_metrics()  # builds the engine
    assert FakeEngine.instances[-1].model_path == str(tmp_path / "model.yaml")


@pytest.mark.asyncio
async def test_semantic_models_path_routes_metrics_across_files(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _write_model(tmp_path / "users.yml", "users_model", ["active_users"])
    _install_file_catalog(
        monkeypatch,
        {"orders": ["order_count"], "users": ["active_users"]},
    )
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    assert [metric.name for metric in await runtime.list_metrics()] == [
        "order_count",
        "active_users",
    ]
    assert [metric.name for metric in await runtime.list_metrics(limit=1, offset=1)] == ["active_users"]
    assert {model.name for model in runtime.list_semantic_models()} == {
        "orders",
        "users",
    }

    await runtime.get_dimensions("active_users")
    users_engine = next(engine for engine in FakeEngine.instances if Path(engine.model_path).name == "users.yml")
    assert users_engine.compile_calls

    await runtime.query_metrics(metrics=["order_count"], dry_run=True)
    orders_engine = next(engine for engine in FakeEngine.instances if Path(engine.model_path).name == "orders.yaml")
    assert orders_engine.compile_calls


@pytest.mark.asyncio
async def test_semantic_models_path_rejects_cross_model_query(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _write_model(tmp_path / "users.yaml", "users_model", ["active_users"])
    _install_file_catalog(
        monkeypatch,
        {"orders": ["order_count"], "users": ["active_users"]},
    )
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    with pytest.raises(SemanticValidationException) as exc:
        await runtime.query_metrics(metrics=["order_count", "active_users"])

    assert exc.value.payload.code == "cross_semantic_model_query_unsupported"
    assert exc.value.payload.metrics == ["order_count", "active_users"]
    assert "order_count -> orders_model" in exc.value.payload.message
    assert "active_users -> users_model" in exc.value.payload.message


@pytest.mark.asyncio
async def test_semantic_models_path_rejects_duplicate_metric_names(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["total"])
    _write_model(tmp_path / "users.yaml", "users_model", ["total"])
    _install_file_catalog(monkeypatch, {"orders": ["total"], "users": ["total"]})
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    with pytest.raises(SemanticCoreException, match=r"metric 'total'.*must be unique"):
        await runtime.list_metrics()


@pytest.mark.asyncio
async def test_catalog_wraps_malformed_sibling_document(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    (tmp_path / "broken.yaml").write_text("semantic_model: [\n")
    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _install_file_catalog(monkeypatch, {"orders": ["order_count"]})
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    with pytest.raises(
        SemanticCoreException,
        match=r"cannot read semantic model .*broken\.yaml",
    ):
        await runtime.list_metrics()


@pytest.mark.asyncio
async def test_semantic_models_path_refreshes_when_file_is_added(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    metrics_by_stem = {"orders": ["order_count"]}
    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _install_file_catalog(monkeypatch, metrics_by_stem)
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))
    assert [metric.name for metric in await runtime.list_metrics()] == ["order_count"]

    metrics_by_stem["users"] = ["active_users"]
    _write_model(tmp_path / "users.yaml", "users_model", ["active_users"])

    assert [metric.name for metric in await runtime.list_metrics()] == [
        "order_count",
        "active_users",
    ]


@pytest.mark.asyncio
async def test_semantic_model_path_remains_an_explicit_single_file_pin(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    orders = tmp_path / "orders.yaml"
    _write_model(orders, "orders_model", ["order_count"])
    _write_model(tmp_path / "users.yaml", "users_model", ["active_users"])
    _install_file_catalog(
        monkeypatch,
        {"orders": ["order_count"], "users": ["active_users"]},
    )
    runtime = DosiRuntime(
        DosiConfig(
            semantic_model_path=str(orders),
            semantic_models_path=str(tmp_path),
        )
    )

    assert [metric.name for metric in await runtime.list_metrics()] == ["order_count"]
    with pytest.raises(SemanticValidationException) as exc:
        await runtime.query_metrics(metrics=["active_users"])
    assert exc.value.payload.code == "unknown_metric"


@pytest.mark.asyncio
async def test_validate_semantic_validates_all_files_and_supports_targeting(tmp_path, fake_binding):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _write_model(tmp_path / "users.yaml", "users_model", ["active_users"])
    validated = []

    def validate(model_text):
        validated.append(model_text)
        if "users_model" in model_text:
            return {
                "valid": False,
                "issues": [
                    {
                        "severity": "error",
                        "code": "invalid_users_model",
                        "location": "semantic_model[0]",
                        "message": "users model is invalid",
                    }
                ],
                "compile_errors": [],
            }
        return {"valid": True, "issues": [], "compile_errors": []}

    fake_binding.validate = validate
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    all_models = await runtime.validate_semantic()
    assert all_models.valid is False
    assert len(validated) == 2
    assert "[semantic_model_file=users.yaml]" in all_models.issues[0].message

    validated.clear()
    orders_only = await runtime.validate_semantic(
        scope="semantic_model",
        semantic_model_name="orders_model",
    )
    assert orders_only.valid is True
    assert len(validated) == 1
    assert "orders_model" in validated[0]


@pytest.mark.asyncio
async def test_validate_semantic_rejects_cross_file_duplicate_identities(tmp_path):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "a.yaml", "shared_model", ["shared_metric"])
    _write_model(tmp_path / "b.yaml", "shared_model", ["shared_metric"])
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    result = await runtime.validate_semantic()

    assert result.valid is False
    messages = [issue.message for issue in result.issues]
    assert any("duplicate_semantic_model" in message for message in messages)
    assert any("duplicate_metric" in message for message in messages)


@pytest.mark.asyncio
async def test_targeted_validation_rejects_duplicate_metric_in_sibling_file(
    tmp_path,
):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["shared_metric"])
    _write_model(tmp_path / "users.yaml", "users_model", ["shared_metric"])
    _write_model(tmp_path / "audit.yaml", "audit_model", ["audit_count"])
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    result = await runtime.validate_semantic(
        scope="semantic_model",
        semantic_model_name="orders_model",
    )

    assert result.valid is False
    assert any("duplicate_metric" in issue.message for issue in result.issues)

    unrelated = await runtime.validate_semantic(
        scope="semantic_model",
        semantic_model_name="audit_model",
    )
    assert unrelated.valid is True
    assert unrelated.issues == []


# ---------------------------------------------------------------------------
# lineage_graph passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lineage_graph_passes_through_engine_payload(make_runtime):
    runtime = make_runtime()
    graphs = await runtime.lineage_graph()
    assert len(graphs) == 1
    assert graphs[0]["version"] == 2
    assert FakeEngine.instances[-1].lineage_calls == [{"redact_sql": False}]


@pytest.mark.asyncio
async def test_lineage_graph_forwards_redact_sql(make_runtime):
    runtime = make_runtime()
    graphs = await runtime.lineage_graph(redact_sql=True)
    assert graphs[0]["nodes"][0]["detail"]["source"] == "<redacted>"
    assert FakeEngine.instances[-1].lineage_calls == [{"redact_sql": True}]


@pytest.mark.asyncio
async def test_lineage_graph_model_filter_selects_owning_file(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _write_model(tmp_path / "users.yml", "users_model", ["active_users"])
    _install_file_catalog(
        monkeypatch,
        {"orders": ["order_count"], "users": ["active_users"]},
    )
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    graphs = await runtime.lineage_graph(model="users_model")

    assert len(graphs) == 1
    called = [e for e in FakeEngine.instances if e.lineage_calls]
    assert [Path(e.model_path).name for e in called] == ["users.yml"]


@pytest.mark.asyncio
async def test_lineage_graph_unknown_model_lists_candidates(tmp_path, monkeypatch):
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

    _write_model(tmp_path / "orders.yaml", "orders_model", ["order_count"])
    _install_file_catalog(monkeypatch, {"orders": ["order_count"]})
    runtime = DosiRuntime(DosiConfig(semantic_models_path=str(tmp_path)))

    with pytest.raises(SemanticCoreException, match=r"'nope' not found.*orders_model"):
        await runtime.lineage_graph(model="nope")


@pytest.mark.asyncio
async def test_lineage_graph_rejects_pre_lineage_binding(make_runtime, monkeypatch):
    runtime = make_runtime()
    monkeypatch.delattr(FakeEngine, "lineage")
    with pytest.raises(SemanticCoreException, match=r"dosi-engine>=0\.1\.9"):
        await runtime.lineage_graph()


def _execute_calls() -> list[dict]:
    return [call for engine in FakeEngine.instances for call in engine.execute_calls]


@pytest.mark.asyncio
async def test_query_metrics_passes_context_filter_only_when_set(make_runtime):
    runtime = make_runtime()
    await runtime.query_metrics(
        metrics=["revenue"],
        dimensions=["orders.status"],
        where="orders.status <> 'void'",
        context_filter="orders.region = 'east'",
    )
    await runtime.query_metrics(metrics=["revenue"], dimensions=["orders.status"])
    scoped, plain = _execute_calls()
    assert scoped["query"]["context_filter"] == "orders.region = 'east'"
    assert scoped["query"]["where_sql"] == "orders.status <> 'void'"
    assert "context_filter" not in plain["query"]


@pytest.mark.asyncio
async def test_result_filter_metric_not_selected_suggests_selecting_it(make_runtime, monkeypatch):
    def execute(self, query, **kwargs):
        raise QueryError(
            'where_sql names metric "order_count", which the query does not select',
            code="result_filter_metric_not_selected",
            metrics=["order_count"],
            candidates=["revenue"],
            hint='add "order_count" to `metrics`',
        )

    monkeypatch.setattr(FakeEngine, "execute", execute)
    with pytest.raises(SemanticValidationException) as exc:
        await make_runtime().query_metrics(metrics=["revenue"], dimensions=["orders.status"], where="order_count > 1")
    payload = exc.value.payload
    assert payload.code == "result_filter_metric_not_selected"
    assert payload.suggested_retry == {"metrics": ["revenue", "order_count"]}
    assert "order_count" in payload.message


def test_artifact_metric_binding_pins_model_content_and_connection(make_runtime, model_file):
    import hashlib

    runtime = make_runtime(datasource="warehouse", connection="warehouse")
    binding = runtime.artifact_metric_binding("revenue")
    assert binding["model_path"] == str(model_file)
    assert binding["datasource"] == "warehouse"
    assert binding["model_revision"] == hashlib.sha256(model_file.read_bytes()).hexdigest()
    model_file.write_text(model_file.read_text() + "# Updated definition snapshot\n", encoding="utf-8")
    revised = runtime.artifact_metric_binding("revenue")
    assert revised["model_revision"] != binding["model_revision"]


def test_artifact_metric_binding_uses_one_catalog_snapshot(make_runtime, model_file, tmp_path, monkeypatch):
    import hashlib

    updated = tmp_path / "updated.yaml"
    updated.write_text("# A later registry snapshot\n", encoding="utf-8")
    snapshots = [
        (((str(model_file), SimpleNamespace(profile_name="original")),), {"revenue": str(model_file)}, {}),
        (((str(updated), SimpleNamespace(profile_name="updated")),), {"revenue": str(updated)}, {}),
    ]
    runtime = make_runtime()
    catalog = MagicMock(side_effect=snapshots)
    monkeypatch.setattr(runtime, "_catalog", catalog)
    assert runtime.artifact_metric_binding("revenue") == {
        "model_path": str(model_file),
        "model_revision": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "datasource": "original",
    }
    catalog.assert_called_once()


def test_artifact_metric_binding_unknown_metric_preserves_validation_error(make_runtime):
    with pytest.raises(SemanticValidationException) as exc:
        make_runtime().artifact_metric_binding("unknown")
    assert exc.value.payload.code == "unknown_metric"
    assert exc.value.payload.metrics == ["unknown"]
