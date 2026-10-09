# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Snapshot producer and reader agree; dependencies do not include available joins."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from datus.agent.node.visual_artifact._insight_snapshot import bake_metric_snapshots, metric_tables
from datus.schemas.analysis_artifacts import SubjectRefs
from datus.utils.artifact_insight import build_artifact_insight


def graph():
    return {
        "nodes": [
            {"id": "m", "kind": "metric_atomic", "name": "revenue", "detail": {"expression": "SUM(orders.amount)"}},
            {"id": "d", "kind": "dataset", "name": "orders"},
            {"id": "t", "kind": "physical_table", "name": "shop.orders"},
            {"id": "j", "kind": "physical_table", "name": "shop.customers"},
        ],
        "edges": [
            {"kind": "aggregates", "source": "d", "target": "m"},
            {"kind": "reads_table", "source": "t", "target": "d"},
            {"kind": "join", "source": "j", "target": "d"},
        ],
    }


def test_dependencies_ignore_available_joins_and_ambiguous_models():
    assert metric_tables([graph()], "revenue") == ["shop.orders"]
    assert metric_tables([graph(), graph()], "revenue") is None
    assert metric_tables([graph()], "unknown") is None
    broken = graph()
    broken["nodes"] = [n for n in broken["nodes"] if n["id"] != "d"]
    assert metric_tables([broken], "revenue") is None


def test_nested_derived_metrics_terminate_on_cycles():
    g = graph()
    for name, source in [("m2", "m"), ("m3", "m2"), ("m4", "m3")]:
        g["nodes"].append({"id": name, "kind": "metric_derived", "name": name})
        g["edges"].append({"kind": "derive_base", "source": source, "target": name})
    g["edges"].append({"kind": "derive_base", "source": "m4", "target": "m2"})
    assert metric_tables([g], "m4") == ["shop.orders"]


def test_dashboard_snapshot_roundtrip_and_render_only_reuse(tmp_path):
    queries = tmp_path / "queries"
    queries.mkdir()
    (queries / "sales.sql.j2").write_text("SELECT amount FROM shop.orders")
    refs = SubjectRefs(metrics=[dict(path=["Sales"], name="revenue")])
    tools = SimpleNamespace(
        runtime=SimpleNamespace(lineage_graph=AsyncMock(return_value=[graph()])),
        get_metric=Mock(
            return_value=SimpleNamespace(
                success=1, result={"name": "revenue", "path": ["Sales"], "description": "Net sales"}
            )
        ),
    )
    assert bake_metric_snapshots(tmp_path, refs, tools) is None
    target = tmp_path / "analysis/metric_snapshots.json"
    original = target.read_text()
    files = {p.relative_to(tmp_path).as_posix(): p.read_text() for p in tmp_path.rglob("*") if p.is_file()}
    manifest = dict(slug="sales", name="Sales", description="Sales", kind="dashboard", created_at="2026-01-01")
    snapshot = build_artifact_insight(manifest, files).metric_details[0]
    assert snapshot.detail["definition"]["expression"] == "SUM(orders.amount)"
    assert snapshot.tables == ["shop.orders"]
    assert bake_metric_snapshots(tmp_path, refs, tools) is None
    assert target.read_text() == original
    tools.get_metric.assert_called_once_with(name="revenue", path=["Sales"])
    # A changed saved query cannot keep the previous model snapshot on failure.
    (queries / "sales.sql.j2").write_text("SELECT amount FROM shop.archive")
    tools.get_metric.side_effect = RuntimeError("model offline")
    assert bake_metric_snapshots(tmp_path, refs, tools) is None
    snapshot = json.loads(target.read_text())["metrics"][0]
    assert snapshot["status"] == "unavailable"
    assert snapshot["tables"] == []


def test_wrong_subject_path_never_captures_same_named_metric(tmp_path):
    (tmp_path / "queries").mkdir()
    tools = SimpleNamespace(
        runtime=None,
        get_metric=Mock(return_value=SimpleNamespace(success=1, result={"name": "revenue", "path": ["Other"]})),
    )
    assert bake_metric_snapshots(tmp_path, SubjectRefs(metrics=[dict(name="revenue", path=["Sales"])]), tools) is None
    assert (
        json.loads((tmp_path / "analysis/metric_snapshots.json").read_text())["metrics"][0]["status"] == "unavailable"
    )
