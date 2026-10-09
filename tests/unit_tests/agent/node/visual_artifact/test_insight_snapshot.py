# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Snapshot producer and reader agree; dependencies do not include available joins."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

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


@pytest.mark.parametrize("default_encoding", ["utf-8", "ascii"])
def test_dashboard_snapshot_roundtrip_and_render_only_reuse(tmp_path, monkeypatch, default_encoding):
    queries = tmp_path / "queries"
    queries.mkdir()
    (queries / "sales.sql.j2").write_text("SELECT amount FROM shop.orders")
    refs = SubjectRefs(metrics=[dict(path=["Sales"], name="revenue")])
    tools = SimpleNamespace(
        runtime=SimpleNamespace(lineage_graph=AsyncMock(return_value=[graph()])),
        get_metric=Mock(
            return_value=SimpleNamespace(
                success=1, result={"name": "revenue", "path": ["Sales"], "description": "Net sales \u00e9"}
            )
        ),
    )
    assert bake_metric_snapshots(tmp_path, refs, tools, artifact_kind="dashboard") is None
    target = tmp_path / "analysis/metric_snapshots.json"
    original = target.read_text(encoding="utf-8")
    files = {
        p.relative_to(tmp_path).as_posix(): p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file()
    }
    manifest = dict(slug="sales", name="Sales", description="Sales", kind="dashboard", created_at="2026-01-01")
    snapshot = build_artifact_insight(manifest, files).metric_details[0]
    assert snapshot.detail["definition"]["expression"] == "SUM(orders.amount)"
    assert snapshot.tables == ["shop.orders"]
    read_text = Path.read_text

    def read_with_locale_default(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or default_encoding, errors=errors)

    # A non-UTF-8 platform default must not recapture an unchanged snapshot.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", read_with_locale_default)
        assert bake_metric_snapshots(tmp_path, refs, tools, artifact_kind="dashboard") is None

    assert target.read_text(encoding="utf-8") == original
    tools.get_metric.assert_called_once_with(name="revenue", path=["Sales"])
    # A changed saved query cannot keep the previous model snapshot on failure.
    (queries / "sales.sql.j2").write_text("SELECT amount FROM shop.archive")
    tools.get_metric.side_effect = RuntimeError("model offline")
    assert bake_metric_snapshots(tmp_path, refs, tools, artifact_kind="dashboard") is None
    snapshot = json.loads(target.read_text())["metrics"][0]
    assert snapshot["status"] == "unavailable"
    assert snapshot["tables"] == []


def test_wrong_subject_path_never_captures_same_named_metric(tmp_path):
    (tmp_path / "queries").mkdir()
    tools = SimpleNamespace(
        runtime=None,
        get_metric=Mock(return_value=SimpleNamespace(success=1, result={"name": "revenue", "path": ["Other"]})),
    )
    assert (
        bake_metric_snapshots(
            tmp_path, SubjectRefs(metrics=[dict(name="revenue", path=["Sales"])]), tools, artifact_kind="report"
        )
        is None
    )
    assert (
        json.loads((tmp_path / "analysis/metric_snapshots.json").read_text())["metrics"][0]["status"] == "unavailable"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["report", "dashboard"])
async def test_snapshot_survives_real_detail_bundle(tmp_path, kind):
    from datus.api.services.dashboard_service import DashboardService
    from datus.api.services.report_service import ReportService

    artifact_dir = tmp_path / f"{kind}s" / "sales"
    queries = artifact_dir / "queries"
    queries.mkdir(parents=True)
    manifest = dict(slug="sales", name="Sales", description="Sales", kind=kind, created_at="2026-01-01")
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    (artifact_dir / "render").mkdir()
    (artifact_dir / "render/app.jsx").write_text('<ChartCard chartId="sales" sqlId="sales" />')
    suffix = ".sql.j2" if kind == "dashboard" else ".sql"
    (queries / f"sales{suffix}").write_text("SELECT amount FROM shop.orders")
    refs = SubjectRefs(metrics=[dict(path=["Sales"], name="revenue")])
    (queries / "sales.brief.json").write_text(
        json.dumps(dict(name="sales", hypothesis="Sales increased", uses=refs.model_dump()))
    )
    if kind == "dashboard":
        (queries / "sales.params.json").write_text(
            json.dumps(
                dict(
                    slug="sales",
                    datasource="retail",
                    params=[],
                    columns=[dict(name="amount", type="number")],
                    sample_params={},
                    sample_row_count=0,
                    saved_at="2026-01-01",
                )
            )
        )
        # Scratch report files must not influence a dashboard snapshot's hash.
        (queries / "probe.json").write_text("{}")
        (queries / "probe.sql").write_text("SELECT 1")
    else:
        (queries / "probe.sql.j2").write_text("SELECT 1")
    (queries / "nested").mkdir()
    (queries / "nested/ignored.brief.json").write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (queries / "external.brief.json").symlink_to(outside)
    tools = SimpleNamespace(
        runtime=None,
        get_metric=Mock(
            return_value=SimpleNamespace(
                success=1,
                result={"name": "revenue", "path": ["Sales"]},
            )
        ),
    )
    assert bake_metric_snapshots(artifact_dir, refs, tools, artifact_kind=kind) is None
    if kind == "dashboard":
        response = await DashboardService(agent_config=None).get_detail(
            project_files_root=tmp_path, dashboard_slug="sales"
        )
    else:
        response = await ReportService().get_detail(project_files_root=tmp_path, report_slug="sales")
    assert response.success is True, response.errorMessage
    bundle = response.data
    files = {f.path: f.content for f in bundle.files}
    assert "queries/external.brief.json" not in files
    assert "queries/nested/ignored.brief.json" not in files
    insight = build_artifact_insight(bundle.manifest.model_dump(), files)
    assert insight.queries[0].brief.uses.metrics == refs.metrics
    assert len(insight.metric_details) == 1
    assert insight.metric_details[0].ref == refs.metrics[0]
    assert insight.warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["report", "dashboard"])
async def test_reference_sql_roundtrip_reuse_and_stale_detection(tmp_path, kind):
    from datus.agent.node.visual_artifact._insight_snapshot import bake_reference_sql_snapshots
    from datus.api.services.dashboard_service import DashboardService
    from datus.api.services.report_service import ReportService

    artifact_dir = tmp_path / f"{kind}s" / "sales"
    queries = artifact_dir / "queries"
    queries.mkdir(parents=True)
    manifest = dict(slug="sales", name="Sales", description="Sales", kind=kind, created_at="2026-01-01")
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    suffix = ".sql.j2" if kind == "dashboard" else ".sql"
    (artifact_dir / "render").mkdir()
    (artifact_dir / "render/app.jsx").write_text("export default function App() { return null; }")
    query_file = queries / f"sales{suffix}"
    query_file.write_text("SELECT amount FROM shop.orders")
    refs = SubjectRefs(reference_sql=[dict(path=["Sales"], name="paid_revenue")])
    (queries / "sales.brief.json").write_text(
        json.dumps(dict(name="sales", hypothesis="Sales", uses=refs.model_dump()))
    )
    original_sql = "-- Revenue \u00e9\nSELECT SUM(amount) FROM shop.orders"
    tools = SimpleNamespace(
        get_reference_sql=Mock(
            return_value=SimpleNamespace(success=1, result=dict(name="paid_revenue", sql=original_sql, summary="Sales"))
        )
    )
    assert bake_reference_sql_snapshots(artifact_dir, refs, tools, artifact_kind=kind) is None

    async def read_bundle():
        if kind == "dashboard":
            result = await DashboardService(agent_config=None).get_detail(
                project_files_root=tmp_path, dashboard_slug="sales"
            )
        else:
            result = await ReportService().get_detail(project_files_root=tmp_path, report_slug="sales")
        assert result.success is True, result.errorMessage
        return build_artifact_insight(result.data.manifest.model_dump(), {f.path: f.content for f in result.data.files})

    insight = await read_bundle()
    assert insight.warnings == []
    saved = insight.reference_sql_details[0]
    assert saved.ref == refs.reference_sql[0]
    assert saved.status == "captured"
    assert saved.sql == original_sql
    assert saved.summary == "Sales"
    tools.get_reference_sql.assert_called_once_with(subject_path=["Sales"], name="paid_revenue")

    # An unchanged query preserves its original reference even if the library changes.
    tools.get_reference_sql.side_effect = RuntimeError("Library unavailable")
    assert bake_reference_sql_snapshots(artifact_dir, refs, tools, artifact_kind=kind) is None
    assert (await read_bundle()).reference_sql_details[0] == saved
    assert tools.get_reference_sql.call_count == 1

    query_file.write_text("SELECT amount FROM shop.archive")
    stale = await read_bundle()
    assert stale.reference_sql_details == []
    assert "analysis/reference_sql_snapshots.json:stale" in stale.warnings
    assert bake_reference_sql_snapshots(artifact_dir, refs, tools, artifact_kind=kind) is None
    unavailable = (await read_bundle()).reference_sql_details[0]
    assert unavailable.status == "unavailable"
    assert unavailable.sql is None


@pytest.mark.parametrize(
    "candidate",
    [None, {}, {"name": "other", "sql": "SELECT 1"}, {"name": "paid", "sql": " "}, {"name": "paid", "sql": 42}],
)
def test_reference_sql_lookup_failure_is_explicit(tmp_path, candidate):
    from datus.agent.node.visual_artifact._insight_snapshot import bake_reference_sql_snapshots

    refs = SubjectRefs(reference_sql=[dict(path=["Sales"], name="paid")])
    tools = SimpleNamespace(
        get_reference_sql=Mock(return_value=SimpleNamespace(success=bool(candidate), result=candidate))
    )
    assert bake_reference_sql_snapshots(tmp_path, refs, tools, artifact_kind="report") is None
    saved = json.loads((tmp_path / "analysis/reference_sql_snapshots.json").read_text(encoding="utf-8"))[
        "reference_sql"
    ][0]
    assert saved["status"] == "unavailable"
    assert saved["sql"] is None
