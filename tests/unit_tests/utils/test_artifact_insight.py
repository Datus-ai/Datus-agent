# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Saved artifact aggregation: no execution and no inferred attribution."""

import json

import pytest

from datus.schemas.gen_visual_dashboard_models import QueryTemplateMetaFile
from datus.utils.artifact_insight import artifact_revision, build_artifact_insight, query_lineage, query_revision


def manifest(kind="report"):
    return dict(slug="sales", name="Sales", description="Sales by region", kind=kind, created_at="2026-01-01T00:00:00Z")


def report_files():
    return {
        "queries/sales.sql": "-- Sales by region\nWITH t AS (SELECT * FROM shop.orders) SELECT amount FROM t",
        "queries/sales.json": json.dumps(
            dict(
                executed_at="2026-01-01T00:00:00Z",
                datasource="retail",
                row_count=1,
                columns=[dict(name="amount", type="number")],
                rows=[dict(amount=12)],
            )
        ),
        "queries/sales.brief.json": json.dumps(
            dict(
                name="sales",
                hypothesis="Sales increased",
                uses=dict(metrics=[dict(path=["Sales"], name="revenue")]),
                caveats="Excludes refunds",
            )
        ),
        "render/app.jsx": '<ChartCard chartId="sales_chart" title="Sales by region" sqlId="queries/sales" />',
    }


def test_report_uses_saved_result_and_literal_bindings():
    files = report_files()
    data = build_artifact_insight(manifest(), files)
    assert data.queries[0].result.rows == [{"amount": 12}]
    assert data.queries[0].lineage.tables == ["shop.orders"]
    assert data.queries[0].brief.uses.metrics[0].path == ["Sales"]
    assert data.blocks[0].query_ids == ["sales"]
    assert data.blocks_status == "partial"
    assert data.metric_details == []  # No live model lookup for old artifacts.
    assert data.source_revision == artifact_revision(manifest(), files)


def test_revision_is_order_independent_and_sensitive_to_sql_and_render():
    files = report_files()
    assert artifact_revision(manifest(), files) == artifact_revision(manifest(), dict(reversed(list(files.items()))))
    changed = {**files, "render/app.jsx": "<main />"}
    assert artifact_revision(manifest(), files) != artifact_revision(manifest(), changed)
    assert query_revision(files) == query_revision(changed)
    assert query_revision(files) != query_revision({**files, "queries/sales.sql": "SELECT 1"})


def test_corrupt_optional_files_degrade_independently():
    files = report_files()
    files["queries/sales.json"] = "not json"
    files["analysis/insights.json"] = "{}"
    data = build_artifact_insight(manifest(), files)
    assert data.queries[0].result is None
    assert data.queries[0].brief.caveats == "Excludes refunds"
    assert data.queries[0].lineage.tables == ["shop.orders"]
    assert data.warnings == ["analysis/insights.json", "queries/sales.json"]


def test_stale_metric_snapshot_not_substituted_into_changed_query():
    files = report_files()
    files["analysis/metric_snapshots.json"] = json.dumps(
        {
            "query_revision": query_revision(files),
            "metrics": [
                dict(
                    ref=dict(path=["Sales"], name="revenue"),
                    captured_at="2026-01-01",
                    status="captured",
                    detail={"description": "Net sales"},
                    tables=["shop.orders"],
                    lineage_status="resolved",
                )
            ],
        }
    )
    assert build_artifact_insight(manifest(), files).metric_details[0].tables == ["shop.orders"]
    files["queries/sales.sql"] += " WHERE amount > 0"
    data = build_artifact_insight(manifest(), files)
    assert data.metric_details == []
    assert "analysis/metric_snapshots.json:stale" in data.warnings


def test_dashboard_only_shows_trial_branch_and_never_report_findings():
    template = QueryTemplateMetaFile(
        slug="sales",
        datasource="retail",
        params=[dict(name="archived", type="boolean")],
        columns=[dict(name="amount", type="number")],
        sample_params={"archived": False},
        sample_row_count=0,
        saved_at="2026-01-01",
    )
    sql = "SELECT amount FROM {% if archived %}shop.archive{% else %}shop.orders{% endif %}"
    data = build_artifact_insight(
        manifest("dashboard"),
        {
            "queries/sales.sql.j2": sql,
            "queries/sales.params.json": template.model_dump_json(),
            "analysis/insights.json": "[]",
        },
    )
    query = data.queries[0]
    assert query.result is None
    assert query.template.sample_row_count == 0
    assert query.lineage.status == "partial"
    assert query.lineage.origin == "sample_parameters"
    assert query.lineage.tables == ["shop.orders"]
    assert data.insights == []


@pytest.mark.parametrize("sql", [None, "not a valid query!", "DELETE FROM orders", "SELECT {{missing}} FROM {{table}}"])
def test_unknown_sql_never_claims_parsed_lineage(sql):
    assert query_lineage(sql, None, None).status == "unavailable"


def test_nested_ctes_and_unions_resolve_only_physical_tables():
    sql = "WITH a AS (SELECT * FROM schema.orders), b AS (WITH c AS (SELECT * FROM a) SELECT * FROM c) SELECT * FROM b UNION ALL SELECT * FROM schema.refunds"
    assert query_lineage(sql, None, "retail").tables == ["schema.orders", "schema.refunds"]


def test_dynamic_and_ambiguous_cards_do_not_invent_query_links():
    files = report_files()
    files["render/app.jsx"] = """<ChartCard chartId="dynamic" sqlId={query} />
    <ChartCard chartId="spread" sqlId="queries/sales" {...props} />
    <BlockHandle handleId="duplicate" sqlId="queries/sales" />
    <ChartCard chartId="duplicate" sqlId="queries/sales" />"""
    blocks = build_artifact_insight(manifest(), files).blocks
    assert [(b.id, b.query_ids) for b in blocks] == [("dynamic", [])]
