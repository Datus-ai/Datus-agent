# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Metric producer/reader contracts, with a compiler double and real SQLite reads."""

import hashlib
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from datus.api.services.dashboard_service import DashboardService, _load_local_template_pair
from datus.schemas.metric_artifact_query import MetricQueryFile, MetricQueryRequest
from datus.tools.func_tool import DashboardArtifactTools, DBFuncTool, ReportArtifactTools
from datus.tools.func_tool.metric_artifact_tools import execute_metric_artifact_query, metric_query_arguments
from datus.tools.semantic_tools.models import QueryResult
from datus.utils.artifact_insight import build_artifact_insight
from datus.utils.exceptions import ErrorCode

REF = {"path": ["Commerce", "Revenue"], "name": "revenue"}


@pytest.fixture
def environment(tmp_path):
    from datus.tools.db_tools.config import SQLiteConfig
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector

    model = tmp_path / "subject" / "semantic_models" / "warehouse" / "shop.yml"
    model.parent.mkdir(parents=True)
    model.write_text("version: '0.2.0.dev0'\nsemantic_model: []\n", encoding="utf-8")
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders(region TEXT, amount REAL)")
        connection.executemany("INSERT INTO orders VALUES (?, ?)", [("North", 12), ("South", 8), ("North", 5)])
    db = DBFuncTool(connector_or_manager=SQLiteConnector(SQLiteConfig(db_path=str(database))))
    db.execute_read_enforced = MagicMock(wraps=db.execute_read_enforced)

    class Runtime:
        def __init__(self):
            self.calls = []

        def artifact_metric_binding(self, name):
            assert name == REF["name"]
            return {
                "model_path": str(model),
                "model_revision": hashlib.sha256(model.read_bytes()).hexdigest(),
                "datasource": "warehouse",
            }

        async def query_metrics(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["metrics"] == ["revenue"] and kwargs["dry_run"] is True
            grouped = bool(kwargs["dimensions"])
            sql = "SELECT " + ("region, " if grouped else "") + "SUM(amount) AS revenue FROM orders"
            if kwargs.get("where"):
                sql += " WHERE " + kwargs["where"]
            if grouped:
                sql += " GROUP BY region ORDER BY region"
            return QueryResult(metadata={"sql": sql, "outputs": [{"name": "revenue", "type": "metric"}]})

        async def lineage_graph(self):
            return [
                {
                    "nodes": [
                        {
                            "id": "metric",
                            "name": "revenue",
                            "kind": "metric_atomic",
                            "detail": {"expression": "SUM(orders.amount)"},
                        }
                    ],
                    "edges": [],
                }
            ]

    runtime = Runtime()
    detail = {**REF, "dimensions": [{"name": "orders.region"}], "description": "Net revenue"}
    semantic = SimpleNamespace(runtime=runtime, get_metric=lambda **kwargs: SimpleNamespace(success=1, result=detail))
    return SimpleNamespace(
        root=tmp_path,
        model=model,
        db=db,
        semantic=semantic,
        runtime=runtime,
        config=SimpleNamespace(project_root=str(tmp_path)),
    )


def _save(env, kind="report", query=None, **kwargs):
    cls = ReportArtifactTools if kind == "report" else DashboardArtifactTools
    tools = cls(agent_config=env.config, db_func_tool=env.db, semantic_tools=env.semantic)
    start = tools.start_new_report if kind == "report" else tools.start_new_dashboard
    assert start("demo", "Metric demo", "A metric backed artifact").success == 1
    save = tools.save_metric_query if kind == "report" else tools.save_metric_query_template
    saved = save(
        name="sales",
        query=query or {"metric": REF},
        goal="Revenue by region",
        hypothesis="North has higher revenue",
        **kwargs,
    )
    return tools, saved


@pytest.mark.parametrize("kind", ["report", "dashboard"])
@pytest.mark.parametrize(
    "arrow_type, expected",
    [
        (pa.int64(), "integer"),
        (pa.float64(), "number"),
        (pa.decimal128(12, 2), "number"),
        (pa.bool_(), "boolean"),
        (pa.date32(), "date"),
        (pa.timestamp("us"), "date"),
        (pa.string(), "string"),
    ],
)
def test_empty_metric_result_preserves_arrow_column_types(environment, kind, arrow_type, expected):
    table = pa.Table.from_batches([], schema=pa.schema([("region", pa.string()), ("revenue", arrow_type)]))
    environment.db.execute_read_enforced.return_value = SimpleNamespace(success=True, sql_return=table)
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, saved = _save(environment, kind, query={"metric": REF, "dimensions": ["orders.region"]}, **options)
    assert saved.success == 1, saved.error
    suffix = ".params.json" if kind == "dashboard" else ".json"
    actual = json.loads((tools.queries_dir / f"sales{suffix}").read_text(encoding="utf-8"))
    assert actual["columns"] == [{"name": "region", "type": "string"}, {"name": "revenue", "type": expected}]
    assert actual.get("sample_row_count", actual.get("row_count")) == 0


def test_report_metric_result_limit_counts_utf8_bytes_and_preserves_saved_files(environment, monkeypatch):
    env = environment
    with sqlite3.connect(env.root / "shop.sqlite") as connection:
        connection.execute("UPDATE orders SET region = ?", ("地区" * 500,))
    query = {"metric": REF, "dimensions": ["orders.region"]}
    tools, saved = _save(env, query=query)
    assert saved.success == 1, saved.error
    before = {path.name: path.read_bytes() for path in tools.queries_dir.iterdir()}
    result = before["sales.json"].decode("utf-8").rstrip("\n")
    cap = (len(result) + len(result.encode("utf-8"))) // 2
    assert len(result) < cap < len(result.encode("utf-8"))
    monkeypatch.setattr("datus.tools.func_tool.report_artifact_tools._MAX_QUERY_BYTES", cap)
    rejected = tools.save_metric_query(
        name="sales", query=query, goal="Revenue by region", hypothesis="North has higher revenue"
    )
    assert rejected.success == 0
    assert "5 MB" in rejected.error
    assert {path.name: path.read_bytes() for path in tools.queries_dir.iterdir()} == before


@pytest.mark.parametrize(
    "source_metric, reusable",
    [(REF, True), (None, True), ({**REF, "name": "other"}, False), ({**REF, "path": ["Other"]}, False)],
)
def test_insight_only_combines_matching_metric_result_identity(environment, source_metric, reusable):
    tools, saved = _save(environment)
    assert saved.success == 1, saved.error
    files = {
        path.relative_to(tools.report_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in tools.report_dir.rglob("*")
        if path.is_file()
    }
    result = json.loads(files["queries/sales.json"])
    if source_metric is None:
        result["source"].pop("metric")
    else:
        result["source"]["metric"] = source_metric
    files["queries/sales.json"] = json.dumps(result)
    insight = build_artifact_insight(json.loads(files["manifest.json"]), files)
    assert (insight.queries[0].metric_query is not None) == reusable
    assert insight.queries[0].source_kind == ("metric" if reusable else "sql")
    assert ("queries/sales.metric.json:source_mismatch" in insight.warnings) == (not reusable)
    assert [detail.ref.model_dump() for detail in insight.metric_details] == ([REF] if reusable else [])


@pytest.mark.parametrize(
    "dimensions, expected",
    [
        ([], [{"revenue": 25.0}]),
        (["orders.region"], [{"region": "North", "revenue": 17.0}, {"region": "South", "revenue": 8.0}]),
    ],
)
def test_report_metric_executes_complete_result_and_preserves_execution_definition(environment, dimensions, expected):
    env = environment
    tools, saved = _save(env, query={"metric": REF, "dimensions": dimensions})
    assert saved.success == 1, saved.error
    result = json.loads((tools.queries_dir / "sales.json").read_text())
    assert result["rows"] == expected
    assert result["row_count"] == len(expected)
    assert result["source"] == {"kind": "metric", "metric": REF}
    assert env.db.execute_read_enforced.call_count == 1
    files = {
        p.relative_to(tools.report_dir).as_posix(): p.read_text() for p in tools.report_dir.rglob("*") if p.is_file()
    }
    insight = build_artifact_insight(json.loads((tools.report_dir / "manifest.json").read_text()), files)
    assert [q.name for q in insight.queries] == ["sales"]
    assert insight.queries[0].source_kind == "metric"
    assert insight.queries[0].sql == result["sql"]
    assert insight.queries[0].lineage.tables == ["orders"]
    assert insight.metric_details[0].origin == "metric_execution"
    assert insight.metric_details[0].detail["definition"]["expression"] == "SUM(orders.amount)"
    assert tools._validate_metric_queries("report") is None


@pytest.mark.asyncio
async def test_dashboard_producer_loads_as_metric_and_live_values_are_bound(environment, monkeypatch):
    env = environment
    tools, saved = _save(
        env,
        "dashboard",
        query={
            "metric": REF,
            "dimensions": ["orders.region"],
            "filters": [{"dimension": "orders.region", "op": "eq", "value": {"param": "region"}}],
        },
        params=[{"name": "region", "type": "string"}],
        sample_params={"region": "South"},
    )
    assert saved.success == 1, saved.error
    loaded = await _load_local_template_pair(env.root, "demo", "sales")
    assert loaded.success is True
    assert isinstance(loaded.data[0], MetricQueryFile)
    import datus.tools.func_tool as func_tools
    import datus.tools.func_tool.semantic_tools as semantic_mod

    monkeypatch.setattr(func_tools, "DBFuncTool", lambda **kwargs: env.db)
    monkeypatch.setattr(semantic_mod, "SemanticTools", lambda **kwargs: env.semantic)
    policy = {"reader": "north"}
    result = await DashboardService(agent_config=env.config).run_query(
        project_files_root=env.root,
        dashboard_slug="demo",
        query_slug="sales",
        params={"region": "North"},
        policy_context=policy,
    )
    assert result.success is True, result.errorMessage
    assert result.data.rows == [{"region": "North", "revenue": 17.0}]
    assert env.db.execute_read_enforced.call_args.kwargs["policy_context"] is policy
    assert tools._validate_metric_queries("dashboard") is None
    assert env.runtime.calls[-1]["where"] == "(orders.region = 'North')"


def test_saved_model_change_refuses_execution_before_warehouse_read(environment):
    env = environment
    tools, saved = _save(env)
    assert saved.success == 1
    recipe = MetricQueryFile.model_validate_json((tools.queries_dir / "sales.metric.json").read_text())
    env.model.write_text("different model", encoding="utf-8")
    with pytest.raises(ValueError, match="METRIC_MODEL_CHANGED"):
        execute_metric_artifact_query(
            recipe, semantic_tools=env.semantic, db_tool=env.db, project_root=env.root, saved=recipe
        )
    assert env.db.execute_read_enforced.call_count == 1


def test_policy_denial_never_persists_metric_or_falls_back_to_sql(environment):
    env = environment
    env.db.execute_read_enforced.return_value = SimpleNamespace(
        success=False, error="Reader denied", error_code=ErrorCode.POLICY_DENIED.code
    )
    tools, saved = _save(env)
    assert not saved.success
    assert "Reader denied" in saved.error
    assert list(tools.queries_dir.iterdir()) == []
    assert env.db.execute_read_enforced.call_count == 1


@pytest.mark.parametrize("change", ["source", "brief", "result"])
def test_metric_bundle_validation_rejects_mismatched_sidecars(environment, change):
    tools, saved = _save(environment)
    assert saved.success == 1
    if change == "source":
        (tools.queries_dir / "sales.sql").write_text("SELECT 1")
    elif change == "brief":
        path = tools.queries_dir / "sales.brief.json"
        data = json.loads(path.read_text())
        data["uses"]["metrics"] = []
        path.write_text(json.dumps(data))
    else:
        path = tools.queries_dir / "sales.json"
        data = json.loads(path.read_text())
        data["sql"] = "SELECT 0"
        path.write_text(json.dumps(data))
    assert "Invalid saved metric query" in tools._validate_metric_queries("report")


def test_sql_fallback_requires_recorded_reason_and_clears_previous_metric_source(environment):
    tools, saved = _save(environment)
    assert saved.success == 1
    args = dict(
        name="sales", sql="SELECT COUNT(*) AS total FROM orders", goal="Custom analysis", hypothesis="There are orders"
    )
    denied = tools.save_query(**args)
    assert not denied.success and "fallback_reason" in denied.error
    fallback = tools.save_query(
        **args, fallback_reason="Needs a count that no existing metric defines", candidate_metrics=[REF]
    )
    assert fallback.success == 1, fallback.error
    brief = json.loads((tools.queries_dir / "sales.brief.json").read_text())
    assert brief["source_selection"]["candidate_metrics"] == [REF]
    assert not (tools.queries_dir / "sales.metric.json").exists()


@pytest.mark.parametrize(
    "op, value, expected",
    [
        ("eq", "O'Reilly", "orders.region = 'O''Reilly'"),
        ("ne", None, "orders.region IS NOT NULL"),
        ("in", [], "1 = 0"),
        ("not_in", [], "1 = 1"),
        ("in", ["North", "South"], "orders.region IN ('North', 'South')"),
        ("gt", 3, "orders.region > 3"),
        ("is_null", None, "orders.region IS NULL"),
        ("eq", True, "orders.region = TRUE"),
    ],
)
def test_filter_bindings_are_values_not_sql(op, value, expected):
    query = MetricQueryRequest(
        metric=REF,
        filters=[{"dimension": "orders.region", "op": op, "value": {"param": "value"} if value is not None else None}],
    )
    args = metric_query_arguments(query, {"value": value})
    assert args["where"] == f"({expected})"
    assert args["metrics"] == ["revenue"]


@pytest.mark.parametrize("op, value", [("in", "North"), ("eq", {}), ("gte", None), ("eq", float("inf"))])
def test_filter_rejects_unstructured_or_nonfinite_values(op, value):
    query = MetricQueryRequest(
        metric=REF,
        filters=[{"dimension": "orders.region", "op": op, "value": {"param": "value"} if value is not None else None}],
    )
    with pytest.raises(ValueError):
        metric_query_arguments(query, {"value": value})


def test_required_parameter_and_identifier_injection_are_rejected():
    query = MetricQueryRequest(metric=REF, time_start={"param": "start_date"})
    with pytest.raises(ValueError, match="missing metric query parameter"):
        metric_query_arguments(query, {})
    with pytest.raises(ValueError):
        MetricQueryRequest(metric=REF, filters=[{"dimension": "orders.region; DROP TABLE orders", "value": "North"}])


def test_optional_bindings_omit_slicers_and_metric_defaults():
    request = MetricQueryRequest.model_validate(
        {
            "metric": REF,
            "filters": [{"dimension": "orders.region", "value": {"param": "region"}}],
            "metric_params": {"threshold": {"param": "threshold"}},
        }
    )
    args = metric_query_arguments(request, {"region": None, "threshold": None})
    assert args["where"] is None
    assert args["params"] == {}


@pytest.mark.parametrize("value", ["a\\' OR 1=1 --", "a\x00b"])
def test_ambiguous_filter_literal_is_rejected(value):
    request = MetricQueryRequest.model_validate(
        {
            "metric": REF,
            "filters": [{"dimension": "orders.region", "value": value}],
        }
    )
    with pytest.raises(ValueError, match="backslashes or NUL"):
        metric_query_arguments(request, {})


def test_pinned_execution_does_not_depend_on_live_subject_index(environment):
    env = environment
    tools, result = _save(env)
    assert result.success == 1, result.error
    saved = MetricQueryFile.model_validate_json((tools.queries_dir / "sales.metric.json").read_text())
    env.semantic.get_metric = MagicMock(side_effect=AssertionError("published subject index unavailable"))
    payload, _ = execute_metric_artifact_query(
        saved,
        semantic_tools=env.semantic,
        db_tool=env.db,
        project_root=env.root,
        saved=saved,
    )
    assert payload.rows == [{"revenue": 25.0}]


def test_optional_dashboard_slicer_can_be_omitted(environment):
    tools, saved = _save(
        environment,
        "dashboard",
        query={
            "metric": REF,
            "dimensions": ["orders.region"],
            "filters": [{"dimension": "orders.region", "value": {"param": "region"}}],
        },
        params=[{"name": "region", "type": "string", "required": False}],
        sample_params={},
    )
    assert saved.success == 1, saved.error
    assert saved.result["row_count"] == 2
    assert environment.runtime.calls[-1]["where"] is None
    assert tools._validate_metric_queries("dashboard") is None


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_same_metric_queries_can_be_resaved_sequentially_but_mixed_revisions_fail_final_validation(environment, kind):
    env = environment
    kwargs = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, saved = _save(env, kind, **kwargs)
    assert saved.success == 1, saved.error
    save = tools.save_metric_query_template if kind == "dashboard" else tools.save_metric_query
    args = dict(
        query={"metric": REF},
        goal="Next sales",
        hypothesis="Revenue changes",
        **kwargs,
    )
    second = save(name="next_sales", **args)
    assert second.success == 1, second.error
    assert tools._validate_metric_queries(kind) is None
    env.model.write_text("version: '0.2.0.dev0'\nsemantic_model: []\n# changed\n", encoding="utf-8")
    first_update = save(name="sales", **args)
    assert first_update.success == 1, first_update.error
    assert "cannot mix model revisions" in tools._validate_metric_queries(kind)
    second_update = save(name="next_sales", **args)
    assert second_update.success == 1, second_update.error
    assert tools._validate_metric_queries(kind) is None
    recipes = [
        MetricQueryFile.model_validate_json(path.read_text(encoding="utf-8"))
        for path in tools.queries_dir.glob("*.metric.json")
    ]
    assert len({recipe.model_revision for recipe in recipes}) == 1
