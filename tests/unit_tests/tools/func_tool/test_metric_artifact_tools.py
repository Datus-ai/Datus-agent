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
    model.write_text("version: 0.2.0.dev0\nname: model\ndatasets: []\n", encoding="utf-8")
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
    semantic = SimpleNamespace(
        runtime=runtime, metric_record=lambda **kwargs: SimpleNamespace(success=1, result=detail)
    )
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
    env.semantic.metric_record = MagicMock(side_effect=AssertionError("published subject index unavailable"))
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
    env.model.write_text(
        "version: 0.2.0.dev0\nname: model\ndatasets: []\ndescription: Revised model\n", encoding="utf-8"
    )
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


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_same_named_metrics_on_different_datasources_keep_independent_revisions(environment, kind, monkeypatch):
    env = environment
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, saved = _save(env, kind, **options)
    assert saved.success == 1, saved.error
    other = env.root / "subject" / "semantic_models" / "other" / "shop.yml"
    other.parent.mkdir(parents=True)
    other.write_text("version: 0.2.0.dev0\nname: model\ndatasets: []\n")
    monkeypatch.setattr(
        env.runtime,
        "artifact_metric_binding",
        lambda name: {
            "model_path": str(other),
            "model_revision": hashlib.sha256(other.read_bytes()).hexdigest(),
            "datasource": "other",
        },
    )
    save = tools.save_metric_query_template if kind == "dashboard" else tools.save_metric_query
    second = save(name="other_sales", query={"metric": REF}, goal="Other sales", hypothesis="Compare", **options)
    assert second.success == 1, second.error
    assert tools._validate_metric_queries(kind) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["report", "dashboard"])
@pytest.mark.parametrize("model_name", [None, "flights.yml", "jeff_shop.yml"])
async def test_queries_share_model_files_and_bundle_preserves_exact_bytes(environment, kind, model_name):
    from datus.api.services.report_service import ReportService
    from datus.utils.metric_artifact_models import resolve_metric_query_snapshot

    env = environment
    content = b"version: '0.2.0.dev0'\r\nname: model\r\ndatasets: []\r\n"
    env.model.write_bytes(content)
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, first = _save(env, kind, **options)
    assert first.success == 1, first.error
    save = tools.save_metric_query_template if kind == "dashboard" else tools.save_metric_query
    second = save(name="total", query={"metric": REF}, goal="Total", hypothesis="Revenue", **options)
    assert second.success == 1, second.error
    artifact_dir = tools.queries_dir.parent
    recipes = [json.loads((tools.queries_dir / f"{name}.metric.json").read_text()) for name in ("sales", "total")]
    assert all(item["schema_version"] == "1.1" and "model_snapshot" not in item for item in recipes)
    assert recipes[0]["model_snapshot_path"] == recipes[1]["model_snapshot_path"]
    revision = hashlib.sha256(content).hexdigest()
    assert all(item["model_revision"] == revision for item in recipes)
    assert recipes[0]["model_snapshot_path"] == f"models/{revision[:16]}.yml"
    assert len(list((artifact_dir / "models").iterdir())) == 1
    assert (artifact_dir / recipes[0]["model_snapshot_path"]).read_bytes() == content
    if model_name is not None:
        renamed = f"models/{model_name}"
        (artifact_dir / recipes[0]["model_snapshot_path"]).rename(artifact_dir / renamed)
        for name, item in zip(("sales", "total"), recipes):
            item["model_snapshot_path"] = renamed
            (tools.queries_dir / f"{name}.metric.json").write_text(json.dumps(item))
    assert tools._validate_metric_queries(kind) is None
    (artifact_dir / "render" / "app.jsx").write_text("export default function App() { return null; }")
    service = ReportService() if kind == "report" else DashboardService(agent_config=env.config)
    detail = await service.get_detail(project_files_root=env.root, **{f"{kind}_slug": "demo"})
    assert detail.success is True, detail.errorMessage
    files = {item.path: item.content for item in detail.data.files}
    assert files[recipes[0]["model_snapshot_path"]].encode() == content
    from datus.agent.node.visual_artifact._artifact_html_renderer import _read_artifact_files
    from datus.agent.node.visual_artifact.dashboard_html_renderer import _DASHBOARD_ARTIFACT_DIRS
    from datus.agent.node.visual_artifact.report_html_renderer import _REPORT_ARTIFACT_DIRS

    spec = _REPORT_ARTIFACT_DIRS if kind == "report" else _DASHBOARD_ARTIFACT_DIRS
    html_files = {item["path"]: item["content"] for item in _read_artifact_files(artifact_dir, spec)}
    assert html_files[recipes[0]["model_snapshot_path"]].encode() == content
    assert (
        json.loads(html_files["queries/sales.metric.json"])["model_snapshot_path"] == recipes[0]["model_snapshot_path"]
    )
    for item in recipes:
        assert (
            resolve_metric_query_snapshot(MetricQueryFile.model_validate(item), files=files).model_snapshot.encode()
            == content
        )


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_multiple_models_with_same_basename_remain_distinct(environment, kind, monkeypatch):
    env = environment
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, saved = _save(env, kind, **options)
    assert saved.success == 1, saved.error
    second_model = env.root / "subject" / "semantic_models" / "other" / "shop.yml"
    second_model.parent.mkdir(parents=True)
    second_model.write_text("version: 0.2.0.dev0\nname: second_model\ndatasets: []\n")
    second_ref = {"path": ["Commerce", "Costs"], "name": "cost"}
    monkeypatch.setattr(
        env.runtime,
        "artifact_metric_binding",
        lambda name: {
            "model_path": str(second_model),
            "model_revision": hashlib.sha256(second_model.read_bytes()).hexdigest(),
            "datasource": "warehouse",
        },
    )
    monkeypatch.setattr(env.semantic, "metric_record", lambda **kwargs: SimpleNamespace(success=1, result=second_ref))

    async def compile_cost(**kwargs):
        assert kwargs["metrics"] == ["cost"]
        return QueryResult(metadata={"sql": "SELECT COUNT(*) AS cost FROM orders"})

    monkeypatch.setattr(env.runtime, "query_metrics", compile_cost)
    save = tools.save_metric_query_template if kind == "dashboard" else tools.save_metric_query
    result = save(name="cost", query={"metric": second_ref}, goal="Cost", hypothesis="Costs", **options)
    assert result.success == 1, result.error
    recipes = [
        MetricQueryFile.model_validate_json(path.read_text()) for path in tools.queries_dir.glob("*.metric.json")
    ]
    assert {item.model_path for item in recipes} == {
        env.model.relative_to(env.root).as_posix(),
        second_model.relative_to(env.root).as_posix(),
    }
    assert len({item.model_snapshot_path for item in recipes}) == 2
    assert {(tools.queries_dir.parent / item.model_snapshot_path).read_bytes() for item in recipes} == {
        env.model.read_bytes(),
        second_model.read_bytes(),
    }
    assert tools._validate_metric_queries(kind) is None


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_conflicting_snapshot_fails_without_changing_recipe_or_model(environment, kind):
    env = environment
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, first = _save(env, kind, **options)
    assert first.success == 1, first.error
    recipe_path = tools.queries_dir / "sales.metric.json"
    original_recipe = recipe_path.read_bytes()
    previous = MetricQueryFile.model_validate_json(original_recipe)
    occupied = tools.queries_dir.parent / previous.model_snapshot_path
    occupied.write_bytes(b"another model occupies this short filename")
    save = tools.save_metric_query_template if kind == "dashboard" else tools.save_metric_query
    saved = save(name="sales", query={"metric": REF}, goal="Revenue", hypothesis="Revenue", **options)
    assert saved.success == 0
    assert "must match model_revision" in saved.error
    assert recipe_path.read_bytes() == original_recipe
    assert occupied.read_bytes() == b"another model occupies this short filename"
    assert list(occupied.parent.iterdir()) == [occupied]


@pytest.mark.parametrize("legacy", [False, True])
def test_published_execution_uses_saved_model_after_studio_changes(environment, monkeypatch, legacy):
    from pathlib import Path

    from datus.tools.func_tool.metric_artifact_tools import execute_published_metric_query
    from datus.tools.semantic_tools.dosi.config import DosiConfig
    from datus.utils.metric_artifact_models import resolve_metric_query_snapshot

    env = environment
    tools, saved = _save(env, "dashboard", params=[], sample_params={})
    assert saved.success == 1, saved.error
    recipe_path = tools.queries_dir / "sales.metric.json"
    query = resolve_metric_query_snapshot(
        MetricQueryFile.model_validate_json(recipe_path.read_text()), artifact_dir=tools.queries_dir.parent
    )
    original = env.model.read_bytes()
    if legacy:
        query = MetricQueryFile.model_validate(
            {**query.model_dump(), "schema_version": "1.0", "model_snapshot_path": None}
        )
    env.model.write_text("Studio model changed")

    class FrozenRuntime:
        def __init__(self, config):
            self.path = Path(config.semantic_model_path)
            assert self.path.read_bytes() == original
            assert config.semantic_models_path is None

        def artifact_metric_binding(self, name):
            assert name == "revenue"
            return {
                "model_path": str(self.path),
                "model_revision": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "datasource": "warehouse",
            }

        async def query_metrics(self, **kwargs):
            return await env.runtime.query_metrics(**kwargs)

    monkeypatch.setattr("datus.tools.semantic_tools.dosi.DosiRuntime", FrozenRuntime)
    policy = {"reader": "alice"}
    result, _ = execute_published_metric_query(
        query, runtime_config=DosiConfig(), db_tool=env.db, params={}, policy_context=policy
    )
    assert result.rows == [{"revenue": 25.0}]
    assert env.db.execute_read_enforced.call_args.kwargs["policy_context"] == policy


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing", "tampered"])
async def test_local_loader_and_final_validation_reject_broken_model_reference(environment, problem):
    tools, saved = _save(environment, "dashboard", params=[], sample_params={})
    assert saved.success == 1
    query = MetricQueryFile.model_validate_json((tools.queries_dir / "sales.metric.json").read_text())
    model = tools.queries_dir.parent / query.model_snapshot_path
    if problem == "missing":
        model.unlink()
    else:
        model.write_text("tampered")
    result = await _load_local_template_pair(environment.root, "demo", "sales")
    assert not result.success and result.errorCode == "TEMPLATE_CORRUPT"
    assert "model snapshot" in tools._validate_metric_queries("dashboard")


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_different_metrics_cannot_mix_revisions_of_one_model(environment, kind, monkeypatch):
    from unittest.mock import AsyncMock

    env = environment
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, saved = _save(env, kind, **options)
    assert saved.success == 1
    env.model.write_text("version: 0.2.0.dev0\nname: model\ndatasets: []\ndescription: Revised model\n")
    other_ref = {"path": ["Commerce"], "name": "cost"}
    binding = env.runtime.artifact_metric_binding("revenue")
    monkeypatch.setattr(env.runtime, "artifact_metric_binding", lambda name: binding)
    monkeypatch.setattr(env.semantic, "metric_record", lambda **kwargs: SimpleNamespace(success=1, result=other_ref))
    monkeypatch.setattr(
        env.runtime,
        "query_metrics",
        AsyncMock(return_value=QueryResult(metadata={"sql": "SELECT COUNT(*) AS cost FROM orders"})),
    )
    save = tools.save_metric_query_template if kind == "dashboard" else tools.save_metric_query
    result = save(name="cost", query={"metric": other_ref}, goal="Cost", hypothesis="Costs", **options)
    assert result.success == 1, result.error
    assert "same model" in tools._validate_metric_queries(kind)


@pytest.mark.parametrize("kind", ["report", "dashboard"])
@pytest.mark.parametrize("operation", ["write", "edit", "delete"])
def test_model_files_are_tool_managed(environment, kind, operation):
    from datus.tools.func_tool.dashboard_artifact_tools import DashboardFilesystemFuncTool
    from datus.tools.func_tool.report_artifact_tools import ReportFilesystemFuncTool

    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, saved = _save(environment, kind, **options)
    assert saved.success == 1
    query = MetricQueryFile.model_validate_json((tools.queries_dir / "sales.metric.json").read_text())
    path = tools.queries_dir.parent / query.model_snapshot_path
    original = path.read_bytes()
    cls = ReportFilesystemFuncTool if kind == "report" else DashboardFilesystemFuncTool
    fs = cls(root_path=str(environment.root))
    relative = path.relative_to(environment.root).as_posix()
    if operation == "write":
        result = fs.write_file(relative, "changed")
    elif operation == "edit":
        result = fs.edit_file(relative, "datasets", "changed")
    else:
        result = fs.delete_file(relative)
    assert result.success == 0
    assert path.read_bytes() == original


def _install_metric_policy(env, monkeypatch, allowed, *, region="North"):
    original_binding = env.runtime.artifact_metric_binding
    env.runtime.artifact_metric_binding = lambda name: {**original_binding(name), "datasets": ["orders"]}
    env.db.agent_config = SimpleNamespace(active_plugin_names=lambda: {"scope"}, get_plugin_profile=lambda _: {})

    contexts = []

    class MetricPolicy:
        def before_metric_read(self, *, datasets, where, policy_context):
            assert datasets == {"orders"}
            contexts.append(policy_context)
            return SimpleNamespace(
                allowed=allowed and policy_context.get("region") == region,
                where=f"orders.region = '{region}'",
                applied_policies=["region_scope"],
                reason="denied",
            )

    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda _: {"scope": lambda _: MetricPolicy()},
    )

    return contexts


def test_metric_policy_is_applied_before_compilation(environment, monkeypatch):
    from datus.tools.read_observer import observe_reads

    env = environment
    _install_metric_policy(env, monkeypatch, True)
    events = []
    with observe_reads(events.append):
        result, _ = execute_metric_artifact_query(
            MetricQueryRequest(metric=REF),
            semantic_tools=env.semantic,
            db_tool=env.db,
            project_root=env.root,
            policy_context={"region": "North"},
        )
    assert result.rows == [{"revenue": 17.0}]
    assert env.runtime.calls[0]["where"] == "orders.region = 'North'"
    assert events[0].phase == "metric_compile"
    assert events[0].applied_policy_ids == ("region_scope",)


def test_native_artifact_policies_apply_once_at_the_final_sql_boundary(environment, monkeypatch):
    from unittest.mock import AsyncMock

    env = environment
    context = {"sql_policy": {"version": "request", "policies": {}}}
    env.db.agent_config = SimpleNamespace(active_plugin_names=lambda: {"sql-policy"}, get_plugin_profile=lambda _: {})

    class NativePolicy:
        def validate_context(self, value):
            assert value == context
            return SimpleNamespace(allowed=True)

        def requires_bound_read(self, value):
            return True

        def before_metric_read(self, **kwargs):
            raise AssertionError("physical policies must not use the former metric-WHERE hook")

        def bind_read(self, **kwargs):
            raise AssertionError("the enforced DB tool owns the execution binding")

    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda _: {"sql-policy": lambda _: NativePolicy()},
    )
    compiler = AsyncMock(wraps=env.runtime.query_metrics)
    env.runtime.compile_for_sql_execution = compiler
    env.runtime.query_metrics = AsyncMock(side_effect=AssertionError("do not apply policies during artifact compile"))
    env.db.execute_read_enforced.return_value = SimpleNamespace(
        success=True, sql_return=pa.table({"region": ["masked"], "revenue": [17.0]})
    )
    result, _ = execute_metric_artifact_query(
        MetricQueryRequest.model_validate({"metric": REF, "dimensions": ["orders.region"]}),
        semantic_tools=env.semantic,
        db_tool=env.db,
        project_root=env.root,
        policy_context=context,
    )
    assert result.rows == [{"region": "masked", "revenue": 17.0}]
    compiler.assert_awaited_once()
    env.runtime.query_metrics.assert_not_called()
    env.db.execute_read_enforced.assert_called_once()
    assert env.db.execute_read_enforced.call_args.kwargs["policy_context"] is context


def test_metric_policy_denial_does_not_compile_or_query(environment, monkeypatch):
    from datus.utils.exceptions import DatusException

    env = environment
    _install_metric_policy(env, monkeypatch, False)
    with pytest.raises(DatusException):
        execute_metric_artifact_query(
            MetricQueryRequest(metric=REF),
            semantic_tools=env.semantic,
            db_tool=env.db,
            project_root=env.root,
            policy_context={"region": "North"},
        )
    assert env.runtime.calls == []
    env.db.execute_read_enforced.assert_not_called()


def test_metric_policy_requires_resolved_model_datasets(environment, monkeypatch):
    from datus.utils.exceptions import DatusException

    env = environment
    _install_metric_policy(env, monkeypatch, True)
    binding = env.runtime.artifact_metric_binding("revenue")
    binding.pop("datasets")
    env.runtime.artifact_metric_binding = lambda _: binding
    with pytest.raises(DatusException):
        execute_metric_artifact_query(
            MetricQueryRequest(metric=REF),
            semantic_tools=env.semantic,
            db_tool=env.db,
            project_root=env.root,
            policy_context={"region": "North"},
        )
    assert env.runtime.calls == []
    env.db.execute_read_enforced.assert_not_called()


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_metric_save_inherits_tool_policy_context(environment, monkeypatch, kind):
    from datus.tools.read_observer import observe_reads

    env = environment
    contexts = _install_metric_policy(env, monkeypatch, True)
    env.db.agent_config.policy_context = {"region": "North"}
    events = []
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    with observe_reads(events.append):
        tools, saved = _save(env, kind, **options)

    assert saved.success == 1, saved.error
    assert contexts == [env.db.policy_context]
    assert env.runtime.calls[0]["where"] == "orders.region = 'North'"
    effective_context = env.db.execute_read_enforced.call_args.kwargs["policy_context"]
    assert effective_context == env.db.policy_context
    assert [event.phase for event in events] == ["metric_compile", "before_sql", "after_sql"]
    assert all(event.policy_context is effective_context for event in events)
    if kind == "report":
        payload = json.loads((tools.queries_dir / "sales.json").read_text())
        assert payload["rows"] == [{"revenue": 17.0}]
    else:
        payload = json.loads((tools.queries_dir / "sales.params.json").read_text())
        assert payload["sample_row_count"] == 1


@pytest.mark.parametrize("explicit_context", [{"region": "North"}, {}])
def test_metric_context_override_never_falls_back_to_tool(environment, monkeypatch, explicit_context):
    from datus.tools.read_observer import observe_reads
    from datus.utils.exceptions import DatusException

    env = environment
    contexts = _install_metric_policy(env, monkeypatch, True)
    env.db.agent_config.policy_context = {"region": "North", "tool_only": True}
    events = []
    kwargs = dict(
        semantic_tools=env.semantic,
        db_tool=env.db,
        project_root=env.root,
        policy_context=explicit_context,
    )
    with observe_reads(events.append):
        if explicit_context:
            result, _ = execute_metric_artifact_query(MetricQueryRequest(metric=REF), **kwargs)
            assert result.rows == [{"revenue": 17.0}]
            assert env.db.execute_read_enforced.call_args.kwargs["policy_context"] is explicit_context
            assert all(event.policy_context is explicit_context for event in events)
        else:
            with pytest.raises(DatusException) as exc:
                execute_metric_artifact_query(MetricQueryRequest(metric=REF), **kwargs)
            assert exc.value.code == ErrorCode.POLICY_DENIED
            assert env.runtime.calls == []
            env.db.execute_read_enforced.assert_not_called()
            assert events == []
    assert contexts == [explicit_context]


@pytest.mark.parametrize("kind", ["report", "dashboard"])
def test_metric_artifact_keeps_policy_sql_out_of_public_content(environment, monkeypatch, kind):
    env = environment
    marker = "private_policy_region_7ad9f2"
    with sqlite3.connect(env.root / "shop.sqlite") as connection:
        connection.execute("UPDATE orders SET region = ? WHERE region = 'North'", (marker,))
    _install_metric_policy(env, monkeypatch, True, region=marker)
    env.db.agent_config.policy_context = {"region": marker}
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}

    tools, saved = _save(env, kind, **options)

    assert saved.success == 1, saved.error
    executed_sql = env.db.execute_read_enforced.call_args.args[0]
    assert marker in executed_sql
    artifact_dir = tools.queries_dir.parent
    files = {p.relative_to(artifact_dir).as_posix(): p.read_text() for p in artifact_dir.rglob("*") if p.is_file()}
    assert marker not in json.dumps(files)
    assert marker not in saved.model_dump_json()
    recipe = MetricQueryFile.model_validate_json(files["queries/sales.metric.json"])
    assert recipe.generated_sql is None
    assert recipe.metric_tables == ["orders"]
    assert tools._validate_metric_queries(kind) is None
    insight = build_artifact_insight(json.loads(files["manifest.json"]), files)
    assert insight.queries[0].source_kind == "metric"
    assert insight.queries[0].lineage.tables == ["orders"]
    assert marker not in insight.model_dump_json()
    output_path, expected = {
        "report": ("queries/sales.json", {"rows": [{"revenue": 17.0}], "sql": None}),
        "dashboard": ("queries/sales.params.json", {"sample_row_count": 1}),
    }[kind]
    output = json.loads(files[output_path])
    assert {key: output.get(key) for key in expected} == expected

    # Re-reading the persisted recipe compiles against the reader's context;
    # no author's SQL is needed, even after reloading the JSON from disk.
    _install_metric_policy(env, monkeypatch, True, region="South")
    result, public_binding = execute_metric_artifact_query(
        recipe,
        saved=recipe,
        semantic_tools=env.semantic,
        db_tool=env.db,
        project_root=env.root,
        policy_context={"region": "South"},
    )
    assert result.rows == [{"revenue": 8.0}]
    assert result.sql is None
    assert public_binding["generated_sql"] is None
    assert marker not in env.db.execute_read_enforced.call_args.args[0]


def test_metric_tool_initializes_without_model_selection(monkeypatch):
    from datus.tools.func_tool.semantic_tools import SemanticTools
    from datus.utils.exceptions import DatusException

    config = SimpleNamespace(active_model=MagicMock(side_effect=DatusException(ErrorCode.MODEL_NOT_SELECTED)))
    monkeypatch.setattr("datus.tools.func_tool.semantic_tools.MetricRAG", lambda *args: None)
    tool = SemanticTools(config)
    config.active_model.assert_not_called()
    with pytest.raises(DatusException) as error:
        _ = tool.compressor
    assert error.value.code == ErrorCode.MODEL_NOT_SELECTED


@pytest.mark.parametrize("kind", ["report", "dashboard"])
@pytest.mark.parametrize("engine", ["sqlite", "duckdb"])
def test_metric_artifact_replays_its_original_database_after_a_turn_switch(environment, kind, engine):
    from collections import OrderedDict

    from datus.configuration.agent_config import DbConfig
    from datus.tools.db_tools.db_manager import DBManager

    env = environment
    if engine == "duckdb":
        import duckdb

        with duckdb.connect(str(env.root / "shop.duckdb")) as connection:
            connection.execute("CREATE TABLE orders AS SELECT 'Main' AS region, 888.0 AS amount")
            for schema, amount in [("tenant_a", 25.0), ("tenant_b", 999.0)]:
                connection.execute(f"CREATE SCHEMA {schema}")
                connection.execute(f"CREATE TABLE {schema}.orders AS SELECT 'Tenant' AS region, ? AS amount", [amount])
        config = DbConfig(type="duckdb", uri=str(env.root / "shop.duckdb"))
    else:
        with sqlite3.connect(env.root / "other.sqlite") as connection:
            connection.execute("CREATE TABLE orders(region TEXT, amount REAL)")
            connection.execute("INSERT INTO orders VALUES ('Other', 999)")
        config = DbConfig(type="sqlite", path_pattern=str(env.root / "*.sqlite"))
    manager = DBManager({"warehouse": config})
    env.db._db_manager = manager
    env.db._is_multi_connector = True
    env.db._default_datasource = "warehouse"
    env.db._default_database = "shop"
    env.db._execution_context = {"datasource": "warehouse", "database": "shop"}
    if engine == "duckdb":
        env.db._execution_context["schema"] = "tenant_a"
    env.db._connector_cache = OrderedDict()
    env.db._connector_cache_size = 8
    env.runtime.config = SimpleNamespace(
        db_config={"database": "shop", **({"schema": "tenant_a"} if engine == "duckdb" else {})}
    )
    options = {"params": [], "sample_params": {}} if kind == "dashboard" else {}
    tools, result = _save(env, kind, **options)
    assert result.success == 1, result.error
    saved = MetricQueryFile.model_validate_json((tools.queries_dir / "sales.metric.json").read_text())
    assert saved.execution_context == {
        "datasource": "warehouse",
        "database": "shop",
        "catalog": "",
        "schema": "tenant_a" if engine == "duckdb" else "",
    }
    env.db._default_database = "shop" if engine == "duckdb" else "other"
    env.db._execution_context = {"datasource": "warehouse", "database": env.db._default_database}
    if engine == "duckdb":
        env.db._execution_context["schema"] = "tenant_b"
    env.runtime.config = SimpleNamespace(db_config={"database": "other"})
    payload, _ = execute_metric_artifact_query(
        saved, saved=saved, semantic_tools=env.semantic, db_tool=env.db, project_root=env.root
    )
    assert payload.rows == [{"revenue": 25.0}]
    current = env.db._get_connector("warehouse")
    assert current.execute_query("SELECT SUM(amount) AS total FROM orders", result_format="list").sql_return == [
        {"total": 999.0}
    ]
    manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".metric.json", ".params.json"])
async def test_dashboard_rejects_a_saved_context_pointing_at_another_datasource(environment, suffix):
    env = environment
    tools, saved = _save(env, "dashboard", params=[], sample_params={})
    assert saved.success == 1, saved.error
    path = tools.queries_dir / f"sales{suffix}"
    content = json.loads(path.read_text())
    content["execution_context"]["datasource"] = "other"
    path.write_text(json.dumps(content))
    env.db.execute_read_enforced.reset_mock()
    service = DashboardService(agent_config=env.config)
    response = await service.run_query(
        project_files_root=env.root, dashboard_slug="demo", query_slug="sales", params={}
    )
    assert response.success is False
    assert response.errorCode == "TEMPLATE_CORRUPT"
    assert "execution context must match" in response.errorMessage
    env.db.execute_read_enforced.assert_not_called()
