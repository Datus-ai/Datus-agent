# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``query_metrics`` filter placement against the real Dosi binding: ``where``
versus ``context_filter`` on a window metric, ``where`` on a selected metric
(HAVING), and the structured refusal each of them can produce."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.tools.func_tool import DBFuncTool, ReportArtifactTools
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.semantic_tools import SemanticTools

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

FIXTURE_DIR = Path(__file__).parents[2] / "data" / "semantic_models" / "window_filters"
MODEL_PATH = FIXTURE_DIR / "model.yaml"
SEED_PATH = FIXTURE_DIR / "seed.sql"

PRODUCT_ITEM = ["scores.product", "scores.item"]
RANKING = ["item_rank", "items_in_product"]


@pytest.fixture
def query_tool(tmp_path, monkeypatch) -> SemanticTools:
    database = tmp_path / "window_filters.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(SEED_PATH.read_text(encoding="utf-8"))

    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="dosi_query_filters_acceptance",
        project_root=str(tmp_path / "workspace"),
        target="mock",
        models={
            "mock": {
                "type": "openai",
                "api_key": "unused",
                "model": "unused",
                "base_url": "http://127.0.0.1:1",
            }
        },
        services={
            "datasources": {
                "window_filters": {
                    "type": "duckdb",
                    "uri": str(database),
                    "default": True,
                }
            },
        },
    )
    config.current_datasource = "window_filters"
    model_dir = Path(config.project_root) / "subject" / "semantic_models" / "window_filters"
    model_dir.mkdir(parents=True)
    (model_dir / "model.yaml").write_text(MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config.path_manager, "semantic_model_path", lambda datasource: model_dir)

    # Metric queries do not use MetricRAG. Replacing that unrelated storage
    # dependency keeps this suite on the real engine binding, SQL execution,
    # and the Agent tool contract.
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        tool = SemanticTools(agent_config=config)

    assert type(tool.runtime).__module__ == "datus.tools.semantic_tools.dosi.runtime"
    return tool


def _rows(tool: SemanticTools, result) -> list[dict[str, str]]:
    """The full rows behind a successful ``query_metrics`` result.

    Read through ``get_query_metrics_result``, the tool an agent uses when the
    compressed preview is not enough, so the test consumes the same contract.
    """
    assert result.success == 1, result.error
    page = tool.get_query_metrics_result(result.result["result_id"])
    assert page.success == 1, page.error
    return list(csv.DictReader(io.StringIO(page.result["csv"])))


def _by_item(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["product"], row["item"]): row for row in rows}


def test_business_verification_compares_explicit_window_queries_without_changing_scope(query_tool):
    """Compare real native ranks, keeping post-window and input filtering explicit."""
    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

    db = DBFuncTool(agent_config=query_tool.agent_config, read_only=True)
    artifact = Path(query_tool.agent_config.project_root) / "subject/semantic_models/window_filters/model.yaml"
    verifier = SemanticVerificationTools(db, query_tool, GenerationEvidence(), lambda: artifact)
    reference = (
        "SELECT product,item,MIN(day) unused_day,"
        "RANK() OVER (PARTITION BY product ORDER BY SUM(score) DESC) r "
        "FROM main.activity_scores GROUP BY product,item"
    )
    selected = "item IN ('b','i')"
    post_filter = f"SELECT * FROM ({reference}) original WHERE {selected}"
    pre_filter = reference.replace(" GROUP BY", f" WHERE {selected} GROUP BY")
    base = {
        "name": "Item rank",
        "metric": "item_rank",
        "reference_column": "r",
        "dimension_mapping": {"product": "scores.product", "item": "scores.item"},
    }
    result = verifier.verify_semantic_outputs(
        [
            {
                **base,
                "reference_sql": post_filter,
                "query": {"dimensions": PRODUCT_ITEM, "where": "scores.item IN ('b','i')"},
            },
            {
                **base,
                "reference_sql": post_filter,
                "query": {"dimensions": PRODUCT_ITEM, "context_filter": "scores.item IN ('b','i')"},
            },
            {
                **base,
                "reference_sql": pre_filter,
                "query": {"dimensions": PRODUCT_ITEM, "context_filter": "scores.item IN ('b','i')"},
            },
            {**base, "reference_sql": "DELETE FROM main.activity_scores", "query": {"dimensions": PRODUCT_ITEM}},
        ]
    )
    assert result.success, result.error
    full_population, changed_population, scoped_population, refused_write = result.result["checks"]
    assert full_population["status"] == "passed" and full_population["reference_groups"] == 2
    assert changed_population["status"] == "failed" and changed_population["value_differences"] == 2
    assert scoped_population["status"] == "passed"
    assert refused_write["status"] == "unverified" and refused_write["reference_status"] == "failed"
    assert db.read_query("SELECT COUNT(*) n FROM main.activity_scores", compress=False).result == [{"n": 10}]


@pytest.mark.parametrize(
    "dimension,grain,mapping",
    [
        ("metric_time", None, "metric_time"),
        ("metric_time", "month", "metric_time"),
        ("scores.day", None, "scores.day"),
        ("scores.day", "month", "scores.day"),
        ("scores.day", "month", "day__month"),
    ],
)
def test_business_verification_normalizes_native_time_dimensions(query_tool, dimension, grain, mapping):
    """A month label and its native date key compare without inventing a query grain."""
    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

    db = DBFuncTool(agent_config=query_tool.agent_config, read_only=True)
    verifier = SemanticVerificationTools(db, query_tool, GenerationEvidence(), lambda: None)
    query = {"dimensions": [dimension], "context_filter": "scores.day = DATE '2025-06-01'"}
    if grain:
        query["time_granularity"] = grain
    result = verifier.verify_semantic_outputs(
        [
            {
                "name": "Score at period start",
                "metric": "score_total",
                "query": query,
                "reference_sql": (
                    "SELECT strftime(day, '%Y-%m') AS month, SUM(score) AS total "
                    "FROM main.activity_scores WHERE day = DATE '2025-06-01' GROUP BY day"
                ),
                "reference_column": "total",
                "dimension_mapping": {"month": mapping},
            }
        ]
    )
    comparison = result.result["checks"][0]
    assert comparison["status"] == "passed", comparison
    assert comparison["reference_groups"] == 1


def test_business_verification_binds_the_queried_model_and_invalidates_changes(query_tool):
    import yaml

    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

    selected = Path(query_tool.runtime.artifact_metric_binding("score_total")["model_path"])
    other = yaml.safe_load(selected.read_text())
    model = other["semantic_model"][0]
    model["name"] = "other_model"
    model["metrics"] = [{**model["metrics"][0], "name": "other_total"}]
    other_path = selected.with_name("other.yaml")
    other_path.write_text(yaml.safe_dump(other))

    evidence = GenerationEvidence()
    db = DBFuncTool(agent_config=query_tool.agent_config, read_only=True)
    verifier = SemanticVerificationTools(db, query_tool, evidence, lambda: selected)
    check = {
        "name": "Total score",
        "metric": "other_total",
        "reference_sql": "SELECT SUM(score) AS total FROM main.activity_scores",
        "reference_column": "total",
    }
    refused = verifier.verify_semantic_outputs([check]).result["checks"][0]
    assert refused["status"] == "unverified" and refused["query_status"] == "not_run"
    assert "not the selected model" in refused["reason"]
    assert not evidence.business_verification_summary(selected)

    # Standalone callers can compare without selecting a model; the runtime
    # still supplies the actual file and revision for their recorded result.
    verifier.artifact_provider = lambda: None
    compared = verifier.verify_semantic_outputs([check]).result["checks"][0]
    assert compared["status"] == "passed", compared
    assert evidence.business_verification_summary(other_path)[0]["status"] == "passed"
    assert not evidence.business_verification_summary(selected)
    model["metrics"][0]["expression"]["dialects"][0]["expression"] = "2 * SUM(scores.score)"
    other_path.write_text(yaml.safe_dump(other))
    assert evidence.business_verification_summary(other_path)[0]["status"] == "unverified"


def test_where_keeps_the_whole_population_ranks_and_context_filter_rescopes_them(query_tool):
    """The same predicate answers two different questions.

    ``where`` shows the selected items' ranks among every item of their
    product: the kept rows carry exactly the rank and partition count the
    unfiltered query gives them. ``context_filter`` ranks the selected items
    against each other only, so each is first of a partition of one.
    """
    unfiltered = _by_item(_rows(query_tool, query_tool.query_metrics(metrics=RANKING, dimensions=PRODUCT_ITEM)))
    selected = {("p1", "b"), ("p2", "i")}

    shown = _by_item(
        _rows(
            query_tool,
            query_tool.query_metrics(metrics=RANKING, dimensions=PRODUCT_ITEM, where="scores.item IN ('b', 'i')"),
        )
    )
    assert set(shown) == selected
    for key, row in shown.items():
        assert (row["item_rank"], row["items_in_product"]) == (
            unfiltered[key]["item_rank"],
            unfiltered[key]["items_in_product"],
        )
    assert all(row["items_in_product"] == "5" for row in shown.values())
    assert [shown[key]["item_rank"] for key in sorted(selected)] == ["3", "2"]

    rescoped = _by_item(
        _rows(
            query_tool,
            query_tool.query_metrics(
                metrics=RANKING, dimensions=PRODUCT_ITEM, context_filter="scores.item IN ('b', 'i')"
            ),
        )
    )
    assert set(rescoped) == selected
    assert all((row["item_rank"], row["items_in_product"]) == ("1", "1") for row in rescoped.values())


def test_where_on_a_selected_metric_filters_the_result_rows(query_tool):
    rows = _rows(
        query_tool,
        query_tool.query_metrics(metrics=["score_total"], dimensions=PRODUCT_ITEM, where="score_total > 20"),
    )
    assert {(row["product"], row["item"]) for row in rows} == {("p1", "d"), ("p1", "e")}
    assert all(float(row["score_total"]) > 20 for row in rows)


def test_where_on_an_unselected_metric_is_refused_with_a_retry(query_tool):
    result = query_tool.query_metrics(metrics=["score_total"], dimensions=PRODUCT_ITEM, where="item_rank > 1")
    assert result.success == 0
    payload = result.result
    assert payload["code"] == "result_filter_metric_not_selected"
    assert payload["suggested_retry"] == {"metrics": ["score_total", "item_rank"]}
    assert "item_rank" in result.error

    retried = query_tool.query_metrics(
        metrics=payload["suggested_retry"]["metrics"], dimensions=PRODUCT_ITEM, where="item_rank > 1"
    )
    rows = _rows(query_tool, retried)
    assert rows, "the suggested retry must be a runnable query"
    assert all(int(row["item_rank"]) > 1 for row in rows)


def test_metric_artifact_uses_real_engine_and_enforced_connector(query_tool, monkeypatch):
    """A saved grouped metric agrees with native execution, including filters."""
    import json

    ref = {"path": ["Scores"], "name": "score_total"}
    monkeypatch.setattr(
        query_tool,
        "get_metric",
        lambda **kwargs: FuncToolResult(
            success=1,
            result={**ref, "dimensions": [{"name": item} for item in PRODUCT_ITEM]},
        ),
    )
    tools = ReportArtifactTools(
        agent_config=query_tool.agent_config,
        db_func_tool=DBFuncTool(agent_config=query_tool.agent_config),
        semantic_tools=query_tool,
    )
    assert tools.start_new_report("metric_engine", "Scores", "Native metric integration").success
    saved = tools.save_metric_query(
        name="scores",
        query={
            "metric": ref,
            "dimensions": PRODUCT_ITEM,
            "filters": [{"dimension": "scores.item", "op": "in", "value": ["b", "i"]}],
        },
        goal="Scores by item",
        hypothesis="Only selected items appear",
    )
    assert saved.success, saved.error
    actual = json.loads((tools.queries_dir / "scores.json").read_text(encoding="utf-8"))
    direct = _rows(
        query_tool,
        query_tool.query_metrics(
            metrics=["score_total"],
            dimensions=PRODUCT_ITEM,
            where="scores.item IN ('b', 'i')",
        ),
    )
    assert {(row["product"], row["item"]): float(row["score_total"]) for row in actual["rows"]} == {
        (row["product"], row["item"]): float(row["score_total"]) for row in direct
    }
    assert actual["source"] == {"kind": "metric", "metric": ref}
    recipe = json.loads((tools.queries_dir / "scores.metric.json").read_text(encoding="utf-8"))
    assert recipe["generated_sql"] == actual["sql"]
    assert recipe["metric_tables"] == ["main.activity_scores"]


@pytest.mark.parametrize("product", ["p2", "no_matching_product"])
@pytest.mark.parametrize("root_model", [False, True])
@pytest.mark.parametrize("snapshot_format", ["short", "full", "named", "inline"])
def test_standalone_published_metric_uses_frozen_model_after_studio_model_changes(
    query_tool, monkeypatch, tmp_path, product, root_model, snapshot_format
):
    """An artifact-only viewer has no Studio files or subject index."""
    from datus.api.models.base_models import Result
    from datus.api.services.dashboard_service import DashboardService
    from datus.schemas.metric_artifact_query import MetricQueryFile
    from datus.tools.func_tool import DashboardArtifactTools
    from datus.utils.async_utils import run_async
    from datus.utils.metric_artifact_models import resolve_metric_query_snapshot

    ref = {"path": ["Scores"], "name": "score_total"}
    monkeypatch.setattr(
        query_tool,
        "get_metric",
        lambda **kwargs: FuncToolResult(
            success=1,
            result={**ref, "dimensions": [{"name": "scores.product"}]},
        ),
    )
    tools = DashboardArtifactTools(
        agent_config=query_tool.agent_config,
        db_func_tool=DBFuncTool(agent_config=query_tool.agent_config),
        semantic_tools=query_tool,
    )
    assert tools.start_new_dashboard("published_scores", "Scores", "Pinned model").success == 1
    saved = tools.save_metric_query_template(
        name="scores",
        query={
            "metric": ref,
            "dimensions": ["scores.product"],
            "filters": [
                {"dimension": "scores.product", "value": {"param": "product"}},
            ],
        },
        goal="Product scores",
        hypothesis="Only selected product appears",
        params=[{"name": "product", "type": "string", "required": True}],
        sample_params={"product": "p1"},
    )
    assert saved.success == 1, saved.error
    recipe = MetricQueryFile.model_validate_json((tools.queries_dir / "scores.metric.json").read_text(encoding="utf-8"))
    assert recipe.model_snapshot_path == f"models/{recipe.model_revision[:16]}.yaml"
    if snapshot_format in ("full", "named"):
        renamed = f"models/{recipe.model_revision}.yaml" if snapshot_format == "full" else "models/flights.yml"
        (tools.queries_dir.parent / recipe.model_snapshot_path).rename(tools.queries_dir.parent / renamed)
        recipe = MetricQueryFile.model_validate({**recipe.model_dump(), "model_snapshot_path": renamed})
    recipe = resolve_metric_query_snapshot(recipe, artifact_dir=tools.queries_dir.parent)
    if snapshot_format == "inline":
        recipe = MetricQueryFile.model_validate(
            {**recipe.model_dump(), "schema_version": "1.0", "model_snapshot_path": None}
        )
    if root_model:
        recipe = recipe.model_copy(update={"model_path": "model.yaml"})
    meta = (tools.queries_dir / "scores.params.json").read_text(encoding="utf-8")
    direct = _rows(
        query_tool,
        query_tool.query_metrics(
            metrics=["score_total"],
            dimensions=["scores.product"],
            where=f"scores.product = '{product}'",
        ),
    )
    Path(query_tool.runtime.artifact_metric_binding("score_total")["model_path"]).unlink()
    # The publication's artifact-only root is empty, and the live model is gone.
    viewer = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "viewer_home"),
        project_name="published_metric_viewer",
        project_root=str(tmp_path / "empty_viewer"),
        target="mock",
        models={"mock": {"type": "openai", "api_key": "unused", "model": "unused", "base_url": "http://127.0.0.1:1"}},
        services={
            "datasources": {
                "window_filters": {"type": "duckdb", "uri": str(tmp_path / "window_filters.duckdb"), "default": True}
            }
        },
    )

    viewer.current_datasource = "window_filters"

    async def loader(version):
        assert version == 3
        return Result(success=True, data=(recipe, meta))

    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        result = run_async(
            DashboardService(agent_config=viewer).run_query(
                project_files_root=Path(viewer.project_root),
                dashboard_slug="published_scores",
                query_slug="scores",
                params={"product": product},
                published_version=3,
                published_template_loader=loader,
            )
        )
    assert result.success is True, result.errorMessage
    assert result.data.rows == [{"product": row["product"], "score_total": float(row["score_total"])} for row in direct]
    assert result.data.source == {"kind": "metric", "metric": ref}
    assert [column.name for column in result.data.columns] == ["product", "score_total"]
    assert result.data.row_count == len(direct)
    assert recipe.model_snapshot == MODEL_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("dimensions", [["scores.product"], PRODUCT_ITEM])
def test_metric_artifact_preserves_native_projection_when_no_rows_match(query_tool, monkeypatch, dimensions):
    """An empty result retains the exact grouped projection from the real engine."""
    import json

    ref = {"path": ["Scores"], "name": "score_total"}
    monkeypatch.setattr(
        query_tool,
        "get_metric",
        lambda **kwargs: FuncToolResult(
            success=1, result={**ref, "dimensions": [{"name": item} for item in PRODUCT_ITEM]}
        ),
    )
    native = query_tool.query_metrics(
        metrics=[ref["name"]], dimensions=dimensions, where="scores.product = 'no_matching_product'"
    )
    assert native.success == 1, native.error
    native_page = query_tool.get_query_metrics_result(native.result["result_id"])
    assert native_page.success == 1, native_page.error
    native_rows = csv.DictReader(io.StringIO(native_page.result["csv"]))
    assert list(native_rows) == []
    tools = ReportArtifactTools(
        agent_config=query_tool.agent_config,
        db_func_tool=DBFuncTool(agent_config=query_tool.agent_config),
        semantic_tools=query_tool,
    )
    assert tools.start_new_report("empty_scores", "Scores", "Empty native result").success == 1
    saved = tools.save_metric_query(
        name="scores",
        query={
            "metric": ref,
            "dimensions": dimensions,
            "filters": [{"dimension": "scores.product", "value": "no_matching_product"}],
        },
        goal="Scores for an absent product",
        hypothesis="An unmatched filter returns an empty result",
    )
    assert saved.success == 1, saved.error
    actual = json.loads((tools.queries_dir / "scores.json").read_text(encoding="utf-8"))
    assert actual["rows"] == []
    assert actual["row_count"] == 0
    assert [column["name"] for column in actual["columns"]] == native_rows.fieldnames
    assert {column["name"]: column["type"] for column in actual["columns"]} == {
        **{item.removeprefix("scores."): "string" for item in dimensions},
        "score_total": "number",
    }
    assert tools._validate_metric_queries("report") is None


def test_published_dashboard_keeps_distinct_native_models(query_tool, monkeypatch):
    """Two model files keep their own calculations after Studio files disappear."""
    import yaml

    from datus.schemas.metric_artifact_query import MetricQueryFile
    from datus.tools.func_tool import DashboardArtifactTools
    from datus.tools.func_tool.metric_artifact_tools import execute_published_metric_query
    from datus.utils.artifact_files import iter_artifact_files
    from datus.utils.metric_artifact_models import resolve_metric_query_snapshot

    model_path = Path(query_tool.runtime.artifact_metric_binding("score_total")["model_path"])
    second = yaml.safe_load(model_path.read_text())
    model = second["semantic_model"][0]
    model["name"] = "double_scores"
    model["metrics"] = [model["metrics"][0]]
    model["metrics"][0]["name"] = "score_double"
    model["metrics"][0]["expression"]["dialects"][0]["expression"] = "2 * SUM(scores.score)"
    second_path = model_path.with_name("double.yaml")
    second_path.write_text(yaml.safe_dump(second), encoding="utf-8")
    monkeypatch.setattr(
        query_tool, "get_metric", lambda name, path: FuncToolResult(success=1, result={"name": name, "path": path})
    )
    db_tool = DBFuncTool(agent_config=query_tool.agent_config)
    tools = DashboardArtifactTools(
        agent_config=query_tool.agent_config, db_func_tool=db_tool, semantic_tools=query_tool
    )
    assert tools.start_new_dashboard("two_models", "Scores", "Distinct models").success == 1
    for metric in ("score_total", "score_double"):
        result = tools.save_metric_query_template(
            name=metric,
            query={"metric": {"path": ["Scores"], "name": metric}},
            goal=metric,
            hypothesis="Each model retains its calculation",
            params=[],
            sample_params={},
        )
        assert result.success == 1, result.error
    assert tools._validate_metric_queries("dashboard") is None
    artifact = tools.queries_dir.parent
    files = {
        path.relative_to(artifact).as_posix(): path.read_bytes().decode("utf-8")
        for path in iter_artifact_files(artifact, "dashboard")
    }
    recipes = [
        MetricQueryFile.model_validate_json(files[f"queries/{metric}.metric.json"])
        for metric in ("score_total", "score_double")
    ]
    assert len({recipe.model_path for recipe in recipes}) == 2
    assert len({recipe.model_snapshot_path for recipe in recipes}) == 2
    model_path.unlink()
    second_path.unlink()
    for recipe, expected in zip(recipes, (151.0, 302.0)):
        resolved = resolve_metric_query_snapshot(recipe, files=files)
        payload, _ = execute_published_metric_query(
            resolved, runtime_config=query_tool.runtime.config, db_tool=db_tool, params={}, policy_context={}
        )
        assert payload.rows == [{recipe.metric.name: expected}]
