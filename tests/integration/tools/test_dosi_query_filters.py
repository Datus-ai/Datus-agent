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
def test_standalone_published_metric_uses_frozen_model_after_studio_model_changes(
    query_tool, monkeypatch, tmp_path, product, root_model
):
    """An artifact-only viewer has no Studio files or subject index."""
    from datus.api.models.base_models import Result
    from datus.api.services.dashboard_service import DashboardService
    from datus.schemas.metric_artifact_query import MetricQueryFile
    from datus.tools.func_tool import DashboardArtifactTools
    from datus.utils.async_utils import run_async

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
    assert tools._validate_metric_queries("report") is None
