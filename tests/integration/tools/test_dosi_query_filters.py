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
