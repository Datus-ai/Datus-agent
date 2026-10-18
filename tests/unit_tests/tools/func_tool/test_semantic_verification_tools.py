"""Read-only result comparisons against independently executed SQLite SQL."""

import json
import sqlite3
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools, compare_output_rows

REFERENCE = """
    SELECT m.name AS market, COUNT(t.id) AS complaint_num
    FROM markets m LEFT JOIN tickets t ON t.market=m.name
    GROUP BY m.name
"""


@pytest.fixture
def setup_verifier(tmp_path):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE markets(name TEXT);
        INSERT INTO markets VALUES ('A'), ('B'), ('C');
        CREATE TABLE tickets(id INTEGER, market TEXT);
        INSERT INTO tickets VALUES (1,'A'),(2,'A'),(3,'B');
    """)
    reads = []

    def execute(sql, **kwargs):
        assert kwargs == {"compress": False}
        reads.append(sql)
        try:
            return FuncToolResult(result=[dict(row) for row in connection.execute(sql)])
        except sqlite3.Error as exc:
            return FuncToolResult(success=0, error=str(exc))

    sql_by_metric = {
        "good": REFERENCE.replace("complaint_num", "good"),
        "bad": "SELECT market, COUNT(*) AS bad FROM tickets GROUP BY market",
    }
    cache = {}

    def query(metrics, **kwargs):
        assert len(metrics) == 1
        assert "limit" not in kwargs and "dry_run" not in kwargs
        metric = metrics[0]
        rows = [dict(row) for row in connection.execute(sql_by_metric[metric])]
        cache[metric] = {"data": rows, "columns": list(rows[0]) if rows else ["market", metric]}
        # The presentation deliberately omits groups; comparison must use full rows.
        return FuncToolResult(
            result={
                "result_id": metric,
                "data": rows[:1],
                "metadata": {
                    "time_dimensions": {
                        name: kwargs.get("time_granularity")
                        for name in kwargs.get("dimensions", [])
                        if name == "metric_time"
                    }
                },
            }
        )

    artifact = tmp_path / "model.yml"
    artifact.write_text("semantic_model: []")

    def binding(metric):
        snapshot = GenerationEvidence._semantic_artifact_state(artifact)
        return {"model_path": snapshot["path"], "model_revision": snapshot["sha256"]}

    semantic = SimpleNamespace(
        query_metrics=MagicMock(side_effect=query),
        get_cached_query_metrics_result=cache.get,
        runtime=SimpleNamespace(artifact_metric_binding=binding),
    )
    evidence = GenerationEvidence()
    verifier = SemanticVerificationTools(SimpleNamespace(read_query=execute), semantic, evidence, lambda: artifact)
    yield SimpleNamespace(
        verifier=verifier,
        evidence=evidence,
        artifact=artifact,
        semantic=semantic,
        connection=connection,
        reads=reads,
        sql_by_metric=sql_by_metric,
    )
    connection.close()


def check(metric="good", **kwargs):
    return {
        "name": "Complaint count",
        "metric": metric,
        "reference_sql": REFERENCE,
        "reference_column": "complaint_num",
        "query": {"dimensions": ["complaints.market"]},
        "dimension_mapping": {"market": "complaints.market"},
        **kwargs,
    }


def test_outputs_are_queried_independently_and_missing_zero_groups_fail(setup_verifier):
    state = setup_verifier
    good, bad = state.verifier.verify_semantic_outputs([check(), check("bad")]).result["checks"]
    assert good["status"] == "passed" and good["reference_groups"] == 3
    assert bad["status"] == "failed" and bad["missing_groups"] == 1
    assert bad["examples"]["missing"] == [["C"]]
    assert len(state.reads) == 2 and state.semantic.query_metrics.call_count == 2
    assert all(call.kwargs["metrics"] in (["good"], ["bad"]) for call in state.semantic.query_metrics.call_args_list)


def test_explicit_results_compare_transformed_keys_and_nested_calculations(setup_verifier):
    state = setup_verifier
    sql = "SELECT COALESCE(market,'unknown') market,n complaint_num FROM (SELECT market,COUNT(*) n FROM tickets GROUP BY market)t"
    state.sql_by_metric["good"] = sql.replace("complaint_num", "good")
    result = state.verifier.verify_semantic_outputs([check(reference_sql=sql)]).result["checks"][0]
    assert result["status"] == "passed" and result["reference_groups"] == 2
    assert state.reads == [sql]


def test_redundant_reference_keys_do_not_change_metric_query_dimensions(setup_verifier):
    state = setup_verifier
    sql = REFERENCE.replace("m.name AS market", "m.name AS market,m.name || '_code' AS market_code")
    request = check(reference_sql=sql)
    result = state.verifier.verify_semantic_outputs([request]).result["checks"][0]
    assert result["status"] == "passed"
    assert state.semantic.query_metrics.call_args.kwargs["dimensions"] == ["complaints.market"]
    assert state.reads == [sql]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["reference_sql", "reference_column"])
async def test_tool_requires_an_explicit_reference_before_executing(setup_verifier, missing):
    state = setup_verifier
    request = check()
    del request[missing]
    tool = state.verifier.available_tools()[0]
    result = await tool.on_invoke_tool(None, json.dumps({"checks": [request]}))
    assert result["success"] == 0 and missing in result["error"]
    state.semantic.query_metrics.assert_not_called()
    assert not state.reads


@pytest.mark.parametrize("stage", ["metric", "reference", "full_result"])
def test_query_and_full_result_errors_are_reported_without_numerical_passes(setup_verifier, stage):
    state = setup_verifier
    if stage == "metric":
        state.semantic.query_metrics.side_effect = lambda **kw: FuncToolResult(success=0, error="Invalid dimension")
        request = check()
    elif stage == "reference":
        request = check(reference_sql="SELECT * FROM unavailable_table")
    else:
        state.semantic.get_cached_query_metrics_result = lambda _: None
        request = check()
    result = state.verifier.verify_semantic_outputs([request]).result["checks"][0]
    assert result["status"] == "unverified"
    assert {
        "metric": "Metric query failed",
        "reference": "Reference query failed",
        "full_result": "Complete metric result",
    }[stage] in result["reason"]


def test_result_key_mapping_and_parameterized_output_aliases(setup_verifier):
    state = setup_verifier
    original = state.semantic.query_metrics.side_effect
    cache = {}

    def query(**kwargs):
        response = original(**kwargs)
        cached = state.semantic.get_cached_query_metrics_result(response.result["result_id"])
        cache.update(
            columns=["Market", "good__brand_KFC"],
            data=[{"Market": row["market"], "good__brand_KFC": row["good"]} for row in cached["data"]],
        )
        return FuncToolResult(
            result={"result_id": "aliased", "metadata": {"outputs": [{"metric": "good", "column": "good__brand_KFC"}]}}
        )

    state.semantic.query_metrics.side_effect = query
    reader = state.semantic.get_cached_query_metrics_result
    state.semantic.get_cached_query_metrics_result = lambda key: cache if key == "aliased" else reader(key)
    result = state.verifier.verify_semantic_outputs(
        [
            check(
                query={"dimensions": ["complaints.Market"], "params": {"brand": "KFC"}},
                dimension_mapping={"market": "Market"},
            )
        ]
    ).result["checks"][0]
    assert result["status"] == "passed"
    assert state.semantic.query_metrics.call_args.kwargs["params"] == {"brand": "KFC"}


def test_latest_check_is_reported_and_model_changes_invalidate_its_result(setup_verifier):
    state = setup_verifier
    state.verifier.verify_semantic_outputs([check(dimension_mapping={"market": "missing"})])
    state.verifier.verify_semantic_outputs([check()])
    summary = state.evidence.business_verification_summary(state.artifact)
    assert len(summary) == 1 and summary[0]["status"] == "passed"
    state.artifact.write_text("semantic_model: [{name: changed}]")
    assert state.evidence.business_verification_summary(state.artifact)[0]["status"] == "unverified"


@pytest.mark.parametrize("change", ["model", "request"])
def test_changes_during_execution_cannot_publish_current_success(setup_verifier, change):
    state = setup_verifier
    original = state.semantic.query_metrics.side_effect

    def changed(**kwargs):
        response = original(**kwargs)
        if change == "model":
            state.artifact.write_text("semantic_model: [{name: changed}]")
        else:
            state.evidence.reset()
        return response

    state.semantic.query_metrics.side_effect = changed
    result = state.verifier.verify_semantic_outputs([check()]).result["checks"][0]
    summary = state.evidence.business_verification_summary(state.artifact)
    if change == "model":
        assert result["status"] == "unverified" and "Model changed" in result["reason"]
        assert summary[0]["status"] == "unverified"
    else:
        assert not summary


@pytest.mark.parametrize(
    "expected,actual,reason",
    [
        ([], [], "Empty"),
        ([{"k": "A", "v": None}], [{"k": "A", "v": None}], "NULL"),
        ([{"k": "A", "v": 1}, {"k": "A", "v": 1}], [{"k": "A", "v": 1}], "duplicate"),
        ([{"k": "A", "v": 1}], [{"k": "A", "wrong": 1}], "missing columns"),
    ],
)
def test_incomparable_results_are_not_passes(expected, actual, reason):
    result = compare_output_rows(expected, actual, "v", "v", {"k": "k"})
    assert result["status"] == "unverified" and reason in result["reason"]


def test_null_zero_extra_groups_and_decimal_tolerance_are_compared_explicitly():
    result = compare_output_rows(
        [{"k": "A", "v": None}, {"k": "B", "v": "0.333333"}],
        [{"k": "A", "v": 0}, {"k": "B", "v": Decimal("0.3333333")}, {"k": "C", "v": 1}],
        "v",
        "v",
        {"k": "k"},
    )
    assert result["status"] == "failed"
    assert result["extra_groups"] == 1 and result["value_differences"] == 1


@pytest.mark.parametrize("expected", [100, 100.0, Decimal("100"), "100"])
@pytest.mark.parametrize(
    "actual,absolute,relative,status",
    [
        ("100.01", "0.1", "0", "passed"),
        ("100.01", "0", "0.001", "passed"),
        ("100.01", "0", "0", "failed"),
        ("100.2", "0.1", "0", "failed"),
    ],
)
def test_numeric_tolerances_do_not_depend_on_connector_result_type(expected, actual, absolute, relative, status):
    result = compare_output_rows(
        [{"v": expected}],
        [{"v": actual}],
        "v",
        "v",
        {},
        absolute_tolerance=Decimal(absolute),
        relative_tolerance=Decimal(relative),
    )
    assert result["status"] == status


def test_explicit_time_key_normalization():
    result = compare_output_rows(
        [{"month": "2026-08", "v": 1}],
        [{"metric_time__month": "2026-08-01", "m": 1}],
        "v",
        "m",
        {"month": "metric_time__month"},
        time_dimensions={"month": "month"},
    )
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    "dimension,actual_key,status",
    [
        ("metric_time", "2026-08-01", "passed"),
        ("metric_time", "2026-08-02", "failed"),
        ("batch_code", "2026-08-01", "failed"),
    ],
)
def test_implicit_metric_time_normalizes_format_without_changing_grain(setup_verifier, dimension, actual_key, status):
    state = setup_verifier
    state.sql_by_metric["good"] = f"SELECT '{actual_key}' AS {dimension}, 1 AS good"
    result = state.verifier.verify_semantic_outputs(
        [
            check(
                reference_sql="SELECT '2026-08' AS month, 1 AS complaint_num",
                query={"dimensions": [dimension]},
                dimension_mapping={"month": dimension},
            )
        ]
    ).result["checks"][0]
    assert result["status"] == status
    assert "time_granularity" not in state.semantic.query_metrics.call_args.kwargs


@pytest.mark.parametrize("value", [float("inf"), "NaN"])
def test_nonfinite_results_are_not_a_numerical_pass(value):
    assert compare_output_rows([{"v": value}], [{"v": value}], "v", "v", {})["status"] == "unverified"


def test_value_difference_count_is_complete_while_examples_are_bounded():
    result = compare_output_rows(
        [{"k": i, "v": 1} for i in range(10)], [{"k": i, "v": 2} for i in range(10)], "v", "v", {"k": "k"}
    )
    assert result["value_differences"] == 10 and len(result["examples"]["different"]) == 5
