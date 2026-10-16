"""Read-only end-to-end verification against independently executed SQLite SQL."""

import json
import sqlite3
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools
from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools, compare_output_rows
from datus.utils.semantic_sources import SemanticSourceAnalyzer

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
    reference_reads = []

    def execute(sql, **kwargs):
        assert kwargs == {"compress": False}
        reference_reads.append(sql)
        return FuncToolResult(result=[dict(row) for row in connection.execute(sql)])

    analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "original.sql", "sql": REFERENCE, "input_kind": "request"}],
        {"markets": ["name"], "tickets": ["id", "market"]},
    )
    discovery = SemanticDiscoveryTools()
    discovery.source_analysis = analysis
    discovery.source_revision = "revision_1"
    discovery._source_dialect = lambda: "sqlite"
    sql_by_metric = {
        "good": REFERENCE.replace("complaint_num", "good"),
        "bad": "SELECT market, COUNT(*) AS bad FROM tickets GROUP BY market",
    }
    cached = {}

    def query(metrics, **kwargs):
        metric = metrics[0]
        if kwargs.get("dry_run"):
            return FuncToolResult(result={"sql": sql_by_metric[metric]})
        sql = sql_by_metric[metric]
        if kwargs.get("where"):
            sql = f"SELECT * FROM ({sql}) complaints WHERE {kwargs['where']}"
        try:
            rows = [dict(row) for row in connection.execute(sql)]
        except sqlite3.Error as exc:
            return FuncToolResult(success=0, error=str(exc))
        cached[metric] = {"data": rows, "columns": list(rows[0]) if rows else ["market", metric]}
        # The preview deliberately omits groups. Verification must use full rows.
        return FuncToolResult(result={"result_id": metric, "data": rows[:1], "metadata": {}})

    semantic = SimpleNamespace(query_metrics=MagicMock(side_effect=query), get_cached_query_metrics_result=cached.get)
    artifact = tmp_path / "model.yml"
    artifact.write_text("semantic_model: []")
    evidence = GenerationEvidence()
    verifier = SemanticVerificationTools(
        SimpleNamespace(read_query=execute), semantic, discovery, evidence, lambda: artifact
    )
    yield SimpleNamespace(
        verifier=verifier,
        evidence=evidence,
        discovery=discovery,
        artifact=artifact,
        semantic=semantic,
        reads=reference_reads,
        source_id=analysis["statements"][0]["outputs"][-1],
        connection=connection,
        sql_by_metric=sql_by_metric,
    )
    connection.close()


def check(state, metric="good", **kwargs):
    return {
        "name": "Complaint count",
        "metric": metric,
        "query": {"dimensions": ["complaints.market"]},
        "source_output_id": state.source_id,
        "dimension_mapping": {"market": "complaints.market"},
        **kwargs,
    }


def test_individual_output_catches_missing_zero_groups_and_reuses_original_execution(setup_verifier):
    state = setup_verifier
    response = state.verifier.verify_semantic_outputs([check(state), check(state, "bad")])
    good, bad = response.result["checks"]
    assert good["status"] == "passed" and good["reference_groups"] == 3
    assert good["category"] == "verified" and good["next_action"] == "none"
    assert bad["status"] == "failed" and bad["missing_groups"] == 1
    assert bad["category"] == "result_mismatch" and bad["next_action"] == "inspect_difference"
    assert bad["examples"]["missing"] == [["C"]]
    assert len(state.reads) == 1
    assert state.semantic.query_metrics.call_count == 2
    assert sum(not c.kwargs.get("dry_run") for c in state.semantic.query_metrics.call_args_list) == 2
    assert all(call.kwargs["metrics"] in (["good"], ["bad"]) for call in state.semantic.query_metrics.call_args_list)


@pytest.mark.asyncio
async def test_baseline_output_is_required_before_executing_queries(setup_verifier):
    state = setup_verifier
    request = check(state)
    del request["source_output_id"]
    tool = state.verifier.available_tools()[0]
    result = await tool.on_invoke_tool(None, json.dumps({"checks": [request]}))
    assert result["success"] == 0 and "source_output_id" in result["error"]
    state.semantic.query_metrics.assert_not_called()
    assert not state.reads


def test_missing_original_dimension_is_unverified_not_a_reaggregated_pass(setup_verifier):
    state = setup_verifier
    result = state.verifier.verify_semantic_outputs([check(state, query={}, dimension_mapping={})]).result["checks"][0]
    assert result["query_status"] == "passed" and result["status"] == "unverified"
    assert result["category"] == "invalid_check" and result["next_action"] == "fix_check"
    assert result["missing_reference_dimensions"] == ["market"]
    assert "Full reference grain" in result["reason"]
    assert not state.reads
    state.semantic.query_metrics.assert_called_once()


def test_unresolved_reference_stops_model_repair_and_repeated_checks(setup_verifier):
    state = setup_verifier
    sql = "SELECT 'ALL' market,n FROM (SELECT market,COUNT(*) n FROM tickets GROUP BY market)t"
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "etl", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    first = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert first["category"] == "reference_unresolved" and first["next_action"] == "record_unverified"
    assert first["query_status"] == "passed"
    again = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert again["unchanged_check"] and not state.reads
    state.semantic.query_metrics.assert_called_once()
    state.discovery.source_revision = "revision_2"
    revised = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert not revised.get("unchanged_check")


def test_compile_failure_does_not_execute_metric_or_reference(setup_verifier):
    state = setup_verifier
    state.semantic.query_metrics.side_effect = lambda **kw: FuncToolResult(
        success=0, error="Dimension is not conformed", result={"code": "not_conformed"}
    )
    result = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert result["category"] == "model_query_error" and result["comparison_status"] == "not_run"
    assert result["next_action"] == "inspect_query"
    assert result["query_diagnostic"]["code"] == "not_conformed"
    assert "dry_run" not in state.semantic.query_metrics.call_args.kwargs and not state.reads


def test_mapped_result_filter_is_applied_after_reference_aggregation(setup_verifier):
    state = setup_verifier
    result = state.verifier.verify_semantic_outputs(
        [check(state, query={"dimensions": ["complaints.market"], "where": "complaints.market IN ('A','C')"})]
    ).result["checks"][0]
    assert result["status"] == "passed" and result["reference_groups"] == 2
    assert "AS reference_result" in state.reads[0]
    assert state.reads[0].index("GROUP BY") < state.reads[0].index("WHERE")


@pytest.mark.parametrize(
    "query,reference_where,category",
    [
        ({"dimensions": ["complaints.market"], "where": "complaints.market='A'"}, "market='B'", "invalid_check"),
        ({"dimensions": ["complaints.market"], "where": "tickets.id > 1"}, None, "model_query_error"),
        ({"dimensions": ["complaints.market"], "context_filter": "complaints.market='A'"}, None, "scope_unconfirmed"),
    ],
)
def test_incompatible_scopes_are_not_reported_as_metric_errors(setup_verifier, query, reference_where, category):
    state = setup_verifier
    result = state.verifier.verify_semantic_outputs(
        [check(state, query=query, reference_where=reference_where)]
    ).result["checks"][0]
    assert result["category"] == category and result["comparison_status"] == "not_run"
    assert not state.reads
    state.semantic.query_metrics.assert_called_once()


def test_time_range_selects_whole_reference_buckets_without_reaggregating(setup_verifier):
    state = setup_verifier
    sql = "SELECT '2024-01' month_time, 1 n UNION ALL SELECT '2024-02',2"
    # The output month is a fixed branch dimension of the UNION.
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "etl", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = "SELECT '2024-02-01' metric_time__month,2 good"
    query = {
        "dimensions": ["metric_time"],
        "time_start": "2024-02-01",
        "time_end": "2024-03-01",
        "time_granularity": "month",
    }
    result = state.verifier.verify_semantic_outputs(
        [check(state, query=query, dimension_mapping={"month_time": "metric_time"})]
    ).result["checks"][0]
    assert result["status"] == "passed" and result["reference_groups"] == 1
    assert "2024-01" in state.reads[0]  # The source input was not restricted.
    state.semantic.query_metrics.reset_mock()
    partial = state.verifier.verify_semantic_outputs(
        [check(state, query={**query, "time_start": "2024-02-15"}, dimension_mapping={"month_time": "metric_time"})]
    ).result["checks"][0]
    assert partial["category"] == "scope_unconfirmed"
    state.semantic.query_metrics.assert_called_once()


def test_model_and_source_changes_invalidate_business_evidence(setup_verifier):
    state = setup_verifier
    state.verifier.verify_semantic_outputs([check(state)])
    assert state.evidence.business_verification_summary(state.artifact, "revision_1")[0]["status"] == "passed"
    assert state.evidence.business_verification_summary(state.artifact, "revision_2")[0]["status"] == "unverified"
    state.artifact.write_text("semantic_model: [{name: revised}]")
    assert state.evidence.business_verification_summary(state.artifact, "revision_1")[0]["status"] == "unverified"
    state.evidence.record_artifact_mutation(state.artifact)
    assert len(state.evidence.business_output_checks) == 1
    assert state.evidence.business_verification_summary(state.artifact, "revision_1")[0]["status"] == "unverified"
    state.evidence.reset()
    assert not state.evidence.business_output_checks


def test_stale_output_id_cannot_silently_select_a_revised_formula(setup_verifier):
    state = setup_verifier
    revised = REFERENCE.replace("COUNT(t.id)", "COUNT(t.id) + 1")
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "original.sql", "sql": revised, "input_kind": "request"}]
    )
    result = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert result["status"] == "unverified" and "Unknown source_output_id" in result["reason"]
    assert not state.reads


def test_template_bindings_are_literals_and_do_not_execute_script_writes(setup_verifier):
    state = setup_verifier
    sql = "INSERT INTO unused(market,n) SELECT market, COUNT(*) n FROM tickets WHERE market='${market}' GROUP BY market"
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "etl", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = "SELECT market,COUNT(*) good FROM tickets WHERE market='A' GROUP BY market"
    result = state.verifier.verify_semantic_outputs([check(state, source_parameters={"market": "A"})]).result["checks"][
        0
    ]
    assert result["status"] == "passed"
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in state.reads)
    assert not state.connection.execute("SELECT name FROM sqlite_master WHERE name='unused'").fetchall()


def test_stored_values_use_confirmed_target_columns_without_reaggregation(setup_verifier):
    state = setup_verifier
    state.connection.executescript(
        "CREATE TABLE summary(area TEXT, total INTEGER); INSERT INTO summary VALUES ('A',2),('B',1),('C',0);"
    )
    sql = "INSERT INTO summary(area,total) " + REFERENCE
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "etl", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    result = state.verifier.verify_semantic_outputs([check(state, reference="stored")]).result["checks"][0]
    assert result["status"] == "passed", result
    assert '"total" AS "complaint_num"' in state.reads[0]
    assert "GROUP BY" not in state.reads[0] and "SUM(" not in state.reads[0]


def test_outer_scalar_aggregate_establishes_its_own_grain(setup_verifier):
    state = setup_verifier
    sql = "SELECT SUM(n) total FROM (SELECT market,COUNT(*) n FROM tickets GROUP BY market)c"
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "etl", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = sql.replace("SUM(n) total", "SUM(n) good")
    result = state.verifier.verify_semantic_outputs([check(state, query={}, dimension_mapping={})]).result["checks"][0]
    assert result["status"] == "passed", result
    assert result["comparison_grain"] == []


def test_union_branch_grain_includes_fixed_branch_dimension(setup_verifier):
    state = setup_verifier
    sql = (
        "SELECT 'KFC' brand, market, COUNT(*) n FROM tickets GROUP BY market "
        "UNION ALL SELECT 'PH' brand, market, COUNT(*) n FROM tickets GROUP BY market"
    )
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "union", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = sql.replace("COUNT(*) n", "COUNT(*) good")
    result = state.verifier.verify_semantic_outputs(
        [
            check(
                state,
                query={"dimensions": ["complaints.brand", "complaints.market"]},
                dimension_mapping={"brand": "complaints.brand", "market": "complaints.market"},
            )
        ]
    ).result["checks"][0]
    assert result["status"] == "passed", result
    assert result["reference_groups"] == 4


def test_change_during_query_is_not_recorded_against_the_new_artifact(setup_verifier):
    state = setup_verifier
    original = state.semantic.query_metrics.side_effect

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        state.artifact.write_text("semantic_model: [{name: concurrent_change}]")
        return result

    state.semantic.query_metrics.side_effect = changed
    state.verifier.verify_semantic_outputs([check(state)])
    assert state.evidence.business_verification_summary(state.artifact, "revision_1")[0]["status"] == "unverified"


def test_late_result_from_previous_request_does_not_pollute_next_request(setup_verifier):
    state = setup_verifier
    original = state.semantic.query_metrics.side_effect

    def reset_request(*args, **kwargs):
        response = original(*args, **kwargs)
        state.evidence.reset()
        return response

    state.semantic.query_metrics.side_effect = reset_request
    state.verifier.verify_semantic_outputs([check(state)])
    assert not state.evidence.business_output_checks and not state.evidence.business_verifications


@pytest.mark.parametrize(
    "expected,actual,reason",
    [
        ([], [], "Empty"),
        ([{"k": "A", "v": None}], [{"k": "A", "v": None}], "NULL"),
        ([{"k": "A", "v": 1}, {"k": "A", "v": 1}], [{"k": "A", "v": 1}], "duplicate"),
    ],
)
def test_insufficient_result_evidence_is_not_a_pass(expected, actual, reason):
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
    integer = compare_output_rows([{"v": 2}], [{"v": 2.0000001}], "v", "v", {})
    assert integer["status"] == "failed"


def test_time_key_normalization_requires_explicit_time_role_and_granularity():
    result = compare_output_rows(
        [{"month": "2026-08", "v": 1}],
        [{"metric_time__month": "2026-08-01", "m": 1}],
        "v",
        "m",
        {"month": "metric_time__month"},
        time_columns={"month", "metric_time__month"},
        time_granularity="month",
    )
    assert result["status"] == "passed"


@pytest.mark.parametrize("value", [float("inf"), "NaN"])
def test_nonfinite_results_are_not_a_numerical_pass(value):
    result = compare_output_rows([{"v": value}], [{"v": value}], "v", "v", {})
    assert result["status"] == "unverified"


def test_empty_generated_result_reports_all_missing_reference_groups():
    result = compare_output_rows([{"k": "A", "v": 1}], [], "v", "v", {"k": "k"})
    assert result["status"] == "failed" and result["missing_groups"] == 1


def test_value_difference_count_is_complete_while_examples_are_bounded():
    expected = [{"k": i, "v": 1} for i in range(10)]
    actual = [{"k": i, "v": 2} for i in range(10)]
    result = compare_output_rows(expected, actual, "v", "v", {"k": "k"})
    assert result["value_differences"] == 10 and len(result["examples"]["different"]) == 5


def test_diagnostic_sql_cannot_claim_original_equivalence(setup_verifier):
    state = setup_verifier
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "agent_sql", "sql": REFERENCE, "input_kind": "inline"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    result = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert result["query_status"] == "passed" and result["reason_code"] == "diagnostic_reference"
    assert not state.reads


def test_duplicate_projected_group_alias_and_automatic_unique_mapping(setup_verifier):
    state = setup_verifier
    sql = "SELECT market, market market_copy, COUNT(*) n FROM tickets GROUP BY market"
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "original", "sql": sql, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = "SELECT market, COUNT(*) good FROM tickets GROUP BY market"
    result = state.verifier.verify_semantic_outputs([check(state)]).result["checks"][0]
    assert result["status"] == "passed" and result["comparison_grain"] == ["market"]

    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "original", "sql": REFERENCE, "input_kind": "request"}]
    )
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = REFERENCE.replace("complaint_num", "good")
    result = state.verifier.verify_semantic_outputs([check(state, dimension_mapping={})]).result["checks"][0]
    assert result["status"] == "passed"


def test_narrative_edits_preserve_business_evidence_and_corrected_mapping_replaces_old_result(setup_verifier):
    state = setup_verifier
    state.artifact.write_text("semantic_model: [{name: model, description: before}]")
    state.verifier.verify_semantic_outputs([check(state, dimension_mapping={"wrong": "complaints.market"})])
    state.verifier.verify_semantic_outputs([check(state)])
    assert len(state.evidence.business_output_checks) == 1
    state.artifact.write_text("semantic_model: [{name: model, description: after, ai_context: {instructions: docs}}]")
    state.evidence.record_artifact_mutation(state.artifact)
    outputs = {o["id"] for o in state.discovery.source_analysis["outputs"]}
    summary = state.evidence.business_verification_summary(state.artifact, "unrelated_sources_added", outputs)
    assert len(summary) == 1 and summary[0]["status"] == "passed"
    state.artifact.write_text("semantic_model: [{name: changed}]")
    assert (
        state.evidence.business_verification_summary(state.artifact, "revision_1", outputs)[0]["category"]
        == "stale_evidence"
    )


def test_declared_timestamp_result_compares_equivalent_encodings(setup_verifier):
    state = setup_verifier
    state.connection.executescript("CREATE TABLE visits(ts TEXT); INSERT INTO visits VALUES ('2026-08-31T10:00:00');")
    state.discovery.source_analysis = SemanticSourceAnalyzer("sqlite").analyze(
        [{"name": "original", "sql": "SELECT MAX(ts) last_visit FROM visits", "input_kind": "request"}]
    )
    state.discovery._semantic_source_cache[()] = {
        "table_name": "visits",
        "schema": {"columns": [{"name": "ts", "type": "DATETIME"}]},
    }
    state.source_id = state.discovery.source_analysis["statements"][0]["outputs"][-1]
    state.sql_by_metric["good"] = "SELECT replace(MAX(ts),'T',' ') good FROM visits"
    result = state.verifier.verify_semantic_outputs([check(state, query={}, dimension_mapping={})]).result["checks"][0]
    assert result["status"] == "passed", result
    raw = compare_output_rows([{"v": "2026-08-31T10:00:00"}], [{"m": "2026-08-31 10:00:00"}], "v", "m", {})
    assert raw["status"] == "failed"  # Ordinary strings are not guessed to be timestamps.
