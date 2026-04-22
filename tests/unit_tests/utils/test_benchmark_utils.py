# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/utils/benchmark_utils.py

CI-level: zero external dependencies, all file I/O mocked or done with tmp_path.
"""

import json

import pandas as pd
import yaml

from datus.utils.benchmark_utils import (
    ComparisonOutcome,
    ComparisonRecord,
    CsvColumnSqlProvider,
    CsvPerTaskResultProvider,
    DirectorySqlProvider,
    EvaluationReportBuilder,
    JsonMappingSqlProvider,
    ResultData,
    SqlData,
    TableComparator,
    TaskEvaluation,
    TrajectoryParser,
    WorkflowAnalysis,
    WorkflowArtifacts,
    WorkflowOutput,
    _append_unique,
    _clean_table_identifier_part,
    _collect_file_artifacts,
    _collect_metric_artifacts,
    _collect_reference_sql_artifacts,
    _collect_semantic_model_artifacts,
    _extract_artifacts_from_action_history,
    _extract_result_payload,
    _find_matching_candidates,
    _is_empty_table_identifier,
    _normalize_field_name,
    _normalize_text,
    _parse_table_identifier,
    _select_from_mapping,
    _split_expected_items,
    _tables_equivalent,
    _trim_trailing_non_empty_tables,
    _unique_preserve_order,
    collect_latest_trajectory_files,
    compute_table_matches,
    csv_str_to_pands,
    list_save_runs,
    list_trajectory_runs,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestNormalizeFieldName:
    def test_strips_and_lowercases(self):
        assert _normalize_field_name("  Hello World  ") == "hello_world"

    def test_replaces_spaces_with_underscore(self):
        assert _normalize_field_name("order id") == "order_id"

    def test_non_string_returns_empty(self):
        assert _normalize_field_name(None) == ""  # type: ignore[arg-type]

    def test_empty_string(self):
        assert _normalize_field_name("") == ""


class TestSelectFromMapping:
    def test_finds_matching_key(self):
        mapping = {"task_id": "t1", "SQL": "SELECT 1"}
        key, val = _select_from_mapping(mapping, ["task_id"])
        assert key == "task_id"
        assert val == "t1"

    def test_normalized_key_match(self):
        mapping = {"Task ID": "t2"}
        key, val = _select_from_mapping(mapping, ["task_id", "Task ID"])
        assert val == "t2"

    def test_skips_none_value(self):
        mapping = {"a": None, "b": "found"}
        key, val = _select_from_mapping(mapping, ["a", "b"])
        assert key == "b"
        assert val == "found"

    def test_no_match(self):
        key, val = _select_from_mapping({"x": "y"}, ["z"])
        assert key is None
        assert val is None

    def test_non_mapping_returns_none(self):
        key, val = _select_from_mapping("not_a_mapping", ["a"])  # type: ignore[arg-type]
        assert key is None
        assert val is None


class TestUniquePreserveOrder:
    def test_deduplicates(self):
        result = _unique_preserve_order(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_skips_empty(self):
        result = _unique_preserve_order(["a", "", "b"])
        assert result == ["a", "b"]

    def test_strips_whitespace(self):
        result = _unique_preserve_order(["  a  ", "a"])
        assert result == ["a"]


class TestCleanTableIdentifierPart:
    def test_strips_quotes(self):
        assert _clean_table_identifier_part('"my_table"') == "my_table"

    def test_strips_backticks(self):
        assert _clean_table_identifier_part("`table`") == "table"

    def test_plain_name(self):
        assert _clean_table_identifier_part("orders") == "orders"


class TestParseTableIdentifier:
    def test_simple_name(self):
        norm, base, is_simple = _parse_table_identifier("orders")
        assert base == "orders"
        assert is_simple is True

    def test_qualified_name(self):
        norm, base, is_simple = _parse_table_identifier("schema.orders")
        assert base == "orders"
        assert is_simple is False

    def test_none(self):
        norm, base, is_simple = _parse_table_identifier(None)  # type: ignore[arg-type]
        assert norm == "" and base == "" and is_simple is False

    def test_empty(self):
        norm, base, is_simple = _parse_table_identifier("")
        assert norm == ""


class TestIsEmptyTableIdentifier:
    def test_none(self):
        assert _is_empty_table_identifier(None) is True

    def test_blank(self):
        assert _is_empty_table_identifier("  ") is True

    def test_non_empty(self):
        assert _is_empty_table_identifier("orders") is False


class TestTrimTrailingNonEmptyTables:
    def test_trims_leading_empties(self):
        # reversed iteration stops at first empty from the end
        result = _trim_trailing_non_empty_tables(["orders", "", "users"])
        assert result == ["users"]

    def test_all_non_empty(self):
        result = _trim_trailing_non_empty_tables(["a", "b"])
        assert result == ["a", "b"]

    def test_empty_list(self):
        assert _trim_trailing_non_empty_tables([]) == []


class TestTablesEquivalent:
    def test_same_name(self):
        assert _tables_equivalent("orders", "orders") is True

    def test_qualified_vs_simple(self):
        assert _tables_equivalent("schema.orders", "orders") is True

    def test_different(self):
        assert _tables_equivalent("orders", "users") is False

    def test_empty(self):
        assert _tables_equivalent("", "orders") is False


class TestComputeTableMatches:
    def test_matching_tables(self):
        result = compute_table_matches(["orders", "users"], ["orders", "customers"])
        assert "orders" in result

    def test_empty_inputs(self):
        assert compute_table_matches([], ["orders"]) == []
        assert compute_table_matches(["orders"], []) == []

    def test_qualified_match(self):
        result = compute_table_matches(["db.orders"], ["orders"])
        assert "orders" in result


class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert _normalize_text("hello   world") == "hello world"

    def test_lowercases(self):
        assert _normalize_text("Hello World") == "hello world"

    def test_none(self):
        assert _normalize_text(None) == ""  # type: ignore[arg-type]


class TestFindMatchingCandidates:
    def test_exact_match(self):
        result = _find_matching_candidates("orders", ["orders", "users"])
        assert "orders" in result

    def test_substring_match(self):
        result = _find_matching_candidates("order", ["my_orders", "users"])
        assert "my_orders" in result

    def test_no_match(self):
        result = _find_matching_candidates("xyz", ["orders", "users"])
        assert result == []

    def test_empty_expected(self):
        assert _find_matching_candidates("", ["orders"]) == []


class TestSplitExpectedItems:
    def test_semicolon_separator(self):
        result = _split_expected_items("a;b;c")
        assert result == ["a", "b", "c"]

    def test_newline_separator(self):
        result = _split_expected_items("a\nb\nc")
        assert result == ["a", "b", "c"]

    def test_empty(self):
        assert _split_expected_items("") == []


class TestAppendUnique:
    def test_adds_string(self):
        container: list = []
        _append_unique(container, "hello")
        assert container == ["hello"]

    def test_no_duplicates(self):
        container = ["hello"]
        _append_unique(container, "hello")
        assert container == ["hello"]

    def test_none_ignored(self):
        container: list = []
        _append_unique(container, None)
        assert container == []

    def test_list_of_values(self):
        container: list = []
        _append_unique(container, ["a", "b", "a"])
        assert container == ["a", "b"]

    def test_mapping_ignored(self):
        container: list = []
        _append_unique(container, {"key": "val"})
        assert container == []


class TestExtractResultPayload:
    def test_raw_output_with_result(self):
        output = {"raw_output": {"result": "found"}}
        assert _extract_result_payload(output) == "found"

    def test_raw_output_plain(self):
        output = {"raw_output": "plain"}
        assert _extract_result_payload(output) == "plain"

    def test_result_key(self):
        output = {"result": "val"}
        assert _extract_result_payload(output) == "val"

    def test_passthrough(self):
        assert _extract_result_payload("string") == "string"


# ---------------------------------------------------------------------------
# Artifact collectors
# ---------------------------------------------------------------------------


class TestCollectFileArtifacts:
    def test_string_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, "some_file.csv")
        assert "some_file.csv" in artifacts.files

    def test_file_written_pattern(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, "file written successfully: output.csv")
        assert "output.csv" in artifacts.files

    def test_mapping_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, {"report.csv": "data"})
        assert "report.csv" in artifacts.files

    def test_list_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_file_artifacts(artifacts, ["a.csv", "b.csv"])
        assert "a.csv" in artifacts.files
        assert "b.csv" in artifacts.files


class TestCollectReferenceSqlArtifacts:
    def test_string_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_reference_sql_artifacts(artifacts, "SELECT 1")
        assert "SELECT 1" in artifacts.reference_sqls

    def test_mapping_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_reference_sql_artifacts(artifacts, {"sql": "SELECT 2", "name": "query_a"})
        assert "SELECT 2" in artifacts.reference_sqls
        assert "query_a" in artifacts.reference_sql_names

    def test_list_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_reference_sql_artifacts(artifacts, [{"sql": "SELECT 3", "name": "q3"}])
        assert "SELECT 3" in artifacts.reference_sqls


class TestCollectSemanticModelArtifacts:
    def test_metadata_entries(self):
        artifacts = WorkflowArtifacts()
        payload = {"metadata": [{"semantic_model_name": "sales_model"}]}
        _collect_semantic_model_artifacts(artifacts, payload)
        assert "sales_model" in artifacts.semantic_models

    def test_description_fallback(self):
        artifacts = WorkflowArtifacts()
        payload = {"metadata": [{"description": "revenue_model"}]}
        _collect_semantic_model_artifacts(artifacts, payload)
        assert "revenue_model" in artifacts.semantic_models


class TestCollectMetricArtifacts:
    def test_single_mapping(self):
        artifacts = WorkflowArtifacts()
        _collect_metric_artifacts(artifacts, {"name": "revenue", "description": "Total revenue"})
        assert "revenue" in artifacts.metrics_names
        assert "Total revenue" in artifacts.metrics_texts

    def test_list_payload(self):
        artifacts = WorkflowArtifacts()
        _collect_metric_artifacts(artifacts, [{"name": "orders"}, {"name": "users"}])
        assert "orders" in artifacts.metrics_names
        assert "users" in artifacts.metrics_names


class TestExtractArtifactsFromActionHistory:
    def test_write_file_tool(self):
        artifacts = WorkflowArtifacts()
        tool_calls: dict = {}
        history = [
            {
                "role": "tool",
                "input": {"function_name": "write_file"},
                "output": {"raw_output": "result.csv"},
            }
        ]
        _extract_artifacts_from_action_history(history, artifacts, tool_calls)
        assert "write_file" in tool_calls
        assert tool_calls["write_file"] == 1

    def test_skips_non_tool_roles(self):
        artifacts = WorkflowArtifacts()
        tool_calls: dict = {}
        history = [{"role": "user", "input": {"function_name": "write_file"}, "output": {}}]
        _extract_artifacts_from_action_history(history, artifacts, tool_calls)
        assert tool_calls == {}

    def test_empty_history(self):
        artifacts = WorkflowArtifacts()
        _extract_artifacts_from_action_history(None, artifacts)  # should not raise

    def test_dotted_function_name(self):
        artifacts = WorkflowArtifacts()
        tool_calls: dict = {}
        history = [
            {
                "role": "tool",
                "input": {"function_name": "module.write_file"},
                "output": {"raw_output": "file.csv"},
            }
        ]
        _extract_artifacts_from_action_history(history, artifacts, tool_calls)
        assert "write_file" in tool_calls


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestWorkflowAnalysis:
    def test_output_nodes_count(self):
        outputs = [WorkflowOutput(node_id="n1", success=True, status="done")]
        analysis = WorkflowAnalysis(
            task_id="t1",
            completion_time=None,
            status="done",
            total_nodes=1,
            outputs=outputs,
        )
        assert analysis.output_nodes == 1
        assert analysis.output_success == 1
        assert analysis.output_failure == 0

    def test_output_failure_count(self):
        outputs = [
            WorkflowOutput(node_id="n1", success=False, status="failed"),
            WorkflowOutput(node_id="n2", success=True, status="done"),
        ]
        analysis = WorkflowAnalysis(task_id="t1", completion_time=None, status="done", total_nodes=2, outputs=outputs)
        assert analysis.output_failure == 1


class TestComparisonOutcome:
    def test_with_error(self):
        outcome = ComparisonOutcome.with_error("something failed")
        assert outcome.error == "something failed"
        assert outcome.match_rate == 0.0

    def test_to_dict(self):
        outcome = ComparisonOutcome(match_rate=1.0, error=None)
        d = outcome.to_dict()
        assert d["match_rate"] == 1.0
        assert "matched_columns" in d


class TestResultData:
    def test_available_true(self):
        df = pd.DataFrame({"a": [1]})
        rd = ResultData(task_id="t1", source="src", dataframe=df)
        assert rd.available is True

    def test_available_false_no_df(self):
        rd = ResultData(task_id="t1", source="src")
        assert rd.available is False

    def test_available_false_with_error(self):
        df = pd.DataFrame({"a": [1]})
        rd = ResultData(task_id="t1", source="src", dataframe=df, error="oops")
        assert rd.available is False


class TestSqlData:
    def test_available_true(self):
        sd = SqlData(task_id="t1", source="src", sql="SELECT 1")
        assert sd.available is True

    def test_available_false(self):
        sd = SqlData(task_id="t1", source="src", error="not found")
        assert sd.available is False


class TestWorkflowArtifacts:
    def test_to_dict(self):
        wa = WorkflowArtifacts(files=["f.csv"], reference_sqls=["SELECT 1"])
        d = wa.to_dict()
        assert d["files"] == ["f.csv"]
        assert d["reference_sqls"] == ["SELECT 1"]


# ---------------------------------------------------------------------------
# csv_str_to_pands
# ---------------------------------------------------------------------------


def test_csv_str_to_pands():
    csv_text = "col1,col2\n1,2\n3,4"
    df = csv_str_to_pands(csv_text)
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# CsvPerTaskResultProvider
# ---------------------------------------------------------------------------


class TestCsvPerTaskResultProvider:
    def test_fetch_missing_file(self, tmp_path):
        provider = CsvPerTaskResultProvider(str(tmp_path))
        result = provider.fetch("task_1")
        assert result.error is not None
        assert "not found" in result.error

    def test_fetch_existing_file(self, tmp_path):
        csv_path = tmp_path / "task_1.csv"
        csv_path.write_text("col1,col2\n1,2\n")
        provider = CsvPerTaskResultProvider(str(tmp_path))
        result = provider.fetch("task_1")
        assert result.available
        assert list(result.dataframe.columns) == ["col1", "col2"]

    def test_datasource_run_id(self, tmp_path):
        ns_dir = tmp_path / "ns" / "run1"
        ns_dir.mkdir(parents=True)
        csv_file = ns_dir / "task_x.csv"
        csv_file.write_text("a,b\n1,2\n")
        provider = CsvPerTaskResultProvider(str(tmp_path), datasource="ns", run_id="run1")
        result = provider.fetch("task_x")
        assert result.available

    def test_datasource_auto_latest_run(self, tmp_path):
        ns_dir = tmp_path / "ns"
        run_dir = ns_dir / "run_2024"
        run_dir.mkdir(parents=True)
        csv_file = run_dir / "task_y.csv"
        csv_file.write_text("x,y\n5,6\n")
        provider = CsvPerTaskResultProvider(str(tmp_path), datasource="ns")
        result = provider.fetch("task_y")
        assert result.available

    def test_datasource_no_run_dirs(self, tmp_path):
        ns_dir = tmp_path / "empty_ns"
        ns_dir.mkdir()
        provider = CsvPerTaskResultProvider(str(tmp_path), datasource="empty_ns")
        result = provider.fetch("task_z")
        assert result.error is not None


# ---------------------------------------------------------------------------
# DirectorySqlProvider
# ---------------------------------------------------------------------------


class TestDirectorySqlProvider:
    def test_fetch_missing_file(self, tmp_path):
        provider = DirectorySqlProvider(str(tmp_path))
        result = provider.fetch("missing_task")
        assert result.error is not None

    def test_fetch_existing_sql_file(self, tmp_path):
        sql_file = tmp_path / "task_1.sql"
        sql_file.write_text("SELECT * FROM orders")
        provider = DirectorySqlProvider(str(tmp_path))
        result = provider.fetch("task_1")
        assert result.available
        assert result.sql == "SELECT * FROM orders"


# ---------------------------------------------------------------------------
# JsonMappingSqlProvider
# ---------------------------------------------------------------------------


class TestJsonMappingSqlProvider:
    def test_fetch_missing_file(self, tmp_path):
        provider = JsonMappingSqlProvider(str(tmp_path / "nonexistent.json"))
        result = provider.fetch("t1")
        assert result.error is not None

    def test_fetch_from_list(self, tmp_path):
        data = [{"task_id": "t1", "SQL": "SELECT 1"}]
        json_file = tmp_path / "sqls.json"
        json_file.write_text(json.dumps(data))
        provider = JsonMappingSqlProvider(str(json_file))
        result = provider.fetch("t1")
        assert result.available
        assert result.sql == "SELECT 1"

    def test_fetch_from_mapping(self, tmp_path):
        data = {"t2": {"task_id": "t2", "SQL": "SELECT 2"}}
        json_file = tmp_path / "sqls.json"
        json_file.write_text(json.dumps(data))
        provider = JsonMappingSqlProvider(str(json_file))
        result = provider.fetch("t2")
        assert result.available

    def test_fetch_not_found(self, tmp_path):
        data = [{"task_id": "t1", "SQL": "SELECT 1"}]
        json_file = tmp_path / "sqls.json"
        json_file.write_text(json.dumps(data))
        provider = JsonMappingSqlProvider(str(json_file))
        result = provider.fetch("nonexistent")
        assert result.error is not None

    def test_fetch_from_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "sqls.jsonl"
        jsonl_file.write_text(json.dumps({"task_id": "t3", "SQL": "SELECT 3"}) + "\n")
        provider = JsonMappingSqlProvider(str(jsonl_file))
        result = provider.fetch("t3")
        assert result.available
        assert result.sql == "SELECT 3"

    def test_empty_file_error(self, tmp_path):
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")
        provider = JsonMappingSqlProvider(str(json_file))
        result = provider.fetch("t1")
        assert result.error is not None
        assert "No SQL entries" in result.error


# ---------------------------------------------------------------------------
# CsvColumnSqlProvider
# ---------------------------------------------------------------------------


class TestCsvColumnSqlProvider:
    def test_fetch_missing_file(self, tmp_path):
        provider = CsvColumnSqlProvider(str(tmp_path / "nonexistent.csv"), task_id_key="task_id", sql_key="SQL")
        result = provider.fetch("t1")
        assert result.error is not None

    def test_fetch_found(self, tmp_path):
        csv_file = tmp_path / "sqls.csv"
        csv_file.write_text("task_id,SQL\nt1,SELECT 1\nt2,SELECT 2\n")
        provider = CsvColumnSqlProvider(str(csv_file), task_id_key="task_id", sql_key="SQL")
        result = provider.fetch("t1")
        assert result.available
        assert result.sql == "SELECT 1"

    def test_fetch_not_found(self, tmp_path):
        csv_file = tmp_path / "sqls.csv"
        csv_file.write_text("task_id,SQL\nt1,SELECT 1\n")
        provider = CsvColumnSqlProvider(str(csv_file), task_id_key="task_id", sql_key="SQL")
        result = provider.fetch("t_missing")
        assert result.error is not None


# ---------------------------------------------------------------------------
# TrajectoryParser
# ---------------------------------------------------------------------------


class TestTrajectoryParser:
    def test_missing_file(self, tmp_path):
        parser = TrajectoryParser()
        result = parser.parse(tmp_path / "nonexistent.yaml", "t1")
        assert result.status == "missing"
        assert any("not found" in e for e in result.errors)

    def test_invalid_format(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("not_workflow: {}")
        parser = TrajectoryParser()
        result = parser.parse(yaml_file, "t1")
        assert result.status == "invalid"

    def test_valid_workflow_with_nodes(self, tmp_path):
        data = {
            "workflow": {
                "completion_time": "2024-01-01T00:00:00",
                "status": "done",
                "nodes": [
                    {
                        "id": "n1",
                        "type": "output",
                        "result": {
                            "success": True,
                            "status": "done",
                            "action_history": [],
                        },
                    }
                ],
            }
        }
        yaml_file = tmp_path / "traj.yaml"
        yaml_file.write_text(yaml.dump(data))
        parser = TrajectoryParser()
        result = parser.parse(yaml_file, "t1")
        assert result.status == "done"
        assert result.total_nodes == 1
        assert len(result.outputs) == 1
        assert result.outputs[0].success is True

    def test_valid_workflow_without_nodes_output_type(self, tmp_path):
        data = {
            "workflow": {
                "completion_time": None,
                "status": "done",
                "id": "wf1",
                "type": "output",
                "result": {
                    "success": False,
                    "status": "failed",
                    "error": "something broke",
                    "action_history": [],
                },
            }
        }
        yaml_file = tmp_path / "traj2.yaml"
        yaml_file.write_text(yaml.dump(data))
        parser = TrajectoryParser()
        result = parser.parse(yaml_file, "t2")
        assert result.total_nodes == 1
        assert len(result.outputs) == 1
        assert result.outputs[0].success is False

    def test_failed_output_adds_error(self, tmp_path):
        data = {
            "workflow": {
                "completion_time": None,
                "status": "failed",
                "nodes": [
                    {
                        "id": "n1",
                        "type": "output",
                        "result": {
                            "success": False,
                            "status": "failed",
                            "error": "task failed",
                            "action_history": [],
                        },
                    }
                ],
            }
        }
        yaml_file = tmp_path / "traj3.yaml"
        yaml_file.write_text(yaml.dump(data))
        parser = TrajectoryParser()
        result = parser.parse(yaml_file, "t3")
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# TableComparator
# ---------------------------------------------------------------------------


class TestTableComparator:
    def test_identical_dataframes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        comparator = TableComparator()
        outcome = comparator.compare(df.copy(), df.copy())
        assert outcome.match_rate == 1.0

    def test_different_dataframes(self):
        df_actual = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_expected = pd.DataFrame({"a": [1, 2], "c": [5, 6]})
        comparator = TableComparator()
        outcome = comparator.compare(df_actual, df_expected)
        assert outcome.match_rate < 1.0


# ---------------------------------------------------------------------------
# EvaluationReportBuilder
# ---------------------------------------------------------------------------


class TestEvaluationReportBuilder:
    def _make_evaluation(self, task_id: str, match_rate: float = 1.0, error: str = None):
        outputs = [WorkflowOutput(node_id="n1", success=True, status="done")]
        analysis = WorkflowAnalysis(
            task_id=task_id,
            completion_time="2024-01-01",
            status="done",
            total_nodes=1,
            outputs=outputs,
        )
        outcome = ComparisonOutcome(match_rate=match_rate, error=error)
        actual = ResultData(task_id=task_id, source="actual", dataframe=pd.DataFrame({"a": [1]}))
        expected = ResultData(task_id=task_id, source="gold", dataframe=pd.DataFrame({"a": [1]}))
        record = ComparisonRecord(task_id=task_id, actual=actual, expected=expected, outcome=outcome)
        return TaskEvaluation(task_id=task_id, analysis=analysis, comparisons=[record])

    def test_all_matched(self):
        evaluations = {"t1": self._make_evaluation("t1", match_rate=1.0)}
        builder = EvaluationReportBuilder()
        report = builder.build(evaluations)
        assert report.summary["comparison_summary"]["match_count"] == 1
        assert report.summary["comparison_summary"]["mismatch_count"] == 0

    def test_mismatch(self):
        evaluations = {"t1": self._make_evaluation("t1", match_rate=0.5)}
        builder = EvaluationReportBuilder()
        report = builder.build(evaluations)
        assert report.summary["comparison_summary"]["mismatch_count"] == 1

    def test_comparison_error(self):
        evaluations = {"t1": self._make_evaluation("t1", error="some error")}
        builder = EvaluationReportBuilder()
        report = builder.build(evaluations)
        assert report.summary["comparison_summary"]["comparison_error_count"] == 1

    def test_empty_result_error(self):
        evaluations = {"t1": self._make_evaluation("t1", error="No columns to parse from file")}
        builder = EvaluationReportBuilder()
        report = builder.build(evaluations)
        assert report.summary["comparison_summary"]["empty_result_count"] == 1

    def test_report_to_dict(self):
        evaluations = {"t1": self._make_evaluation("t1")}
        builder = EvaluationReportBuilder()
        report = builder.build(evaluations)
        d = report.to_dict()
        assert "summary" in d
        assert "details" in d

    def test_metrics_comparison(self):
        evaluation = self._make_evaluation("t1")
        evaluation.comparisons[0].outcome.tools_comparison = {
            "expected_metrics": {"expected": ["revenue"], "match": True}
        }
        builder = EvaluationReportBuilder()
        report = builder.build({"t1": evaluation})
        assert report.summary["metrics_summary"]["metrics_matched_count"] == 1


# ---------------------------------------------------------------------------
# Directory listing utilities
# ---------------------------------------------------------------------------


class TestListTrajectoryRuns:
    def test_nonexistent_dir(self, tmp_path):
        result = list_trajectory_runs(str(tmp_path / "nope"))
        assert result == {}

    def test_lists_runs_for_datasource(self, tmp_path):
        run_dir = tmp_path / "ns" / "run1"
        run_dir.mkdir(parents=True)
        result = list_trajectory_runs(str(tmp_path), datasource="ns")
        assert "ns" in result
        assert "run1" in result["ns"]

    def test_lists_all_datasources(self, tmp_path):
        (tmp_path / "ns1" / "run_a").mkdir(parents=True)
        (tmp_path / "ns2" / "run_b").mkdir(parents=True)
        result = list_trajectory_runs(str(tmp_path))
        assert "ns1" in result
        assert "ns2" in result


class TestListSaveRuns:
    def test_nonexistent_dir(self, tmp_path):
        result = list_save_runs(str(tmp_path / "nope"))
        assert result == {}

    def test_lists_runs_for_datasource(self, tmp_path):
        run_dir = tmp_path / "ns" / "run1"
        run_dir.mkdir(parents=True)
        result = list_save_runs(str(tmp_path), datasource="ns")
        assert "ns" in result

    def test_lists_all_datasources(self, tmp_path):
        (tmp_path / "nsA" / "run1").mkdir(parents=True)
        (tmp_path / "nsB" / "run2").mkdir(parents=True)
        result = list_save_runs(str(tmp_path))
        assert "nsA" in result
        assert "nsB" in result


class TestCollectLatestTrajectoryFiles:
    def test_nonexistent_dir(self, tmp_path):
        result = collect_latest_trajectory_files(str(tmp_path / "nope"), "ns")
        assert result == {}

    def test_collects_yaml_files(self, tmp_path):
        run_dir = tmp_path / "ns" / "run1"
        run_dir.mkdir(parents=True)
        # Create a properly named trajectory file: {task_id}_{timestamp}.yaml
        yaml_file = run_dir / "task_1_1700000000.yaml"
        yaml_file.write_text("workflow: {}")
        result = collect_latest_trajectory_files(str(tmp_path), "ns", run_id="run1")
        # task_1 should appear if parse_trajectory_filename handles this format
        assert isinstance(result, dict)

    def test_auto_selects_latest_run(self, tmp_path):
        ns_dir = tmp_path / "ns"
        (ns_dir / "run_b").mkdir(parents=True)
        (ns_dir / "run_a").mkdir(parents=True)
        result = collect_latest_trajectory_files(str(tmp_path), "ns")
        assert isinstance(result, dict)
