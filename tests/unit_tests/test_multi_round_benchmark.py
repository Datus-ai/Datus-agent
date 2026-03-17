# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/multi_round_benchmark.py

CI-level: zero external deps, all I/O and agent calls mocked.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from datus.multi_round_benchmark import (
    STATUS_COLUMN_MISMATCH,
    STATUS_GEN_SQL_FAILED,
    STATUS_GOLD_SQL_FAILED,
    STATUS_MATCH_FAILED,
    STATUS_MATCHED,
    STATUS_NOT_EXECUTED,
    STATUS_RESULT_MISMATCH,
    STATUS_TABLE_MISMATCH,
    _parse_duration_seconds,
    build_agent_args,
    build_status_matrix,
    classify_task_status,
    export_summary_excel,
    override_round_paths,
    resolve_task_ids,
    sanitize_group_name,
    setup_base_parser_args,
)

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# sanitize_group_name
# ---------------------------------------------------------------------------


class TestSanitizeGroupName:
    def test_clean_name_unchanged(self):
        assert sanitize_group_name("my_workflow") == "my_workflow"

    def test_spaces_replaced_with_underscore(self):
        assert sanitize_group_name("my workflow") == "my_workflow"

    def test_special_chars_replaced(self):
        result = sanitize_group_name("a/b.c!d")
        assert "/" not in result
        assert "." not in result
        assert "!" not in result

    def test_empty_string_returns_workflow(self):
        assert sanitize_group_name("") == "workflow"

    def test_only_spaces_returns_workflow(self):
        assert sanitize_group_name("   ") == "workflow"

    def test_hyphens_and_underscores_preserved(self):
        result = sanitize_group_name("my-workflow_v2")
        assert result == "my-workflow_v2"

    def test_alphanumeric_preserved(self):
        result = sanitize_group_name("workflow123")
        assert result == "workflow123"


# ---------------------------------------------------------------------------
# _parse_duration_seconds
# ---------------------------------------------------------------------------


class TestParseDurationSeconds:
    def test_none_returns_none(self):
        assert _parse_duration_seconds(None) is None

    def test_float_value(self):
        assert _parse_duration_seconds(1.5) == 1.5

    def test_int_value(self):
        assert _parse_duration_seconds(100) == 100.0

    def test_string_number(self):
        assert _parse_duration_seconds("3.14") == 3.14

    def test_invalid_string_returns_none(self):
        assert _parse_duration_seconds("abc") is None

    def test_empty_string_returns_none(self):
        assert _parse_duration_seconds("") is None


# ---------------------------------------------------------------------------
# classify_task_status
# ---------------------------------------------------------------------------


class TestClassifyTaskStatus:
    def test_no_evaluation_returns_not_executed(self):
        assert classify_task_status("1", None) == STATUS_NOT_EXECUTED

    def test_empty_evaluation_returns_not_executed(self):
        assert classify_task_status("1", {}) == STATUS_NOT_EXECUTED

    def test_no_details_returns_not_executed(self):
        assert classify_task_status("1", {"details": {}}) == STATUS_NOT_EXECUTED

    def test_detail_not_dict_returns_not_executed(self):
        assert classify_task_status("1", {"details": {"1": "some_string"}}) == STATUS_NOT_EXECUTED

    def test_no_comparisons_with_errors_returns_gen_sql_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [],
                    "errors": ["some error"],
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_GEN_SQL_FAILED

    def test_no_comparisons_no_output_returns_gen_sql_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [],
                    "output_success_count": 0,
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_GEN_SQL_FAILED

    def test_no_comparisons_match_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [],
                    "output_success_count": 2,
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_MATCH_FAILED

    def test_actual_unavailable_returns_gen_sql_failed(self):
        evaluation = {
            "details": {"1": {"comparison_results": [{"comparison": {"error": "Actual result unavailable"}}]}}
        }
        assert classify_task_status("1", evaluation) == STATUS_GEN_SQL_FAILED

    def test_gold_unavailable_returns_gold_sql_failed(self):
        evaluation = {
            "details": {"1": {"comparison_results": [{"comparison": {"error": "Gold standard unavailable"}}]}}
        }
        assert classify_task_status("1", evaluation) == STATUS_GOLD_SQL_FAILED

    def test_table_mismatch_returns_table_mismatch(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1", "t2"],
                                "match_rate": 0.5,
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_TABLE_MISMATCH

    def test_match_rate_1_returns_matched(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "match_rate": 1.0,
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_MATCHED

    def test_match_rate_0_returns_result_mismatch(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "match_rate": 0.0,
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_RESULT_MISMATCH

    def test_actual_sql_error_returns_gen_sql_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "actual_sql_error": "syntax error",
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_GEN_SQL_FAILED

    def test_gold_sql_error_returns_gold_sql_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "sql_error": "syntax error",
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_GOLD_SQL_FAILED

    def test_missing_columns_returns_column_mismatch(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "missing_columns": ["col1"],
                                "extra_columns": [],
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_COLUMN_MISMATCH

    def test_no_columns_to_parse_error_returns_gen_sql_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "error": "No columns to parse from file",
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_GEN_SQL_FAILED

    def test_file_not_found_error_returns_gen_sql_failed(self):
        evaluation = {
            "details": {
                "1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "error": "File not found",
                            }
                        }
                    ]
                }
            }
        }
        assert classify_task_status("1", evaluation) == STATUS_GEN_SQL_FAILED


# ---------------------------------------------------------------------------
# build_status_matrix
# ---------------------------------------------------------------------------


class TestBuildStatusMatrix:
    def test_empty_tasks(self):
        matrix = build_status_matrix([], [])
        assert matrix == {}

    def test_single_task_single_report(self):
        report = {
            "details": {
                "task1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["t1"],
                                "expected_tables": ["t1"],
                                "match_rate": 1.0,
                            }
                        }
                    ]
                }
            }
        }
        matrix = build_status_matrix([report], ["task1"])
        assert "task1" in matrix
        assert matrix["task1"] == [STATUS_MATCHED]

    def test_none_report_gives_not_executed(self):
        matrix = build_status_matrix([None], ["task1"])
        assert matrix["task1"] == [STATUS_NOT_EXECUTED]

    def test_multiple_rounds(self):
        report_matched = {
            "details": {
                "t1": {
                    "comparison_results": [
                        {
                            "comparison": {
                                "matched_tables": ["x"],
                                "expected_tables": ["x"],
                                "match_rate": 1.0,
                            }
                        }
                    ]
                }
            }
        }
        matrix = build_status_matrix([report_matched, None], ["t1"])
        assert matrix["t1"] == [STATUS_MATCHED, STATUS_NOT_EXECUTED]


# ---------------------------------------------------------------------------
# export_summary_excel
# ---------------------------------------------------------------------------


class TestExportSummaryExcel:
    def test_creates_excel_file(self, tmp_path):
        matrix = {
            "task1": [STATUS_MATCHED, STATUS_MATCHED],
            "task2": [STATUS_GEN_SQL_FAILED, STATUS_MATCHED],
        }
        path = export_summary_excel(
            matrix=matrix,
            round_count=2,
            integration_root=tmp_path,
            workflow_slug="test_wf",
            round_durations=[10.0, 20.5],
            run_id="20250101_1200",
        )
        assert path.exists()
        assert path.suffix == ".xlsx"

    def test_empty_matrix_creates_file(self, tmp_path):
        path = export_summary_excel(
            matrix={},
            round_count=2,
            integration_root=tmp_path,
            workflow_slug="wf",
            round_durations=[],
            run_id="run1",
        )
        assert path.exists()

    def test_none_durations_handled(self, tmp_path):
        matrix = {"t1": [STATUS_MATCHED]}
        path = export_summary_excel(
            matrix=matrix,
            round_count=1,
            integration_root=tmp_path,
            workflow_slug="wf",
            round_durations=[None],
            run_id="run2",
        )
        assert path.exists()

    def test_round_count_0_no_divide_by_zero(self, tmp_path):
        matrix = {"t1": []}
        path = export_summary_excel(
            matrix=matrix,
            round_count=0,
            integration_root=tmp_path,
            workflow_slug="wf",
            round_durations=[],
            run_id="run3",
        )
        assert path.exists()


# ---------------------------------------------------------------------------
# override_round_paths
# ---------------------------------------------------------------------------


class TestOverrideRoundPaths:
    def test_creates_save_and_trajectory_dirs(self, tmp_path):
        mock_config = MagicMock()
        override_round_paths(mock_config, tmp_path)
        assert (tmp_path / "save").exists()
        assert (tmp_path / "trajectory").exists()
        mock_config._save_dir = str(tmp_path / "save")
        mock_config._trajectory_dir = str(tmp_path / "trajectory")


# ---------------------------------------------------------------------------
# build_agent_args
# ---------------------------------------------------------------------------


class TestBuildAgentArgs:
    def _make_cli_args(self, **overrides):
        defaults = dict(
            benchmark="bird_dev",
            namespace="ns1",
            workflow="reflection",
            max_steps=30,
            workers=1,
            summary_report_file=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_basic_build(self, tmp_path):
        cli_args = self._make_cli_args()
        task_ids = ["1", "2", "3"]
        round_dir = tmp_path / "round_0"
        result = build_agent_args(cli_args, task_ids, round_dir, 0, "run1")
        assert result.benchmark == "bird_dev"
        assert result.namespace == "ns1"
        assert result.workflow == "reflection"
        assert result.max_steps == 30
        assert result.max_workers == 1
        assert result.task_ids == ["1", "2", "3"]

    def test_empty_task_ids_gives_none(self, tmp_path):
        cli_args = self._make_cli_args()
        result = build_agent_args(cli_args, [], tmp_path, 0, "run1")
        assert result.task_ids is None

    def test_output_file_contains_run_id(self, tmp_path):
        cli_args = self._make_cli_args()
        result = build_agent_args(cli_args, ["t1"], tmp_path, 2, "runABC")
        assert "runABC" in result.output_file
        assert "2" in result.output_file


# ---------------------------------------------------------------------------
# resolve_task_ids - explicit ids
# ---------------------------------------------------------------------------


class TestResolveTaskIdsExplicit:
    def test_explicit_ids_returned(self):
        mock_config = MagicMock()
        ids = resolve_task_ids(mock_config, "bird_dev", ["1", "2", "3"])
        assert ids == ["1", "2", "3"]

    def test_comma_separated_ids(self):
        mock_config = MagicMock()
        ids = resolve_task_ids(mock_config, "bird_dev", ["1,2,3"])
        assert ids == ["1", "2", "3"]

    def test_empty_explicit_ids_raises(self):
        mock_config = MagicMock()
        with pytest.raises(ValueError, match="No valid task ids"):
            resolve_task_ids(mock_config, "bird_dev", [""])

    def test_none_explicit_ids_loads_from_benchmark(self):
        mock_config = MagicMock()
        mock_benchmark_config = MagicMock()
        mock_benchmark_config.question_id_key = "task_id"
        mock_config.benchmark_config.return_value = mock_benchmark_config

        with patch("datus.multi_round_benchmark.load_benchmark_tasks") as mock_load:
            mock_load.return_value = [{"task_id": "t1"}, {"task_id": "t2"}]
            ids = resolve_task_ids(mock_config, "bird_dev", None)
        assert ids == ["t1", "t2"]

    def test_no_tasks_from_benchmark_raises(self):
        mock_config = MagicMock()
        mock_benchmark_config = MagicMock()
        mock_benchmark_config.question_id_key = "task_id"
        mock_config.benchmark_config.return_value = mock_benchmark_config

        with patch("datus.multi_round_benchmark.load_benchmark_tasks") as mock_load:
            mock_load.return_value = []
            with pytest.raises(ValueError, match="Could not resolve any task ids"):
                resolve_task_ids(mock_config, "bird_dev", None)


# ---------------------------------------------------------------------------
# setup_base_parser_args
# ---------------------------------------------------------------------------


class TestSetupBaseParserArgs:
    def test_adds_required_args(self):
        parser = argparse.ArgumentParser()
        setup_base_parser_args(parser)
        # namespace and benchmark are required
        args = parser.parse_args(["--namespace", "ns1", "--benchmark", "b1"])
        assert args.namespace == "ns1"
        assert args.benchmark == "b1"
        assert args.round == 4  # default
        assert args.workers == 1  # default

    def test_default_workflow(self):
        parser = argparse.ArgumentParser()
        setup_base_parser_args(parser)
        args = parser.parse_args(["--namespace", "ns", "--benchmark", "bm"])
        assert args.workflow == "reflection"
