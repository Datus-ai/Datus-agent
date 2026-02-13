import csv
import json
from pathlib import Path
from typing import Optional

import pytest
import yaml

from datus.configuration.agent_config import AgentConfig, BenchmarkConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.benchmark_utils import evaluate_benchmark_and_report
from datus.utils.constants import DBType


@pytest.fixture
def agent_config(tmp_path: Path) -> AgentConfig:
    agent_config = load_agent_config(namespace="bird_school", home=tmp_path)
    return agent_config


def _write_result_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _write_gold_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "task_id",
            "question",
            "gold_sql",
            "expected_answer",
            "answer_rows",
            "file",
            "expected_table",
            "expected_sql",
            "semantic_model",
            "expected_metrics",
            "expected_knowledge",
        ]
        writer.writerow(header)
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_sql(path: Path, sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql, encoding="utf-8")


def _write_trajectory(path: Path, task_id: str, tool_actions: Optional[list[dict]] = None) -> None:
    payload = {
        "workflow": {
            "completion_time": 1,
            "status": "completed",
            "nodes": [
                {
                    "id": f"{task_id}-output",
                    "type": "output",
                    "result": {
                        "success": True,
                        "status": "completed",
                        "action_history": tool_actions or [],
                    },
                }
            ],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True)


def _benchmark_root(agent_config: AgentConfig, relative_path: str) -> Path:
    return Path(agent_config.home) / "benchmark" / relative_path


def _match_tool_actions() -> list[dict]:
    return [
        {
            "action_id": "tool_write_file",
            "role": "tool",
            "action_type": "write_file",
            "input": {"function_name": "write_file"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": "File written successfully: /workspace/sql/task-123.sql",
                },
            },
            "status": "success",
        },
        {
            "action_id": "tool_search_reference_sql",
            "role": "tool",
            "action_type": "search_reference_sql",
            "input": {"function_name": "search_reference_sql"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": [{"name": "customer_totals", "sql": "SELECT name, total FROM customers"}],
                },
            },
            "status": "success",
        },
        {
            "action_id": "tool_search_table",
            "role": "tool",
            "action_type": "search_table",
            "input": {"function_name": "search_table"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": {"metadata": [{"semantic_model_name": "customer_semantic_model"}]},
                },
            },
            "status": "success",
        },
        {
            "action_id": "tool_search_metrics",
            "role": "tool",
            "action_type": "search_metrics",
            "input": {"function_name": "search_metrics"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": [{"name": "Metric A", "description": "Total customers"}],
                },
            },
            "status": "success",
        },
    ]


def _mismatch_tool_actions() -> list[dict]:
    return [
        {
            "action_id": "tool_write_file_mismatch",
            "role": "tool",
            "action_type": "write_file",
            "input": {"function_name": "write_file"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": "File written successfully: /workspace/sql/task-456.sql",
                },
            },
            "status": "success",
        },
        {
            "action_id": "tool_search_reference_sql_mismatch",
            "role": "tool",
            "action_type": "search_reference_sql",
            "input": {"function_name": "search_reference_sql"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": [{"name": "other_reference", "sql": "SELECT name, total FROM other_table"}],
                },
            },
            "status": "success",
        },
        {
            "action_id": "tool_search_table_mismatch",
            "role": "tool",
            "action_type": "search_table",
            "input": {"function_name": "search_table"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": {"metadata": [{"semantic_model_name": "other_semantic"}]},
                },
            },
            "status": "success",
        },
        {
            "action_id": "tool_search_metrics_mismatch",
            "role": "tool",
            "action_type": "search_metrics",
            "input": {"function_name": "search_metrics"},
            "output": {
                "success": True,
                "raw_output": {
                    "success": 1,
                    "error": None,
                    "result": [{"name": "Another Metric", "description": "Another metric text"}],
                },
            },
            "status": "success",
        },
    ]


def _assert_report_structure(report: dict[str, object]) -> None:
    assert report["status"] == "success"
    details = report["details"]
    assert set(details.keys()) == {"task-123", "task-456"}

    match_comparison = details["task-123"]["comparison_results"][0]["comparison"]
    assert match_comparison["match_rate"] == 1.0
    assert match_comparison["missing_columns"] == []
    assert {tuple(pair) for pair in match_comparison["matched_columns"]} == {("name", "name"), ("total", "total")}
    actual_tables = [table.lower() for table in match_comparison["actual_tables"]]
    expected_tables = [table.lower() for table in match_comparison["expected_tables"]]
    matched_tables = [table.lower() for table in match_comparison["matched_tables"]]
    assert any("customers" in table for table in actual_tables)
    assert any("customers" in table for table in expected_tables)
    assert matched_tables and all("customers" in table for table in matched_tables)
    artifacts_match = match_comparison["tools_comparison"]
    assert artifacts_match["expected_file"]["match"] is True
    assert any("task-123" in value for value in artifacts_match["expected_file"]["matched_actual"])
    assert artifacts_match["expected_sql"]["match"] is True
    assert artifacts_match["expected_semantic_model"]["match"] is True
    assert artifacts_match["expected_metrics"]["match"] is True
    assert not artifacts_match["expected_metrics"]["missing_expected"]

    mismatch_comparison = details["task-456"]["comparison_results"][0]["comparison"]
    assert mismatch_comparison["match_rate"] < 1.0
    assert mismatch_comparison["missing_columns"] == ["total"]
    assert {tuple(pair) for pair in mismatch_comparison["matched_columns"]} == {("name", "name")}
    mismatch_actual_tables = [table.lower() for table in mismatch_comparison["actual_tables"]]
    mismatch_expected_tables = [table.lower() for table in mismatch_comparison["expected_tables"]]
    assert any("other_table" in table for table in mismatch_actual_tables)
    assert any("customers" in table for table in mismatch_expected_tables)
    assert mismatch_comparison["matched_tables"] == []
    artifacts_mismatch = mismatch_comparison["tools_comparison"]
    assert artifacts_mismatch["expected_file"]["match"] is False
    assert artifacts_mismatch["expected_sql"]["match"] is False
    assert artifacts_mismatch["expected_sql"]["matched_actual"] == []
    assert artifacts_mismatch["expected_semantic_model"]["match"] is False
    assert artifacts_mismatch["expected_metrics"]["match"] is False
    assert artifacts_mismatch["expected_metrics"]["missing_expected"]


@pytest.mark.parametrize("db_type", [DBType.SQLITE, DBType.SNOWFLAKE])
def test_evaluate_benchmark_and_report_with_csv_manifest(
    agent_config: AgentConfig, tmp_path: Path, db_type: DBType
) -> None:
    benchmark_name = "csv_benchmark_evaluation"
    benchmark_config = BenchmarkConfig(
        benchmark_path=benchmark_name,
        question_file="gold.csv",
        question_id_key="task_id",
        question_key="question",
        gold_result_path="gold.csv",
        gold_result_key="expected_answer",
        gold_sql_path="gold.csv",
        gold_sql_key="gold_sql",
    )
    agent_config.benchmark_configs[benchmark_name] = benchmark_config
    benchmark_path = Path(agent_config.benchmark_path(benchmark_name))
    # Arrange gold answers
    gold_file = benchmark_path / "gold.csv"
    expected_answer = "name,total\nAlice,1\nBob,2"
    _write_gold_csv(
        gold_file,
        [
            [
                "task-123",
                "Question one",
                "SELECT name, total FROM customers",
                expected_answer,
                "2",
                "/workspace/sql/task-123.sql",
                "customers",
                "customer_totals",
                "customer_semantic_model",
                "Metric A",
                "",
            ],
            [
                "task-456",
                "Question two",
                "SELECT name, total FROM customers",
                expected_answer,
                "2",
                "/workspace/sql/task-999.sql",
                "customers",
                "expected_reference",
                "target_semantic_model",
                "Target Metric",
                "",
            ],
        ],
    )

    # Arrange agent results (CSV + SQL)
    result_dir = Path(agent_config.output_dir)
    _write_result_csv(
        result_dir / "task-123.csv",
        ["name", "total"],
        [["Alice", "1"], ["Bob", "2"]],
    )
    _write_sql(result_dir / "task-123.sql", "SELECT name, total FROM customers;")

    _write_result_csv(
        result_dir / "task-456.csv",
        ["name", "total"],
        [["Alice", "50"], ["Bob", "60"]],
    )
    _write_sql(result_dir / "task-456.sql", "SELECT name, total FROM other_table;")

    # Arrange trajectories
    trajectory_dir = Path(agent_config.trajectory_dir)
    _write_trajectory(trajectory_dir / "task-123_1.yaml", "task-123", _match_tool_actions())
    _write_trajectory(trajectory_dir / "task-456_1.yaml", "task-456", _mismatch_tool_actions())

    agent_config.db_type = db_type

    # Act
    report = evaluate_benchmark_and_report(
        agent_config=agent_config,
        benchmark_platform=benchmark_name,
    )
    _assert_report_structure(report)


@pytest.mark.parametrize("db_type", [DBType.SQLITE, DBType.SNOWFLAKE])
def test_evaluate_benchmark_and_report_with_jsonl_manifest(
    agent_config: AgentConfig, tmp_path: Path, db_type: DBType
) -> None:
    benchmark_name = "jsonl_benchmark_evaluation"
    benchmark_config = BenchmarkConfig(
        benchmark_path=benchmark_name,
        question_file="tasks.jsonl",
        question_id_key="task_id",
        question_key="prompt",
        db_key="db_id",
        gold_sql_key="gold_sql",
        gold_result_key="expected_answer",
        gold_result_path="tasks.jsonl",
        gold_sql_path="tasks.jsonl",
    )
    agent_config.benchmark_configs[benchmark_name] = benchmark_config
    benchmark_path = Path(agent_config.benchmark_path(benchmark_name))

    expected_answer = "name,total\nAlice,1\nBob,2"
    manifest = benchmark_path / "tasks.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "task_id": "task-123",
                "prompt": "Question one",
                "db_id": "customers_db",
                "gold_sql": "SELECT name, total FROM customers",
                "expected_answer": expected_answer,
                "file": "/workspace/sql/task-123.sql",
                "expected_table": "customers",
                "expected_sql": "customer_totals",
                "semantic_model": "customer_semantic_model",
                "expected_metrics": "Metric A",
            },
            {
                "task_id": "task-456",
                "prompt": "Question two",
                "db_id": "customers_db",
                "gold_sql": "SELECT name, total FROM customers",
                "expected_answer": expected_answer,
                "file": "/workspace/sql/task-999.sql",
                "expected_table": "customers",
                "expected_sql": "expected_reference",
                "semantic_model": "target_semantic_model",
                "expected_metrics": "Target Metric",
            },
        ],
    )

    result_dir = Path(agent_config.output_dir)
    _write_result_csv(
        result_dir / "task-123.csv",
        ["name", "total"],
        [["Alice", "1"], ["Bob", "2"]],
    )
    _write_sql(result_dir / "task-123.sql", "SELECT name, total FROM customers;")

    _write_result_csv(
        result_dir / "task-456.csv",
        ["name", "total"],
        [["Alice", "50"], ["Bob", "60"]],
    )
    _write_sql(result_dir / "task-456.sql", "SELECT name, total FROM other_table;")

    trajectory_dir = Path(agent_config.trajectory_dir)
    _write_trajectory(trajectory_dir / "task-123_1.yaml", "task-123", _match_tool_actions())
    _write_trajectory(trajectory_dir / "task-456_1.yaml", "task-456", _mismatch_tool_actions())

    agent_config.db_type = db_type
    report = evaluate_benchmark_and_report(
        agent_config=agent_config,
        benchmark_platform=benchmark_name,
    )
    _assert_report_structure(report)
