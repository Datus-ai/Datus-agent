import csv
from pathlib import Path

import pytest
import yaml

from datus.utils.benchmark_utils import evaluate_sub_agent_and_report
from datus.utils.constants import DBType


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
            "SQL",
            "Question ",
            "Gold sql",
            "Expected answer",
            "answer rows",
            "File",
            "Expected table",
            "Expected sql",
            "semantic_model",
            "expected metrics",
            "expected knowledge",
        ]
        writer.writerow(header)
        writer.writerows(rows)


def _write_sql(path: Path, sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sql, encoding="utf-8")


def _write_trajectory(path: Path, task_id: str) -> None:
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
                    },
                }
            ],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)


@pytest.mark.parametrize("db_type", [DBType.SQLITE, DBType.SNOWFLAKE])
def test_evaluate_sub_agent_and_report(tmp_path: Path, db_type: str) -> None:
    # Arrange gold answers
    gold_file = tmp_path / "gold.csv"
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
                "",
                "customers",
                "SELECT name, total FROM customers",
                "",
                "",
                "",
            ],
            [
                "task-456",
                "Question two",
                "SELECT name, total FROM customers",
                expected_answer,
                "2",
                "",
                "customers",
                "SELECT name, total FROM customers",
                "",
                "",
                "",
            ],
        ],
    )

    # Arrange agent results (CSV + SQL)
    result_dir = tmp_path / "results"
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
    trajectory_dir = tmp_path / "trajectories"
    _write_trajectory(trajectory_dir / "task-123_1.yaml", "task-123")
    _write_trajectory(trajectory_dir / "task-456_1.yaml", "task-456")

    # Act
    report = evaluate_sub_agent_and_report(
        benchmark_path=str(gold_file),
        trajectory_dir=str(trajectory_dir),
        result_path=str(result_dir),
        db_type=db_type,
    )

    # Assert successful evaluation structure
    print("$$$", report)
    assert report["status"] == "success"
    details = report["details"]
    assert set(details.keys()) == {"task-123", "task-456"}

    # Task with perfect match: results, tables, and columns align
    match_comparison = details["task-123"]["comparison_results"][0]["comparison"]
    assert match_comparison["match_rate"] == 1.0
    assert match_comparison["un_matched_columns"] == []
    assert {tuple(pair) for pair in match_comparison["matched_columns"]} == {("name", "name"), ("total", "total")}
    actual_tables = [table.lower() for table in match_comparison["actual_tables"]]
    expected_tables = [table.lower() for table in match_comparison["expected_tables"]]
    matched_tables = [table.lower() for table in match_comparison["matched_tables"]]
    assert any("customers" in table for table in actual_tables)
    assert any("customers" in table for table in expected_tables)
    assert matched_tables and all("customers" in table for table in matched_tables)

    # Task with mismatched totals and table name
    mismatch_comparison = details["task-456"]["comparison_results"][0]["comparison"]
    assert mismatch_comparison["match_rate"] < 1.0
    assert mismatch_comparison["un_matched_columns"] == ["total"]
    assert {tuple(pair) for pair in mismatch_comparison["matched_columns"]} == {("name", "name")}
    mismatch_actual_tables = [table.lower() for table in mismatch_comparison["actual_tables"]]
    mismatch_expected_tables = [table.lower() for table in mismatch_comparison["expected_tables"]]
    assert any("other_table" in table for table in mismatch_actual_tables)
    assert any("customers" in table for table in mismatch_expected_tables)
    assert mismatch_comparison["matched_tables"] == []
