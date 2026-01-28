# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from datus.agent.agent import Agent
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.benchmark_utils import load_benchmark_tasks
from datus.utils.time_utils import format_duration_human

STATUS_MATCHED = "Matched"
STATUS_GEN_SQL_FAILED = "Gen SQL Failed"
STATUS_GOLD_SQL_FAILED = "Gold SQL Failed"
STATUS_MATCH_FAILED = "Match Failed"
STATUS_RESULT_MISMATCH = "Result Mismatch"
STATUS_TABLE_MISMATCH = "Table Mismatch"
STATUS_COLUMN_MISMATCH = "Column Mismatch"
STATUS_NOT_EXECUTED = "Not Executed"
TASK_SUCCESS_RATE_HEADER = "Matching Rate"
SUMMARY_ROW_LABEL = "Summary of Matching Rate"
ROUND_DURATION_ROW_LABEL = "Round Duration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-round benchmark + evaluation cycles.")
    parser.add_argument("--config", default="", help="Path to agent config file.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    setup_base_parser_args(parser)
    return parser.parse_args()


def setup_base_parser_args(parser: argparse.ArgumentParser):
    parser.add_argument("--namespace", required=True, help="Namespace to benchmark, e.g. bird_sqlite.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name, e.g. bird_dev.")
    parser.add_argument("--workflow", default="reflection", help="Workflow plan to execute (default: reflection)")
    parser.add_argument(
        "--max_steps", "--max-steps", type=int, default=30, help="Maximum steps per workflow execution (default: 30)"
    )
    parser.add_argument(
        "--round",
        "--max_round",
        "--max-round",
        type=int,
        default=4,
        help="Number of benchmark iterations to run (default: 4)",
    )
    parser.add_argument(
        "--group_name",
        "--group-name",
        type=str,
        help="The name of the integration test group. If it is empty, the name of the workflow will be used.",
    )
    parser.add_argument(
        "--task_ids",
        "--task-ids",
        nargs="*",
        default=None,
        help="Explicit task ids to benchmark and evaluate (space/comma separated)",
    )
    parser.add_argument(
        "--workers",
        "--max_workers",
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution (default: 1)",
    )
    parser.add_argument(
        "--summary_report_file",
        "--summary-report-file",
        type=str,
        default=None,
        help="Path to summary report file. Reports will be appended to this file for each round.",
    )
    parser.add_argument(
        "--delete_history",
        "--delete-history",
        action="store_true",
        help="Delete existing round output directory before each round starts",
    )


def sanitize_group_name(workflow: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", workflow.strip())
    return slug or "workflow"


def resolve_task_ids(
    agent_config: AgentConfig,
    benchmark: str,
    explicit_ids: Optional[Sequence[str]],
) -> List[str]:
    if explicit_ids:
        task_ids: List[str] = []
        for value in explicit_ids:
            if not value:
                continue
            task_ids.extend([item.strip() for item in str(value).split(",") if item is not None])
        if not task_ids:
            raise ValueError("No valid task ids parsed from --task_ids.")
        return task_ids

    benchmark_config = agent_config.benchmark_config(benchmark)
    task_id_key = benchmark_config.question_id_key or "_task_id"

    ids: List[str] = []
    for task in load_benchmark_tasks(agent_config, benchmark):
        task_id = task.get(task_id_key)
        if task_id is None:
            continue
        ids.append(str(task_id))

    if not ids:
        raise ValueError("Could not resolve any task ids from benchmark data.")
    return ids


def override_round_paths(agent_config: AgentConfig, round_dir: Path) -> None:
    save_dir = round_dir / "save"
    trajectory_dir = round_dir / "trajectory"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)
    agent_config._save_dir = str(save_dir)
    agent_config._trajectory_dir = str(trajectory_dir)


def build_agent_args(
    cli_args: argparse.Namespace,
    task_ids: Sequence[str],
    round_dir: Path,
    round_idx: int,
    run_id: str,
) -> argparse.Namespace:
    evaluation_file = round_dir / f"evaluation_round_{run_id}_{round_idx}.json"
    target_task_ids = None if not task_ids else list(task_ids)
    common_kwargs = {
        # "components": ["metrics", "metadata", "table_lineage", "document"],
        "load_cp": None,
        "max_steps": cli_args.max_steps,
        "benchmark": cli_args.benchmark,
        "namespace": cli_args.namespace,
        "benchmark_task_ids": target_task_ids,
        "task_ids": target_task_ids,
        "catalog": "",
        "database": "",
        "subject_path": None,
        "current_date": None,
        "workflow": cli_args.workflow,
        "output_file": str(evaluation_file),
        "max_workers": cli_args.workers,
        "summary_report_file": cli_args.summary_report_file,
    }
    return argparse.Namespace(**common_kwargs)


def _parse_duration_seconds(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_single_round(
    agent_config: AgentConfig,
    round_idx: int,
    args: argparse.Namespace,
    base_home: str,
    group_slug: str,
    target_task_ids: Sequence[str],
    run_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    integration_root = Path(base_home) / "integration"
    integration_root.mkdir(parents=True, exist_ok=True)
    round_dir = integration_root / f"{group_slug}_{round_idx}"
    if round_dir.exists() and args.delete_history:
        shutil.rmtree(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)

    override_round_paths(agent_config, round_dir)

    agent_args = build_agent_args(args, target_task_ids, round_dir, round_idx, run_id)
    db_manager = db_manager_instance(agent_config.namespaces)
    agent = Agent(args=agent_args, agent_config=agent_config, db_manager=db_manager)

    print(f"[Round {round_idx}] Starting benchmark -> {round_dir}")
    benchmark_result = agent.benchmark(run_id=run_id) or {}
    benchmark_duration = _parse_duration_seconds(benchmark_result.get("time_spends_seconds"))
    print(f"[Round {round_idx}] Finished benchmark: {benchmark_result}")
    print(f"[Round {round_idx}] Benchmark finished, running evaluation...")
    evaluation_result = agent.evaluation(log_summary=False) or {}
    if evaluation_result.get("status") == "success":
        with open(agent_args.output_file, "r") as f:
            return json.load(f), benchmark_duration
    else:
        print(f"⚠️⚠️⚠️Failed evaluation for round {round_idx}: {evaluation_result.get('message')}")
        return {}, benchmark_duration


def classify_task_status(task_id: str, evaluation: Optional[Dict[str, object]]) -> str:
    if not evaluation:
        return STATUS_NOT_EXECUTED
    details: Dict[str, object] = evaluation.get("details") or {}
    detail = details.get(str(task_id))
    if not isinstance(detail, dict):
        return STATUS_NOT_EXECUTED

    comparisons = detail.get("comparison_results") or []
    if not comparisons:
        if detail.get("errors") or detail.get("output_success_count", 0) == 0:
            return STATUS_GEN_SQL_FAILED
        return STATUS_MATCH_FAILED

    for record in comparisons:
        comparison = (record or {}).get("comparison") if isinstance(record, dict) else None
        if not isinstance(comparison, dict):
            continue
        if comparison.get("error") == "Actual result unavailable":
            return STATUS_GEN_SQL_FAILED
        if comparison.get("error") == "Gold standard unavailable":
            return STATUS_GOLD_SQL_FAILED
        matched_tables = comparison.get("matched_tables") or []
        expected_tables = comparison.get("expected_tables") or []
        if len(matched_tables) != len(expected_tables):
            return STATUS_TABLE_MISMATCH

        match_rate = comparison.get("match_rate")
        try:
            if match_rate is not None:
                if float(match_rate) >= 0.999:
                    return STATUS_MATCHED
                if float(match_rate) <= 0.00001:
                    return STATUS_RESULT_MISMATCH
        except (TypeError, ValueError):
            pass
    column_issue = False
    for record in comparisons:
        comparison = (record or {}).get("comparison") if isinstance(record, dict) else None
        if not isinstance(comparison, dict):
            continue
        error_text = comparison.get("error")
        if error_text:
            if "No columns to parse" in error_text or "file not found" in error_text.lower():
                return STATUS_GEN_SQL_FAILED
            continue
        if comparison.get("actual_sql_error"):
            return STATUS_GEN_SQL_FAILED
        if comparison.get("sql_error"):
            return STATUS_GOLD_SQL_FAILED
        missing = comparison.get("missing_columns") or []
        extra = comparison.get("extra_columns") or []
        if missing or extra:
            column_issue = True
    if column_issue:
        return STATUS_COLUMN_MISMATCH
    return STATUS_TABLE_MISMATCH


def build_status_matrix(
    reports: Sequence[Optional[Dict[str, object]]], task_ids: Sequence[str]
) -> Dict[str, List[str]]:
    matrix: Dict[str, List[str]] = {task_id: [] for task_id in task_ids}
    for task_id in task_ids:
        for report in reports:
            matrix[task_id].append(classify_task_status(task_id, report))
    return matrix


def export_summary_excel(
    matrix: Dict[str, List[str]],
    round_count: int,
    integration_root: Path,
    workflow_slug: str,
    round_durations: Sequence[Optional[float]],
    run_id: str,
) -> Path:
    def normalize_statuses(statuses: List[str]) -> List[str]:
        normalized: List[str] = []
        for idx in range(round_count):
            normalized.append(statuses[idx] if idx < len(statuses) else STATUS_NOT_EXECUTED)
        return normalized

    def format_percentage(rate: float) -> str:
        return f"{rate * 100:.2f}%"

    rows: List[Dict[str, str]] = []
    round_match_counts = [0] * round_count
    for task_id, statuses in matrix.items():
        normalized_statuses = normalize_statuses(statuses)
        row = {"task_id": task_id}
        for idx in range(round_count):
            header = f"round_{idx}"
            status_value = normalized_statuses[idx]
            row[header] = status_value
            if status_value == STATUS_MATCHED:
                round_match_counts[idx] += 1
        success_count = sum(1 for status in normalized_statuses if status == STATUS_MATCHED)
        denominator = round_count if round_count else 1
        success_rate = success_count / denominator
        row[TASK_SUCCESS_RATE_HEADER] = format_percentage(success_rate)
        rows.append(row)
    total_tasks = len(matrix) if matrix else 0
    summary_row: Dict[str, str] = {"task_id": SUMMARY_ROW_LABEL}
    for idx in range(round_count):
        denominator = total_tasks if total_tasks else 1
        rate = round_match_counts[idx] / denominator
        summary_row[f"round_{idx}"] = format_percentage(rate)
    total_denominator = (total_tasks * round_count) if total_tasks and round_count else 1
    total_success_rate = sum(round_match_counts) / total_denominator
    summary_row[TASK_SUCCESS_RATE_HEADER] = format_percentage(total_success_rate)
    rows.append(summary_row)
    if round_durations:
        duration_row: Dict[str, str] = {"task_id": ROUND_DURATION_ROW_LABEL}
        for idx in range(round_count):
            duration_value = round_durations[idx] if idx < len(round_durations) else None
            duration_row[f"round_{idx}"] = (
                format_duration_human(duration_value) if duration_value is not None else "N/A"
            )
        duration_row[TASK_SUCCESS_RATE_HEADER] = ""
        rows.append(duration_row)
    df = pd.DataFrame(rows)
    excel_path = integration_root / f"{workflow_slug}_summary_{run_id}.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path


def multi_benchmark(args: argparse.Namespace):
    initial_config = load_agent_config(**vars(args))
    base_home = str(Path(initial_config.home if initial_config.home else "~/.datus").expanduser())
    group_slug = sanitize_group_name(args.group_name or args.workflow)
    integration_root = Path(base_home) / "integration"
    integration_root.mkdir(parents=True, exist_ok=True)

    target_task_ids = resolve_task_ids(
        initial_config,
        args.benchmark,
        args.task_ids,
    )
    print(f"Benchmark task ids ({len(target_task_ids)}): {', '.join(target_task_ids)}")

    reports: List[Optional[Dict[str, object]]] = []
    round_durations: List[Optional[float]] = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    for round_idx in range(args.round):
        try:
            report, duration = run_single_round(
                agent_config=initial_config,
                round_idx=round_idx,
                args=args,
                base_home=base_home,
                group_slug=group_slug,
                target_task_ids=target_task_ids,
                run_id=run_id,
            )
        except Exception as exc:  # pragma: no cover - surfaced for manual runs
            print(f"[Round {round_idx}] Failed with error: {exc}")
            reports.append(None)
            round_durations.append(None)
            continue
        reports.append(report)
        round_durations.append(duration)

    status_matrix = build_status_matrix(reports, target_task_ids)
    summary_path = export_summary_excel(
        status_matrix, len(reports), integration_root, group_slug, round_durations, run_id
    )
    print(f"Multi-round summary exported to {summary_path}")


def main():
    args = parse_args()
    from datus.utils.loggings import configure_logging

    configure_logging(args.debug)
    multi_benchmark(args)


if __name__ == "__main__":
    main()
