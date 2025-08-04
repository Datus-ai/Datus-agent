"""
Benchmark evaluation utilities for comparing SQL execution results.

This module provides utilities for:
- Executing SQL queries and saving results
- Comparing CSV results with gold standards
- Evaluating benchmark accuracy
- Generating evaluation reports
"""

import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import yaml

from datus.tools.db_tools import BaseSqlConnector
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def get_latest_trajectory_files(save_dir: str) -> Dict[str, str]:
    """Get the latest trajectory file for each task ID.

    Args:
        save_dir: Directory containing trajectory YAML files

    Returns:
        Dict mapping task_id to latest trajectory file path
    """
    if not os.path.exists(save_dir):
        return {}

    yaml_files = glob.glob(os.path.join(save_dir, "*.yaml"))
    file_groups = defaultdict(list)

    for filepath in yaml_files:
        filename = os.path.basename(filepath)
        task_id, timestamp = parse_trajectory_filename(filename)
        if task_id and timestamp:
            file_groups[task_id].append((timestamp, filepath))

    latest_files = {}
    for task_id, files in file_groups.items():
        files.sort(key=lambda x: x[0], reverse=True)
        _, latest_filepath = files[0]
        latest_files[task_id] = latest_filepath

    return latest_files


def parse_trajectory_filename(filename: str):
    """Parse trajectory filename to extract task ID and timestamp.

    Args:
        filename: Trajectory filename like "task_id_timestamp.yaml"

    Returns:
        Tuple of (task_id, timestamp) or (None, None) if parsing fails
    """
    base_name = os.path.splitext(filename)[0]
    last_underscore_idx = base_name.rfind("_")
    if last_underscore_idx == -1:
        return None, None

    task_id = base_name[:last_underscore_idx]
    timestamp_str = base_name[last_underscore_idx + 1 :]

    try:
        timestamp = float(timestamp_str)
        return task_id, timestamp
    except ValueError:
        return None, None


def analyze_trajectory_file(filepath: str, task_id: str, gold_path: str, result_dir: str) -> Dict:
    """Analyze a single trajectory file and compare with gold standard.

    Args:
        filepath: Path to trajectory YAML file
        task_id: Task identifier
        gold_path: Path to gold standard directory
        result_dir: Directory containing actual results

    Returns:
        Dict containing analysis results
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "workflow" not in data:
            return {"error": "Invalid trajectory file format"}

        workflow = data["workflow"]
        if workflow is None:
            return {"error": "Workflow is None"}

        results = {
            "total_nodes": 0,
            "output_nodes": 0,
            "output_success": 0,
            "output_failure": 0,
            "errors": [],
            "node_types": defaultdict(int),
            "completion_time": workflow.get("completion_time"),
            "status": workflow.get("status", "unknown"),
            "comparison_results": [],
        }

        nodes = workflow.get("nodes", [])
        if not nodes and workflow and "type" in workflow:
            # Handle single workflow node case
            results["total_nodes"] = 1
            node_type = workflow.get("type", "unknown")
            results["node_types"][node_type] += 1

            if node_type == "output":
                results["output_nodes"] = 1
                result = workflow.get("result", {}) or {}
                success = result.get("success", False)
                status = result.get("status", "unknown").lower()

                if success and status not in ["pending", "failed", "error"]:
                    results["output_success"] += 1
                    # Compare with gold standard
                    comparison = compare_with_gold_standard(task_id, gold_path, result_dir)
                    if comparison:
                        results["comparison_results"].append(comparison)
                else:
                    results["output_failure"] += 1
                    error_info = result.get("error", "Unknown error") if not success else f"Output status is '{status}'"
                    results["errors"].append(f"workflow: {error_info}")
        else:
            # Handle multiple nodes case
            results["total_nodes"] = len(nodes)
            for node in nodes:
                if node is None:
                    continue

                node_type = node.get("type", "unknown")
                results["node_types"][node_type] += 1

                if node_type == "output":
                    results["output_nodes"] += 1
                    result = node.get("result", {}) or {}
                    success = result.get("success", False)
                    status = result.get("status", "unknown").lower()

                    if success and status not in ["pending", "failed", "error"]:
                        results["output_success"] += 1
                        # Compare with gold standard
                        comparison = compare_with_gold_standard(task_id, gold_path, result_dir)
                        if comparison:
                            results["comparison_results"].append(comparison)
                    else:
                        results["output_failure"] += 1
                        error_info = (
                            result.get("error", "Unknown error") if not success else f"Output status is '{status}'"
                        )
                        results["errors"].append(f"node {node.get('id', 'unknown')}: {error_info}")

        results["node_types"] = dict(results["node_types"])
        return results

    except Exception as e:
        return {"error": f"Failed to analyze trajectory file: {str(e)}"}


def compare_with_gold_standard(task_id: str, gold_path: str, result_dir: str) -> Dict:
    """Compare execution results with gold standard.

    Args:
        task_id: Task identifier
        gold_path: Path to gold standard directory
        result_dir: Directory containing actual results

    Returns:
        Dict containing comparison results
    """
    actual_csv = os.path.join(result_dir, f"{task_id}.csv")
    gold_csv = os.path.join(gold_path, "exec_result", f"{task_id}.csv")

    comparison_result = {
        "task_id": task_id,
        "actual_file_exists": os.path.exists(actual_csv),
        "gold_file_exists": os.path.exists(gold_csv),
        "actual_path": actual_csv,
        "gold_path": gold_csv,
        "comparison": None,
    }

    if not comparison_result["actual_file_exists"]:
        comparison_result["comparison"] = {"error": f"Actual result file not found: {actual_csv}"}
        return comparison_result

    if not comparison_result["gold_file_exists"]:
        comparison_result["comparison"] = {"error": f"Gold standard file not found: {gold_csv}"}
        return comparison_result

    comparison_result["comparison"] = compare_csv_results(actual_csv, gold_csv)
    return comparison_result


def compare_csv_results(actual_path: str, expected_path: str) -> Dict:
    """Compare two CSV files using smart comparison method.

    Args:
        actual_path: Path to actual results CSV
        expected_path: Path to expected results CSV

    Returns:
        Dict containing comparison results
    """
    comparison_result = {
        "match": False,
        "actual_file_exists": True,
        "expected_file_exists": True,
        "actual_shape": None,
        "expected_shape": None,
        "actual_preview": None,
        "expected_preview": None,
        "error": None,
    }

    try:
        # Load actual results
        actual_df, actual_error = load_csv_data(actual_path)
        if actual_error:
            comparison_result["error"] = f"Actual file error: {actual_error}"
            comparison_result["actual_file_exists"] = False
            return comparison_result

        # Load expected results
        expected_df, expected_error = load_csv_data(expected_path)
        if expected_error:
            comparison_result["error"] = f"Expected file error: {expected_error}"
            comparison_result["expected_file_exists"] = False
            return comparison_result

        comparison_result["actual_shape"] = actual_df.shape
        comparison_result["expected_shape"] = expected_df.shape
        comparison_result["actual_preview"] = preview_dataframe(actual_df)
        comparison_result["expected_preview"] = preview_dataframe(expected_df)

        # Use smart comparison method
        score = compare_pandas_tables(actual_df, expected_df, ignore_order=True)
        comparison_result["match"] = score == 1

    except Exception as e:
        comparison_result["error"] = f"Comparison error: {str(e)}"

    return comparison_result


def load_csv_data(filepath: str):
    """Load CSV data and return pandas DataFrame.

    Args:
        filepath: Path to CSV file

    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        df = pd.read_csv(filepath)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"


def preview_dataframe(df, max_rows: int = 3, max_cols: int = 5) -> str:
    """Preview dataframe content with truncation.

    Args:
        df: pandas DataFrame to preview
        max_rows: Maximum number of rows to show
        max_cols: Maximum number of columns to show

    Returns:
        String representation of DataFrame preview
    """
    if df is None:
        return "No data"

    preview_df = df.head(max_rows)
    if len(df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
        truncated_cols = True
    else:
        truncated_cols = False

    result_lines = []
    headers = list(preview_df.columns)
    if truncated_cols:
        headers.append("...")
    result_lines.append(" | ".join(str(h) for h in headers))
    result_lines.append("-" * len(result_lines[0]))

    for _, row in preview_df.iterrows():
        row_values = [str(v) for v in row.values]
        if truncated_cols:
            row_values.append("...")
        result_lines.append(" | ".join(row_values))

    if len(df) > max_rows:
        result_lines.append("...")

    return "\n       ".join(result_lines)


def compare_pandas_tables(pred_df, gold_df, ignore_order: bool = False) -> int:
    """Smart comparison of two pandas tables.

    Args:
        pred_df: Predicted results DataFrame
        gold_df: Gold standard DataFrame
        ignore_order: Whether to ignore row order in comparison

    Returns:
        1 if tables match, 0 otherwise
    """
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1, v2 = (
                sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
            )
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    # Transpose and convert to lists for comparison
    t_gold_list = gold_df.transpose().values.tolist()
    t_pred_list = pred_df.transpose().values.tolist()

    score = 1
    for gold_col in t_gold_list:
        if not any(vectors_match(gold_col, pred_col, ignore_order_=ignore_order) for pred_col in t_pred_list):
            score = 0
            break

    return score


def generate_evaluation_report(analysis_results: Dict) -> Dict:
    """Generate evaluation report with accuracy metrics.

    Args:
        analysis_results: Dict mapping task_id to analysis results

    Returns:
        Dict containing evaluation report
    """
    total_files = len(analysis_results)
    total_output_success = 0
    total_output_failure = 0
    total_output_nodes = 0

    total_comparisons = 0
    successful_matches = 0
    mismatches = 0
    comparison_errors = 0
    empty_result_errors = 0

    failed_task_ids = []
    matched_task_ids = []
    mismatched_task_ids = []
    empty_result_task_ids = []

    for task_id, result in analysis_results.items():
        if "error" in result:
            failed_task_ids.append(task_id)
        else:
            total_output_nodes += result["output_nodes"]
            total_output_success += result["output_success"]
            total_output_failure += result["output_failure"]

            if result["output_failure"] > 0:
                failed_task_ids.append(task_id)

            for comp_result in result.get("comparison_results", []):
                if comp_result.get("comparison"):
                    total_comparisons += 1
                    comp = comp_result["comparison"]
                    if comp.get("error"):
                        error_msg = comp.get("error", "")
                        if "No columns to parse from file" in error_msg:
                            empty_result_errors += 1
                            empty_result_task_ids.append(task_id)
                        else:
                            comparison_errors += 1
                    else:
                        if comp.get("match"):
                            successful_matches += 1
                            matched_task_ids.append(task_id)
                        else:
                            mismatches += 1
                            mismatched_task_ids.append(task_id)

    # Calculate success rate
    success_rate = (total_output_success / total_output_nodes * 100) if total_output_nodes > 0 else 0.0

    # Calculate match rate
    match_rate = (successful_matches / total_comparisons * 100) if total_comparisons > 0 else 0.0

    report = {
        "status": "success",
        "generated_time": datetime.now().isoformat(),
        "summary": {
            "total_files": total_files,
            "total_output_nodes": total_output_nodes,
            "total_output_success": total_output_success,
            "total_output_failure": total_output_failure,
            "success_rate": round(success_rate, 2),
            "comparison_summary": {
                "total_comparisons": total_comparisons,
                "successful_matches": successful_matches,
                "mismatches": mismatches,
                "comparison_errors": comparison_errors,
                "empty_result_errors": empty_result_errors,
                "match_rate": round(match_rate, 2),
            },
        },
        "task_ids": {
            "failed_task_ids": ",".join(map(str, sorted(failed_task_ids))),
            "matched_task_ids": ",".join(map(str, sorted(matched_task_ids))),
            "mismatched_task_ids": ",".join(map(str, sorted(mismatched_task_ids))),
            "empty_result_task_ids": ",".join(map(str, sorted(empty_result_task_ids))),
        },
        "details": analysis_results,
    }

    return report


def evaluate_benchmark_accuracy(
    benchmark_path: str,
    trajectory_dir: str,
    current_namespace: str,
    output_dir: str,
    target_task_ids: Optional[Set[str]] = None,
) -> Dict:
    """Evaluate benchmark accuracy by comparing generated results with gold standard.

    Args:
        benchmark_path: Path to benchmark directory containing testing_set.csv
        trajectory_dir: Directory containing trajectory files
        current_namespace: Current namespace for logging
        output_dir: Directory containing actual results
        target_task_ids: Optional set of specific task IDs to evaluate

    Returns:
        Dict containing evaluation results and accuracy metrics
    """
    logger.info("Starting benchmark accuracy evaluation")

    # Get latest trajectory files from save directory
    latest_files = get_latest_trajectory_files(trajectory_dir)

    if not latest_files:
        logger.warning(f"No trajectory files found in {trajectory_dir}")
        return {"status": "error", "message": "No trajectory files found for evaluation"}

    # Filter by target task IDs if specified
    if target_task_ids:
        latest_files = {task_id: filepath for task_id, filepath in latest_files.items() if task_id in target_task_ids}

    analysis_results = {}
    gold_path = os.path.join(benchmark_path, "gold")

    # Analyze each trajectory file
    for task_id, filepath in latest_files.items():
        logger.info(f"Analyzing trajectory file for task {task_id}")
        result = analyze_trajectory_file(filepath, task_id, gold_path, output_dir)
        analysis_results[task_id] = result

    # Generate evaluation report
    report = generate_evaluation_report(analysis_results)

    logger.info(f"Benchmark evaluation completed. Analyzed {len(analysis_results)} tasks")
    return report


def execute_sql_and_save_results(
    sql_query: str, task_id: str, sql_connector: BaseSqlConnector, output_dir: str = None
) -> Dict:
    """Execute SQL query and save results to CSV file for benchmark evaluation.

    Args:
        sql_query: The SQL query to execute
        task_id: Task identifier for file naming
        sql_connector: The connection of db
        output_dir: Output directory path

    Returns:
        Dict with execution results and file path
    """
    if not output_dir:
        raise ValueError("output_dir is required")

    # Create gold standard directory structure
    gold_dir = os.path.join(output_dir, "gold", "exec_result")
    os.makedirs(gold_dir, exist_ok=True)

    try:
        # Execute SQL using provided function
        query_result = sql_connector.execute_arrow(sql_query)
        if not query_result.success:
            return {"success": False, "error": query_result.error or "", "output_path": None, "rows_returned": 0}

        query_result = query_result.sql_return
        results = {
            "success": True,
            "columns": query_result.column_names,
            "results": query_result.to_pylist(),
            "error": None,
        }
        # Save results to CSV
        output_path = os.path.join(gold_dir, f"{task_id}.csv")
        save_results_to_csv(results, output_path)

        return {
            "success": results["success"],
            "output_path": output_path,
            "rows_returned": len(results["results"]) if results["success"] else 0,
            "error": results.get("error"),
        }

    except Exception as e:
        logger.error(f"Failed to execute SQL for task {task_id}: {str(e)}")
        return {"success": False, "error": str(e), "output_path": None, "rows_returned": 0}


def save_results_to_csv(results: Dict, output_path: str):
    """Save query results to CSV file.

    Args:
        results: Dict containing query results with keys: success, columns, results, error
        output_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if results["success"] and "results" in results:
        if results["columns"] and results["results"]:
            df = pd.DataFrame(results["results"], columns=results["columns"])
        elif results["results"]:
            # If no column names, create generic column names
            num_cols = len(results["results"][0]) if results["results"] else 0
            df = pd.DataFrame(results["results"], columns=[f"col_{i}" for i in range(num_cols)])
        else:
            df = pd.DataFrame()
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        # Handle error cases
        if results["error"]:
            error_df = pd.DataFrame([["error", results["error"]]], columns=["status", "message"])
        else:
            error_df = pd.DataFrame(columns=["status", "message"])
        error_df.to_csv(output_path, index=False, encoding="utf-8")


def execute_duckdb_query(namespace_config, sql_query: str) -> Dict:
    """Execute DuckDB query and return results."""
    try:
        import duckdb

        db_path = namespace_config.uri if hasattr(namespace_config, "uri") else namespace_config.get("uri", "")
        if not db_path:
            raise Exception("DuckDB URI not found in namespace config")

        conn = duckdb.connect(db_path)
        result_df = conn.execute(sql_query).df()
        conn.close()

        if result_df is not None and not result_df.empty:
            column_names = result_df.columns.tolist()
            results = result_df.values.tolist()
            return {"success": True, "columns": column_names, "results": results, "error": None}
        else:
            return {"success": True, "columns": [], "results": [], "error": None}

    except Exception as e:
        return {"success": False, "columns": [], "results": [], "error": str(e)}


def generate_gold_standard_results(
    tasks: List[Dict], benchmark_path: str, sql_connector: BaseSqlConnector, target_task_ids: Optional[Set[str]] = None
) -> Dict:
    """Generate gold standard results by executing expected SQL queries.

    Phase 1: Execute expected SQL queries for each task and save results as gold standard.

    Args:
        tasks: List of task dictionaries containing question_id and sql
        benchmark_path: Path to benchmark directory for saving gold standard results
        target_task_ids: Optional set of specific task IDs to process

    Returns:
        Dict mapping task_id to execution results
    """
    gold_results = {}

    # Get appropriate executor function based on DB type

    for task in tasks:
        task_id = str(task["question_id"])
        if target_task_ids and task_id not in target_task_ids:
            continue

        logger.info(f"Generating gold standard for task {task_id}")
        try:
            gold_result = execute_sql_and_save_results(task["sql"], task_id, sql_connector, benchmark_path)
            gold_results[task_id] = gold_result

            if gold_result["success"]:
                logger.info(f"Gold standard for task {task_id} created: {gold_result['rows_returned']} rows")
            else:
                logger.warning(f"Failed to create gold standard for task {task_id}: {gold_result.get('error')}")

        except Exception as e:
            logger.error(f"Error generating gold standard for task {task_id}: {str(e)}")
            gold_results[task_id] = {"success": False, "error": str(e), "rows_returned": 0}

    logger.info(f"Phase 1 completed. Generated gold standards for {len(gold_results)} tasks")
    return gold_results


def evaluate_and_report_accuracy(
    benchmark_path: str,
    trajectory_dir: str,
    current_namespace: str,
    output_dir: str,
    target_task_ids: Optional[Set[str]] = None,
    output_file: Optional[str] = None,
) -> Dict:
    """Evaluate benchmark accuracy and generate comprehensive report.

    Phase 3: Evaluate accuracy against gold standard and generate detailed report.

    Args:
        benchmark_path: Path to benchmark directory containing testing_set.csv
        trajectory_dir: Directory containing trajectory files
        current_namespace: Current namespace configuration
        output_dir: Output directory for results
        target_task_ids: Optional set of specific task IDs to evaluate
        output_file: Optional output file path. If provided, saves detailed report to file

    Returns:
        Dict containing evaluation results and accuracy metrics
    """
    try:
        # Get accuracy report using existing function
        accuracy_report = evaluate_benchmark_accuracy(
            benchmark_path, trajectory_dir, current_namespace, output_dir, target_task_ids
        )

        # Log summary results
        if accuracy_report.get("status") == "success":
            _log_accuracy_summary(accuracy_report)

            # Save detailed report to file only if output_file is specified
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(accuracy_report, f, ensure_ascii=False, indent=2)
                logger.info(f"Detailed accuracy report saved to: {output_file}")

        else:
            logger.error(f"Accuracy evaluation failed: {accuracy_report.get('message')}")

        return accuracy_report

    except Exception as e:
        logger.error(f"Error during accuracy evaluation: {str(e)}")
        return {"status": "error", "message": f"Accuracy evaluation failed: {str(e)}"}


def _log_accuracy_summary(accuracy_report: Dict):
    """Log formatted accuracy summary results in a comprehensive, readable format.

    Args:
        accuracy_report: The accuracy report dictionary containing summary data
    """
    from datetime import datetime

    summary = accuracy_report["summary"]
    task_ids = accuracy_report.get("task_ids", {})
    comp_summary = summary.get("comparison_summary", {})

    # Parse task ID strings into lists for better formatting
    failed_ids = [id.strip() for id in task_ids.get("failed_task_ids", "").split(",") if id.strip()]
    matched_ids = [id.strip() for id in task_ids.get("matched_task_ids", "").split(",") if id.strip()]
    mismatched_ids = [id.strip() for id in task_ids.get("mismatched_task_ids", "").split(",") if id.strip()]
    empty_result_ids = [id.strip() for id in task_ids.get("empty_result_task_ids", "").split(",") if id.strip()]

    # Build comprehensive report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BENCHMARK ACCURACY EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total tasks analyzed: {summary['total_files']}")

    if summary.get("success_rate") is not None:
        success_rate = summary["success_rate"]
        report_lines.append(f"Execution success rate: {success_rate:.1f}%")

    if comp_summary:
        match_rate = comp_summary.get("match_rate", 0)
        report_lines.append(f"Result comparison match rate: {match_rate:.1f}%")

    report_lines.append("")

    # Detailed Statistics
    report_lines.append("DETAILED STATISTICS")
    report_lines.append("-" * 40)

    if comp_summary:
        total_comparisons = comp_summary.get("total_comparisons", 0)
        successful_matches = comp_summary.get("successful_matches", 0)
        mismatches = comp_summary.get("mismatches", 0)
        comparison_errors = comp_summary.get("comparison_errors", 0)
        empty_result_errors = comp_summary.get("empty_result_errors", 0)

        report_lines.append(f"Total comparisons performed: {total_comparisons}")
        report_lines.append(f"Successful matches: {successful_matches}")
        report_lines.append(f"Mismatches: {mismatches}")
        report_lines.append(f"Comparison errors: {comparison_errors}")
        report_lines.append(f"Empty result errors: {empty_result_errors}")

        if total_comparisons > 0:
            mismatch_rate = (mismatches / total_comparisons) * 100
            error_rate = ((comparison_errors + empty_result_errors) / total_comparisons) * 100
            report_lines.append(f"Mismatch rate: {mismatch_rate:.1f}%")
            report_lines.append(f"Error rate: {error_rate:.1f}%")

    report_lines.append("")

    # Task Breakdown by Category
    report_lines.append("TASK BREAKDOWN BY CATEGORY")
    report_lines.append("-" * 40)

    def format_task_list(task_list, max_display=10):
        if not task_list:
            return "None"

        # Try to sort as integers first, fallback to string sorting
        try:
            sorted_list = sorted(task_list, key=int)
        except ValueError:
            # If any task_id can't be converted to int, sort as strings
            sorted_list = sorted(task_list, key=str)

        if len(task_list) <= max_display:
            return ", ".join(sorted_list)
        else:
            displayed = sorted_list[:max_display]
            return f"{', '.join(displayed)} ... (+{len(task_list) - max_display} more)"

    report_lines.append(f"Matched tasks ({len(matched_ids)}):")
    report_lines.append(f"   {format_task_list(matched_ids)}")
    report_lines.append("")

    report_lines.append(f"Mismatched tasks ({len(mismatched_ids)}):")
    report_lines.append(f"   {format_task_list(mismatched_ids)}")
    report_lines.append("")

    report_lines.append(f"Failed tasks ({len(failed_ids)}):")
    report_lines.append(f"   {format_task_list(failed_ids)}")
    report_lines.append("")

    if empty_result_ids:
        report_lines.append(f"Empty result tasks ({len(empty_result_ids)}):")
        report_lines.append(f"   {format_task_list(empty_result_ids)}")
        report_lines.append("")

    # Additional Statistics
    report_lines.append("ADDITIONAL STATISTICS")
    report_lines.append("-" * 40)

    total_tasks = summary["total_files"]
    success_count = len(matched_ids)

    if total_tasks > 0:
        overall_success_rate = (success_count / total_tasks) * 100
        report_lines.append(f"Overall success rate: {overall_success_rate:.1f}%")

    report_lines.append(f"Successful tasks: {success_count}")
    report_lines.append(f"Failed tasks: {len(failed_ids)}")
    report_lines.append(f"Mismatched tasks: {len(mismatched_ids)}")

    if empty_result_ids:
        report_lines.append(f"Empty result tasks: {len(empty_result_ids)}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Output the complete report as a single log entry
    full_report = "\n".join(report_lines)
    logger.info(f"\n{full_report}")


def load_bird_dev_tasks(benchmark_path: str) -> List[Dict[str, Any]]:
    file_path = os.path.join(benchmark_path, "dev.json")
    if not os.path.exists(file_path):
        raise DatusException(
            code=ErrorCode.COMMON_FILE_NOT_FOUND,
            message_args={"file_name": file_path, "config_name": "Bird-dev benchmark"},
        )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in file '{file_path}': {str(e)}")
        raise DatusException(
            ErrorCode.COMMON_JSON_PARSE_ERROR, message_args={"file_path": file_path, "error_detail": str(e)}
        )
