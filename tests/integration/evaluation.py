import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd
import yaml


def parse_filename(filename):
    base_name = os.path.splitext(filename)[0]
    last_underscore_idx = base_name.rfind("_")
    if last_underscore_idx == -1:
        return None, None

    task_id = base_name[:last_underscore_idx]
    last_underscore_idx_next = last_underscore_idx + 1
    timestamp_str = base_name[last_underscore_idx_next:]

    try:
        timestamp = float(timestamp_str)
        return task_id, timestamp
    except ValueError:
        return None, None


def get_latest_files(save_dir):
    yaml_files = glob.glob(os.path.join(save_dir, "*.yaml"))

    file_groups = defaultdict(list)

    for filepath in yaml_files:
        filename = os.path.basename(filepath)
        task_id, timestamp = parse_filename(filename)

        if task_id and timestamp:
            file_groups[task_id].append((timestamp, filepath))

    latest_files = {}
    for task_id, files in file_groups.items():
        files.sort(key=lambda x: x[0], reverse=True)
        latest_timestamp, latest_filepath = files[0]
        latest_files[task_id] = latest_filepath

    return latest_files


def load_csv_data(filepath):
    """Load CSV data and return pandas DataFrame"""
    try:
        df = pd.read_csv(filepath)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"


def compare_pandas_table(pred, gold, ignore_order=False):
    """
    Smart comparison of two pandas tables, based on spider_evaluation.py implementation

    Args:
        pred (DataFrame): Predicted result table
        gold (DataFrame): Gold standard table
        ignore_order (bool, optional): Whether to ignore row order. Defaults to False.

    Returns:
        int: 1 for match, 0 for no match
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

    gold_cols = gold
    pred_cols = pred

    # Transpose and convert to lists for comparison
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()

    score = 1
    for _, gold_col in enumerate(t_gold_list):
        if not any(vectors_match(gold_col, pred_col, ignore_order_=ignore_order) for pred_col in t_pred_list):
            score = 0
            break
        else:
            for _, pred_col in enumerate(t_pred_list):
                if vectors_match(gold_col, pred_col, ignore_order_=ignore_order):
                    break

    return score


def format_shape(shape):
    """Format shape tuple to string like 1x1"""
    if shape is None:
        return "Unknown"
    return f"{shape[0]}x{shape[1]}"


def preview_dataframe(df, max_rows=3, max_cols=5):
    """Preview dataframe content with truncation"""
    if df is None:
        return "No data"

    # Limit rows and columns for preview
    preview_df = df.head(max_rows)
    if len(df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
        truncated_cols = True
    else:
        truncated_cols = False

    # Convert to string representation
    result_lines = []

    # Headers
    headers = list(preview_df.columns)
    if truncated_cols:
        headers.append("...")
    result_lines.append(" | ".join(str(h) for h in headers))

    # Separator
    result_lines.append("-" * len(result_lines[0]))

    # Data rows
    for _, row in preview_df.iterrows():
        row_values = [str(v) for v in row.values]
        if truncated_cols:
            row_values.append("...")
        result_lines.append(" | ".join(row_values))

    # Add truncation indicator for rows
    if len(df) > max_rows:
        result_lines.append("...")

    return "\n       ".join(result_lines)


def compare_csv_results(actual_path, expected_path):
    """Use smart comparison method to compare CSV results"""
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
        score = compare_pandas_table(actual_df, expected_df, ignore_order=True)
        comparison_result["match"] = score == 1

    except Exception as e:
        comparison_result["error"] = f"Comparison error: {str(e)}"

    return comparison_result


def analyze_yaml_file(
    filepath, workdir, namespace, enable_comparison=False, target_task_id=None, gold_path=None, result_dir="output"
):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return {"error": "file format error: empty or invalid YAML"}

        if "workflow" not in data:
            return {"error": "file format error: missing workflow field"}

        workflow = data["workflow"]
        if workflow is None:
            return {"error": "file format error: workflow is None"}

        results = {
            "total_nodes": 0,
            "output_nodes": 0,
            "output_success": 0,
            "output_failure": 0,
            "errors": [],
            "node_types": defaultdict(int),
            "completion_time": workflow.get("completion_time") if workflow else None,
            "status": workflow.get("status", "unknown") if workflow else "unknown",
            "comparison_results": [],
        }

        nodes = workflow.get("nodes", []) if workflow else []

        if not nodes and workflow and "type" in workflow:
            results["total_nodes"] = 1
            node_type = workflow.get("type", "unknown")
            results["node_types"][node_type] += 1

            if node_type == "output":
                results["output_nodes"] = 1
                result = workflow.get("result", {})
                if result is None:
                    result = {}

                success = result.get("success", False)
                status = result.get("status", "unknown").lower()

                if success and status not in ["pending", "failed", "error"]:
                    results["output_success"] += 1

                    if enable_comparison:
                        task = workflow.get("task", {})
                        if task is None:
                            task = {}
                        task_id = task.get("id", "") if task else ""

                        if task_id and (target_task_id is None or task_id == target_task_id):
                            comparison = compare_with_gold_standard(task_id, workdir, namespace, gold_path, result_dir)
                            if comparison:
                                results["comparison_results"].append(comparison)
                else:
                    results["output_failure"] += 1

                    if not success:
                        error_info = result.get("error", "Unknown error")
                    else:
                        error_info = f"Output status is '{status}', not successful"

                    results["errors"].append(f"workflow: {error_info}")
            else:
                result = workflow.get("result", {})
                if result is None:
                    result = {}
                if result and not result.get("success", True):
                    error_info = result.get("error", "Unknown error")
                    results["errors"].append(f"workflow {node_type}: {error_info}")
        else:
            results["total_nodes"] = len(nodes)

            for node in nodes:
                if node is None:
                    continue

                node_type = node.get("type", "unknown")
                results["node_types"][node_type] += 1

                if node_type == "output":
                    results["output_nodes"] += 1

                    result = node.get("result", {})
                    if result is None:
                        result = {}

                    success = result.get("success", False)
                    status = result.get("status", "unknown").lower()

                    if success and status not in ["pending", "failed", "error"]:
                        results["output_success"] += 1

                        if enable_comparison:
                            task = workflow.get("task", {})
                            if task is None:
                                task = {}
                            task_id = task.get("id", "") if task else ""

                            if task_id and (target_task_id is None or task_id == target_task_id):
                                comparison = compare_with_gold_standard(
                                    task_id, workdir, namespace, gold_path, result_dir
                                )
                                if comparison:
                                    results["comparison_results"].append(comparison)
                    else:
                        results["output_failure"] += 1

                        if not success:
                            error_info = result.get("error", "Unknown error")
                        else:
                            error_info = f"Output status is '{status}', not successful"

                        results["errors"].append(f"node {node.get('id', 'unknown')}: {error_info}")

        results["node_types"] = dict(results["node_types"])

        return results

    except Exception as e:
        import traceback

        return {"error": f"parse file failed: {str(e)}\nTraceback: {traceback.format_exc()}"}


def compare_with_gold_standard(task_id, workdir, namespace, gold_path, result_dir="output"):
    """Compare execution results with gold standard"""
    actual_csv = os.path.join(workdir, result_dir, namespace, f"{task_id}.csv")
    gold_csv = os.path.join(workdir, gold_path, "exec_result", f"{task_id}.csv")

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


def parse_integration_script(script_path):
    """Parse run_integration.sh and extract all test commands and task IDs"""
    test_commands = []
    task_ids = []

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("(cd") and "benchmark_task_id" in line:
                match = re.search(r"--benchmark_task_id\s+(\S+)", line)
                if match:
                    task_id = match.group(1).rstrip(")")
                    task_ids.append(task_id)
                    test_commands.append(line)

    except Exception as e:
        print(f"Error parsing integration script: {e}")
        return [], []

    return test_commands, task_ids


def clean_successful_tests_and_generate_rerun_script(workdir, namespace, analysis_results):
    """Clean failed test files and generate rerun script for failed tests"""
    script_path = os.path.join(workdir, "tests", "integration", "run_integration.sh")

    if not os.path.exists(script_path):
        print(f"Warning: run_integration.sh not found at {script_path}")
        return

    test_commands, all_task_ids = parse_integration_script(script_path)

    if not test_commands:
        print("No test commands found in run_integration.sh")
        return

    print(f"Found {len(test_commands)} test commands in run_integration.sh")

    successful_tasks = []
    failed_tasks = []

    for task_id in all_task_ids:
        if task_id in analysis_results:
            result = analysis_results[task_id]
            if "error" not in result and result.get("output_failure", 0) == 0 and result.get("output_success", 0) > 0:
                successful_tasks.append(task_id)
            else:
                failed_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)

    print(f"Successful tasks: {len(successful_tasks)}")
    print(f"Failed tasks: {len(failed_tasks)}")

    files_to_delete = []
    save_dir = os.path.join(workdir, "save")

    for task_id in failed_tasks:
        yaml_pattern = os.path.join(save_dir, f"{task_id}_*.yaml")
        matching_files = glob.glob(yaml_pattern)
        files_to_delete.extend(matching_files)

    print(f"Failed test files to delete: {len(files_to_delete)}")

    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted failed test: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    print(f"Successfully deleted {deleted_count} failed test files")

    failed_commands = []
    for cmd, task_id in zip(test_commands, all_task_ids):
        if task_id in failed_tasks:
            failed_commands.append(cmd)

    rerun_script_path = os.path.join(workdir, "tests", "integration", "rerun_failed_integration.sh")

    try:
        with open(rerun_script_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Rerun failed integration tests\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Failed tasks: {len(failed_tasks)}\n")
            f.write(f"# Successful tasks (will be skipped): {len(successful_tasks)}\n\n")

            for cmd in failed_commands:
                f.write(cmd + "\n")

        os.chmod(rerun_script_path, 0o755)
        print(f"Generated rerun script: {rerun_script_path}")
        print(f"Script contains {len(failed_commands)} failed test commands")
        print("Successful tests will be skipped because their YAML files are preserved")

    except Exception as e:
        print(f"Failed to generate rerun script: {e}")


def generate_report(analysis_results, namespace, output_file=None, json_output=False, enable_comparison=False):
    if json_output:
        # Collect task IDs for different categories
        failed_task_ids = []
        matched_task_ids = []
        mismatched_task_ids = []
        empty_result_task_ids = []

        json_report = {
            "namespace": namespace,
            "generated_time": datetime.now().isoformat(),
            "summary": {
                "total_files": len(analysis_results),
                "total_output_nodes": 0,
                "total_output_success": 0,
                "total_output_failure": 0,
                "success_rate": 0.0,
            },
            "files": analysis_results,
        }

        if enable_comparison:
            json_report["summary"]["comparison_summary"] = {
                "total_comparisons": 0,
                "successful_matches": 0,
                "mismatches": 0,
                "comparison_errors": 0,
                "empty_result_errors": 0,
            }

        total_comparisons = 0
        successful_matches = 0
        mismatches = 0
        comparison_errors = 0
        empty_result_errors = 0

        for task_id, result in analysis_results.items():
            if "error" in result:
                failed_task_ids.append(task_id)
            else:
                json_report["summary"]["total_output_nodes"] += result["output_nodes"]
                json_report["summary"]["total_output_success"] += result["output_success"]
                json_report["summary"]["total_output_failure"] += result["output_failure"]

                # Add to failed_task_ids if there are output failures
                if result["output_failure"] > 0:
                    failed_task_ids.append(task_id)

                if enable_comparison:
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

        # Add task ID lists to JSON report
        json_report["task_ids"] = {
            "failed_task_ids": ",".join(map(str, sorted(failed_task_ids))),
            "matched_task_ids": ",".join(map(str, sorted(matched_task_ids))),
            "mismatched_task_ids": ",".join(map(str, sorted(mismatched_task_ids))),
            "empty_result_task_ids": ",".join(map(str, sorted(empty_result_task_ids))),
        }

        if enable_comparison:
            json_report["summary"]["comparison_summary"].update(
                {
                    "total_comparisons": total_comparisons,
                    "successful_matches": successful_matches,
                    "mismatches": mismatches,
                    "comparison_errors": comparison_errors,
                    "empty_result_errors": empty_result_errors,
                }
            )

        if json_report["summary"]["total_output_nodes"] > 0:
            json_report["summary"]["success_rate"] = (
                json_report["summary"]["total_output_success"] / json_report["summary"]["total_output_nodes"]
            ) * 100

        json_str = json.dumps(json_report, ensure_ascii=False, indent=2)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"JSON report save to: {output_file}")
        else:
            print(json_str)
        return

    # Collect task IDs for different categories
    failed_task_ids = []
    matched_task_ids = []
    mismatched_task_ids = []
    empty_result_task_ids = []

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"Workflow Evaluation Report - {namespace}")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    total_files = len(analysis_results)
    total_output_success = 0
    total_output_failure = 0
    total_output_nodes = 0
    failed_files = []
    all_node_types = defaultdict(int)

    total_comparisons = 0
    successful_matches = 0
    mismatches = 0
    comparison_errors = 0
    empty_result_errors = 0

    report_lines.append(f"Total files analyzed: {total_files}")
    report_lines.append("")

    for task_id, result in analysis_results.items():
        if "error" in result:
            report_lines.append(f"❌ {task_id}: {result['error']}")
            failed_files.append(task_id)
            failed_task_ids.append(task_id)
        else:
            total_output_nodes += result["output_nodes"]
            total_output_success += result["output_success"]
            total_output_failure += result["output_failure"]

            # Add to failed_task_ids if there are output failures
            if result["output_failure"] > 0:
                failed_task_ids.append(task_id)

            for node_type, count in result["node_types"].items():
                all_node_types[node_type] += count

            status = "✅" if result["output_failure"] == 0 else "⚠️"
            report_lines.append(f"{status} {task_id}:")
            report_lines.append(f"   Total nodes: {result['total_nodes']}")
            report_lines.append(f"   Output nodes: {result['output_nodes']}")
            report_lines.append(f"   Success: {result['output_success']}")
            report_lines.append(f"   Failure: {result['output_failure']}")
            report_lines.append(f"   Workflow status: {result['status']}")

            if result.get("completion_time"):
                completion_time = datetime.fromtimestamp(result["completion_time"])
                report_lines.append(f"   Completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

            if result["node_types"]:
                type_summary = ", ".join([f"{k}: {v}" for k, v in result["node_types"].items()])
                report_lines.append(f"   Node types: {type_summary}")

            if enable_comparison and result.get("comparison_results"):
                report_lines.append("   Result comparison:")
                for comp_result in result["comparison_results"]:
                    total_comparisons += 1
                    comp = comp_result.get("comparison", {})

                    if comp.get("error"):
                        error_msg = comp.get("error", "")
                        if "No columns to parse from file" in error_msg:
                            empty_result_errors += 1
                            empty_result_task_ids.append(task_id)
                            report_lines.append(f"     🔍 {comp_result['task_id']}: Empty result - {comp['error']}")
                        else:
                            comparison_errors += 1
                            report_lines.append(f"     ❌ {comp_result['task_id']}: {comp['error']}")
                    else:
                        match_result = comp.get("match", False)

                        if match_result:
                            successful_matches += 1
                            matched_task_ids.append(task_id)
                            report_lines.append(f"     ✅ {comp_result['task_id']}: Results match")
                        else:
                            mismatches += 1
                            mismatched_task_ids.append(task_id)
                            report_lines.append(f"     ❌ {comp_result['task_id']}: Results don't match")

                        # Display table shape information
                        actual_shape = comp.get("actual_shape")
                        expected_shape = comp.get("expected_shape")
                        if actual_shape and expected_shape:
                            report_lines.append(f"       Actual shape: {format_shape(actual_shape)}")
                            report_lines.append(f"       Expected shape: {format_shape(expected_shape)}")

                        # Display result preview
                        if not match_result:
                            actual_preview = comp.get("actual_preview")
                            expected_preview = comp.get("expected_preview")
                            if actual_preview:
                                report_lines.append("       Actual result preview:")
                                report_lines.append(f"       {actual_preview}")
                            if expected_preview:
                                report_lines.append("       Expected result preview:")
                                report_lines.append(f"       {expected_preview}")

            if result["errors"]:
                report_lines.append("   Error details:")
                for error in result["errors"]:
                    report_lines.append(f"     • {error}")
            report_lines.append("")

    report_lines.append("=" * 60)
    report_lines.append("Summary Statistics")
    report_lines.append("=" * 60)
    report_lines.append(f"Total output nodes: {total_output_nodes}")
    report_lines.append(f"Execution success: {total_output_success}")
    report_lines.append(f"Execution failure: {total_output_failure}")

    if total_output_nodes > 0:
        success_rate = (total_output_success / total_output_nodes) * 100
        report_lines.append(f"Success rate: {success_rate:.2f}%")
    else:
        report_lines.append("Success rate: N/A (no output nodes found)")

    if enable_comparison and total_comparisons > 0:
        report_lines.append("")
        report_lines.append("Result comparison summary:")
        report_lines.append(f"   Total comparisons: {total_comparisons}")
        report_lines.append(f"   Successful matches: {successful_matches}")
        report_lines.append(f"   Mismatches: {mismatches}")
        report_lines.append(f"   Empty result errors: {empty_result_errors}")
        report_lines.append(f"   Other comparison errors: {comparison_errors}")

        if total_comparisons > 0:
            match_rate = (successful_matches / total_comparisons) * 100
            report_lines.append(f"   Match rate: {match_rate:.2f}%")

            mismatch_rate = (mismatches / total_comparisons) * 100
            report_lines.append(f"   Mismatch rate: {mismatch_rate:.2f}%")

            empty_error_rate = (empty_result_errors / total_comparisons) * 100
            report_lines.append(f"   Empty result error rate: {empty_error_rate:.2f}%")

    # Add task ID lists to report
    report_lines.append("")
    report_lines.append("Task ID Lists:")
    report_lines.append(f"   failed_task_ids: {','.join(map(str, sorted(failed_task_ids)))}")
    report_lines.append(f"   matched_task_ids: {','.join(map(str, sorted(matched_task_ids)))}")
    report_lines.append(f"   mismatched_task_ids: {','.join(map(str, sorted(mismatched_task_ids)))}")
    report_lines.append(f"   empty_result_task_ids: {','.join(map(str, sorted(empty_result_task_ids)))}")

    if failed_files:
        report_lines.append(f"\nFile parsing failures: {len(failed_files)}")
        for failed_file in failed_files:
            report_lines.append(f"  • {failed_file}")

    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
        print("\n" + report_text)
    else:
        print(report_text)


def main():
    parser = argparse.ArgumentParser(description="evaluate script")
    parser.add_argument("--namespace", required=True, help="namespace (example: snowflake)")
    parser.add_argument("--workdir", required=True, help="working directory")
    parser.add_argument("--gold-path", required=True, help="path to gold standard files directory")
    parser.add_argument("--result-dir", default="output", help="result directory (default: output)")
    parser.add_argument("--save-dir", default="save", help="save directory containing trajectory files (default: save)")
    parser.add_argument("--output", help="output file")
    parser.add_argument("--json", action="store_true", help="json format")
    parser.add_argument(
        "--enable-comparison",
        action="store_true",
        help="enable result comparison with gold standard (default: disabled for performance)",
    )
    parser.add_argument("--task-id", help="specific task ID to compare (only effective with --enable-comparison)")
    parser.add_argument(
        "--clean-and-rerun",
        action="store_true",
        help="delete failed test files and generate rerun script for failed tests (successful tests will be skipped)",
    )

    args = parser.parse_args()

    save_dir = os.path.join(args.workdir, args.save_dir)

    if not os.path.exists(save_dir):
        print(f"Err: save dir not exists: {save_dir}")
        return 1

    latest_files = get_latest_files(save_dir)

    if not latest_files:
        print(f"waring: could not find yaml file in {save_dir}")
        return 1

    if args.task_id:
        if args.task_id in latest_files:
            latest_files = {args.task_id: latest_files[args.task_id]}
        else:
            print(f"Error: Task ID '{args.task_id}' not found in available files.")
            print(f"Available task IDs: {list(latest_files.keys())}")
            return 1

    analysis_results = {}
    for task_id, filepath in latest_files.items():
        if not args.json:
            print(f"Analyze file: {os.path.basename(filepath)}")
        result = analyze_yaml_file(
            filepath,
            args.workdir,
            args.namespace,
            args.enable_comparison,
            args.task_id,
            args.gold_path,
            args.result_dir,
        )
        analysis_results[task_id] = result

    generate_report(analysis_results, args.namespace, args.output, args.json, args.enable_comparison)

    if args.clean_and_rerun:
        print("\n" + "=" * 60)
        print("Cleaning failed tests and generating rerun script...")
        print("=" * 60)
        clean_successful_tests_and_generate_rerun_script(args.workdir, args.namespace, analysis_results)

    return 0


if __name__ == "__main__":
    exit(main())
