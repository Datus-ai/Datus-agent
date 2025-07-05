import argparse
import glob
import os
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
    try:
        df = pd.read_csv(filepath)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"


def format_shape(shape):
    if shape is None:
        return "Unknown"
    return f"{shape[0]}x{shape[1]}"


def preview_dataframe(df, max_rows=3, max_cols=5):
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


def analyze_yaml_file(filepath, workdir, namespace, agent_num, result_dir="output"):
    """Analyze a single YAML trajectory file"""
    result = {
        "filepath": filepath,
        "task_id": None,
        "total_nodes": 0,
        "output_nodes": 0,
        "output_success": 0,
        "output_failure": 0,
        "status": "unknown",
        "completion_time": None,
        "node_types": defaultdict(int),
        "errors": []
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            result["errors"].append("Empty YAML file")
            return result

        filename = os.path.basename(filepath)
        task_id, _ = parse_filename(filename)
        result["task_id"] = task_id

        if "workflow" in data:
            workflow = data["workflow"]
            result["status"] = workflow.get("status", "unknown")

            if "completion_time" in workflow:
                result["completion_time"] = workflow["completion_time"]

        if "nodes" in data:
            nodes = data["nodes"]
            result["total_nodes"] = len(nodes)

            for node_id, node_data in nodes.items():
                node_type = node_data.get("type", "unknown")
                result["node_types"][node_type] += 1

                if node_type == "output":
                    result["output_nodes"] += 1

                    if "result" in node_data:
                        node_result = node_data["result"]
                        if isinstance(node_result, dict) and node_result.get("success"):
                            result["output_success"] += 1
                        else:
                            result["output_failure"] += 1
                    else:
                        result["output_failure"] += 1

    except Exception as e:
        result["errors"].append(f"Error parsing YAML: {str(e)}")

    return result


def generate_report(analysis_results, namespace, agent_num, output_file=None):
    report_lines = []
    
    # Header
    report_lines.append("=" * 60)
    report_lines.append(f"Multi-Agent Evaluation Report - Agent {agent_num}")
    report_lines.append(f"Namespace: {namespace}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    total_tasks = len(analysis_results)
    total_output_nodes = 0
    total_output_success = 0
    total_output_failure = 0
    failed_task_ids = []
    successful_task_ids = []
    all_node_types = defaultdict(int)

    for task_id, result in analysis_results.items():
        total_output_nodes += result["output_nodes"]
        total_output_success += result["output_success"]
        total_output_failure += result["output_failure"]

        if result["output_failure"] > 0:
            failed_task_ids.append(task_id)
        else:
            successful_task_ids.append(task_id)

        for node_type, count in result["node_types"].items():
            all_node_types[node_type] += count

        status = "✅" if result["output_failure"] == 0 else "⚠️"
        report_lines.append(f"{status} Task {task_id}:")
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

        if result["errors"]:
            report_lines.append("   Errors:")
            for error in result["errors"]:
                report_lines.append(f"     • {error}")
        report_lines.append("")

    report_lines.append("=" * 60)
    report_lines.append("Summary Statistics")
    report_lines.append("=" * 60)
    report_lines.append(f"Total tasks: {total_tasks}")
    report_lines.append(f"Total output nodes: {total_output_nodes}")
    report_lines.append(f"Execution success: {total_output_success}")
    report_lines.append(f"Execution failure: {total_output_failure}")

    if total_output_nodes > 0:
        success_rate = (total_output_success / total_output_nodes) * 100
        report_lines.append(f"Success rate: {success_rate:.2f}%")
    else:
        report_lines.append("Success rate: N/A (no output nodes found)")

    if all_node_types:
        report_lines.append("")
        report_lines.append("Node Type Distribution:")
        for node_type, count in sorted(all_node_types.items()):
            report_lines.append(f"   {node_type}: {count}")

    report_lines.append("")
    report_lines.append("Task Lists:")
    report_lines.append(f"   Successful tasks: {','.join(map(str, sorted(successful_task_ids)))}")
    report_lines.append(f"   Failed tasks: {','.join(map(str, sorted(failed_task_ids)))}")

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
    parser = argparse.ArgumentParser(description="Multi-agent evaluation script")
    parser.add_argument("--namespace", required=True, help="namespace (example: bird_sqlite)")
    parser.add_argument("--workdir", required=True, help="working directory")
    parser.add_argument("--agent", type=int, required=True, help="agent number (1, 2, 3, etc.)")
    parser.add_argument("--output", help="output report file")

    args = parser.parse_args()

    save_dir = os.path.join(args.workdir, "multi", f"agent{args.agent}_save")
    output_dir = os.path.join(args.workdir, "multi", f"agent{args.agent}_output")

    if not os.path.exists(save_dir):
        print(f"Error: Save directory not found: {save_dir}")
        return 1

    if not os.path.exists(output_dir):
        print(f"Warning: Output directory not found: {output_dir}")

    latest_files = get_latest_files(save_dir)

    if not latest_files:
        print(f"Warning: No YAML files found in {save_dir}")
        return 1

    print(f"Found {len(latest_files)} trajectory files for agent {args.agent}")

    analysis_results = {}
    for task_id, filepath in latest_files.items():
        print(f"Analyzing file: {os.path.basename(filepath)}")
        result = analyze_yaml_file(filepath, args.workdir, args.namespace, args.agent)
        analysis_results[task_id] = result

    generate_report(analysis_results, args.namespace, args.agent, args.output)

    return 0


if __name__ == "__main__":
    exit(main()) 