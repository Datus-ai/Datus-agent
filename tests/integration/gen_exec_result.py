import argparse
import glob
import json
import os
import re
import sys

import pandas as pd
import pyarrow as pa
from utils import fix_path

from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import db_manager_instance


def resolve_env(value: str) -> str:
    """Resolve environment variables in a string"""
    if not value or not isinstance(value, str):
        return value

    pattern = r"\${([^}]+)}"

    def replace_env(match):
        env_var = match.group(1)
        return os.getenv(env_var, f"<MISSING:{env_var}>")

    return re.sub(pattern, replace_env, value)


def get_benchmark_path(config, benchmark):
    """Get benchmark path from config"""
    benchmark_config = config.get("agent", {}).get("benchmark", {})

    if benchmark not in benchmark_config:
        raise Exception(f"Benchmark '{benchmark}' not found in config")

    benchmark_path = benchmark_config[benchmark].get("benchmark_path")
    if not benchmark_path:
        raise Exception(f"benchmark_path not found in '{benchmark}'")

    return benchmark_path


def get_namespace_config(config, namespace):
    """Get namespace config from config"""
    namespace_config = config.get("agent", {}).get("namespace", {})

    if namespace not in namespace_config:
        raise Exception(f"Namespace '{namespace}' not found in config")

    return namespace_config[namespace]


def parse_dev_json(dev_json_path):
    """Parse dev.json file and return SQL statements and database mappings"""
    if not os.path.exists(dev_json_path):
        raise FileNotFoundError(f"dev.json file not found: {dev_json_path}")

    with open(dev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sql_data = []
    for item in data:
        sql_data.append({"question_id": item["question_id"], "sql": item["SQL"], "db_id": item["db_id"]})

    return sql_data


def parse_dev_sql(dev_sql_path):
    """Parse dev.sql file and return SQL statements and database mappings"""
    if not os.path.exists(dev_sql_path):
        raise FileNotFoundError(f"dev.sql file not found: {dev_sql_path}")
    sql_data = []
    with open(dev_sql_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    sql_query = parts[0].strip()
                    db_id = parts[1].strip()
                    sql_data.append({"question_id": line_no, "sql": sql_query, "db_id": db_id})
                else:
                    print(f"Warning: Line {line_no} has incorrect format, skipping")

    return sql_data


def find_sqlite_database(path_pattern, db_id):
    """Find SQLite database file based on path pattern and database ID"""
    sqlite_files = glob.glob(path_pattern, recursive=True)

    matching_files = []
    for file_path in sqlite_files:
        if file_path.endswith(f"{db_id}.sqlite"):
            matching_files.append(file_path)

    if matching_files:
        return matching_files[0]

    # Try alternative pattern
    base_path = path_pattern.split("/**")[0]
    alt_pattern = f"{base_path}/{db_id}/{db_id}.sqlite"
    alt_files = glob.glob(alt_pattern)
    if alt_files:
        return alt_files[0]

    raise FileNotFoundError(f"Database file {db_id}.sqlite not found in path: {path_pattern}")


def save_results_to_csv(results, output_path):
    """Save results to CSV file using pandas DataFrame"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if results.success and results.sql_return:
        # Create DataFrame from results
        if isinstance(results.sql_return, pd.DataFrame):
            df: pd.DataFrame = results.sql_return
        elif isinstance(results.sql_return, pa.Table):
            df: pd.DataFrame = results.sql_return.to_pandas()
        elif isinstance(results.sql_return, list):
            df: pd.DataFrame = pd.DataFrame(data=results.sql_return)
        else:
            # Empty results but successful
            print("Unsupported types:", results.sql_return)
            df = pd.DataFrame()

        # Save DataFrame to CSV
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        # Handle error cases
        if results.error:
            error_df = pd.DataFrame([["error", results["error"]]], columns=["status", "message"])
        else:
            # there are columns but no data, pandas dataframe will ignore header when comparing,
            # so column names don't matter
            error_df = pd.DataFrame(columns=["status", "message"])

        error_df.to_csv(output_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate SQL execution results")
    parser.add_argument("--namespace", required=True, help="Namespace (e.g., bird_sqlite)")
    parser.add_argument("--benchmark", required=True, help="Benchmark (e.g., bird_dev)")
    parser.add_argument("--type", required=True, help="Type (e.g., bird)")
    parser.add_argument("--workdir", required=True, help="Working directory path")
    parser.add_argument("--task-id", type=int, dest="task_id", help="Task ID (optional, process all if not specified)")
    parser.add_argument("--config", default="conf/agent.yml", help="Config file path")

    args = parser.parse_args()

    try:
        config_path = fix_path(args.workdir, args.config)

        config = load_agent_config(config=config_path, namespace=args.namespace, benchmark=args.benchmark)

        benchmark_path = config.benchmark_path(args.benchmark)

        full_benchmark_path = fix_path(args.workdir, benchmark_path)

        if args.type == "bird":
            # Try dev.json first, then fallback to dev.sql
            dev_json_path = os.path.join(full_benchmark_path, "dev.json")
            dev_sql_path = os.path.join(full_benchmark_path, "dev.sql")

            if os.path.exists(dev_json_path):
                sql_data = parse_dev_json(dev_json_path)
                print(f"Using dev.json with {len(sql_data)} questions")
            elif os.path.exists(dev_sql_path):
                sql_data = parse_dev_sql(dev_sql_path)
                print(f"Using dev.sql with {len(sql_data)} questions")
            else:
                raise FileNotFoundError("Neither dev.json nor dev.sql found")
        else:
            raise Exception(f"Unsupported type: {args.type}")

        gold_dir = os.path.join(full_benchmark_path, "gold", "exec_result")
        os.makedirs(gold_dir, exist_ok=True)
        db_manager = db_manager_instance(config.namespaces)
        if args.task_id is not None:
            # Process single task
            task_id = args.task_id
            task_data = None
            for data in sql_data:
                if data["question_id"] == task_id:
                    task_data = data
                    break

            if task_data is None:
                print(f"Error: Task ID {task_id} not found")
                return

            print(f"Processing task {task_id}: database={task_data['db_id']}")
            try:
                sql_connector = db_manager.get_conn(args.namespace, task_data["db_id"])
                print(f"Executing SQL: {task_data['sql'][:100]}...")
                results = sql_connector.execute_arrow(task_data["sql"])

                if results.success:
                    output_path = os.path.join(gold_dir, f"{task_id}.csv")
                    print(f"Task {task_id} completed, results saved to: {output_path}")
                    print(f"Returned {results.row_count} rows")
                    save_results_to_csv(results, output_path)

                else:
                    print(f"Task {task_id} failed: {results['error']}")
            except Exception as e:
                print(f"Task {task_id} processing failed: {e}")
        else:
            # Process all tasks
            print(f"Processing all {len(sql_data)} tasks...")

            for task_data in sql_data:
                task_id = task_data["question_id"]
                print(f"Processing task {task_id}/{len(sql_data)}: database={task_data['db_id']}")
                try:
                    sql_connector = db_manager.get_conn(args.namespace, task_data["db_id"])
                    results = sql_connector.execute_arrow(task_data["sql"])
                    output_path = os.path.join(gold_dir, f"{task_id}.csv")

                    if results["success"]:
                        print(f"Task {task_id} completed, returned {len(results['results'])} rows")
                        save_results_to_csv(results, output_path)

                    else:
                        print(f"Task {task_id} failed: {results['error']}")

                except Exception as e:
                    print(f"Task {task_id} processing failed: {e}")

            print("All tasks completed")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
