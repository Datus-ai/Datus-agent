import argparse
import csv
import glob
import json
import os
import sqlite3
import sys

import pandas as pd
import yaml

from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.utils.constants import DBType


def load_config(config_path):
    """Load configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


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


def parse_success_story_csv(csv_path):
    """Parse success_story.csv file and return SQL statements"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"success_story.csv file not found: {csv_path}")

    sql_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, 1):
            if "sql" in row and row["sql"].strip():
                sql_data.append(
                    {
                        "question_id": line_no,
                        "sql": row["sql"].strip(),
                        "question": row.get("question", "").strip() if "question" in row else "",
                    }
                )

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


def execute_sql_query(db_path, sql_query):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(sql_query)

        column_names = [description[0] for description in cursor.description] if cursor.description else []
        results = cursor.fetchall()

        conn.close()

        return {"success": True, "columns": column_names, "results": results, "error": None}

    except Exception as e:
        return {"success": False, "columns": [], "results": [], "error": str(e)}


def execute_duckdb_query(namespace_config, sql_query):
    """Execute DuckDB query and return results"""
    try:
        db_path = namespace_config.get("uri", "")
        if not db_path:
            raise Exception("DuckDB URI not found in namespace config")

        # Create DuckDB connector
        connector = DuckdbConnector(db_path)

        # Execute query and get results as DataFrame
        result_df = connector.execute_query(sql_query)

        # Convert DataFrame to the expected format
        if result_df is not None and not result_df.empty:
            column_names = result_df.columns.tolist()
            results = result_df.values.tolist()
            return {"success": True, "columns": column_names, "results": results, "error": None}
        else:
            return {"success": True, "columns": [], "results": [], "error": None}

    except Exception as e:
        return {"success": False, "columns": [], "results": [], "error": str(e)}


def save_results_to_csv(results, output_path):
    """Save results to CSV file using pandas DataFrame"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if results["success"] and results["results"]:
        # Create DataFrame from results
        if results["columns"] and results["results"]:
            df = pd.DataFrame(results["results"], columns=results["columns"])
        elif results["results"]:
            # If no column names, create generic column names
            num_cols = len(results["results"][0]) if results["results"] else 0
            df = pd.DataFrame(results["results"], columns=[f"col_{i}" for i in range(num_cols)])
        else:
            # Empty results but successful
            df = pd.DataFrame()

        # Save DataFrame to CSV
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        # Handle error cases
        if results["error"]:
            error_df = pd.DataFrame([["error", results["error"]]], columns=["status", "message"])
        else:
            error_df = pd.DataFrame([["message", "no results"]], columns=["status", "message"])

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
        config_path = os.path.join(args.workdir, args.config)
        config = load_config(config_path)

        benchmark_path = get_benchmark_path(config, args.benchmark)
        full_benchmark_path = os.path.join(args.workdir, benchmark_path)

        namespace_config = get_namespace_config(config, args.namespace)

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
        elif args.type == "semantic_layer":
            # Parse success_story.csv file
            success_story_path = os.path.join(full_benchmark_path, "testing_set.csv")
            if not os.path.exists(success_story_path):
                raise FileNotFoundError(f"success_story.csv file not found: {success_story_path}")

            sql_data = parse_success_story_csv(success_story_path)
            print(f"Using success_story.csv with {len(sql_data)} questions")
        else:
            raise Exception(f"Unsupported type: {args.type}")

        gold_dir = os.path.join(full_benchmark_path, "gold", "exec_result")
        os.makedirs(gold_dir, exist_ok=True)

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

            if args.type == "semantic_layer":
                print(f"Processing task {task_id}")
            else:
                print(f"Processing task {task_id}: database={task_data['db_id']}")

            if namespace_config.get("type") == DBType.SQLITE:
                path_pattern = namespace_config.get("path_pattern", "")
                full_path_pattern = os.path.join(args.workdir, path_pattern)
                db_path = find_sqlite_database(full_path_pattern, task_data["db_id"])
                print(f"Executing SQL: {task_data['sql'][:100]}...")
                results = execute_sql_query(db_path, task_data["sql"])
            elif namespace_config.get("type") == DBType.DUCKDB:
                print(f"Executing SQL: {task_data['sql'][:100]}...")
                results = execute_duckdb_query(namespace_config, task_data["sql"])
            else:
                raise Exception(f"Unsupported database type: {namespace_config.get('type')}")

            output_path = os.path.join(gold_dir, f"{task_id}.csv")
            save_results_to_csv(results, output_path)

            if results["success"]:
                print(f"Task {task_id} completed, results saved to: {output_path}")
                print(f"Returned {len(results['results'])} rows")
            else:
                print(f"Task {task_id} failed: {results['error']}")
        else:
            # Process all tasks
            print(f"Processing all {len(sql_data)} tasks...")

            for task_data in sql_data:
                task_id = task_data["question_id"]
                if args.type == "semantic_layer":
                    print(f"Processing task {task_id}/{len(sql_data)}")
                else:
                    print(f"Processing task {task_id}/{len(sql_data)}: database={task_data['db_id']}")

                try:
                    if namespace_config.get("type") == DBType.SQLITE:
                        path_pattern = namespace_config.get("path_pattern", "")
                        full_path_pattern = os.path.join(args.workdir, path_pattern)
                        db_path = find_sqlite_database(full_path_pattern, task_data["db_id"])
                        results = execute_sql_query(db_path, task_data["sql"])
                    elif namespace_config.get("type") == DBType.DUCKDB:
                        results = execute_duckdb_query(namespace_config, task_data["sql"])
                    else:
                        raise Exception(f"Unsupported database type: {namespace_config.get('type')}")

                    output_path = os.path.join(gold_dir, f"{task_id}.csv")
                    save_results_to_csv(results, output_path)

                    if results["success"]:
                        print(f"Task {task_id} completed, returned {len(results['results'])} rows")
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
