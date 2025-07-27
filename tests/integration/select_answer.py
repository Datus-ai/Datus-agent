# flake8: noqa
import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger
from datus.utils.path_utils import get_files_from_glob_pattern

logger = get_logger(__name__)


def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"


def compare_pandas_table(pred, gold, ignore_order=False):
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


def compare_csv_results(actual_path, expected_path):
    comparison_result = {
        "match": False,
        "actual_file_exists": True,
        "expected_file_exists": True,
        "actual_shape": None,
        "expected_shape": None,
        "error": None,
    }

    try:
        actual_df, actual_error = load_csv_data(actual_path)
        if actual_error:
            comparison_result["error"] = f"Actual file error: {actual_error}"
            comparison_result["actual_file_exists"] = False
            return comparison_result

        expected_df, expected_error = load_csv_data(expected_path)
        if expected_error:
            comparison_result["error"] = f"Expected file error: {expected_error}"
            comparison_result["expected_file_exists"] = False
            return comparison_result

        comparison_result["actual_shape"] = actual_df.shape
        comparison_result["expected_shape"] = expected_df.shape

        score = compare_pandas_table(actual_df, expected_df, ignore_order=True)
        comparison_result["match"] = score == 1

    except Exception as e:
        comparison_result["error"] = f"Comparison error: {str(e)}"

    return comparison_result


def compare_with_gold_standard(task_id, workdir, namespace, gold_path, result_dir="output"):
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


class AgentAnswerSelector:
    def __init__(self, workdir: str, namespace: str, agent_count: int, gold_path: str = None):
        self.workdir = Path(workdir)
        self.namespace = namespace
        self.agent_count = agent_count
        self.gold_path = gold_path
        self.multi_dir = self.workdir / "multi"

        config_path = self.workdir / "conf" / "agent.yml"
        original_cwd = os.getcwd()
        os.chdir(self.workdir)

        try:
            self.agent_config = load_agent_config(config=str(config_path), namespace=self.namespace)
        except Exception as e:
            os.chdir(original_cwd)
            logger.error(f"Error loading agent config: {e}")
            sys.exit(1)
        finally:
            os.chdir(original_cwd)

        try:
            self.model = LLMBaseModel.create_model(self.agent_config)
            print("Using Select Model:" + self.model.model_config.model)
        except Exception as e:
            logger.error(f"Error creating LLM model: {e}")
            sys.exit(1)

    def load_agent_outputs(self, task_id: str) -> Dict[str, Dict]:
        agent_outputs = {}

        for i in range(1, self.agent_count + 1):
            output_dir = self.multi_dir / f"agent{i}_output" / self.namespace
            json_file = output_dir / f"{task_id}.json"

            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        agent_outputs[f"agent{i}"] = data
                        logger.info(f"Loaded output for agent{i}: {json_file}")
                except Exception as e:
                    logger.error(f"Error loading output for agent{i}: {e}")
            else:
                logger.warning(f"Output file not found for agent{i}: {json_file}")

        return agent_outputs

    def check_agent_gold_matches(self, task_id: str) -> Dict[str, bool]:
        agent_matches = {}

        if not self.gold_path:
            logger.warning("Gold path not provided, skipping gold comparison")
            return agent_matches

        for i in range(1, self.agent_count + 1):
            agent_name = f"agent{i}"
            result_dir = f"multi/agent{i}_output"

            try:
                comparison_result = compare_with_gold_standard(
                    task_id, str(self.workdir), self.namespace, self.gold_path, result_dir
                )

                if comparison_result["comparison"] and not comparison_result["comparison"].get("error"):
                    agent_matches[agent_name] = comparison_result["comparison"]["match"]
                else:
                    error_msg = (
                        comparison_result["comparison"].get("error", "Unknown error")
                        if comparison_result["comparison"]
                        else "No comparison result"
                    )
                    logger.warning(f"Gold comparison failed for {agent_name}: {error_msg}")
                    agent_matches[agent_name] = False

            except Exception as e:
                logger.warning(f"Error comparing {agent_name} with gold: {e}")
                agent_matches[agent_name] = False

        return agent_matches

    def truncate_sql_result(self, sql_result: str, max_length: int = 2000) -> str:
        if len(sql_result) <= max_length:
            return sql_result

        return sql_result[:max_length] + "\n... (Result truncated)"

    def extract_tables_from_sql(self, sql: str) -> List[str]:
        if not sql:
            return []

        sql_lower = sql.lower()

        patterns = [
            r"from\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"join\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"update\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"insert\s+into\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"delete\s+from\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]

        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, sql_lower)
            tables.update(matches)

        return list(tables)

    def get_sqlite_database_path(self, database_name: str) -> str:
        if hasattr(self.agent_config, "namespaces") and self.namespace in self.agent_config.namespaces:
            namespace_configs = self.agent_config.namespaces[self.namespace]

            if len(namespace_configs) > 1:
                if database_name in namespace_configs:
                    db_config = namespace_configs[database_name]
                    if db_config.uri and db_config.uri.startswith("sqlite:///"):
                        return db_config.uri[10:]
            else:
                db_config = list(namespace_configs.values())[0]

                if hasattr(db_config, "path_pattern") and db_config.path_pattern:
                    from datus.utils.constants import DBType

                    glob_results = get_files_from_glob_pattern(db_config.path_pattern, str(DBType.SQLITE))

                    for db_path in glob_results:
                        if db_path["name"] == database_name:
                            file_path = db_path["uri"]
                            if file_path.startswith("sqlite:///"):
                                return file_path[10:]
                            elif file_path.startswith("DBType.SQLITE:///"):
                                return file_path[17:]
                            else:
                                logger.warning(f"Unknown URI format: {file_path}")
                                return ""
                elif hasattr(db_config, "uri") and db_config.uri:
                    if db_config.uri.startswith("sqlite:///"):
                        return db_config.uri[10:]

        standard_path = os.path.join(
            str(self.workdir), "benchmark/bird/dev_20240627/dev_databases", database_name, f"{database_name}.sqlite"
        )
        if os.path.exists(standard_path):
            return standard_path

        logger.warning(f"Cannot find SQLite file for database: {database_name}")
        return ""

    def get_sqlite_table_metadata(self, database_name: str) -> str:
        db_path = self.get_sqlite_database_path(database_name)
        if not db_path or not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return ""

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = cursor.fetchall()

            metadata_parts = []

            for (table_name,) in tables:
                cursor.execute(f'PRAGMA table_info("{table_name}");')
                columns = cursor.fetchall()

                if not columns:
                    continue

                table_info = f"\nTable: {table_name}\n"
                table_info += "Columns:\n"

                for cid, name, col_type, notnull, default_value, pk in columns:
                    nullable = "NOT NULL" if notnull else "NULL"
                    primary_key = "PRIMARY KEY" if pk else ""
                    default = f"DEFAULT {default_value}" if default_value is not None else ""

                    column_desc = f"  - {name}: {col_type} {nullable} {primary_key} {default}".strip()
                    table_info += column_desc + "\n"

                cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
                foreign_keys = cursor.fetchall()
                if foreign_keys:
                    table_info += "Foreign Keys:\n"
                    for fk in foreign_keys:
                        table_info += f"  - {fk[3]} -> {fk[2]}.{fk[4]}\n"

                metadata_parts.append(table_info)

            conn.close()
            return "\n".join(metadata_parts)

        except Exception as e:
            logger.warning(f"Error getting SQLite metadata for {database_name}: {e}")
            return ""

    def load_database_description(self, database_name: str) -> str:
        return self.get_sqlite_table_metadata(database_name)

    def extract_relevant_schema_info(self, sql_queries: List[str], database_metadata: str) -> str:
        if not database_metadata or not sql_queries:
            return ""

        valid_queries = [q for q in sql_queries if q.strip()]
        if not valid_queries:
            return ""

        prompt = f"""Extract and format database schema information relevant to the given SQL queries. The database metadata provided is directly from SQLite PRAGMA table_info commands.

SQL Queries:
{chr(10).join([f"- {sql}" for sql in valid_queries])}

Database Metadata (from SQLite):
{database_metadata}

Required Output Format:

## Tables Used
For each table mentioned in the SQL queries, provide:
- Table name: [name]
- Relevant columns: [list of column names used in queries]
- Column types: [column_name: data_type]
- Primary keys: [column_name] (if any)
- Constraints: [NOT NULL/NULL status]

## Join Relationships
For each join operation in the SQL queries, provide:
- Join: [table1].[column1] = [table2].[column2]
- Data types: [table1].[column1] ([type]) = [table2].[column2] ([type])
- Compatibility: [Compatible/Incompatible based on types]

## Column Details
For each column referenced in WHERE clauses, SELECT, or GROUP BY, provide:
- Column: [table].[column]
- Data type: [type]
- Constraints: [NOT NULL/NULL, PRIMARY KEY status]
- Default values: [if any]

## Foreign Key Constraints
List any foreign key relationships from the metadata:
- [table1].[column1] → [table2].[column2]

Output only factual information from the SQLite metadata. Focus on tables and columns actually used in the SQL queries.
"""

        print("------------------------\n" + prompt + "------------------------\n")
        try:
            response = self.model.generate(prompt)
            return response
        except Exception as e:
            logger.warning(f"Error extracting schema info: {e}")
            return ""

    def create_comparison_prompt(self, task_id: str, agent_outputs: Dict[str, Dict]) -> str:
        if not agent_outputs:
            return ""

        first_agent = list(agent_outputs.keys())[0]
        instruction = agent_outputs[first_agent].get("instruction", "")
        database_name = agent_outputs[first_agent].get("database_name", "")

        sql_queries = []
        for agent_name, output in agent_outputs.items():
            gen_sql_final = output.get("gen_sql_final", "")
            if gen_sql_final:
                sql_queries.append(gen_sql_final)

        database_metadata = self.load_database_description(database_name)
        relevant_schema_info = ""

        if database_metadata:
            relevant_schema_info = self.extract_relevant_schema_info(sql_queries, database_metadata)

        prompt = f"""Please analyze the following task's different agent answers and select the best answer.

Task ID: {task_id}
Database: {database_name}
Task Description: {instruction}
"""

        if relevant_schema_info:
            prompt += f"""
Relevant Database Schema Information (from SQLite metadata):
{relevant_schema_info}
"""

        prompt += """
Here are the answers from different agents:

"""

        for agent_name, output in agent_outputs.items():
            gen_sql_final = output.get("gen_sql_final", "")
            sql_result_final = output.get("sql_result_final", "")

            truncated_result = self.truncate_sql_result(sql_result_final)

            prompt += f"""
{agent_name}:
- SQL Query: {gen_sql_final}
- Execution Result: {truncated_result}
- Finished: {output.get("finished", False)}
- Row Count: {output.get("row_count", "Unknown")}

"""

        prompt += """
Please evaluate and select the best answer based on the following criteria:
1. SQL query correctness and logic (considering the database schema)
2. Execution result reasonableness
3. Whether the task was successfully completed
4. Query efficiency and code quality
5. Proper use of table relationships and constraints

Please return results in JSON format, including:
{
    "best_agent": "name of the selected best agent",
    "reason": "detailed reason for selection",
    "score_analysis": {
        "agent1": {"score": score(1-10), "reason": "scoring reason"},
        "agent2": {"score": score(1-10), "reason": "scoring reason"},
        ...
    }
}
"""

        return prompt

    def select_best_answer(self, task_id: str) -> Optional[Dict]:
        logger.info(f"Starting to process task: {task_id}")

        agent_outputs = self.load_agent_outputs(task_id)

        if not agent_outputs:
            logger.error(f"No agent outputs found for task {task_id}")
            return None

        agent_gold_matches = self.check_agent_gold_matches(task_id)

        answer_found = any(agent_gold_matches.values()) if agent_gold_matches else False

        if len(agent_outputs) == 1:
            logger.info(f"Only one agent output found for task {task_id}, returning directly")
            agent_name = list(agent_outputs.keys())[0]
            is_selected_agent_right = agent_gold_matches.get(agent_name, False)

            return {
                "task_id": task_id,
                "best_agent": agent_name,
                "reason": "Only one agent output available",
                "agent_outputs": agent_outputs,
                "score_analysis": {agent_name: {"score": 10, "reason": "Single output"}},
                "agent_gold_matches": agent_gold_matches,
                "answer_found": answer_found,
                "is_selected_agent_right": is_selected_agent_right,
            }

        prompt = self.create_comparison_prompt(task_id, agent_outputs)

        try:
            logger.info(f"Calling LLM to compare answers for task {task_id}...")
            print("------------------------\n" + prompt + "------------------------\n")
            response = self.model.generate_with_json_output(prompt)

            if not response or "best_agent" not in response:
                logger.error(f"Invalid LLM response for task {task_id}: {response}")
                best_agent = list(agent_outputs.keys())[0] if agent_outputs else "Unknown"
                response = {
                    "best_agent": best_agent,
                    "reason": "LLM response invalid, selected first available agent",
                    "score_analysis": {
                        agent: {"score": 5, "reason": "Default selection due to LLM failure"}
                        for agent in agent_outputs.keys()
                    },
                }

            best_agent = response.get("best_agent", "Unknown")
            is_selected_agent_right = agent_gold_matches.get(best_agent, False)

            result = {
                "task_id": task_id,
                "agent_outputs": agent_outputs,
                "agent_gold_matches": agent_gold_matches,
                "answer_found": answer_found,
                "is_selected_agent_right": is_selected_agent_right,
                **response,
            }

            logger.info(f"Best answer for task {task_id}: {best_agent}")
            logger.info(f"Answer found: {answer_found}")
            logger.info(f"Selected agent is right: {is_selected_agent_right}")

            return result

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            sys.exit(1)

    def copy_best_agent_files(self, task_id: str, best_agent: str) -> tuple[Path, Path]:
        try:
            best_output_dir = self.multi_dir / "best_agent_output" / self.namespace
            best_save_dir = self.multi_dir / "best_agent_save"

            best_output_dir.mkdir(parents=True, exist_ok=True)
            best_save_dir.mkdir(parents=True, exist_ok=True)

            source_output_dir = self.multi_dir / f"{best_agent}_output" / self.namespace
            for ext in [".json", ".csv", ".sql"]:
                source_file = source_output_dir / f"{task_id}{ext}"
                if source_file.exists():
                    dest_file = best_output_dir / f"{task_id}{ext}"
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"Copied {source_file} to {dest_file}")

            source_save_dir = self.multi_dir / f"{best_agent}_save"
            if source_save_dir.exists():
                for save_file in source_save_dir.glob(f"{task_id}_*.yaml"):
                    dest_file = best_save_dir / save_file.name
                    shutil.copy2(save_file, dest_file)
                    logger.info(f"Copied {save_file} to {dest_file}")

            return best_output_dir, best_save_dir
        except Exception as e:
            logger.warning(f"Error copying best agent files for task {task_id}: {e}")
            best_output_dir = self.multi_dir / "best_agent_output" / self.namespace
            best_save_dir = self.multi_dir / "best_agent_save"
            best_output_dir.mkdir(parents=True, exist_ok=True)
            best_save_dir.mkdir(parents=True, exist_ok=True)
            return best_output_dir, best_save_dir

    def save_results(self, results: List[Dict], output_file: str):
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.warning(f"Error saving results: {e}")
            backup_file = f"selection_results_backup_{task_id if 'task_id' in str(output_file) else 'unknown'}.json"
            try:
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Results saved to backup file: {backup_file}")
            except Exception as backup_e:
                logger.error(f"Failed to save backup results: {backup_e}")

    def generate_summary(self, results: List[Dict]) -> Dict:
        if not results:
            return {"total_tasks": 0, "agent_wins": {}}

        agent_wins = {}
        total_tasks = len(results)

        for result in results:
            best_agent = result.get("best_agent", "Unknown")
            agent_wins[best_agent] = agent_wins.get(best_agent, 0) + 1

        summary = {
            "total_tasks": total_tasks,
            "agent_wins": agent_wins,
            "win_rates": {agent: wins / total_tasks * 100 for agent, wins in agent_wins.items()},
        }

        return summary


def main():
    parser = argparse.ArgumentParser(description="Agent Answer Selection Tool")
    parser.add_argument("--workdir", required=True, help="Working directory path")
    parser.add_argument("--namespace", required=True, help="Dataset namespace (e.g., bird_sqlite)")
    parser.add_argument("--agent", type=int, required=True, help="Number of agents")
    parser.add_argument("--task-id", required=True, help="Task ID (required)")
    parser.add_argument("--gold-path", help="Path to gold standard files")
    parser.add_argument(
        "--output",
        default="selection_results.json",
        help="Output file name (default: selection_results_${task_id}.json)",
    )

    args = parser.parse_args()

    workdir = Path(args.workdir)
    if not workdir.exists():
        logger.error(f"Working directory does not exist: {workdir}")
        sys.exit(1)

    multi_dir = workdir / "multi"
    if not multi_dir.exists():
        logger.error(f"Multi directory does not exist: {multi_dir}")
        sys.exit(1)

    task_id = args.task_id
    gold_path = args.gold_path

    selector = AgentAnswerSelector(
        workdir=str(workdir), namespace=args.namespace, agent_count=args.agent, gold_path=gold_path
    )

    result = selector.select_best_answer(task_id)

    if not result:
        logger.error(f"Failed to process task {task_id} - no agent outputs available")
        sys.exit(1)

    best_agent = result.get("best_agent", "Unknown")
    if best_agent == "Unknown":
        logger.warning(f"No best agent identified for task {task_id}, but continuing with available results")

    best_output_dir, best_save_dir = selector.copy_best_agent_files(task_id, best_agent)

    if args.output == "selection_results.json":
        output_filename = f"selection_results_{task_id}.json"
    else:
        output_filename = args.output
    output_file = best_output_dir / output_filename
    selector.save_results([result], str(output_file))

    print(f"\n=== Task {task_id} Selection Results ===")
    print(f"Best Agent: {result.get('best_agent', 'Unknown')}")
    print(f"Selection Reason: {result.get('reason', 'Not provided')}")
    print(f"Answer Found: {result.get('answer_found', False)}")
    print(f"Selected Agent is Right: {result.get('is_selected_agent_right', False)}")

    score_analysis = result.get("score_analysis", {})
    if score_analysis:
        print("\n=== Score Analysis ===")
        for agent, analysis in score_analysis.items():
            print(f"{agent}: {analysis.get('score', 0)}/10 - {analysis.get('reason', 'No reason')}")

    agent_gold_matches = result.get("agent_gold_matches", {})
    if agent_gold_matches:
        print("\n=== Gold Standard Matches ===")
        for agent, match in agent_gold_matches.items():
            print(f"{agent}: {'✓' if match else '✗'}")

    print("\nBest agent files copied to:")
    print(f"  Output: {best_output_dir}")
    print(f"  Save: {best_save_dir}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
