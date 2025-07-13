# flake8: noqa
"""
Agent Answer Selection Tool

This script compares answers from different agents and uses a large language model
to select the best answer for each task.

Usage:
    python select_answer.py --workdir=/path/to/workdir --namespace=bird_sqlite --agent=3
    --task-id=0 --gold-path=benchmark/bird/dev_20240627/gold
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datus.configuration.agent_config_loader import load_agent_config
from datus.models.base import LLMBaseModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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


def compare_csv_results(actual_path, expected_path):
    """Use smart comparison method to compare CSV results"""
    comparison_result = {
        "match": False,
        "actual_file_exists": True,
        "expected_file_exists": True,
        "actual_shape": None,
        "expected_shape": None,
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

        # Use smart comparison method
        score = compare_pandas_table(actual_df, expected_df, ignore_order=True)
        comparison_result["match"] = score == 1

    except Exception as e:
        comparison_result["error"] = f"Comparison error: {str(e)}"

    return comparison_result


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


class AgentAnswerSelector:
    """Tool for selecting the best answer from different agents"""

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
        finally:
            os.chdir(original_cwd)

        self.model = LLMBaseModel.create_model(self.agent_config)
        print("Using Select Model:" + self.model.model_config.model)

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
        """Check which agents match with gold standard"""
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
                    agent_matches[agent_name] = False
                    logger.warning(f"Gold comparison failed for {agent_name}: {comparison_result['comparison']}")

            except Exception as e:
                logger.error(f"Error comparing {agent_name} with gold: {e}")
                agent_matches[agent_name] = False

        return agent_matches

    def truncate_sql_result(self, sql_result: str, max_length: int = 2000) -> str:
        if len(sql_result) <= max_length:
            return sql_result

        # Truncate and add ellipsis
        return sql_result[:max_length] + "\n... (Result truncated)"

    def create_comparison_prompt(self, task_id: str, agent_outputs: Dict[str, Dict]) -> str:
        if not agent_outputs:
            return ""

        first_agent = list(agent_outputs.keys())[0]
        instruction = agent_outputs[first_agent].get("instruction", "")
        database_name = agent_outputs[first_agent].get("database_name", "")

        prompt = f"""Please analyze the following task's different agent answers and select the best answer.

Task ID: {task_id}
Database: {database_name}
Task Description: {instruction}

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
1. SQL query correctness and logic
2. Execution result reasonableness
3. Whether the task was successfully completed
4. Query efficiency and code quality

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

        # Check which agents match with gold standard
        agent_gold_matches = self.check_agent_gold_matches(task_id)

        # Determine answer_found
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
            response = self.model.generate_with_json_output(prompt)

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
            return None

    def copy_best_agent_files(self, task_id: str, best_agent: str) -> tuple[Path, Path]:
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

    def save_results(self, results: List[Dict], output_file: str):
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

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

    if result:
        best_agent = result.get("best_agent", "Unknown")

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
    else:
        print(f"Failed to process task {task_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
