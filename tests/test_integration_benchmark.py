import argparse
import random
from typing import List

import pytest

from datus.agent.agent import Agent
from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.utils.benchmark_utils import load_benchmark_tasks
from tests.conftest import load_acceptance_config


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_acceptance_config(namespace="bird_sqlite", home="~/.datus")


@pytest.fixture
def db_manager(agent_config: AgentConfig) -> DBManager:
    return db_manager_instance(agent_config.namespaces)


def test_benchmark_bird(agent_config: AgentConfig, db_manager: DBManager):
    # random some task_ids
    tasks = []
    for task in load_benchmark_tasks(agent_config, "bird_dev"):
        tasks.append(task)
    task_ids = set(generate_unique_random_numbers(30, len(tasks)))
    target_task_ids = [str(item) for item in task_ids]
    args = argparse.Namespace(
        **{
            "components": ["metrics", "metadata", "table_lineage", "document"],
            "load_cp": None,
            "max_steps": 10,
            "benchmark": "bird_dev",
            "namespace": "bird_sqlite",
            "benchmark_task_ids": target_task_ids,  # for benchmark
            "task_ids": target_task_ids,  # for evaluation
            "task_db_name": "",
            "task_schema": "",
            "metric_meta": "",
            "subject_path": "",
            "task_ext_knowledge": "",
            "current_date": None,
            "output_file": "evaluation_details.json",
        }
    )
    agent = Agent(args=args, agent_config=agent_config, db_manager=db_manager)

    agent.benchmark()

    agent.evaluation()


def generate_unique_random_numbers(count, max_val) -> List[int]:
    """
    Generate non-repeating random numbers within a specified range

    Args:
        count (int): Number of random numbers to generate
        max_val (int): Maximum value of the range of random numbers (not included)

    Returns:
        list: random numbers
    """
    return random.sample(range(0, max_val), count)
