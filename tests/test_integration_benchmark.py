import argparse
import random
from typing import List

import pytest

from datus.agent.agent import Agent
from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import SqlTask
from datus.tools.db_tools.db_manager import DBManager, db_manager_instance
from datus.utils.benchmark_utils import load_bird_dev_tasks
from datus.utils.constants import DBType
from tests.conftest import load_acceptance_config


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_acceptance_config(namespace="bird_sqlite")


@pytest.fixture
def db_manager(agent_config: AgentConfig) -> DBManager:
    return db_manager_instance(agent_config.namespaces)


def test_benchmark_bird(agent_config: AgentConfig, db_manager: DBManager):
    args = argparse.Namespace(
        **{"components": ["metrics", "metadata", "table_lineage", "document"], "load_cp": None, "max_steps": 10}
    )
    agent = Agent(args=args, agent_config=agent_config, db_manager=db_manager)
    # random some task_ids
    tasks = load_bird_dev_tasks(agent_config.benchmark_path("bird_dev"))
    task_ids = set(generate_unique_random_numbers(10, len(tasks)))
    for task in tasks:
        task_id = task["question_id"]
        if task_id not in task_ids:
            continue
        sql_task = SqlTask(
            id=str(task_id),
            database_type=DBType.SQLITE,
            task=task["question"],
            database_name=task["db_id"],
            external_knowledge=task["evidence"],
            output_dir="tests/output/bird_dev",
        )
        agent.run(sql_task=sql_task, check_storage=True)


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
