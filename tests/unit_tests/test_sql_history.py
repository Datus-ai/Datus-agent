import json

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.sql_history import SqlHistoryRAG, sql_history_rag_by_configuration


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(namespace="bird_sqlite", benchmark="bird_dev")


@pytest.fixture
def rag_storage(agent_config: AgentConfig) -> SqlHistoryRAG:
    rag_storage = sql_history_rag_by_configuration(agent_config)
    return rag_storage


# @pytest.mark.parametrize("task_id", ["1", "2", "3", "4"])
@pytest.mark.parametrize("task_id", ["1"])
def test_search_by_task_id(task_id: str, rag_storage: SqlHistoryRAG, agent_config: AgentConfig):
    benchmark_path = agent_config.benchmark_path("bird_dev")
    with open(f"{benchmark_path}/dev.json", mode="r", encoding="utf-8") as f:
        tasks = json.load(f)
        for task in tasks:
            if str(task["question_id"]) != task_id:
                continue
            question = task["question"]
            result = rag_storage.search_sql_history_by_summary(query_text=question)
            print(result)
            assert result is not None and len(result) > 0
