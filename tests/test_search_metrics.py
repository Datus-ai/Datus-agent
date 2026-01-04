import json

import pytest

from datus.agent.node import Node
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import Metric, SqlTask
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)


@pytest.fixture
def agent_config() -> AgentConfig:
    agent_config = load_acceptance_config(namespace="bird_school")
    agent_config.rag_base_path = "tests/data"
    return agent_config


class TestNode:
    def test_vector_and_scalar_query(self, agent_config: AgentConfig):
        sql_task = SqlTask(
            id="test_task_2",
            database_type=DBType.DUCKDB,
            task="test task 2",
            catalog_name="",
            database_name="",
            schema_name="",
            subject_path=["RGM_voice"],
        )

        node = Node.new_instance(
            node_id="search_metrics",
            description="Search Metrics",
            node_type=NodeType.TYPE_SEARCH_METRICS,
            input_data=SearchMetricsInput(
                input_text="Calculate the cancellation rate by transaction type (Quick Buy or Not Quick Buy).",
                sql_task=sql_task,
                database_type=DBType.DUCKDB,
            ),
            agent_config=agent_config,
        )
        node.run()
        print(f"result {node.result}")
        assert node.result is not None, "Expected node.result to be populated, but got None"

    def test_empty_vector_and_scalar_query(self, agent_config: AgentConfig):
        sql_task = SqlTask(
            id="test_task",
            database_type=DBType.DUCKDB,
            task="test task",
            catalog_name="",
            database_name="",
            schema_name="",
            subject_path=[],
        )

        node = Node.new_instance(
            node_id="search_metrics",
            description="Search Metrics",
            node_type=NodeType.TYPE_SEARCH_METRICS,
            input_data=SearchMetricsInput(
                # input_text="Calculate the cancellation rate by transaction type (Quick Buy or Not Quick Buy).",
                input_text="",
                sql_task=sql_task,
                database_type=DBType.DUCKDB,
            ),
            agent_config=agent_config,
        )
        node.execute()
        print(f"result {node.result}")
        assert node.result is not None, node.result is None


class TestRag:
    @pytest.fixture
    def metrics_rag(self, agent_config: AgentConfig) -> MetricRAG:
        return MetricRAG(agent_config)

    @pytest.fixture
    def semantic_rag(self, agent_config: AgentConfig) -> SemanticModelRAG:
        return SemanticModelRAG(agent_config)

    def test_pure_scalar_query(self, metrics_rag: MetricRAG, semantic_rag: SemanticModelRAG):
        semantic_rag.storage._ensure_table_ready()
        result = semantic_rag.storage.table.search().to_list()
        assert len(result) > 0
        metrics_rag.storage._ensure_table_ready()

        result = metrics_rag.storage.table.search().to_list()
        assert len(result) > 0


def test_json():
    metric = Metric(
        name="metric_name",
        description="A test metric for JSON serialization",
    )
    json_str = json.dumps(metric.__dict__)
    print(f"json:{json_str}")
    assert json.loads(json_str) == metric.__dict__
