import json

import pytest

from datus.schemas.node_models import Metric, SqlTask
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.storage.metric.store import SemanticMetricsRAG, qualify_name, rag_by_configuration
from datus.tools.metric_tools.search_metrics import SearchMetricsTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)


@pytest.fixture(scope="module")
def metrics_rag() -> SemanticMetricsRAG:
    agent_config = load_acceptance_config(namespace="bird_school")
    agent_config.rag_base_path = "tests/data"
    return rag_by_configuration(agent_config)


@pytest.fixture
def search_metrics_tool(metrics_rag: SemanticMetricsRAG) -> SearchMetricsTool:
    return SearchMetricsTool(store=metrics_rag)


@pytest.fixture
def build_empty_pure_scalar_input() -> SearchMetricsInput:
    sql_task = SqlTask(
        id="test_task",
        database_type=DBType.DUCKDB,
        task="test task",
        catalog_name="",
        database_name="",
        schema_name="",
        layer1="",
        layer2="",
        domain="",
    )

    input_param = SearchMetricsInput(
        # input_text="Calculate the cancellation rate by transaction type (Quick Buy or Not Quick Buy).",
        input_text="",
        sql_task=sql_task,
        database_type=DBType.DUCKDB,
    )

    return input_param


@pytest.fixture
def build_some_value_pure_scalar_input() -> SearchMetricsInput:
    sql_task = SqlTask(
        id="test_task_2",
        database_type=DBType.DUCKDB,
        task="test task 2",
        catalog_name="",
        database_name="",
        schema_name="",
        layer1="",
        layer2="",
        domain="RGM_voice",
    )

    input_param = SearchMetricsInput(
        input_text="Calculate the cancellation rate by transaction type (Quick Buy or Not Quick Buy).",
        sql_task=sql_task,
        database_type=DBType.DUCKDB,
    )

    return input_param


def test_vector_and_scalar_query(search_metrics_tool, build_some_value_pure_scalar_input):
    result = search_metrics_tool.execute(build_some_value_pure_scalar_input)
    print(f"result {result}")
    assert result is not None, result is None


def test_empty_vector_and_scalar_query(search_metrics_tool, build_empty_pure_scalar_input):
    result = search_metrics_tool.execute(build_empty_pure_scalar_input)
    print(f"result {result}")
    assert result is not None, result is None


def test_pure_scalar_query(search_metrics_tool):
    search_metrics_tool.store.semantic_model_storage._ensure_table_ready()
    result = (
        search_metrics_tool.store.semantic_model_storage.table.search()
        .where("catalog_database_schema like '%_%_%'")
        .to_list()
    )
    assert len(result) > 0
    search_metrics_tool.store.metric_storage._ensure_table_ready()

    result = (
        search_metrics_tool.store.metric_storage.table.search().where("domain_layer1_layer2 like '%_%_%'").to_list()
    )
    print(f"result: {result}")
    assert len(result) > 0


def test_qualify_name_with_none_or_blank():
    first = None
    second = ""
    three = "23456"
    full_name = qualify_name([first, second, three])
    assert "%_%_23456" == full_name


def test_invalid_input(search_metrics_tool):
    """Test input validation"""
    logger.info("Testing missing SearchMetricsInput parameter")

    # Test missing SearchMetricsInput
    with pytest.raises(AttributeError, match="'dict' object has no attribute 'sql_task'"):
        search_metrics_tool.execute({})


def test_json():
    metric = Metric(
        name="metric_name",
        sql_query="SELECT metric_name FROM metrics",
        description="a description of this metric",
        constraint="a constraint of this metric",
    )
    json_str = json.dumps(metric.__dict__)
    print(f"json:{json_str}")
    assert json.loads(json_str) == metric.__dict__
