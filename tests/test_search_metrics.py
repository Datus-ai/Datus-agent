import json

import pytest

from datus.schemas.generate_semantic_model_node_models import SemanticModelMeta
from datus.schemas.node_models import Metrics
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.storage.metric.store import qualify_name
from datus.tools.metric_tools.search_metric import SearchMetricsTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.fixture
def search_metrics_tool() -> SearchMetricsTool:
    return SearchMetricsTool()


@pytest.fixture
def build_empty_pure_scalar_input() -> SearchMetricsInput:
    semantic_model_meta = SemanticModelMeta(
        catalog_name="",
        database_name="",
        schema_name="",
        table_name="",
        layer1="",
        layer2="",
        domain="",
    )

    input_param = SearchMetricsInput(
        # input_text="Calculate the cancellation rate by transaction type (Quick Buy or Not Quick Buy).",
        input_text="",
        semantic_model_meta=semantic_model_meta,
        database_type="duckdb",
    )

    return input_param


@pytest.fixture
def build_some_value_pure_scalar_input() -> SearchMetricsInput:
    semantic_model_meta = SemanticModelMeta(
        catalog_name="",
        database_name="",
        schema_name="",
        table_name="",
        layer1="",
        layer2="",
        domain="RGM_voice",
    )

    input_param = SearchMetricsInput(
        input_text="Calculate the cancellation rate by transaction type (Quick Buy or Not Quick Buy).",
        semantic_model_meta=semantic_model_meta,
        database_type="duckdb",
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
    result = (
        search_metrics_tool.store.semantic_model_storage.table.search()
        .where("catalog_database_schema like '%.%.%'")
        .to_list()
    )
    print(f"result: {result}")
    assert len(result) > 0
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
    assert "%.%.23456" == full_name


def test_invalid_input(search_metrics_tool):
    """Test input validation"""
    logger.info("Testing missing SearchMetricsInput parameter")

    # Test missing SearchMetricsInput
    with pytest.raises(AttributeError, match="'dict' object has no attribute 'input_text'"):
        search_metrics_tool.execute({})


def test_json():
    metric = Metrics(
        metric_name="metric_name",
        metric_sql_query="SELECT metric_name FROM metrics",
        metric_value='{"name": "cancellation_rate", "owners": ["support@transformdata.io"], "type": "ratio", '
        '"type_params": {"numerator": "cancellations_usd", "denominator": "transaction_amount_usd"}}',
    )
    json_str = json.dumps(metric.__dict__)
    print(f"json:{json_str}")
    assert json.loads(json_str) == metric.__dict__
