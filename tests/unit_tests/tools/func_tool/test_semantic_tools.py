"""
Test cases for SemanticTools utility functions and query_metrics compression.
"""

import json
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.attribution_utils import (
    AttributionValidationErrorPayload,
    AttributionValidationException,
)
from datus.tools.func_tool.base import FuncToolResult, normalize_null, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.semantic_tools import _run_async
from datus.tools.semantic_tools.models import QueryResult, ValidationResult


class _Severity(Enum):
    ERROR = "error"


class TestSemanticToolsGenerationEvidence:
    def test_missing_success_key_is_not_success(self):
        evidence = GenerationEvidence()

        evidence.record_validation_result({"result": {"valid": True, "issues": []}})
        evidence.record_metric_dry_run(["revenue"], {"result": {"metadata": {"sql": "SELECT 1"}}})

        assert evidence.validation_passed is False
        assert evidence.metric_dry_run_passed is False
        assert evidence.metric_sqls == {}

    def test_attr_payload_metadata_is_recorded(self):
        evidence = GenerationEvidence()
        payload = Mock()
        payload.metadata = {"sql": "SELECT 1"}
        result = FuncToolResult(success=1, result=payload)

        evidence.record_metric_dry_run(["revenue"], result)

        assert evidence.metric_dry_run_passed is True
        assert evidence.metric_sqls == {"revenue": "SELECT 1"}

    def test_single_sql_fallback_not_fanned_out_to_multiple_metrics(self):
        evidence = GenerationEvidence()
        result = FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}})

        evidence.record_metric_dry_run(["revenue", "cost"], result)

        assert evidence.metric_dry_run_passed is True
        assert evidence.metric_sqls == {"__query_metrics_dry_run__": "SELECT 1"}
        assert evidence.has_metric_dry_run(["revenue", "cost"]) is True

    def test_dry_run_success_without_sql_metadata_records_coverage(self):
        evidence = GenerationEvidence()
        result = FuncToolResult(success=1, result={"metadata": {}})

        evidence.record_metric_dry_run(["revenue"], result)

        assert evidence.metric_dry_run_passed is True
        assert evidence.metric_sqls == {}
        assert evidence.has_metric_dry_run(["revenue"]) is True

    def test_queryability_contract_requires_one_complete_grouped_dry_run(self):
        evidence = GenerationEvidence()
        evidence.set_metric_queryability_contracts(
            [
                {
                    "contract_id": "orders:group_1",
                    "source_id": "orders",
                    "metric_output_ids": ["orders:revenue", "orders:count"],
                    "dimensions": ["orders.order_date", "orders.region"],
                    "time_grain": "month",
                }
            ]
        )
        evidence.bind_metric_output_names(
            [
                {"output_id": "orders:revenue", "metric_name": "revenue_total"},
                {"output_id": "orders:count", "metric_name": "order_count"},
            ]
        )
        result = FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}})

        evidence.record_metric_dry_run(
            ["revenue_total"],
            result,
            dimensions=["orders.order_date", "orders.region"],
            time_granularity="month",
        )
        evidence.record_metric_dry_run(
            ["order_count"],
            result,
            dimensions=["orders.order_date", "orders.region"],
            time_granularity="month",
        )

        assert evidence.has_required_queryability_dry_runs(["revenue_total", "order_count"]) is False

        evidence.record_metric_dry_run(
            ["revenue_total", "order_count"],
            result,
            dimensions=["orders.order_date", "orders.region"],
            time_granularity="month",
        )

        assert evidence.has_required_queryability_dry_runs(["revenue_total", "order_count"]) is True


class TestNormalizeNull:
    """Tests for normalize_null utility function."""

    @pytest.mark.parametrize(
        "value",
        [None, "null", "None", "NULL", "Null", "NONE", "none", "", "  ", "\t"],
    )
    def test_null_variants_return_none(self, value):
        assert normalize_null(value) is None

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("2024-01-01", "2024-01-01"),
            ("hello", "hello"),
            (42, 42),
            (0, 0),
        ],
    )
    def test_valid_value_passes_through(self, value, expected):
        assert normalize_null(value) == expected


@pytest.fixture
def semantic_tools():
    """Create a SemanticTools instance with mocked dependencies."""
    with (
        patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-4o"
        mock_config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type
        mock_config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": "ns1"}
        tool = SemanticTools(agent_config=mock_config, adapter_type="mock_adapter")
        return tool


@pytest.fixture
def mock_adapter(semantic_tools):
    """Set up a mock adapter on the SemanticTools instance."""
    adapter = Mock()
    semantic_tools._adapter = adapter
    return adapter


@pytest.mark.usefixtures("mock_adapter")
class TestQueryMetricsCompression:
    """Test cases for query_metrics with DataCompressor integration."""

    def test_tool_schema_exposes_osi_half_open_time_range(self, semantic_tools):
        schema = trans_to_function_tool(semantic_tools.query_metrics).params_json_schema

        start_description = schema["properties"]["time_start"]["description"].lower()
        end_description = schema["properties"]["time_end"]["description"].lower()
        assert "inclusive" in start_description
        assert "exclusive" in end_description
        assert "2024-02-01" in end_description

    def test_query_metrics_success_with_compression(self, semantic_tools, mock_adapter):
        """Test that query_metrics returns compressed data on success."""
        query_result = QueryResult(
            columns=["date", "revenue", "orders"],
            data=[
                {"date": "2024-01-01", "revenue": 1000, "orders": 50},
                {"date": "2024-01-02", "revenue": 1200, "orders": 60},
            ],
            metadata={"execution_time": 0.5},
        )
        mock_adapter.query_metrics = Mock(return_value=query_result)

        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            return_value=query_result,
        ):
            result = semantic_tools.query_metrics(
                metrics=["revenue", "orders"],
                dimensions=["date"],
            )

        assert isinstance(result, FuncToolResult)
        assert result.success == 1
        assert result.error is None

        # Verify result structure contains compression metadata
        result_dict = result.result
        assert "columns" in result_dict
        assert "data" in result_dict
        assert "metadata" in result_dict
        assert result_dict["result_id"] == result_dict["metadata"]["_full_result_cache_key"]

        # Verify data is now a compressed dict (not raw list)
        compressed_data = result_dict["data"]
        assert isinstance(compressed_data, dict)
        assert "original_rows" in compressed_data
        assert "original_columns" in compressed_data
        assert "is_compressed" in compressed_data
        assert "compressed_data" in compressed_data
        assert "removed_columns" in compressed_data
        assert "compression_type" in compressed_data

        # Verify metadata is preserved
        assert result_dict["columns"] == ["date", "revenue", "orders"]
        assert result_dict["metadata"]["execution_time"] == 0.5
        assert result_dict["metadata"]["_full_result_cache_key"]
        assert result_dict["metadata"]["_full_result_cached"] is True
        assert result_dict["metadata"]["_full_result_row_count"] == 2
        assert "complete uncompressed query result is cached" in result_dict["metadata"]["_full_result_note"]

    def test_query_metrics_small_data_not_compressed(self, semantic_tools):
        """Test that small data within token threshold is not compressed."""
        query_result = QueryResult(
            columns=["id", "value"],
            data=[
                {"id": 1, "value": 100},
                {"id": 2, "value": 200},
            ],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["value"])

        compressed_data = result.result["data"]
        assert compressed_data["original_rows"] == 2
        assert compressed_data["is_compressed"] is False
        assert compressed_data["compression_type"] == "none"

    def test_query_metrics_large_data_row_compressed(self, semantic_tools):
        """Test that data exceeding 20 rows triggers row compression."""
        rows = [{"id": i, "value": i * 100} for i in range(50)]
        query_result = QueryResult(
            columns=["id", "value"],
            data=rows,
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["value"])

        compressed_data = result.result["data"]
        assert compressed_data["original_rows"] == 50
        assert compressed_data["is_compressed"] is True
        assert compressed_data["compression_type"] in ("rows", "rows_and_columns")

        cache_key = result.result["metadata"]["_full_result_cache_key"]
        assert result.result["result_id"] == cache_key
        cached_result = semantic_tools.get_cached_query_metrics_result(cache_key)
        assert cached_result["row_count"] == 50
        assert result.result["metadata"]["_full_result_row_count"] == 50
        assert "10,1000" in cached_result["csv"]
        assert "49,4900" in cached_result["csv"]
        assert "..." not in cached_result["csv"]

    def test_query_metrics_full_result_cache_is_bounded(self, semantic_tools):
        semantic_tools.MAX_QUERY_METRICS_RESULT_CACHE_SIZE = 2

        first = semantic_tools._cache_query_metrics_result(["id"], [{"id": 1}])
        second = semantic_tools._cache_query_metrics_result(["id"], [{"id": 2}])
        third = semantic_tools._cache_query_metrics_result(["id"], [{"id": 3}])

        assert semantic_tools.get_cached_query_metrics_result(first) is None
        assert semantic_tools.get_cached_query_metrics_result(second)["row_count"] == 1
        assert semantic_tools.get_cached_query_metrics_result(third)["row_count"] == 1

    def test_query_metrics_result_cache_helpers_handle_supported_data_shapes(self, semantic_tools):
        class NumRows:
            num_rows = "7"

        class BadNumRows:
            num_rows = "bad"

        class ShapeRows:
            shape = (3, 2)

        class BadShapeRows:
            shape = ()

        class CsvLike:
            def to_csv(self, index=False):
                assert index is False
                return "x\n1\n"

        class ToPandasLike:
            def to_pandas(self):
                return CsvLike()

        assert semantic_tools._query_data_row_count(None) == 0
        assert semantic_tools._query_data_row_count(NumRows()) == 7
        assert semantic_tools._query_data_row_count(BadNumRows()) == 0
        assert semantic_tools._query_data_row_count(ShapeRows()) == 3
        assert semantic_tools._query_data_row_count(BadShapeRows()) == 0
        assert semantic_tools._query_data_row_count(1) == 0

        assert semantic_tools._query_data_to_csv(["x"], None) == ""
        assert semantic_tools._query_data_to_csv(["x"], ToPandasLike()) == "x\n1\n"
        assert semantic_tools._query_data_to_csv(["x"], [{"x": 1}, (2,), 3]) == "x\r\n1\r\n2\r\n3\r\n"
        assert semantic_tools._cache_query_metrics_result(["x"], None) is None
        with patch.object(semantic_tools, "_query_data_to_csv", return_value=""):
            assert semantic_tools._cache_query_metrics_result(["x"], [{"x": 1}]) is None

    def test_query_metrics_empty_data(self, semantic_tools):
        """Test query_metrics with empty result set."""
        query_result = QueryResult(
            columns=[],
            data=[],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["value"])

        compressed_data = result.result["data"]
        assert compressed_data["original_rows"] == 0
        assert compressed_data["is_compressed"] is False
        assert compressed_data["compression_type"] == "none"

    def test_query_metrics_no_adapter(self, semantic_tools):
        """Test query_metrics returns error when no adapter is configured."""
        semantic_tools._adapter = None
        semantic_tools.adapter_type = None

        result = semantic_tools.query_metrics(metrics=["revenue"])

        assert result.success == 0
        assert "adapter" in result.error.lower()

    @pytest.mark.parametrize("metrics", [[], ["null", "", None], ""])
    def test_query_metrics_rejects_empty_metrics_before_adapter_call(self, semantic_tools, mock_adapter, metrics):
        """MetricFlow otherwise raises a cryptic ComputeMetricsNode assertion."""
        result = semantic_tools.query_metrics(metrics=metrics)

        assert result.success == 0
        assert "at least one metric name" in result.error
        mock_adapter.query_metrics.assert_not_called()

    def test_query_metrics_normalizes_string_arguments(self, semantic_tools, mock_adapter):
        """LLM tool calls may send a single string even when the schema says list."""
        query_result = QueryResult(columns=["revenue"], data=[{"revenue": 10}], metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics="revenue",
                dimensions="metric_time__day",
                path="Finance",
                order_by="-revenue",
            )

        assert result.success == 1
        mock_adapter.query_metrics.assert_called_once_with(
            metrics=["revenue"],
            dimensions=["metric_time__day"],
            path=["Finance"],
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=["-revenue"],
            dry_run=False,
        )

    def test_query_metrics_runs_warehouse_dry_run_for_compiled_sql(self, semantic_tools, mock_adapter):
        query_result = QueryResult(
            columns=["sql"],
            data=[{"sql": "SELECT COUNT(*) FROM orders"}],
            metadata={"sql": "SELECT COUNT(*) FROM orders"},
        )
        calls = []
        semantic_tools._warehouse_dry_run_provider = lambda sql: (
            calls.append(sql) or {"status": "success", "datasource": "warehouse"}
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["order_count"], dry_run=True)

        assert result.success == 1
        assert calls == ["SELECT COUNT(*) FROM orders"]
        assert result.result["metadata"]["warehouse_dry_run"] == {
            "status": "success",
            "datasource": "warehouse",
        }

    def test_query_metrics_returns_failure_when_warehouse_dry_run_fails(self, semantic_tools, mock_adapter):
        query_result = QueryResult(
            columns=["sql"],
            data=[{"sql": "SELECT COUNT(*) FROM missing_orders"}],
            metadata={"sql": "SELECT COUNT(*) FROM missing_orders"},
        )
        semantic_tools._warehouse_dry_run_provider = lambda _sql: {
            "status": "failed",
            "error": "table not found",
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["order_count"], dry_run=True)

        assert result.success == 0
        assert result.error == "Warehouse dry-run failed: table not found"
        assert result.result["metadata"]["warehouse_dry_run"]["status"] == "failed"

    def test_query_metrics_delegates_dimension_validation_to_adapter(self, semantic_tools, mock_adapter):
        """Dimension metadata is advisory; the backend validates the requested query."""
        query_result = QueryResult(
            columns=["supplier_nation", "discount_rate"],
            data=[{"supplier_nation": "CN", "discount_rate": 0.1}],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["shipped_revenue", "discount_rate"],
                dimensions=["supplier_nation"],
            )

        assert result.success == 1
        mock_adapter.get_dimensions.assert_not_called()
        mock_adapter.query_metrics.assert_called_once_with(
            metrics=["shipped_revenue", "discount_rate"],
            dimensions=["supplier_nation"],
            path=None,
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_delegates_time_granularity_validation_to_adapter(self, semantic_tools, mock_adapter):
        """Advertised grains are hints; the adapter validates explicit requests."""
        query_result = QueryResult(
            columns=["order_date__month", "orders"],
            data=[{"order_date__month": "2024-01-01", "orders": 10}],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["orders"],
                dimensions=["order_date"],
                time_granularity="month",
            )

        assert result.success == 1
        mock_adapter.get_dimensions.assert_not_called()
        mock_adapter.query_metrics.assert_called_once_with(
            metrics=["orders"],
            dimensions=["order_date"],
            path=None,
            time_start=None,
            time_end=None,
            time_granularity="month",
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_adapter_exception(self, semantic_tools):
        """Test query_metrics handles adapter exceptions gracefully."""
        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=Exception("Connection timeout"),
        ):
            result = semantic_tools.query_metrics(metrics=["revenue"])

        assert result.success == 0
        assert "Connection timeout" in result.error

    def test_query_metrics_preserves_columns_and_metadata(self, semantic_tools):
        """Test that columns and metadata are preserved unchanged after compression."""
        query_result = QueryResult(
            columns=["metric_time__day", "revenue", "cost"],
            data=[{"metric_time__day": "2024-01-01", "revenue": 500, "cost": 200}],
            metadata={"sql": "SELECT ...", "row_count": 1},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["revenue", "cost"],
                dimensions=["metric_time__day"],
            )

        assert result.result["columns"] == ["metric_time__day", "revenue", "cost"]
        assert result.result["metadata"]["sql"] == "SELECT ..."
        assert result.result["metadata"]["row_count"] == 1
        assert result.result["metadata"]["_full_result_cache_key"]

    def test_query_metrics_dry_run_records_generation_evidence(self, semantic_tools):
        """Successful dry-run evidence gates metric publishing."""
        evidence = GenerationEvidence()
        semantic_tools.generation_evidence = evidence
        query_result = QueryResult(
            columns=[],
            data=[],
            metadata={"sql": "SELECT SUM(revenue) AS revenue FROM orders"},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["revenue"],
                dimensions=["customer_segment"],
                time_granularity="month",
                dry_run=True,
            )

        assert result.success == 1
        assert evidence.metric_dry_run_passed is True
        assert len(evidence.metric_dry_run_queries) == 1
        dry_run = evidence.metric_dry_run_queries[0]
        assert dry_run["metrics"] == ["revenue"]
        assert dry_run["dimensions"] == ["customer_segment"]
        assert dry_run["time_granularity"] == "month"
        assert dry_run["time_granularity_explicit"] is True
        assert dry_run["sql"] == "SELECT SUM(revenue) AS revenue FROM orders"
        assert evidence.metric_sqls == {"revenue": "SELECT SUM(revenue) AS revenue FROM orders"}

    def test_query_metrics_non_dry_run_does_not_record_publish_evidence(self, semantic_tools):
        evidence = GenerationEvidence()
        semantic_tools.generation_evidence = evidence
        query_result = QueryResult(columns=[], data=[], metadata={"sql": "SELECT 1"})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"], dry_run=False)

        assert result.success == 1
        assert evidence.metric_dry_run_passed is False
        assert evidence.metric_sqls == {}

    def test_query_metrics_drops_non_serializable_metadata(self, semantic_tools):
        """Test that non-JSON-serializable metadata values are dropped."""

        class FakePlan:
            def __str__(self):
                return "<FakePlan: node1 -> node2>"

        query_result = QueryResult(
            columns=["revenue"],
            data=[{"revenue": 100}],
            metadata={"dataflow_plan": FakePlan(), "sql": "SELECT 1", "count": 42},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"])

        assert result.success == 1
        meta = result.result["metadata"]
        # Non-serializable entries are dropped; serializable ones pass through.
        assert "dataflow_plan" not in meta
        assert meta["sql"] == "SELECT 1"
        assert meta["count"] == 42

    def test_query_metrics_compressed_data_contains_original_columns(self, semantic_tools):
        """Test that compressed result includes original column names."""
        query_result = QueryResult(
            columns=["date", "revenue", "orders", "customers"],
            data=[
                {"date": "2024-01-01", "revenue": 1000, "orders": 50, "customers": 30},
            ],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"])

        compressed_data = result.result["data"]
        assert set(compressed_data["original_columns"]) == {"date", "revenue", "orders", "customers"}

    def test_query_metrics_passes_all_parameters(self, semantic_tools, mock_adapter):
        """Test that all parameters are correctly passed to the adapter."""
        query_result = QueryResult(columns=["x"], data=[{"x": 1}], metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["revenue"],
                dimensions=["region"],
                path=["Finance"],
                time_start="2024-01-01",
                time_end="2024-02-01",
                time_granularity="day",
                where="region = 'US'",
                limit=100,
                order_by=["-revenue"],
                dry_run=True,
            )

            # Verify adapter.query_metrics was called with correct parameters
            mock_adapter.query_metrics.assert_called_once_with(
                metrics=["revenue"],
                dimensions=["region"],
                path=["Finance"],
                time_start="2024-01-01",
                time_end="2024-02-01",
                time_granularity="day",
                where="region = 'US'",
                limit=100,
                order_by=["-revenue"],
                dry_run=True,
            )

            # Verify result is successful with compressed data
            assert result.success == 1
            assert result.result["data"]["original_rows"] == 1
            assert result.result["data"]["original_columns"] == ["x"]

    def test_query_metrics_preserves_join_filtered_rows_metadata(self, semantic_tools):
        """Adapter-reported unmatched-row counts must reach the tool result metadata.

        The ask_metrics prompt instructs the model to disclose
        `join_policy_filtered_rows` to the user; this protects that contract.
        """
        query_result = QueryResult(
            columns=["x"],
            data=[{"x": 1}],
            metadata={"join_policy": "match_only", "join_policy_filtered_rows": 3},
        )

        class Adapter:
            def get_dimensions(self, metric_name, path=None):
                return []

            def query_metrics(self, metrics, **kwargs):
                return query_result

        semantic_tools._adapter = Adapter()
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["order_count"])

        assert result.success == 1
        assert result.result["metadata"]["join_policy"] == "match_only"
        assert result.result["metadata"]["join_policy_filtered_rows"] == 3

    def test_metric_datasets_maps_names_from_catalog_metadata(self, semantic_tools):
        """The adapter reports which datasets each metric reads."""
        metrics = [
            SimpleNamespace(name="revenue", metadata={"datasets": ["orders"]}),
            SimpleNamespace(name="signups", metadata={"datasets": ["users"]}),
        ]
        semantic_tools._adapter = SimpleNamespace(list_metrics=lambda **kwargs: metrics)
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=metrics):
            assert semantic_tools.metric_datasets() == {"revenue": ["orders"], "signups": ["users"]}

    def test_metric_datasets_reads_every_page(self, semantic_tools):
        """A truncated read would hide metrics from a policy that scopes by dataset."""
        from datus.tools.func_tool.semantic_tools import _METRIC_DATASETS_PAGE_SIZE

        first = [
            SimpleNamespace(name=f"m{i}", metadata={"datasets": ["orders"]}) for i in range(_METRIC_DATASETS_PAGE_SIZE)
        ]
        second = [SimpleNamespace(name="tail", metadata={"datasets": ["users"]})]
        offsets = []

        def list_metrics(limit, offset):
            offsets.append(offset)
            return first if offset == 0 else second

        semantic_tools._adapter = SimpleNamespace(list_metrics=list_metrics)
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            mapping = semantic_tools.metric_datasets()

        assert offsets == [0, _METRIC_DATASETS_PAGE_SIZE]
        assert mapping["tail"] == ["users"]
        assert len(mapping) == _METRIC_DATASETS_PAGE_SIZE + 1

    def test_metric_datasets_gives_up_on_an_adapter_that_ignores_offset(self, semantic_tools):
        """An incomplete map must not look like a complete one."""
        from datus.tools.func_tool.semantic_tools import _METRIC_DATASETS_PAGE_SIZE

        page = [
            SimpleNamespace(name=f"m{i}", metadata={"datasets": ["orders"]}) for i in range(_METRIC_DATASETS_PAGE_SIZE)
        ]
        semantic_tools._adapter = SimpleNamespace(list_metrics=lambda limit, offset: page)
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() is None

    def test_metric_datasets_keeps_metrics_without_dataset_information(self, semantic_tools):
        """A metric reported without dataset information maps to an empty list."""
        metrics = [SimpleNamespace(name="orphan", metadata={})]
        semantic_tools._adapter = SimpleNamespace(list_metrics=lambda **kwargs: metrics)
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=metrics):
            assert semantic_tools.metric_datasets() == {"orphan": []}

    def test_metric_datasets_without_an_adapter_is_none(self, semantic_tools):
        """An unavailable catalog is distinct from a readable empty one."""
        with patch.object(type(semantic_tools), "adapter", property(lambda self: None)):
            assert semantic_tools.metric_datasets() is None

    def test_metric_datasets_prefers_the_adapter_lightweight_accessor(self, semantic_tools):
        """Adapters may skip building a MetricDefinition per metric."""

        def fail_list_metrics(**kwargs):
            raise AssertionError("should not fall back to list_metrics")

        semantic_tools._adapter = SimpleNamespace(
            metric_datasets=lambda: {"revenue": ["orders"]},
            list_metrics=fail_list_metrics,
        )
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() == {"revenue": ["orders"]}


# ---------------------------------------------------------------------------
# Extended fixtures (no adapter_type)
# ---------------------------------------------------------------------------


@pytest.fixture
def semantic_tools_ext():
    """Create a SemanticTools instance WITHOUT adapter_type (for tests that require no adapter)."""
    with (
        patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        config = Mock()
        config.active_model.return_value.model = "gpt-4o"
        config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type
        config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": "ns1"}
        tool = SemanticTools(agent_config=config)
        return tool


@pytest.fixture
def semantic_tools_with_adapter():
    with (
        patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        config = Mock()
        config.active_model.return_value.model = "gpt-4o"
        config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type
        config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": "ns1"}
        tool = SemanticTools(agent_config=config, adapter_type="metricflow")
        mock_adapter = Mock()
        tool._adapter = mock_adapter
        return tool, mock_adapter


# ---------------------------------------------------------------------------
# Extended tests
# ---------------------------------------------------------------------------


class TestRunAsync:
    def test_delegates_to_run_async_utility(self):
        mock_coro = Mock()
        with patch("datus.utils.async_utils.run_async", return_value="result") as mock_run:
            result = _run_async(mock_coro)
        mock_run.assert_called_once_with(mock_coro)
        assert result == "result"


class TestAllToolsName:
    def test_returns_expected_names(self):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        names = SemanticTools.all_tools_name()
        assert "list_metrics" in names
        assert "get_dimensions" in names
        assert "query_metrics" in names
        assert "validate_semantic" in names
        assert "attribution_analyze" in names


class TestAvailableTools:
    def test_no_adapter_returns_no_tools(self, semantic_tools_ext):
        with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = semantic_tools_ext.available_tools()
        assert tools == []

    def test_default_metricflow_adapter_does_not_load_during_tool_registration(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                side_effect=RuntimeError("adapter unavailable"),
            ),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type or "metricflow"
            config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": "ns1"}
            tool = SemanticTools(agent_config=config)

            with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:

                def _mock_tool(func):
                    tool = Mock()
                    tool.name = func.__name__
                    return tool

                mock_trans.side_effect = _mock_tool
                tools = tool.available_tools()

        names = [tool.name for tool in tools]
        assert names == [
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "validate_semantic",
            "attribution_analyze",
        ]

    def test_with_adapter_adds_validate_and_attribution_tools(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            tool = SemanticTools(agent_config=config)
            tool._adapter = Mock()  # Set adapter (also enables attribution_tool)

            with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:
                mock_trans.side_effect = lambda f: Mock(name=f.__name__)
                tools = tool.available_tools()
        # 3 base + validate_semantic + attribution_analyze (both enabled when adapter is set)
        assert len(tools) == 5

    def test_configured_adapter_load_failure_is_reported_when_tool_runs(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                side_effect=RuntimeError("bad yaml"),
            ),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type
            config.build_semantic_adapter_config.side_effect = lambda adapter_type=None: {"datasource": "ns1"}
            tool = SemanticTools(agent_config=config, adapter_type="metricflow")

            with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:

                def _mock_tool(func):
                    tool = Mock()
                    tool.name = func.__name__
                    return tool

                mock_trans.side_effect = _mock_tool
                tools = tool.available_tools()

            names = [tool.name for tool in tools]
            assert names == [
                "list_metrics",
                "get_dimensions",
                "query_metrics",
                "validate_semantic",
                "attribution_analyze",
            ]

            result = tool.validate_semantic()
            assert result.success == 0
            assert "bad yaml" in result.error


class TestRuntimeDbContext:
    def test_normalize_runtime_context_handles_empty_and_aliases(self):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        assert SemanticTools._normalize_runtime_db_context(None) == {}
        assert SemanticTools._normalize_runtime_db_context(
            {
                "catalog_name": " runtime_catalog ",
                "database_name": " runtime_db ",
                "db_schema": " runtime_schema ",
            }
        ) == {
            "catalog_name": "runtime_catalog",
            "catalog": "runtime_catalog",
            "database_name": "runtime_db",
            "database": "runtime_db",
            "db_schema": "runtime_schema",
            "schema": "runtime_schema",
        }
        assert SemanticTools._normalize_runtime_db_context({"schema_name": "runtime_schema"}) == {
            "schema_name": "runtime_schema",
            "schema": "runtime_schema",
        }

    def test_adapter_config_receives_runtime_context_and_reloads_when_it_changes(self):
        runtime_context = {"datasource": "college_exam", "database": "db_one"}
        captured_builder_calls = []
        adapter_config = object()
        adapter_one = Mock()
        adapter_two = Mock()

        def build_config(adapter_type=None, database_name=None, runtime_db_context=None):
            captured_builder_calls.append(
                {
                    "adapter_type": adapter_type,
                    "database_name": database_name,
                    "runtime_db_context": dict(runtime_db_context or {}),
                }
            )
            return adapter_config

        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry.get_metadata", return_value=None),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                side_effect=[adapter_one, adapter_two],
            ) as create_adapter,
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.current_datasource = "college_exam"
            config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type
            config.build_semantic_adapter_config = build_config
            tool = SemanticTools(
                agent_config=config,
                adapter_type="metricflow",
                runtime_db_context_provider=lambda: runtime_context,
            )

            assert tool.adapter is adapter_one
            assert tool.adapter is adapter_one
            runtime_context["database"] = "db_two"
            assert tool.adapter is adapter_two

        assert create_adapter.call_count == 2
        assert captured_builder_calls == [
            {
                "adapter_type": "metricflow",
                "database_name": "college_exam",
                "runtime_db_context": {"datasource": "college_exam", "database": "db_one"},
            },
            {
                "adapter_type": "metricflow",
                "database_name": "college_exam",
                "runtime_db_context": {"datasource": "college_exam", "database": "db_two"},
            },
        ]

    def test_adapter_config_uses_agent_config_runtime_context_when_provider_absent(self):
        captured_builder_calls = []
        adapter_config = object()
        adapter = Mock()

        def build_config(adapter_type=None, database_name=None, runtime_db_context=None):
            captured_builder_calls.append(
                {
                    "adapter_type": adapter_type,
                    "database_name": database_name,
                    "runtime_db_context": dict(runtime_db_context or {}),
                }
            )
            return adapter_config

        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry.get_metadata", return_value=None),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                return_value=adapter,
            ) as create_adapter,
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.current_datasource = "static_ds"
            config.runtime_db_context.return_value = {
                "datasource": "runtime_ds",
                "database": "agent_ctx_db",
            }
            config.resolve_semantic_adapter.side_effect = lambda adapter_type=None: adapter_type
            config.build_semantic_adapter_config = build_config

            tool = SemanticTools(agent_config=config, adapter_type="metricflow")

            assert tool.adapter is adapter

        assert create_adapter.call_count == 1
        assert captured_builder_calls == [
            {
                "adapter_type": "metricflow",
                "database_name": "runtime_ds",
                "runtime_db_context": {"datasource": "runtime_ds", "database": "agent_ctx_db"},
            }
        ]

    def test_runtime_context_provider_failure_returns_empty_context(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            tool = SemanticTools(
                agent_config=config,
                adapter_type="metricflow",
                runtime_db_context_provider=Mock(side_effect=RuntimeError("boom")),
            )

        assert tool._runtime_db_context() == {}

    def test_agent_config_runtime_context_failure_returns_empty_context(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.runtime_db_context.side_effect = RuntimeError("boom")
            tool = SemanticTools(agent_config=config, adapter_type="metricflow")

        assert tool._runtime_db_context() == {}

    def test_static_runtime_context_overrides_agent_config_context_and_is_idempotent(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.runtime_db_context.return_value = {"database": "agent_db"}
            tool = SemanticTools(agent_config=config, adapter_type="metricflow")

        tool._adapter = Mock()
        tool._attribution_tool = Mock()
        tool._adapter_context_key = ("metricflow", "ds", "", "", "")
        tool.set_runtime_db_context({"database_name": "static_db"})

        assert tool._runtime_db_context() == {
            "database_name": "static_db",
            "database": "static_db",
        }
        assert tool._adapter is None
        assert tool._attribution_tool is None
        assert tool._adapter_context_key is None

        adapter = Mock()
        tool._adapter = adapter
        tool.set_runtime_db_context({"database_name": "static_db"})

        assert tool._adapter is adapter

    def test_adapter_uses_default_config_when_builder_absent(self, tmp_path):
        adapter = Mock()
        path_manager = SimpleNamespace(semantic_model_path=lambda datasource: tmp_path / datasource)
        config = SimpleNamespace(
            active_model=lambda: SimpleNamespace(model="gpt-4o"),
            current_datasource="runtime_ds",
            path_manager=path_manager,
        )

        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry.get_metadata", return_value=None),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                return_value=adapter,
            ) as create_adapter,
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            tool = SemanticTools(agent_config=config, adapter_type="metricflow")

            assert tool.adapter is adapter

        adapter_config = create_adapter.call_args.args[1]
        assert adapter_config.datasource == "runtime_ds"

    def test_adapter_uses_metadata_config_when_builder_absent(self, tmp_path):
        adapter = Mock()

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        config = SimpleNamespace(
            active_model=lambda: SimpleNamespace(model="gpt-4o"),
            current_datasource="runtime_ds",
            path_manager=SimpleNamespace(semantic_model_path=lambda datasource: tmp_path / datasource),
        )
        metadata = SimpleNamespace(config_class=FakeConfig)

        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry.get_metadata", return_value=metadata),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                return_value=adapter,
            ) as create_adapter,
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            tool = SemanticTools(agent_config=config, adapter_type="metricflow")

            assert tool.adapter is adapter

        adapter_config = create_adapter.call_args.args[1]
        assert adapter_config.kwargs["datasource"] == "runtime_ds"
        assert adapter_config.kwargs["semantic_models_path"] == str(tmp_path / "runtime_ds")

    def test_validate_semantic_initializes_adapter_with_runtime_database(self, tmp_path):
        from datus.configuration.agent_config import AgentConfig, NodeConfig
        from datus.tools.func_tool.semantic_tools import SemanticTools

        captured_configs = []

        class FakeAdapter:
            async def validate_semantic(self, scope="all"):
                return ValidationResult(valid=True, issues=[])

        def create_adapter(adapter_type, adapter_config):
            captured_configs.append(adapter_config)
            return FakeAdapter()

        config = AgentConfig(
            nodes={"test": NodeConfig(model="test-model", input=None)},
            home=str(tmp_path / "h"),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            services={
                "datasources": {
                    "college_exam": {
                        "type": "mysql",
                        "host": "mysql",
                        "username": "user",
                        "password": "pass",
                        "default": True,
                    },
                },
                "semantic_layer": {"metricflow": {"datasource": "college_exam"}},
            },
            skip_init_dirs=True,
        )

        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry.get_metadata", return_value=None),
            patch(
                "datus.tools.func_tool.semantic_tools.semantic_adapter_registry.create_adapter",
                side_effect=create_adapter,
            ),
        ):
            tool = SemanticTools(
                agent_config=config,
                adapter_type="metricflow",
                runtime_db_context_provider=lambda: {
                    "datasource": "college_exam",
                    "database": "college_exam",
                },
            )

            result = tool.validate_semantic(scope="semantic_model")

        assert result.success == 1
        assert result.result["valid"] is True
        assert captured_configs
        first_config = captured_configs[0]
        assert first_config.datasource == "college_exam"
        assert first_config.db_config["type"] == "mysql"
        assert first_config.db_config["host"] == "mysql"
        assert first_config.db_config["database"] == "college_exam"


class TestListMetrics:
    def test_no_adapter_returns_error(self, semantic_tools_ext):
        result = semantic_tools_ext.list_metrics()

        assert result.success == 0
        assert "semantic adapter" in result.error.lower()

    def test_success_from_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_metric = Mock()
        mock_metric.name = "orders"
        mock_metric.description = "Order count"
        mock_metric.type = "count"
        mock_metric.dimensions = []
        mock_metric.measures = []
        mock_metric.unit = None
        mock_metric.format = None
        mock_metric.path = ["Sales"]
        mock_metric.metadata = {
            "inputs": [{"name": "orders", "offset_window": "1 month"}],
            "non_serializable": object(),
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [
            {
                "name": "orders",
                "description": "Order count",
                "type": "count",
                "dimensions": [],
                "measures": [],
                "unit": None,
                "format": None,
                "path": ["Sales"],
                "metadata": {"inputs": [{"name": "orders", "offset_window": "1 month"}]},
            }
        ]
        assert envelope["total"] is None
        assert envelope["has_more"] is False
        assert envelope["extra"] is None
        mock_adapter.list_metrics.assert_called_once_with(path=None, limit=100, offset=0)
        # Contract: list_metrics MUST NOT carry compressor artefacts anymore.
        assert "compressed_data" not in envelope
        assert "original_rows" not in envelope

    def test_passes_path_and_pagination_to_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        metrics = []
        for name in ("m1", "m2", "m3"):
            metric = Mock()
            metric.name = name
            metric.description = ""
            metric.type = ""
            metric.dimensions = []
            metric.measures = []
            metric.unit = None
            metric.format = None
            metric.path = ["Finance"]
            metrics.append(metric)

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=metrics):
            result = tool.list_metrics(path=["Finance"], limit=3, offset=2)

        assert result.success == 1
        envelope = result.result
        assert [row["name"] for row in envelope["items"]] == ["m1", "m2", "m3"]
        assert envelope["total"] is None
        assert envelope["has_more"] is True
        assert envelope["extra"] == {"next_offset": 5}
        mock_adapter.list_metrics.assert_called_once_with(path=["Finance"], limit=3, offset=2)

    def test_drops_null_path_placeholders(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[]):
            result = tool.list_metrics(path=[None, "", "null"], limit=50, offset=0)

        assert result.success == 1
        mock_adapter.list_metrics.assert_called_once_with(path=None, limit=50, offset=0)

    def test_ignores_non_dict_metric_metadata(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        mock_metric = Mock()
        mock_metric.name = "orders"
        mock_metric.description = ""
        mock_metric.type = "count"
        mock_metric.dimensions = []
        mock_metric.measures = []
        mock_metric.unit = None
        mock_metric.format = None
        mock_metric.path = ["Sales"]
        mock_metric.metadata = "not a metadata dict"

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        assert result.result["items"][0]["metadata"] == {}

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("adapter error")):
            result = tool.list_metrics()

        assert result.success == 0
        assert "adapter error" in result.error


class TestGetDimensions:
    def test_with_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=["date", "region"]):
            result = tool.get_dimensions("revenue")

        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"name": "date"}, {"name": "region"}]
        assert envelope["total"] == 2
        assert envelope["has_more"] is False
        assert envelope["extra"] == {
            "time_dimension": None,
            "time_granularities": [],
        }

    def test_returns_time_query_capabilities_in_existing_envelope(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        dimensions = [
            {
                "name": "event_month",
                "type": "time",
                "is_primary_time": True,
                "time_granularities": ["month", "quarter", "year"],
            },
            {"name": "event_type", "type": "categorical"},
        ]

        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            return_value=dimensions,
        ):
            result = tool.get_dimensions("event_total")

        assert result.success == 1
        assert result.result["extra"] == {
            "time_dimension": "event_month",
            "time_granularities": ["month", "quarter", "year"],
        }
        assert "is_primary_time" not in result.result["items"][0]
        assert "time_granularities" not in result.result["items"][0]

    def test_no_adapter_returns_error(self, semantic_tools_ext):
        result = semantic_tools_ext.get_dimensions("revenue")

        assert result.success == 0
        assert "semantic adapter" in result.error.lower()

    def test_with_path_passes_to_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=["date"]):
            result = tool.get_dimensions("revenue", path=["Finance"])

        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"name": "date"}]
        mock_adapter.get_dimensions.assert_called_once_with(metric_name="revenue", path=["Finance"])

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("conn error")):
            result = tool.get_dimensions("revenue")

        assert result.success == 0
        assert "conn error" in result.error


class TestValidateSemantic:
    def test_no_adapter_returns_error(self, semantic_tools_ext):
        result = semantic_tools_ext.validate_semantic()
        assert result.success == 0
        assert "adapter" in result.error.lower()

    def test_valid_result(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        mock_validation = Mock()
        mock_validation.valid = True
        mock_validation.issues = []

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            with patch.object(tool, "_reload_adapter", return_value=True):
                result = tool.validate_semantic()

        assert result.success == 1
        assert result.result["valid"] is True
        assert result.result["issues"] == []
        assert evidence.validation_passed is True

    def test_invalid_result(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        mock_issue = Mock()
        mock_issue.model_dump.return_value = {"severity": "error", "message": "bad config"}
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = [mock_issue]

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            result = tool.validate_semantic()

        assert result.success == 0
        assert result.result["valid"] is False
        assert len(result.result["issues"]) == 1
        assert "1 validation errors" in result.error
        assert "bad config" in result.error
        assert evidence.validation_passed is False

    def test_invalid_result_is_compact_for_large_backend_errors(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = []
        for index in range(20):
            issue = Mock()
            issue.model_dump.return_value = {
                "severity": "error",
                "message": f"issue {index}: " + ("x" * 5000),
            }
            mock_validation.issues.append(issue)

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            result = tool.validate_semantic()

        assert result.success == 0
        assert result.result["issue_count"] == 20
        assert len(result.result["issues"]) == 9
        assert len(json.dumps(result.result, ensure_ascii=False)) < 8_000
        assert len(result.error) < 2_500
        assert "additional validation issue" in result.result["issues"][-1]["message"]

    def test_all_scope_keeps_no_metrics_validation_error(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        mock_issue = Mock()
        mock_issue.model_dump.return_value = {
            "severity": "error",
            "message": "No metrics present in the model.",
        }
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = [mock_issue]

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            result = tool.validate_semantic()

        assert result.success == 0
        assert result.result["valid"] is False
        assert result.result["issues"] == [{"severity": "error", "message": "No metrics present in the model."}]
        assert result.result["ignored_issues"] == []
        assert evidence.validation_passed is False

    def test_semantic_model_scope_ignores_no_metrics_validation_error(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        mock_issue = Mock()
        mock_issue.model_dump.return_value = {
            "severity": "error",
            "message": "No metrics present in the model.",
        }
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = [mock_issue]

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            with patch.object(tool, "_reload_adapter", return_value=True):
                result = tool.validate_semantic(scope="semantic_model")

        assert result.success == 1
        assert result.result["valid"] is True
        assert result.result["issues"] == []
        assert result.result["ignored_issues"] == [{"severity": "error", "message": "No metrics present in the model."}]
        assert result.result["scope"] == "semantic_model"
        assert evidence.validation_passed is True

    def test_semantic_model_scope_keeps_real_validation_errors(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        no_metrics_issue = Mock()
        no_metrics_issue.model_dump.return_value = {
            "severity": "error",
            "message": "No metrics present in the model.",
        }
        duplicate_issue = Mock()
        duplicate_issue.model_dump.return_value = {
            "severity": "error",
            "message": "Element ac_code has already been used as Dimension",
        }
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = [no_metrics_issue, duplicate_issue]

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            result = tool.validate_semantic(scope="semantic_model")

        assert result.success == 0
        assert result.result["valid"] is False
        assert result.result["issues"] == [
            {"severity": "error", "message": "Element ac_code has already been used as Dimension"}
        ]
        assert result.result["ignored_issues"] == [{"severity": "error", "message": "No metrics present in the model."}]
        assert "1 validation errors" in result.error
        assert "Element ac_code" in result.error
        assert evidence.validation_passed is False

    def test_semantic_model_scope_treats_enum_severity_as_error(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        mock_issue = Mock()
        mock_issue.model_dump.return_value = {
            "severity": _Severity.ERROR,
            "message": "bad enum severity",
        }
        mock_issue.model_dump.side_effect = lambda mode=None: {
            "severity": _Severity.ERROR.value if mode == "json" else _Severity.ERROR,
            "message": "bad enum severity",
        }
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = [mock_issue]

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            result = tool.validate_semantic(scope="semantic_model")

        assert result.success == 0
        assert result.result["issues"] == [{"severity": "error", "message": "bad enum severity"}]
        assert result.result["ignored_issues"] == []
        assert evidence.validation_passed is False

    def test_invalid_scope_returns_error(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        result = tool.validate_semantic(scope="unknown")

        assert result.success == 0
        assert "scope must be one of" in result.error

    def test_passes_checks_and_baseline_to_supported_adapter(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        calls = {}

        class _Adapter:
            async def validate_semantic(
                self,
                scope="all",
                semantic_model_name=None,
                checks=None,
                baseline_artifact=None,
            ):
                calls["scope"] = scope
                calls["semantic_model_name"] = semantic_model_name
                calls["checks"] = checks
                calls["baseline_artifact"] = baseline_artifact
                result = Mock()
                result.valid = True
                result.issues = []
                return result

        tool._adapter = _Adapter()
        baseline = {"version": "0.2.0.dev0", "semantic_model": [{"name": "shop", "datasets": []}]}

        with patch.object(tool, "_reload_adapter", return_value=True):
            result = tool.validate_semantic(
                semantic_model_name="shop",
                checks="authoring_quality,mutation_guard",
                baseline_artifact_json=json.dumps(baseline),
            )

        assert result.success == 1
        assert result.result["checks"] == ["authoring_quality", "mutation_guard"]
        assert calls == {
            "scope": "all",
            "semantic_model_name": "shop",
            "checks": ["authoring_quality", "mutation_guard"],
            "baseline_artifact": baseline,
        }

    def test_passes_validation_options_to_kwargs_adapter(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        calls = {}

        class _Adapter:
            async def validate_semantic(self, **kwargs):
                calls.update(kwargs)
                result = Mock()
                result.valid = True
                result.issues = []
                return result

        tool._adapter = _Adapter()
        baseline = {"semantic_model": [{"name": "commerce"}]}

        with patch.object(tool, "_reload_adapter", return_value=True):
            result = tool.validate_semantic(
                scope="semantic_model",
                semantic_model_name="commerce",
                checks=["authoring_quality"],
                baseline_artifact_json=json.dumps(baseline),
            )

        assert result.success == 1
        assert calls == {
            "scope": "semantic_model",
            "semantic_model_name": "commerce",
            "checks": ["authoring_quality"],
            "baseline_artifact": baseline,
        }

    def test_records_target_artifact_validation_evidence(self, semantic_tools_with_adapter, tmp_path):
        tool, _ = semantic_tools_with_adapter
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        class _Adapter:
            async def validate_semantic(self, scope="all", semantic_model_name=None):
                result = Mock()
                result.valid = True
                result.issues = []
                return result

        tool._adapter = _Adapter()
        with (
            patch.object(tool, "_reload_adapter", return_value=True),
            patch.object(
                tool,
                "_semantic_model_artifact_evidence",
                return_value={
                    "semantic_model_name": "commerce",
                    "semantic_model_file": str(artifact),
                },
            ),
        ):
            result = tool.validate_semantic(
                scope="semantic_model",
                semantic_model_name="commerce",
            )

        assert result.success == 1
        assert evidence.semantic_artifact_validation_passed("commerce", artifact)

    def test_resolves_target_artifact_validation_evidence(self, semantic_tools_with_adapter, tmp_path):
        tool, _ = semantic_tools_with_adapter
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")
        tool.adapter_type = "osi"

        with patch(
            "datus.agent.node.semantic_authoring.discover_osi_semantic_models",
            return_value=[
                {
                    "semantic_model_name": "commerce",
                    "absolute_path": str(artifact),
                }
            ],
        ):
            result = tool._semantic_model_artifact_evidence("commerce")

        assert result["semantic_model_name"] == "commerce"
        assert result["semantic_model_file"] == str(artifact.resolve())
        assert len(result["semantic_model_file_sha256"]) == 64

    def test_dosi_resolves_osi_target_artifact_evidence(self, semantic_tools_with_adapter, tmp_path):
        tool, _ = semantic_tools_with_adapter
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")
        tool.adapter_type = "dosi"

        with patch(
            "datus.agent.node.semantic_authoring.discover_osi_semantic_models",
            return_value=[
                {
                    "semantic_model_name": "commerce",
                    "absolute_path": str(artifact),
                }
            ],
        ):
            result = tool._semantic_model_artifact_evidence("commerce")

        assert result["semantic_model_name"] == "commerce"
        assert result["semantic_model_file"] == str(artifact.resolve())

    def test_rejects_target_when_adapter_does_not_support_it(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        class _Adapter:
            async def validate_semantic(self, scope="all"):
                result = Mock()
                result.valid = True
                result.issues = []
                return result

        tool._adapter = _Adapter()

        result = tool.validate_semantic(semantic_model_name="shop")

        assert result.success == 0
        assert "Targeted semantic-model validation is not supported" in result.error

    def test_rejects_checks_when_adapter_does_not_support_them(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        class _Adapter:
            async def validate_semantic(self, scope="all"):
                result = Mock()
                result.valid = True
                result.issues = []
                return result

        tool._adapter = _Adapter()

        result = tool.validate_semantic(checks=["authoring_quality"])

        assert result.success == 0
        assert "checks are not supported" in result.error

    def test_rejects_invalid_baseline_json(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        result = tool.validate_semantic(baseline_artifact_json="{bad")

        assert result.success == 0
        assert "baseline_artifact_json must be valid JSON" in result.error

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("adapter crash")):
            result = tool.validate_semantic()

        assert result.success == 0
        assert "adapter crash" in result.error


class TestAttributionAnalyze:
    def test_tool_schema_exposes_drilldown_guardrail_parameters(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        schema = trans_to_function_tool(tool.attribution_analyze).params_json_schema

        assert {"where", "path", "max_dimension_values"}.issubset(schema["properties"])
        assert "exclusive" in schema["properties"]["baseline_end"]["description"].lower()
        assert "exclusive" in schema["properties"]["current_end"]["description"].lower()

    def test_tool_description_is_explicitly_non_causal(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter

        description = " ".join(trans_to_function_tool(tool.attribution_analyze).description.lower().split())

        assert "descriptive dimension analysis" in description
        assert "do not establish causation" in description
        assert "root cause analysis" not in description
        assert "failed and truncated dimensions are excluded" in description

    def test_no_attribution_tool_returns_error(self, semantic_tools_ext):
        result = semantic_tools_ext.attribution_analyze(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_start="2024-01-01",
            baseline_end="2024-01-08",
            current_start="2024-01-08",
            current_end="2024-01-15",
        )
        assert result.success == 0
        assert "semantic adapter" in result.error.lower()

    def test_success_with_dict_anomaly_context(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "dimension_ranking": [],
            "selected_dimensions": [],
            "top_dimension_values": {},
            "warnings": [{"code": "UNEQUAL_WINDOWS", "message": "not equal"}],
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_result):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-08",
                current_start="2024-01-08",
                current_end="2024-01-15",
                anomaly_context={"rule": "3sigma", "observed_change_pct": 20.0},
                where="region = 'US'",
                path=["sales"],
                max_dimension_values=25,
            )

        assert result.success == 1
        assert result.result["warnings"][0]["code"] == "UNEQUAL_WINDOWS"
        mock_attribution.attribution_analyze.assert_called_once_with(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_start="2024-01-01",
            baseline_end="2024-01-08",
            current_start="2024-01-08",
            current_end="2024-01-15",
            anomaly_context={"rule": "3sigma", "observed_change_pct": 20.0},
            max_selected_dimensions=3,
            top_n_values=10,
            where="region = 'US'",
            path=["sales"],
            max_dimension_values=25,
        )

    def test_success_none_anomaly_context(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "dimension_ranking": [],
            "dimension_analysis_status": "unavailable",
            "per_dimension": {
                "region": {
                    "error": {
                        "code": "DIMENSION_QUERY_FAILED",
                        "message": "region query failed",
                    }
                }
            },
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_result):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-08",
                current_start="2024-01-08",
                current_end="2024-01-15",
                anomaly_context=None,
            )

        assert result.success == 1
        assert result.result["dimension_analysis_status"] == "unavailable"
        assert result.result["per_dimension"]["region"]["error"] == {
            "code": "DIMENSION_QUERY_FAILED",
            "message": "region query failed",
        }

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("analysis failed")):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-08",
                current_start="2024-01-08",
                current_end="2024-01-15",
            )

        assert result.success == 0
        assert "analysis failed" in result.error

    def test_validation_exception_returns_structured_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        tool._attribution_tool = Mock()
        payload = AttributionValidationErrorPayload(
            code="MULTI_ROW_TOTAL",
            message="Expected one total row.",
            period="baseline",
            columns=["revenue"],
            row_count=2,
        )

        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=AttributionValidationException(payload),
        ):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-08",
                current_start="2024-01-08",
                current_end="2024-01-15",
            )

        assert result.success == 0
        assert result.error == "Expected one total row."
        assert result.result == payload.model_dump()


class TestExtractDbConfig:
    """Tests for _extract_db_config helper method."""

    def test_returns_none_when_datasource_not_found(self, semantic_tools):
        """Should return None when the database config cannot be resolved."""
        semantic_tools.agent_config.current_db_config.side_effect = Exception("missing")
        result = semantic_tools._extract_db_config("missing_ns")
        assert result is None

    def test_extracts_and_filters_db_config(self, semantic_tools):
        """Should extract db_config, stringify values, and exclude filtered keys."""
        mock_db_config = Mock()
        mock_db_config.to_dict.return_value = {
            "db_type": "mysql",
            "host": "localhost",
            "port": 3306,
            "password": "secret",
            "role": "ANALYST",
            "private_key_file": "/tmp/rsa_key.p8",
            "private_key_file_pwd": 1234,
            "extra": "skip",
            "path_pattern": "skip",
            "catalog": "skip",
        }
        semantic_tools.agent_config.current_db_config.return_value = mock_db_config

        result = semantic_tools._extract_db_config("ns1")

        assert result["db_type"] == "mysql"
        assert result["host"] == "localhost"
        assert result["port"] == "3306"
        assert result["role"] == "ANALYST"
        assert result["private_key_file"] == "/tmp/rsa_key.p8"
        assert result["private_key_file_pwd"] == "1234"
        assert "extra" not in result
        assert "path_pattern" not in result
        assert result["catalog"] == "skip"


class TestReloadAdapter:
    def test_no_adapter_type_returns_false(self, semantic_tools_ext):
        result = semantic_tools_ext._reload_adapter()
        assert result is False

    def test_reload_success(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        new_adapter = Mock()
        # After clearing, the property should return a new adapter
        with patch.object(type(tool), "adapter", new_callable=lambda: property(lambda self: new_adapter)):
            result = tool._reload_adapter()
        assert result is True

    def test_reload_adapter_fails_returns_false(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        tool._adapter = None

        # Simulate adapter load failure
        with patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry") as mock_registry:
            mock_registry.get_metadata.return_value = None
            mock_registry.create_adapter.side_effect = Exception("config missing")

            result = tool._reload_adapter()

        assert result is False


class TestCompressorModelName:
    """Verify that SemanticTools uses agent_config's model name for DataCompressor."""

    def test_compressor_uses_agent_config_model(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "deepseek/deepseek-chat"
            tool = SemanticTools(agent_config=config)
            assert tool.compressor.model_name == "deepseek/deepseek-chat"

    def test_list_metrics_returns_envelope_without_compressor(self, semantic_tools_with_adapter):
        """list_metrics returns the canonical FuncToolListResult envelope.

        Regression: list_metrics used to wrap rows in DataCompressor output
        (``{original_rows, compressed_data, ...}``) regardless of size.
        After the envelope migration it returns ``{items, total, has_more,
        extra}`` with NO compressor artefacts — list_* never compresses.
        """
        tool, _ = semantic_tools_with_adapter
        mock_metric = Mock()
        mock_metric.name = "orders"
        mock_metric.description = ""
        mock_metric.type = "count"
        mock_metric.dimensions = []
        mock_metric.measures = []
        mock_metric.unit = None
        mock_metric.format = None
        mock_metric.path = []
        mock_metric.metadata = {}

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        envelope = result.result
        assert set(envelope.keys()) == {"items", "total", "has_more", "extra"}
        assert envelope["items"][0]["name"] == "orders"
        # No compressor residue leaks through.
        assert "original_rows" not in envelope
        assert "compressed_data" not in envelope
        assert "compression_type" not in envelope
