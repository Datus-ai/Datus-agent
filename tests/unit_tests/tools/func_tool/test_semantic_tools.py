"""
Test cases for SemanticTools utility functions and query_metrics compression.
"""

from enum import Enum
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.base import FuncToolResult, normalize_null
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.metric_queryability import extract_metric_queryability_contracts
from datus.tools.func_tool.semantic_tools import _run_async
from datus.tools.semantic_tools.models import QueryResult


class _Severity(Enum):
    ERROR = "error"


class TestGenerationEvidence:
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

    def test_queryability_contract_requires_grouped_dry_run(self):
        evidence = GenerationEvidence()
        evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["revenue_total"],
                    "dimension_hints": ["customer_segment"],
                }
            ]
        )
        result = FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}})

        evidence.record_metric_dry_run(["revenue_total"], result)

        assert evidence.has_metric_dry_run(["revenue_total"]) is True
        assert evidence.has_required_queryability_dry_runs(["revenue_total"]) is False

        evidence.record_metric_dry_run(["revenue_total"], result, dimensions=["customer__segment_name"])

        assert evidence.has_required_queryability_dry_runs(["revenue_total"]) is True

    def test_queryability_contract_rejects_partial_dimension_token_match(self):
        evidence = GenerationEvidence()
        evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["revenue_total"],
                    "dimension_hints": ["customer_segment"],
                }
            ]
        )

        evidence.record_metric_dry_run(
            ["revenue_total"],
            FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}}),
            dimensions=["customer_region"],
        )

        assert evidence.has_required_queryability_dry_runs(["revenue_total"]) is False

    def test_queryability_contract_time_hint_requires_metric_time_dimension_and_grain(self):
        result = FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}})

        evidence = GenerationEvidence()
        evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["order_count"],
                    "dimension_hints": ["order_date"],
                }
            ]
        )
        evidence.record_metric_dry_run(["order_count"], result, time_granularity="month")

        assert evidence.has_required_queryability_dry_runs(["order_count"]) is False

        evidence = GenerationEvidence()
        evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["order_count"],
                    "dimension_hints": ["order_date"],
                }
            ]
        )
        evidence.record_metric_dry_run(["order_count"], result, dimensions=["metric_time__month"])

        assert evidence.has_required_queryability_dry_runs(["order_count"]) is False

        evidence = GenerationEvidence()
        evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["order_count"],
                    "dimension_hints": ["order_date"],
                }
            ]
        )
        evidence.record_metric_dry_run(
            ["order_count"],
            result,
            dimensions=["metric_time__month"],
            time_granularity="month",
        )

        assert evidence.has_required_queryability_dry_runs(["order_count"]) is True

    def test_extracts_grouped_metric_queryability_contract_from_sql(self):
        contracts = extract_metric_queryability_contracts(
            """
            SQL:
            SELECT n.n_name AS supplier_nation, SUM(l.l_extendedprice) AS shipped_revenue
            FROM lineitem l
            JOIN supplier s ON l.l_suppkey = s.s_suppkey
            JOIN nation n ON s.s_nationkey = n.n_nationkey
            GROUP BY n.n_name;
            """
        )

        assert contracts == [
            {
                "source": "sql_1",
                "dimension_hints": ["supplier_nation"],
                "metric_hints": ["shipped_revenue"],
                "sql": (
                    "SELECT n.n_name AS supplier_nation, SUM(l.l_extendedprice) AS shipped_revenue\n"
                    "            FROM lineitem l\n"
                    "            JOIN supplier s ON l.l_suppkey = s.s_suppkey\n"
                    "            JOIN nation n ON s.s_nationkey = n.n_nationkey\n"
                    "            GROUP BY n.n_name"
                ),
            }
        ]

    def test_extracts_grouped_contract_from_dialect_fenced_sql(self):
        contracts = extract_metric_queryability_contracts(
            """
            ```snowflake
            SELECT customer_segment, SUM(revenue) AS revenue_total
            FROM orders
            GROUP BY customer_segment;
            ```
            """
        )

        assert contracts == [
            {
                "source": "sql_1",
                "dimension_hints": ["customer_segment"],
                "metric_hints": ["revenue_total"],
                "sql": (
                    "SELECT customer_segment, SUM(revenue) AS revenue_total\n"
                    "            FROM orders\n"
                    "            GROUP BY customer_segment"
                ),
            }
        ]

    def test_extracts_contract_from_final_select_not_grouped_cte(self):
        contracts = extract_metric_queryability_contracts(
            """
            WITH daily AS (
                SELECT order_date, customer_segment, SUM(revenue) AS day_revenue
                FROM orders
                GROUP BY order_date, customer_segment
            )
            SELECT customer_segment, SUM(day_revenue) AS revenue_total
            FROM daily
            GROUP BY customer_segment;
            """
        )

        assert len(contracts) == 1
        assert contracts[0]["source"] == "sql_1"
        assert contracts[0]["dimension_hints"] == ["customer_segment"]
        assert contracts[0]["metric_hints"] == ["revenue_total"]
        assert contracts[0]["sql"].startswith("WITH daily AS")

    def test_ignores_nested_group_when_final_select_is_ungrouped(self):
        contracts = extract_metric_queryability_contracts(
            """
            WITH grouped AS (
                SELECT customer_segment, SUM(revenue) AS revenue
                FROM orders
                GROUP BY customer_segment
            )
            SELECT SUM(revenue) AS revenue_total FROM grouped;
            """
        )

        assert contracts == []


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
            side_effect=[
                [{"name": "date"}],
                [{"name": "date"}],
                query_result,
            ],
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
        assert result_dict["metadata"] == {"execution_time": 0.5}

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

    def test_query_metrics_rejects_dimensions_not_common_to_all_metrics(self, semantic_tools, mock_adapter):
        """Preflight reports incompatible metric/dimension combinations before adapter query."""
        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=[
                [{"name": "ship_date"}, {"name": "ship_mode"}],
                [{"name": "ship_date"}, {"name": "ship_mode"}],
                [{"name": "ship_date"}, {"name": "supplier_nation"}],
            ],
        ):
            result = semantic_tools.query_metrics(
                metrics=["shipped_revenue", "discount_amount", "discount_rate"],
                dimensions=["supplier_nation"],
            )

        assert result.success == 0
        assert "dimension preflight failed" in result.error
        assert result.result["invalid_dimensions"] == [
            {
                "name": "supplier_nation",
                "unsupported_metrics": ["shipped_revenue", "discount_amount"],
                "supported_metrics": ["discount_rate"],
            }
        ]
        assert result.result["common_dimensions"] == ["ship_date"]
        assert result.result["suggested_metric_groups"] == [
            {"metrics": ["shipped_revenue", "discount_amount"], "dimensions": []},
            {"metrics": ["discount_rate"], "dimensions": ["supplier_nation"]},
        ]
        mock_adapter.query_metrics.assert_not_called()

    def test_query_metrics_preflight_preserves_metric_time_in_retry_guidance(self, semantic_tools, mock_adapter):
        """Preflight retry guidance keeps requested metric-time dimensions."""
        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=[
                [{"name": "ship_date"}, {"name": "ship_mode"}],
                [{"name": "ship_date"}, {"name": "supplier_nation"}],
            ],
        ):
            result = semantic_tools.query_metrics(
                metrics=["shipped_revenue", "discount_rate"],
                dimensions=["metric_time__month", "supplier_nation"],
            )

        assert result.success == 0
        assert result.result["common_dimensions"] == ["metric_time__month", "ship_date"]
        assert result.result["suggested_metric_groups"] == [
            {"metrics": ["shipped_revenue"], "dimensions": ["metric_time__month"]},
            {"metrics": ["discount_rate"], "dimensions": ["metric_time__month", "supplier_nation"]},
        ]
        mock_adapter.query_metrics.assert_not_called()

    def test_query_metrics_preflight_allows_time_grain_alias_for_known_time_dimension(self, semantic_tools):
        query_result = QueryResult(
            columns=["metric_time__month", "orders"],
            data=[{"metric_time__month": "2024-01-01", "orders": 10}],
            metadata={},
        )

        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=[
                [{"name": "order_date", "type": "TIME"}],
                query_result,
            ],
        ):
            result = semantic_tools.query_metrics(
                metrics=["orders"],
                dimensions=["order_date__month"],
                time_granularity="month",
            )

        assert result.success == 1

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
        assert result.result["metadata"] == {"sql": "SELECT ...", "row_count": 1}

    def test_query_metrics_dry_run_records_generation_evidence(self, semantic_tools):
        """Successful dry-run evidence gates metric publishing."""
        evidence = GenerationEvidence()
        semantic_tools.generation_evidence = evidence
        query_result = QueryResult(
            columns=[],
            data=[],
            metadata={"sql": "SELECT SUM(revenue) AS revenue FROM orders"},
        )

        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=[
                [{"name": "customer_segment"}],
                query_result,
            ],
        ):
            result = semantic_tools.query_metrics(
                metrics=["revenue"],
                dimensions=["customer_segment"],
                time_granularity="month",
                dry_run=True,
            )

        assert result.success == 1
        assert evidence.metric_dry_run_passed is True
        assert evidence.metric_dry_run_queries == [
            {
                "metrics": ["revenue"],
                "dimensions": ["customer_segment"],
                "time_granularity": "month",
            }
        ]
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

        with patch(
            "datus.tools.func_tool.semantic_tools._run_async",
            side_effect=[
                [{"name": "region"}],
                query_result,
            ],
        ):
            result = semantic_tools.query_metrics(
                metrics=["revenue"],
                dimensions=["region"],
                path=["Finance"],
                time_start="2024-01-01",
                time_end="2024-01-31",
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
                time_end="2024-01-31",
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
    def test_no_adapter_returns_three_tools(self, semantic_tools_ext):
        with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = semantic_tools_ext.available_tools()
        assert len(tools) == 3

    def test_default_metricflow_adapter_exposes_validate_tool(self):
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
        assert "validate_semantic" in names

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

    def test_configured_adapter_load_failure_still_exposes_validate(self):
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
            assert "validate_semantic" in names
            assert "attribution_analyze" not in names

            result = tool.validate_semantic()
            assert result.success == 0
            assert "bad yaml" in result.error


class TestListMetrics:
    def test_success_from_storage(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = [
            {
                "name": "orders",
                "description": "Order count",
                "metric_type": "count",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
                "subject_path": ["Sales"],
            }
        ]

        result = semantic_tools_ext.list_metrics()

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
            }
        ]
        assert envelope["total"] == 1
        assert envelope["has_more"] is False
        assert envelope["extra"] is None
        # Contract: list_metrics MUST NOT carry compressor artefacts anymore.
        assert "compressed_data" not in envelope
        assert "original_rows" not in envelope

    def test_empty_storage_no_adapter_returns_empty_envelope(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = []

        result = semantic_tools_ext.list_metrics()

        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == []
        assert envelope["total"] == 0
        assert envelope["has_more"] is False
        assert envelope["extra"] is None

    def test_path_filter_applied(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = [
            {
                "name": "m1",
                "subject_path": ["Finance"],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            },
            {
                "name": "m2",
                "subject_path": ["Sales"],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            },
        ]

        result = semantic_tools_ext.list_metrics(path=["Finance"])

        assert result.success == 1
        envelope = result.result
        names = [row["name"] for row in envelope["items"]]
        assert names == ["m1"]
        assert envelope["total"] == 1
        assert envelope["has_more"] is False

    def test_pagination(self, semantic_tools_ext):
        metrics = [
            {
                "name": f"m{i}",
                "subject_path": [],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            }
            for i in range(10)
        ]
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = metrics

        result = semantic_tools_ext.list_metrics(limit=3, offset=2)

        assert result.success == 1
        envelope = result.result
        assert [row["name"] for row in envelope["items"]] == ["m2", "m3", "m4"]
        assert envelope["total"] == 10
        assert envelope["has_more"] is True
        assert envelope["extra"] == {"next_offset": 5}

    def test_pagination_last_page(self, semantic_tools_ext):
        metrics = [
            {
                "name": f"m{i}",
                "subject_path": [],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            }
            for i in range(5)
        ]
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = metrics

        result = semantic_tools_ext.list_metrics(limit=3, offset=3)

        assert result.success == 1
        envelope = result.result
        assert [row["name"] for row in envelope["items"]] == ["m3", "m4"]
        assert envelope["total"] == 5
        assert envelope["has_more"] is False
        assert envelope["extra"] is None

    def test_falls_back_to_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        tool.metric_rag.search_all_metrics.return_value = []

        mock_metric = Mock()
        mock_metric.name = "revenue"
        mock_metric.description = "Revenue metric"
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        envelope = result.result
        assert len(envelope["items"]) == 1
        assert envelope["items"][0]["name"] == "revenue"
        # Adapter path has no upstream total — envelope signals unknown via None.
        assert envelope["total"] is None
        # has_more heuristic: len(items) == limit (1) < default limit (100) → False.
        assert envelope["has_more"] is False

    def test_exception_returns_failure(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.side_effect = Exception("db error")

        result = semantic_tools_ext.list_metrics()

        assert result.success == 0
        assert "db error" in result.error


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

    def test_no_adapter_from_storage(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = [
            {"name": "revenue", "dimensions": ["date", "channel"]}
        ]

        result = semantic_tools_ext.get_dimensions("revenue")

        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"name": "date"}, {"name": "channel"}]
        assert envelope["total"] == 2

    def test_no_adapter_metric_not_found(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = []

        result = semantic_tools_ext.get_dimensions("nonexistent")

        assert result.success == 0
        assert "not found" in result.error
        # Even the error path returns an envelope shape, keeping the
        # consumer contract uniform.
        assert result.result == {"items": [], "total": 0, "has_more": False, "extra": None}

    def test_with_path_filter(self, semantic_tools_ext):
        mock_storage = Mock()
        semantic_tools_ext.metric_rag.storage = mock_storage
        mock_storage.search_all_metrics.return_value = [{"name": "revenue", "dimensions": ["date"]}]

        result = semantic_tools_ext.get_dimensions("revenue", path=["Finance"])

        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"name": "date"}]

    def test_exception_returns_failure(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.side_effect = Exception("conn error")

        result = semantic_tools_ext.get_dimensions("revenue")

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

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("adapter crash")):
            result = tool.validate_semantic()

        assert result.success == 0
        assert "adapter crash" in result.error


class TestAttributionAnalyze:
    def test_no_attribution_tool_returns_error(self, semantic_tools_ext):
        result = semantic_tools_ext.attribution_analyze(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_start="2024-01-01",
            baseline_end="2024-01-07",
            current_start="2024-01-08",
            current_end="2024-01-14",
        )
        assert result.success == 0
        assert "Attribution tool not available" in result.error

    def test_success_with_dict_anomaly_context(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "dimension_ranking": [],
            "selected_dimensions": [],
            "top_dimension_values": {},
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_result):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-07",
                current_start="2024-01-08",
                current_end="2024-01-14",
                anomaly_context={"rule": "3sigma", "observed_change_pct": 20.0},
            )

        assert result.success == 1

    def test_success_none_anomaly_context(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        mock_result = Mock()
        mock_result.model_dump.return_value = {"dimension_ranking": []}

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_result):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-07",
                current_start="2024-01-08",
                current_end="2024-01-14",
                anomaly_context=None,
            )

        assert result.success == 1

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("analysis failed")):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-07",
                current_start="2024-01-08",
                current_end="2024-01-14",
            )

        assert result.success == 0
        assert "analysis failed" in result.error


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
            "logic_name": "skip",
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
        assert "logic_name" not in result
        assert "path_pattern" not in result
        assert "catalog" not in result


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

    def test_list_metrics_returns_envelope_without_compressor(self, semantic_tools_ext):
        """list_metrics returns the canonical FuncToolListResult envelope.

        Regression: list_metrics used to wrap rows in DataCompressor output
        (``{original_rows, compressed_data, ...}``) regardless of size.
        After the envelope migration it returns ``{items, total, has_more,
        extra}`` with NO compressor artefacts — list_* never compresses.
        """
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = [
            {
                "name": "orders",
                "description": "",
                "metric_type": "count",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
                "subject_path": [],
            }
        ]
        result = semantic_tools_ext.list_metrics()
        assert result.success == 1
        envelope = result.result
        assert set(envelope.keys()) == {"items", "total", "has_more", "extra"}
        assert envelope["items"][0]["name"] == "orders"
        # No compressor residue leaks through.
        assert "original_rows" not in envelope
        assert "compressed_data" not in envelope
        assert "compression_type" not in envelope
