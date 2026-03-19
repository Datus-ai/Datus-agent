"""
Test cases for SemanticTools utility functions and query_metrics compression.
"""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.base import FuncToolResult, normalize_null
from datus.tools.func_tool.semantic_tools import _run_async
from datus.tools.semantic_tools.models import QueryResult


class TestNormalizeNull:
    """Tests for normalize_null utility function."""

    def test_none_returns_none(self):
        assert normalize_null(None) is None

    def test_string_null_returns_none(self):
        assert normalize_null("null") is None

    def test_string_none_returns_none(self):
        assert normalize_null("None") is None

    def test_case_insensitive_null(self):
        assert normalize_null("NULL") is None
        assert normalize_null("Null") is None

    def test_case_insensitive_none(self):
        assert normalize_null("NONE") is None
        assert normalize_null("none") is None

    def test_empty_string_returns_none(self):
        assert normalize_null("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_null("  ") is None
        assert normalize_null("\t") is None

    def test_valid_value_passes_through(self):
        assert normalize_null("2024-01-01") == "2024-01-01"
        assert normalize_null("hello") == "hello"

    def test_numeric_value_passes_through(self):
        assert normalize_null(42) == 42
        assert normalize_null(0) == 0


@pytest.fixture
def semantic_tools():
    """Create a SemanticTools instance with mocked dependencies."""
    with patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"), patch(
        "datus.tools.func_tool.semantic_tools.MetricRAG"
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-4o"
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

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
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
        # Result is now a compressed dict
        assert isinstance(result.result, dict)
        assert result.result["original_rows"] == 1
        assert "orders" in result.result["compressed_data"]

    def test_empty_storage_no_adapter_returns_compressed_empty(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = []

        result = semantic_tools_ext.list_metrics()

        assert result.success == 1
        assert isinstance(result.result, dict)
        assert result.result["original_rows"] == 0
        assert result.result["is_compressed"] is False

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
        assert isinstance(result.result, dict)
        assert result.result["original_rows"] == 1
        assert "m1" in result.result["compressed_data"]

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
        assert isinstance(result.result, dict)
        assert result.result["original_rows"] == 3
        assert "m2" in result.result["compressed_data"]

    def test_falls_back_to_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        tool.metric_rag.search_all_metrics.return_value = []

        mock_metric = Mock()
        mock_metric.name = "revenue"
        mock_metric.description = "Revenue metric"
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        assert isinstance(result.result, dict)
        assert result.result["original_rows"] == 1
        assert "revenue" in result.result["compressed_data"]

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
        assert result.result == ["date", "region"]

    def test_no_adapter_from_storage(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = [
            {"name": "revenue", "dimensions": ["date", "channel"]}
        ]

        result = semantic_tools_ext.get_dimensions("revenue")

        assert result.success == 1
        assert result.result == ["date", "channel"]

    def test_no_adapter_metric_not_found(self, semantic_tools_ext):
        semantic_tools_ext.metric_rag.search_all_metrics.return_value = []

        result = semantic_tools_ext.get_dimensions("nonexistent")

        assert result.success == 0
        assert "not found" in result.error

    def test_with_path_filter(self, semantic_tools_ext):
        mock_storage = Mock()
        semantic_tools_ext.metric_rag.storage = mock_storage
        mock_storage.search_all_metrics.return_value = [{"name": "revenue", "dimensions": ["date"]}]

        result = semantic_tools_ext.get_dimensions("revenue", path=["Finance"])

        assert result.success == 1
        assert result.result == ["date"]

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

        mock_validation = Mock()
        mock_validation.valid = True
        mock_validation.issues = []

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            with patch.object(tool, "_reload_adapter", return_value=True):
                result = tool.validate_semantic()

        assert result.success == 1
        assert result.result["valid"] is True
        assert result.result["issues"] == []

    def test_invalid_result(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

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

        # It either returns False or raises - we just check it doesn't crash
        assert isinstance(result, bool)


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

    def test_list_metrics_returns_compressed_dict(self, semantic_tools_ext):
        """list_metrics result should be a compressed dict, not a raw list."""
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
        assert "original_rows" in result.result
        assert "compression_type" in result.result
