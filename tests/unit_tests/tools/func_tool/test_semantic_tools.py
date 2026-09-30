"""
Test cases for SemanticTools utility functions and query_metrics compression.
"""

import hashlib
import json
from enum import Enum
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from datus.tools.func_tool.base import FuncToolResult, normalize_null, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.semantic_tools import SemanticTools, _run_async
from datus.tools.semantic_tools.models import (
    AttributionRequest,
    AttributionWindow,
    QueryResult,
)


class _Severity(Enum):
    ERROR = "error"


class TestSemanticToolsGenerationEvidence:
    def test_missing_success_key_is_not_success(self):
        evidence = GenerationEvidence()

        evidence.record_validation_result({"result": {"valid": True, "issues": []}})
        evidence.record_metric_dry_run(["revenue"], {"result": {"metadata": {"sql": "SELECT 1"}}})

        assert evidence.validation_passed is False
        assert evidence.metric_sqls == {}

    def test_attr_payload_metadata_is_recorded(self):
        evidence = GenerationEvidence()
        payload = Mock()
        payload.metadata = {"sql": "SELECT 1"}
        result = FuncToolResult(success=1, result=payload)

        evidence.record_metric_dry_run(["revenue"], result)

        assert evidence.metric_sqls == {"revenue": "SELECT 1"}

    def test_single_sql_fallback_not_fanned_out_to_multiple_metrics(self):
        evidence = GenerationEvidence()
        result = FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}})

        evidence.record_metric_dry_run(["revenue", "cost"], result)

        assert evidence.metric_sqls == {"__query_metrics_dry_run__": "SELECT 1"}


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
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        mock_config = Mock()
        mock_config.active_model.return_value.model = "gpt-4o"
        mock_config.current_datasource = "ns1"
        mock_config.runtime_db_context.return_value = {}
        mock_config.current_db_config.return_value = None
        mock_config.path_manager.semantic_model_path.return_value = "/tmp/models"
        tool = SemanticTools(agent_config=mock_config)
        return tool


@pytest.fixture
def mock_runtime(semantic_tools):
    """Set up a mock Dosi runtime on the SemanticTools instance."""
    runtime = Mock()
    semantic_tools._runtime = runtime
    return runtime


@pytest.mark.usefixtures("mock_runtime")
class TestQueryMetricsCompression:
    """Test cases for query_metrics with DataCompressor integration."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "node_module, node_class",
        [
            ("gen_sql_agentic_node", "GenSQLAgenticNode"),
            ("gen_report_agentic_node", "GenReportAgenticNode"),
            ("gen_visual_report_agentic_node", "GenVisualReportAgenticNode"),
            ("gen_visual_dashboard_agentic_node", "GenVisualDashboardAgenticNode"),
        ],
    )
    async def test_named_registration_preserves_parameter_schema(self, semantic_tools, node_module, node_class):
        node_type = getattr(import_module(f"datus.agent.node.{node_module}"), node_class)
        node = object.__new__(node_type)
        node.tools = []
        node.semantic_tools = semantic_tools
        node._setup_specific_tool_method("semantic_tools", "query_metrics")

        assert [tool.name for tool in node.tools] == ["query_metrics"]
        await self._assert_parameterized_query_executes(semantic_tools, node.tools[0])

    async def _assert_parameterized_query_executes(self, semantic_tools, query_tool):
        calls = []

        class Runtime:
            async def query_metrics(self, metrics, params=None, **kwargs):
                calls.append((metrics, params))
                return QueryResult(columns=["revenue"], data=[{"revenue": 100}])

        semantic_tools._runtime = Runtime()
        params = {"regions": ["APAC", "EMEA"], "threshold": 100}
        assert query_tool.strict_json_schema is False
        result = await query_tool.on_invoke_tool(None, json.dumps({"metrics": ["revenue"], "params": params}))

        assert result["success"] == 1
        assert result["result"]["columns"] == ["revenue"]
        assert calls == [(["revenue"], params)]

    def test_tool_schema_exposes_dosi_time_range_and_query_arguments(self, semantic_tools):
        schema = trans_to_function_tool(semantic_tools.query_metrics).params_json_schema

        start_description = schema["properties"]["time_start"]["description"].lower()
        end_description = schema["properties"]["time_end"]["description"].lower()
        assert "inclusive" in start_description
        assert "exclusive" in end_description
        assert "yyyy-mm-dd" in start_description
        assert "yyyy-mm-dd" in end_description
        assert "path" not in schema["properties"]

    def test_query_metrics_success_with_compression(self, semantic_tools, mock_runtime):
        """Test that query_metrics returns compressed data on success."""
        query_result = QueryResult(
            columns=["date", "revenue", "orders"],
            data=[
                {"date": "2024-01-01", "revenue": 1000, "orders": 50},
                {"date": "2024-01-02", "revenue": 1200, "orders": 60},
            ],
            metadata={"execution_time": 0.5},
        )
        mock_runtime.query_metrics = Mock(return_value=query_result)

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

    def test_query_metrics_passes_dosi_parameter_bindings(self, semantic_tools):
        calls = {}

        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, metrics, params=None, **kwargs):
                calls["metrics"] = metrics
                calls["params"] = params
                return QueryResult(columns=["revenue"], data=[{"revenue": 1}])

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["revenue"],
            params={"regions": ["APAC", "EMEA"], "threshold": 100},
        )

        assert result.success == 1
        assert calls == {
            "metrics": ["revenue"],
            "params": {"regions": ["APAC", "EMEA"], "threshold": 100},
        }

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

    def _query_fifty_rows(self, semantic_tools):
        rows = [{"id": i, "value": i * 100} for i in range(50)]
        query_result = QueryResult(columns=["id", "value"], data=rows, metadata={})
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            return semantic_tools.query_metrics(metrics=["value"])

    def test_get_query_metrics_result_pages_the_full_rows(self, semantic_tools):
        """The rows the compressed preview left out are reachable, a page at a time."""
        result_id = self._query_fifty_rows(semantic_tools).result["result_id"]

        first = semantic_tools.get_query_metrics_result(result_id, offset=0, limit=20)
        last = semantic_tools.get_query_metrics_result(result_id, offset=40, limit=20)

        assert first.success == 1
        assert first.result["row_count"] == 50
        assert first.result["returned"] == 20
        assert first.result["has_more"] is True
        assert first.result["csv"].splitlines()[0] == "id,value"
        assert first.result["csv"].splitlines()[1:] == [f"{i},{i * 100}" for i in range(20)]

        assert last.result["returned"] == 10
        assert last.result["has_more"] is False
        assert last.result["csv"].splitlines()[-1] == "49,4900"

    def test_get_query_metrics_result_serves_a_cell_past_the_csv_field_limit(self, semantic_tools):
        """``csv.reader`` refuses a field over ``csv.field_size_limit()`` (128 KiB by
        default); a page must not depend on re-reading the cached CSV."""
        import csv as csv_module

        big = "x" * (csv_module.field_size_limit() + 1)
        rows = [{"id": 0, "note": "small"}, {"id": 1, "note": big}]
        query_result = QueryResult(columns=["id", "note"], data=rows, metadata={})
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result_id = semantic_tools.query_metrics(metrics=["note"]).result["result_id"]

        page = semantic_tools.get_query_metrics_result(result_id, offset=1, limit=1)

        assert page.success == 1
        assert page.result["returned"] == 1
        assert big in page.result["csv"]

    def test_get_query_metrics_result_serializes_only_the_page(self, semantic_tools):
        """The rows before ``offset`` are sliced away, not parsed and discarded."""
        result_id = self._query_fifty_rows(semantic_tools).result["result_id"]
        seen = []
        real = SemanticTools._query_data_to_csv

        def spy(columns, data):
            seen.append(len(data))
            return real(columns, data)

        with patch.object(SemanticTools, "_query_data_to_csv", side_effect=spy):
            semantic_tools.get_query_metrics_result(result_id, offset=45, limit=3)

        assert seen == [3]

    @pytest.mark.parametrize("shape", ["pyarrow", "pandas"])
    def test_get_query_metrics_result_slices_tabular_results(self, semantic_tools, shape):
        import pandas as pd
        import pyarrow as pa

        frame = pd.DataFrame({"id": list(range(10)), "value": [i * 100 for i in range(10)]})
        data = pa.Table.from_pandas(frame, preserve_index=False) if shape == "pyarrow" else frame
        result_id = semantic_tools._cache_query_metrics_result(["id", "value"], data)

        page = semantic_tools.get_query_metrics_result(result_id, offset=8, limit=5)

        assert page.result["returned"] == 2
        assert page.result["has_more"] is False
        assert page.result["csv"].splitlines() == ["id,value", "8,800", "9,900"]

    def test_get_query_metrics_result_cuts_a_page_to_the_character_budget(self, semantic_tools):
        """A row cap does not bound a page's size; ``returned``/``has_more`` must
        describe the rows that actually fit, so paging on from them loses none."""
        semantic_tools.MAX_QUERY_METRICS_RESULT_PAGE_CHARS = 80
        result_id = self._query_fifty_rows(semantic_tools).result["result_id"]

        rows, offset = [], 0
        while True:
            page = semantic_tools.get_query_metrics_result(result_id, offset=offset, limit=1000).result
            assert len(page["csv"]) <= 80
            assert page["returned"] < 50
            rows.extend(page["csv"].splitlines()[1:])
            offset += page["returned"]
            if not page["has_more"]:
                break

        assert rows == [f"{i},{i * 100}" for i in range(50)]

    def test_get_query_metrics_result_still_advances_past_one_oversized_row(self, semantic_tools):
        semantic_tools.MAX_QUERY_METRICS_RESULT_PAGE_CHARS = 10
        rows = [{"id": 0, "note": "x" * 100}, {"id": 1, "note": "y"}]
        result_id = semantic_tools._cache_query_metrics_result(["id", "note"], rows)

        page = semantic_tools.get_query_metrics_result(result_id, offset=0, limit=2).result

        assert page["returned"] == 1
        assert page["has_more"] is True
        assert "x" * 100 in page["csv"]

    def test_get_query_metrics_result_caps_the_page(self, semantic_tools):
        semantic_tools.MAX_QUERY_METRICS_RESULT_PAGE = 5
        result_id = self._query_fifty_rows(semantic_tools).result["result_id"]

        page = semantic_tools.get_query_metrics_result(result_id, limit=1000)

        assert page.result["returned"] == 5
        assert page.result["has_more"] is True

    def test_get_query_metrics_result_refuses_an_unknown_id(self, semantic_tools):
        result = semantic_tools.get_query_metrics_result("query_metrics:1")

        assert result.success == 0
        assert "Unknown or expired" in result.error

    def test_result_ids_cannot_be_guessed_from_one_another(self, semantic_tools):
        """``get_query_metrics_result`` serves whoever names a key, and one instance
        can serve many MCP clients — a counter would let them read each other's."""
        first = semantic_tools._cache_query_metrics_result(["id"], [{"id": 1}])
        second = semantic_tools._cache_query_metrics_result(["id"], [{"id": 2}])

        assert first != second
        assert first.startswith("query_metrics:") and second.startswith("query_metrics:")
        assert first not in {"query_metrics:1", "query_metrics:2"}
        assert len(first.split(":", 1)[1]) >= 16

    def test_get_query_metrics_result_is_not_a_chat_tool(self, semantic_tools):
        assert "get_query_metrics_result" not in SemanticTools.all_tools_name()

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

    @pytest.mark.parametrize("metrics", [[], ["null", "", None], ""])
    def test_query_metrics_rejects_empty_metrics_before_runtime_call(self, semantic_tools, mock_runtime, metrics):
        """Legacy otherwise raises a cryptic ComputeMetricsNode assertion."""
        result = semantic_tools.query_metrics(metrics=metrics)

        assert result.success == 0
        assert "at least one metric name" in result.error
        mock_runtime.query_metrics.assert_not_called()

    def test_query_metrics_normalizes_string_arguments(self, semantic_tools, mock_runtime):
        """LLM tool calls may send a single string even when the schema says list."""
        query_result = QueryResult(columns=["revenue"], data=[{"revenue": 10}], metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics="revenue",
                dimensions="metric_time",
                order_by="-revenue",
            )

        assert result.success == 1
        mock_runtime.query_metrics.assert_called_once_with(
            metrics=["revenue"],
            dimensions=["metric_time"],
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=["-revenue"],
            dry_run=False,
        )

    @pytest.mark.parametrize("limit", ["", " ", "null", "None"])
    def test_query_metrics_normalizes_null_limit(self, semantic_tools, mock_runtime, limit):
        """LLM null placeholders must not reach runtimes as a present limit."""
        query_result = QueryResult(columns=["revenue"], data=[{"revenue": 10}], metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"], limit=limit)

        assert result.success == 1
        assert mock_runtime.query_metrics.call_args.kwargs["limit"] is None

    def test_query_metrics_runs_warehouse_dry_run_for_compiled_sql(self, semantic_tools, mock_runtime):
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

    def test_query_metrics_returns_failure_when_warehouse_dry_run_fails(self, semantic_tools, mock_runtime):
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

    def test_query_metrics_delegates_dimension_validation_to_runtime(self, semantic_tools, mock_runtime):
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
        mock_runtime.get_dimensions.assert_not_called()
        mock_runtime.query_metrics.assert_called_once_with(
            metrics=["shipped_revenue", "discount_rate"],
            dimensions=["supplier_nation"],
            time_start=None,
            time_end=None,
            time_granularity=None,
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_delegates_time_granularity_validation_to_runtime(self, semantic_tools, mock_runtime):
        """Advertised grains are hints; the runtime validates explicit requests."""
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
        mock_runtime.get_dimensions.assert_not_called()
        mock_runtime.query_metrics.assert_called_once_with(
            metrics=["orders"],
            dimensions=["order_date"],
            time_start=None,
            time_end=None,
            time_granularity="month",
            where=None,
            limit=None,
            order_by=None,
            dry_run=False,
        )

    def test_query_metrics_runtime_exception(self, semantic_tools):
        """Test query_metrics handles runtime exceptions gracefully."""
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
        assert result.result["metadata"]["row_count"] == 1
        assert result.result["metadata"]["_full_result_cache_key"]

    def test_query_metrics_drops_the_compiled_sql_body(self, semantic_tools):
        """A non-dry-run result does not carry the compiled SQL.

        The SQL dominates the payload while the rows it produced sit beside it,
        and nothing outside dry-run publish evidence reads it.
        """
        query_result = QueryResult(
            columns=["revenue"],
            data=[{"revenue": 500}],
            metadata={"sql": "SELECT " + "x, " * 5000 + "1", "row_count": 1},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"])

        metadata = result.result["metadata"]
        assert "sql" not in metadata
        assert metadata["row_count"] == 1

    def test_query_metrics_dry_run_keeps_the_compiled_sql_body(self, semantic_tools):
        """dry_run exists to return the SQL, and publish evidence hashes it."""
        query_result = QueryResult(columns=["sql"], data=[], metadata={"sql": "SELECT 1", "dry_run": True})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"], dry_run=True)

        assert result.result["metadata"]["sql"] == "SELECT 1"
        assert "sql_sha256" not in result.result["metadata"]

    def test_query_metrics_columns_match_the_rows_after_column_compression(self, semantic_tools):
        """``columns`` describes what ``data`` holds, never the pre-compression set.

        Reporting a dropped column would send the caller looking for a column
        that is not in the payload. Driven with a deliberately tiny budget so
        column compression is guaranteed to fire regardless of the production
        budget.
        """
        from datus.utils.compress_utils import DataCompressor

        semantic_tools.compressor = DataCompressor(model_name="gpt-4o", token_threshold=32)
        columns = [f"metric_{index}" for index in range(12)]
        query_result = QueryResult(
            columns=columns,
            data=[{name: 987654321 + index for index, name in enumerate(columns)} for _ in range(5)],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=columns)

        compressed = result.result["data"]
        removed = set(compressed["removed_columns"])
        assert removed, "expected this payload to exceed the budget and drop columns"
        assert result.result["columns"] == [name for name in columns if name not in removed]
        assert not removed & set(result.result["columns"])

    def test_query_metrics_gives_up_the_last_requested_metric_first(self, semantic_tools):
        """Columns are dropped from the far end of the request, not the middle.

        Result column order follows runtime compilation, so dropping from the
        middle discards whichever metric happens to land there — in practice the
        one asked for first. The caller's own ordering is the only statement of
        importance available.
        """
        from datus.utils.compress_utils import DataCompressor

        semantic_tools.compressor = DataCompressor(model_name="gpt-4o", token_threshold=32)
        requested = [f"metric_{index}" for index in range(8)]
        # The runtime emits them in an unrelated order.
        emitted = requested[4:] + requested[:4]
        query_result = QueryResult(
            columns=emitted,
            data=[{name: 987654321 for name in emitted} for _ in range(4)],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=requested)

        removed = result.result["data"]["removed_columns"]
        assert removed, "expected this payload to exceed the budget and drop columns"
        # Whatever survives, the first-requested metric outlives the last one.
        assert requested[0] not in removed
        assert requested[-1] in removed
        # Drops proceed from the end of the request backwards.
        assert removed == requested[: -len(removed) - 1 : -1]

    def test_query_metrics_budget_keeps_a_wide_metric_row_intact(self, semantic_tools):
        """The production budget must not drop metric columns from a normal query.

        16 metrics across 32 grouped rows is an ordinary decomposition query; the
        previous 1024-token budget dropped metric columns from it, which is what
        made a composite metric unverifiable against its inputs.
        """
        columns = ["metric_time__month", "area", "brand"] + [f"metric_{index}" for index in range(16)]
        rows = [
            {name: (f"v{row}" if index < 3 else row * 100 + index) for index, name in enumerate(columns)}
            for row in range(32)
        ]
        query_result = QueryResult(columns=columns, data=rows, metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=columns[3:], dimensions=columns[:3])

        assert result.result["data"]["removed_columns"] == []
        assert result.result["columns"] == columns

    def test_query_metrics_dry_run_records_compiled_sql(self, semantic_tools):
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
        assert evidence.metric_sqls == {"revenue": "SELECT SUM(revenue) AS revenue FROM orders"}

    def test_query_metrics_non_dry_run_does_not_record_metric_sql(self, semantic_tools):
        evidence = GenerationEvidence()
        semantic_tools.generation_evidence = evidence
        query_result = QueryResult(columns=[], data=[], metadata={"sql": "SELECT 1"})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"], dry_run=False)

        assert result.success == 1
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
        assert "sql" not in meta
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

    def test_query_metrics_passes_dosi_parameters(self, semantic_tools, mock_runtime):
        """Test that Dosi query parameters reach the runtime."""
        query_result = QueryResult(columns=["x"], data=[{"x": 1}], metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["revenue"],
                dimensions=["region"],
                time_start="2024-01-01",
                time_end="2024-02-01",
                time_granularity="day",
                where="region = 'US'",
                limit=100,
                order_by=["-revenue"],
                dry_run=True,
            )

            # Verify runtime.query_metrics was called with correct parameters
            mock_runtime.query_metrics.assert_called_once_with(
                metrics=["revenue"],
                dimensions=["region"],
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
        """Runtime-reported unmatched-row counts must reach the tool result metadata.

        The ask_metrics prompt instructs the model to disclose
        `join_policy_filtered_rows` to the user; this protects that contract.
        """
        query_result = QueryResult(
            columns=["x"],
            data=[{"x": 1}],
            metadata={"join_policy": "match_only", "join_policy_filtered_rows": 3},
        )

        class Runtime:
            def get_dimensions(self, metric_name, path=None):
                return []

            def query_metrics(self, metrics, **kwargs):
                return query_result

        semantic_tools._runtime = Runtime()
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["order_count"])

        assert result.success == 1
        assert result.result["metadata"]["join_policy"] == "match_only"
        assert result.result["metadata"]["join_policy_filtered_rows"] == 3

    def test_metric_datasets_maps_names_from_catalog_metadata(self, semantic_tools):
        """The runtime reports which datasets each metric reads."""
        metrics = [
            SimpleNamespace(name="revenue", metadata={"datasets": ["orders"]}),
            SimpleNamespace(name="signups", metadata={"datasets": ["users"]}),
        ]
        semantic_tools._runtime = SimpleNamespace(list_metrics=lambda limit, offset: metrics if offset == 0 else [])
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() == {"revenue": ["orders"], "signups": ["users"]}

    def test_metric_datasets_reads_until_an_empty_page(self, semantic_tools):
        """A truncated read would hide metrics from a policy that scopes by dataset."""
        pages = [
            [SimpleNamespace(name="a", metadata={"datasets": ["orders"]})],
            [SimpleNamespace(name="b", metadata={"datasets": ["users"]})],
            [],
        ]
        offsets = []

        def list_metrics(limit, offset):
            offsets.append(offset)
            return pages[len(offsets) - 1]

        semantic_tools._runtime = SimpleNamespace(list_metrics=list_metrics)
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            mapping = semantic_tools.metric_datasets()

        assert offsets == [0, 1, 2]
        assert mapping == {"a": ["orders"], "b": ["users"]}

    def test_metric_datasets_keeps_paging_when_the_runtime_caps_the_page(self, semantic_tools):
        """An runtime may honour offset while returning fewer rows than requested."""
        served = []

        def list_metrics(limit, offset):
            served.append((limit, offset))
            if offset >= 3:
                return []
            return [SimpleNamespace(name=f"m{offset}", metadata={"datasets": ["orders"]})]

        semantic_tools._runtime = SimpleNamespace(list_metrics=list_metrics)
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            mapping = semantic_tools.metric_datasets()

        assert [offset for _, offset in served] == [0, 1, 2, 3]
        assert sorted(mapping) == ["m0", "m1", "m2"]

    def test_metric_datasets_gives_up_on_an_runtime_that_ignores_offset(self, semantic_tools):
        """An incomplete map must not look like a complete one."""
        semantic_tools._runtime = SimpleNamespace(
            list_metrics=lambda limit, offset: [SimpleNamespace(name="m", metadata={"datasets": ["orders"]})]
        )
        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro),
            patch.object(SemanticTools, "_metric_catalog_paging", return_value=(500, 3)),
        ):
            assert semantic_tools.metric_datasets() is None

    def test_metric_datasets_accepts_a_catalog_that_exactly_fills_the_bound(self, semantic_tools):
        """Filling the page bound is not the same as exceeding it."""
        max_pages = 3

        def list_metrics(limit, offset):
            if offset >= max_pages:
                return []
            return [SimpleNamespace(name=f"m{offset}", metadata={"datasets": ["orders"]})]

        semantic_tools._runtime = SimpleNamespace(list_metrics=list_metrics)
        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro),
            patch.object(SemanticTools, "_metric_catalog_paging", return_value=(1, max_pages)),
        ):
            mapping = semantic_tools.metric_datasets()

        assert sorted(mapping) == ["m0", "m1", "m2"]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("orders", ["orders"]),
            (["orders", "users"], ["orders", "users"]),
            (["orders", "", "  "], ["orders"]),
            (["orders", None, 1], ["orders"]),
            (None, []),
            (42, []),
        ],
    )
    def test_metric_datasets_normalizes_the_reported_shape(self, semantic_tools, raw, expected):
        """A lone string names one dataset; iterating it would yield characters."""
        metrics = [SimpleNamespace(name="revenue", metadata={"datasets": raw})]
        semantic_tools._runtime = SimpleNamespace(list_metrics=lambda limit, offset: metrics if offset == 0 else [])
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() == {"revenue": expected}

    def test_metric_datasets_keeps_metrics_without_dataset_information(self, semantic_tools):
        """A metric reported without dataset information maps to an empty list."""
        metrics = [SimpleNamespace(name="orphan", metadata={})]
        semantic_tools._runtime = SimpleNamespace(list_metrics=lambda limit, offset: metrics if offset == 0 else [])
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() == {"orphan": []}

    def test_metric_datasets_without_an_runtime_is_none(self, semantic_tools):
        """An unavailable catalog is distinct from a readable empty one."""
        with patch.object(type(semantic_tools), "runtime", property(lambda self: None)):
            assert semantic_tools.metric_datasets() is None

    @pytest.mark.parametrize("reported", [None, ["orders"], "orders"])
    def test_metric_datasets_rejects_a_non_mapping_from_the_accessor(self, semantic_tools, reported):
        """An invalid provider result must not reach the transformer context."""
        semantic_tools._runtime = SimpleNamespace(metric_datasets=lambda: reported, list_metrics=lambda **kwargs: [])
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() is None

    def test_metric_datasets_skips_unnamed_metrics_from_the_accessor(self, semantic_tools):
        """Both read paths drop metrics without a usable name."""
        semantic_tools._runtime = SimpleNamespace(
            metric_datasets=lambda: {None: ["x"], "": ["y"], " revenue ": ["orders"]},
            list_metrics=lambda **kwargs: [],
        )
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() == {"revenue": ["orders"]}

    def test_metric_datasets_prefers_the_runtime_lightweight_accessor(self, semantic_tools):
        """The runtime may skip building a MetricDefinition per metric."""

        def fail_list_metrics(**kwargs):
            raise AssertionError("should not fall back to list_metrics")

        semantic_tools._runtime = SimpleNamespace(
            metric_datasets=lambda: {"revenue": ["orders"]},
            list_metrics=fail_list_metrics,
        )
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: coro):
            assert semantic_tools.metric_datasets() == {"revenue": ["orders"]}


@pytest.fixture
def semantic_tools_with_runtime():
    with (
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        config = Mock()
        config.active_model.return_value.model = "gpt-4o"
        config.current_datasource = "ns1"
        config.runtime_db_context.return_value = {}
        config.current_db_config.return_value = None
        config.path_manager.semantic_model_path.return_value = "/tmp/models"
        tool = SemanticTools(agent_config=config)
        mock_runtime = Mock()
        tool._runtime = mock_runtime
        return tool, mock_runtime


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
        assert "get_metric" in names
        assert "query_metrics" in names
        assert "validate_semantic" in names
        assert "attribution_analyze" in names


class TestAvailableTools:
    def test_dosi_runtime_does_not_load_during_tool_registration(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch(
                "datus.tools.func_tool.semantic_tools.DosiRuntime",
                side_effect=RuntimeError("engine unavailable"),
            ) as runtime,
        ):
            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.resolve_semantic_runtime.return_value = "dosi"
            tool = SemanticTools(agent_config=config)
            with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as convert:
                convert.side_effect = lambda function: SimpleNamespace(name=function.__name__)
                names = [registered.name for registered in tool.available_tools()]
        assert names == ["list_metrics", "get_metric", "query_metrics", "validate_semantic", "attribution_analyze"]
        runtime.assert_not_called()

    def test_configured_runtime_load_failure_is_reported_when_tool_runs(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.DosiRuntime", side_effect=RuntimeError("bad yaml")),
        ):
            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.current_datasource = "ns1"
            config.current_db_config.return_value = None
            config.runtime_db_context.return_value = {}
            config.path_manager.semantic_model_path.return_value = "/tmp/models"
            tool = SemanticTools(agent_config=config)
            result = tool.validate_semantic()
        assert result.success == 0
        assert "bad yaml" in result.error


class TestRuntimeDbContext:
    def test_normalize_runtime_context_handles_empty_and_aliases(self):
        assert SemanticTools._normalize_runtime_db_context(None) == {}
        assert SemanticTools._normalize_runtime_db_context({"database_name": " runtime_db "}) == {
            "database_name": "runtime_db",
            "database": "runtime_db",
        }

    def test_runtime_change_rebuilds_dosi_runtime(self):
        context = {"datasource": "warehouse", "database": "one"}

        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.DosiRuntime") as runtime,
        ):
            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.current_datasource = "warehouse"
            config.current_db_config.return_value = None
            config.path_manager.semantic_model_path.return_value = "/tmp/models"
            tool = SemanticTools(config, runtime_db_context_provider=lambda: context)
            first = tool.runtime
            assert tool.runtime is first
            assert runtime.call_args.args[0].db_config["database"] == "one"
            context["database"] = "two"
            assert tool.runtime is first
            assert runtime.call_count == 2
            assert runtime.call_args.args[0].db_config["database"] == "two"

    def test_selected_model_path_rebuilds_runtime(self, tmp_path):
        selected = {"path": str(tmp_path / "orders.yml")}
        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.DosiRuntime") as runtime,
        ):
            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            config.current_datasource = "warehouse"
            config.current_db_config.return_value = None
            config.runtime_db_context.return_value = {}
            config.path_manager.semantic_model_path.return_value = str(tmp_path)
            tool = SemanticTools(config, semantic_model_path_provider=lambda: selected["path"])
            first = tool.runtime
            assert runtime.call_args.args[0].semantic_model_path == selected["path"]
            selected["path"] = str(tmp_path / "finance.yml")
            assert tool.runtime is first
            assert runtime.call_count == 2
            assert runtime.call_args.args[0].semantic_model_path == selected["path"]

    def test_runtime_context_provider_failure_returns_empty_context(self):
        with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            tool = SemanticTools(config, runtime_db_context_provider=Mock(side_effect=RuntimeError("boom")))
        assert tool._runtime_db_context() == {}

    def test_static_runtime_context_invalidates_cached_runtime(self):
        with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
            config = Mock()
            config.active_model.return_value.model = "gpt-4o"
            tool = SemanticTools(config)
        tool._runtime = Mock()
        tool.set_runtime_db_context({"database_name": "next"})
        assert tool._runtime is None
        assert tool._runtime_db_context()["database"] == "next"


class TestListMetrics:
    def test_success_from_runtime(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = [{"name": "orders", "subject_path": ["Commerce", "Orders"]}]
        mock_metric = Mock()
        mock_metric.name = "orders"
        mock_metric.description = "Order count"
        mock_metric.type = "count"
        mock_metric.dimensions = []
        mock_metric.measures = []
        mock_metric.unit = None
        mock_metric.format = None
        mock_metric.path = ["Ignored", "Runtime", "Path"]
        mock_metric.metadata = {
            "base_kind": "aggregate",
            "time_dimension": "orders.ordered_at",
            "non_serializable": object(),
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        envelope = result.result
        # A summary row: identity and derivation only. ``measures`` (compiled
        # internal names), the runtime's always-empty ``dimensions``, and unset
        # unit/format stay out — get_metric carries the per-metric detail.
        assert envelope["items"] == [
            {
                "name": "orders",
                "description": "Order count",
                "kind": "aggregate",
                "time_dimension": "orders.ordered_at",
                # The knowledge-base path wins over the runtime's own ``path``.
                "path": ["Commerce", "Orders"],
            }
        ]
        assert envelope["total"] is None
        assert envelope["has_more"] is False
        assert envelope["extra"] is None
        mock_runtime.list_metrics.assert_called_once_with(path=None, limit=200, offset=0)
        # Contract: list_metrics MUST NOT carry compressor artefacts anymore.
        assert "compressed_data" not in envelope
        assert "original_rows" not in envelope

    @pytest.mark.parametrize(
        "limit, offset, expected_limit, expected_offset",
        [
            ("200", "0", 200, 0),  # both stringified, as a model actually sent them
            ("50", 0, 50, 0),
            (50, "10", 50, 10),
            ("abc", 0, 200, 0),  # unusable -> default, not a failed call
            (None, None, 200, 0),
            ("null", "none", 200, 0),
            (0, -5, 1, 0),  # clamped: a zero page or negative offset means nothing
        ],
    )
    def test_paging_bounds_accept_what_a_model_actually_sends(
        self, semantic_tools_with_runtime, limit, offset, expected_limit, expected_offset
    ):
        """A schema declaring ``int`` does not stop a model sending ``"200"``.

        Runtimes slice and add with these values, so a string arrives as
        ``TypeError: slice indices must be integers`` — a failed call whose error
        tells the caller nothing about what to do differently.
        """
        tool, mock_runtime = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[]):
            result = tool.list_metrics(limit=limit, offset=offset)

        assert result.success == 1
        assert mock_runtime.list_metrics.call_args.kwargs["limit"] == expected_limit
        assert mock_runtime.list_metrics.call_args.kwargs["offset"] == expected_offset

    def test_paging_bounds_reach_the_runtime_as_ints(self, semantic_tools_with_runtime):
        """Coercion must produce real ints — ``"200"`` slices nothing."""
        tool, mock_runtime = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[]):
            tool.list_metrics(limit="200", offset="0")

        kwargs = mock_runtime.list_metrics.call_args.kwargs
        assert type(kwargs["limit"]) is int
        assert type(kwargs["offset"]) is int

    def test_summary_row_carries_the_dependency_edges(self, semantic_tools_with_runtime):
        """derive_expr / derive_base are what make a composite metric decomposable.

        Without them a caller sees that a metric is derived but not from what, so
        it cannot walk from a total down to the inputs that moved it.
        """
        tool, _ = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = []
        composed = Mock()
        composed.name = "area_score"
        composed.description = "Weighted total"
        composed.type = "expression"
        composed.path = None
        composed.unit = None
        composed.format = None
        composed.metadata = {
            "base_kind": "expression",
            "derive_family": "compose",
            "derive_expr": "kp_tel * 0.1 + sla_score * 0.1",
        }
        ranked = Mock()
        ranked.name = "store_issue_num_rn"
        ranked.description = "Rank"
        ranked.type = "ratio"
        ranked.path = None
        ranked.unit = None
        ranked.format = None
        ranked.metadata = {
            "base_kind": "ratio",
            "derive_family": "window",
            "derive_base": "store_issue_num",
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[composed, ranked]):
            result = tool.list_metrics()

        items = {item["name"]: item for item in result.result["items"]}
        assert items["area_score"]["derive_expr"] == "kp_tel * 0.1 + sla_score * 0.1"
        assert items["area_score"]["derive_family"] == "compose"
        assert "derive_base" not in items["area_score"]
        assert items["store_issue_num_rn"]["derive_base"] == "store_issue_num"
        assert items["store_issue_num_rn"]["derive_family"] == "window"
        assert "derive_expr" not in items["store_issue_num_rn"]

    def test_filters_path_with_kb_and_does_not_pass_path_to_runtime(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = [
            {"name": "M1", "subject_path": ["Finance"]},
            {"name": "m2", "subject_path": ["Sales"]},
            {"name": "m3", "subject_path": ["Finance", "Revenue"]},
            {"name": "", "subject_path": ["Finance"]},
        ]
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
            metric.path = None
            metrics.append(metric)

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", return_value=metrics),
            patch.object(SemanticTools, "_metric_catalog_paging", return_value=(100, 10)),
        ):
            result = tool.list_metrics(path=["Finance"], limit=10, offset=0)

        assert result.success == 1
        envelope = result.result
        assert [(row["name"], row["path"]) for row in envelope["items"]] == [
            ("m1", ["Finance"]),
            ("m3", ["Finance", "Revenue"]),
        ]
        assert envelope["total"] == 2
        assert envelope["has_more"] is False
        assert envelope["extra"] is None
        mock_runtime.list_metrics.assert_called_once_with(path=None, limit=100, offset=0)
        tool.metric_rag.search_all_metrics.assert_called_once_with(select_fields=["name"])

    def test_path_filtering_happens_before_pagination(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = [
            {"name": name, "subject_path": ["Metrics", "flights"]} for name in ("m1", "m3", "m5")
        ]
        pages = [
            [SimpleNamespace(name="m1", description="", path=None)],
            [SimpleNamespace(name="m2", description="", path=None)],
            [SimpleNamespace(name="m3", description="", path=None)],
            [SimpleNamespace(name="m4", description="", path=None)],
            [SimpleNamespace(name="m5", description="", path=None)],
        ]

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=pages),
            patch.object(SemanticTools, "_metric_catalog_paging", return_value=(1, 10)),
        ):
            result = tool.list_metrics(path=["Metrics", "flights"], limit=1, offset=1)

        assert result.success == 1
        assert [row["name"] for row in result.result["items"]] == ["m3"]
        assert result.result["total"] == 3
        assert result.result["has_more"] is True
        assert result.result["extra"] == {"next_offset": 2}
        assert mock_runtime.list_metrics.call_args_list == [
            call(path=None, limit=1, offset=0),
            call(path=None, limit=1, offset=1),
            call(path=None, limit=1, offset=2),
            call(path=None, limit=1, offset=3),
            call(path=None, limit=1, offset=4),
        ]

    def test_unknown_kb_path_returns_empty_without_reading_runtime(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = []

        result = tool.list_metrics(path=["Metrics", "missing"])

        assert result.success == 1
        assert result.result == {"items": [], "total": 0, "has_more": False, "extra": None}
        mock_runtime.list_metrics.assert_not_called()

    def test_path_query_excludes_kb_metric_missing_from_runtime(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = [
            {"name": "flight_count", "subject_path": ["Metrics", "flights"]},
            {"name": "stale_metric", "subject_path": ["Metrics", "flights"]},
        ]
        metric = SimpleNamespace(name="flight_count", description="Flights", path=None)

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=[[metric], []]),
            patch.object(SemanticTools, "_metric_catalog_paging", return_value=(100, 10)),
        ):
            result = tool.list_metrics(path=["Metrics", "flights"])

        assert result.success == 1
        assert [(row["name"], row["path"]) for row in result.result["items"]] == [
            ("flight_count", ["Metrics", "flights"])
        ]
        assert result.result["total"] == 1

    @pytest.mark.parametrize("extra_page,expected_success", [([], 1), ([SimpleNamespace(name="m2")], 0)])
    def test_path_query_handles_catalog_page_cap(self, semantic_tools_with_runtime, extra_page, expected_success):
        tool, _ = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = [{"name": "missing", "subject_path": ["Metrics", "flights"]}]
        unrelated = SimpleNamespace(name="m1", description="")

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=[[unrelated], extra_page]),
            patch.object(SemanticTools, "_metric_catalog_paging", return_value=(1, 1)),
        ):
            result = tool.list_metrics(path=["Metrics", "flights"])

        assert result.success == expected_success
        if expected_success:
            assert result.result["items"] == []
            assert result.result["total"] == 0
        else:
            assert "error_code=400001" in result.error
            assert "cannot apply the knowledge-base subject path safely" in result.error

    def test_drops_null_path_placeholders(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = []

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[]):
            result = tool.list_metrics(path=[None, "", "null"], limit=50, offset=0)

        assert result.success == 1
        mock_runtime.list_metrics.assert_called_once_with(path=None, limit=50, offset=0)

    def test_ignores_non_dict_metric_metadata(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.return_value = []
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
        item = result.result["items"][0]
        # Unusable metadata degrades to the runtime's own ``type`` rather than
        # failing the listing.
        assert item["kind"] == "count"
        assert not any(key.startswith("derive_") for key in item)
        # The KB knows no path for this metric, so the runtime's own path must
        # not leak in as a substitute.
        assert item.get("path") is None

    def test_no_path_keeps_runtime_available_when_kb_read_fails(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.side_effect = RuntimeError("KB unavailable")
        mock_metric = SimpleNamespace(name="orders", description="Order count", path=["Yaml", "Path"])

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        assert result.result["items"][0]["name"] == "orders"
        assert result.result["items"][0].get("path") is None
        mock_runtime.list_metrics.assert_called_once_with(path=None, limit=200, offset=0)

    def test_path_query_fails_when_kb_read_fails(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.side_effect = RuntimeError("KB unavailable")

        result = tool.list_metrics(path=["Metrics", "orders"])

        assert result.success == 0
        assert "KB unavailable" in result.error
        mock_runtime.list_metrics.assert_not_called()

    def test_exception_returns_failure(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("runtime error")):
            result = tool.list_metrics()

        assert result.success == 0
        assert "runtime error" in result.error


def _detail_metric(name, metadata, measures=("m",)):
    metric = Mock()
    metric.name = name
    metric.description = f"{name} description"
    metric.type = metadata.get("base_kind")
    metric.measures = list(measures)
    metric.unit = None
    metric.format = None
    metric.path = None
    metric.metadata = metadata
    return metric


class TestGetMetric:
    RANKED = {
        "base_kind": "ratio",
        "derive_family": "window",
        "derive_base": "store_issue_num",
        "time_dimension": "cell.month_key",
        "window": {
            "base": "store_issue_num",
            "rank": {
                "function": "rank",
                "order": {"by": "value", "direction": "asc"},
                "partition": {"mode": "query_dimensions_except", "exclude": ["cell.brand"]},
            },
        },
        "datasets": ["repair"],
    }

    @staticmethod
    def _wire(metrics, dimensions=("cell.brand", "cell.area"), dimension_rows=None):
        """Route the two runtime coroutines this tool awaits, in call order."""
        dimension_rows = dimension_rows if dimension_rows is not None else [{"name": name} for name in dimensions]

        def dispatch(coro):
            # ``list_metrics`` is awaited once for the catalog, then
            # ``get_dimensions`` once for the resolved metric.
            dispatch.calls += 1
            return metrics if dispatch.calls == 1 else dimension_rows

        dispatch.calls = 0
        return patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=dispatch)

    @pytest.mark.parametrize("name", ["", "   ", None, "null"])
    def test_rejects_a_missing_name(self, semantic_tools_with_runtime, name):
        tool, _ = semantic_tools_with_runtime

        result = tool.get_metric(name=name)

        assert result.success == 0
        assert "requires a metric name" in result.error

    def test_returns_detail_and_queryable_dimensions(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric("store_issue_num_rn", self.RANKED)

        with self._wire([metric]):
            result = tool.get_metric(name="store_issue_num_rn")

        assert result.success == 1
        item = result.result
        assert item["derive_base"] == "store_issue_num"
        assert item["window"]["rank"]["order"]["direction"] == "asc"
        assert item["datasets"] == ["repair"]
        assert [dimension["name"] for dimension in item["dimensions"]] == ["cell.brand", "cell.area"]

    def test_omits_compiled_measure_names(self, semantic_tools_with_runtime):
        """Compiled measure names are unusable as tool arguments and dominate
        the row, so they stay out of the detail exactly as they stay out of the
        summary."""
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric("area_score", {"base_kind": "expression"}, measures=["a_very_long_compiled_name"] * 21)

        with self._wire([metric]):
            result = tool.get_metric(name="area_score")

        assert "measures" not in result.result

    def test_passes_through_the_dimensions_the_runtime_says_are_required(self, semantic_tools_with_runtime):
        """The requirement is the runtime's answer, carried verbatim.

        Which dimensions a metric needs before its partitions mean anything can
        depend on metrics the runtime never published — a composite inherits the
        requirement from its inputs. Deriving it here from the partition rule
        would report "no requirement" for exactly those metrics, so the field is
        passed through and never reconstructed.
        """
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric(
            "area_score",
            {"base_kind": "expression", "derive_family": "compose", "required_dimensions": ["cell.brand", "cell.area"]},
        )

        with self._wire([metric]):
            result = tool.get_metric(name="area_score")

        assert result.result["required_dimensions"] == ["cell.brand", "cell.area"]

    def test_keeps_the_field_absent_when_the_runtime_reports_nothing(self, semantic_tools_with_runtime):
        """A metric whose window excludes a dimension but that publishes no
        requirement must not grow one here: absence means the runtime did not
        say, and inventing an answer from the rule is what got it wrong."""
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric("store_issue_num_rn", self.RANKED)

        with self._wire([metric]):
            result = tool.get_metric(name="store_issue_num_rn")

        assert "required_dimensions" not in result.result

    def test_unknown_name_fails_with_the_name(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric("kpi_issues", {"base_kind": "aggregate"})

        with self._wire([metric]):
            result = tool.get_metric(name="no_such_metric")

        assert result.success == 0
        assert "no_such_metric" in result.error
        assert "list_metrics" in result.error

    def test_dimension_failure_keeps_the_rest_of_the_detail(self, semantic_tools_with_runtime):
        """Detail a caller can use should survive one failing sub-query."""
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric("kpi_issues", {"base_kind": "aggregate"})

        def dispatch(coro):
            dispatch.calls += 1
            if dispatch.calls == 1:
                return [metric]
            raise RuntimeError("planner unavailable")

        dispatch.calls = 0
        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=dispatch):
            result = tool.get_metric(name="kpi_issues")

        assert result.success == 1
        assert result.result["kind"] == "aggregate"
        assert "planner unavailable" in result.result["dimensions_error"]
        assert "dimensions" not in result.result

    def test_catalog_failure_is_reported(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("catalog down")):
            result = tool.get_metric(name="kpi_issues")

        assert result.success == 0
        assert "catalog down" in result.error

    def test_promotes_the_time_axis_off_the_dimension_rows(self, semantic_tools_with_runtime):
        """The time contract is one answer per metric, not per dimension.

        The runtime reports it on whichever dimension is the primary time axis;
        repeating it on every row would restate the same answer N times.
        """
        tool, _ = semantic_tools_with_runtime
        metric = _detail_metric("event_total", {"base_kind": "aggregate"})
        dimensions = [
            {
                "name": "event_month",
                "type": "time",
                "is_primary_time": True,
                "time_granularities": ["month", "quarter", "year"],
                "recommended": True,
                "recommendation_source": "inferred:time",
            },
            {
                "name": "event_id",
                "type": "categorical",
                "recommended": False,
                "recommendation_source": "inferred:primary_key",
            },
        ]

        with self._wire([metric], dimension_rows=dimensions):
            result = tool.get_metric(name="event_total")

        assert result.success == 1
        assert result.result["time_dimension"] == "event_month"
        assert result.result["time_granularities"] == ["month", "quarter", "year"]
        for row in result.result["dimensions"]:
            assert "is_primary_time" not in row
            assert "time_granularities" not in row
        # The grouping recommendation is per dimension, so unlike the time axis
        # it stays on the row instead of being promoted to the metric.
        assert result.result["dimensions"][0]["recommended"] is True
        assert result.result["dimensions"][0]["recommendation_source"] == "inferred:time"
        assert result.result["dimensions"][1]["recommended"] is False
        assert result.result["dimensions"][1]["recommendation_source"] == "inferred:primary_key"

    def test_describes_a_metric_that_only_a_later_catalog_page_holds(self, semantic_tools_with_runtime):
        """Every name list_metrics can hand out, get_metric has to accept.

        list_metrics pages the catalog, so it will report names past the first
        page. Resolving those against a single bounded read answers "unknown
        metric" for a metric the caller was just told exists.
        """
        tool, _ = semantic_tools_with_runtime
        wanted = _detail_metric("revenue", {"base_kind": "aggregate"})
        pages = [
            [_detail_metric(f"filler_{index}", {"base_kind": "aggregate"}) for index in range(3)],
            [wanted],
            [],
        ]
        dimension_rows = [{"name": "cell.brand"}]

        def dispatch(coro):
            if pages:
                return pages.pop(0)
            return dimension_rows

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=dispatch):
            result = tool.get_metric(name="revenue")

        assert result.success == 1
        assert result.result["name"] == "revenue"

    def test_unknown_name_fails_once_the_catalog_runs_out(self, semantic_tools_with_runtime):
        """Paging to the end of the catalog is an unknown metric, not a failure
        to read it: the caller needs to fix the name, not retry."""
        tool, _ = semantic_tools_with_runtime
        pages = [[_detail_metric("revenue", {"base_kind": "aggregate"})], []]

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=lambda coro: pages.pop(0)):
            result = tool.get_metric(name="no_such_metric")

        assert result.success == 0
        assert "no_such_metric" in result.error
        assert "list_metrics" in result.error

    def test_resolves_the_name_against_the_whole_catalog_not_the_path(self, semantic_tools_with_runtime):
        """Name resolution must scope exactly the way list_metrics scopes it.

        list_metrics filters by knowledge-base subject path and asks the runtime
        for its unfiltered catalog. Narrowing the runtime read by path here would
        make get_metric reject names list_metrics had just handed out under that
        same path.
        """
        tool, mock_runtime = semantic_tools_with_runtime
        metric = _detail_metric("revenue", {"base_kind": "aggregate"})

        with self._wire([metric]):
            result = tool.get_metric(name="revenue", path=["Finance"])

        assert result.success == 1
        assert mock_runtime.list_metrics.call_args.kwargs["path"] is None
        mock_runtime.get_dimensions.assert_called_once_with(metric_name="revenue", path=["Finance"])


class TestValidateSemantic:
    def test_valid_result(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        mock_validation = Mock()
        mock_validation.valid = True
        mock_validation.issues = []

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            with patch.object(tool, "_reload_runtime", return_value=True):
                result = tool.validate_semantic()

        assert result.success == 1
        assert result.result["valid"] is True
        assert result.result["issues"] == []
        assert evidence.validation_passed is True

    def test_records_compiled_evidence_without_exposing_descriptions(self, semantic_tools_with_runtime, tmp_path):
        tool, _ = semantic_tools_with_runtime
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")
        calls = {}
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence
        tool._semantic_metric_names_provider = lambda: ["revenue"]

        class _Runtime:
            async def validate_semantic(self, scope="all", metric_names=None):
                calls["metric_names"] = metric_names
                return SimpleNamespace(
                    valid=True,
                    issues=[],
                    metadata={
                        "contract_digest": "sha256:" + hashlib.sha256(b"contract").hexdigest(),
                        "artifact_sha256": {
                            str(artifact): "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
                        },
                        "compiled_metrics": [
                            {
                                "name": "revenue",
                                "kind": "aggregate",
                                "datasets": ["orders"],
                            }
                        ],
                        "compiled_metric_digests": {"revenue": "sha256:" + hashlib.sha256(b"revenue").hexdigest()},
                    },
                )

        tool._runtime = _Runtime()
        with patch.object(tool, "_reload_runtime", return_value=True):
            result = tool.validate_semantic()

        assert result.success == 1
        assert calls["metric_names"] == ["revenue"]
        assert result.result["compiled_metric_count"] == 1
        assert "compiled_metrics" not in result.result
        assert evidence.compiled_validation_passed(artifact, ["revenue"])

    def test_invalid_result(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
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

    def test_invalid_result_is_compact_for_large_backend_errors(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
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

    def test_all_scope_keeps_no_metrics_validation_error(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
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

    def test_semantic_model_scope_ignores_no_metrics_validation_error(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
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
            with patch.object(tool, "_reload_runtime", return_value=True):
                result = tool.validate_semantic(scope="semantic_model")

        assert result.success == 1
        assert result.result["valid"] is True
        assert result.result["issues"] == []
        assert result.result["ignored_issues"] == [{"severity": "error", "message": "No metrics present in the model."}]
        assert result.result["scope"] == "semantic_model"
        assert evidence.validation_passed is True

    def test_semantic_model_scope_keeps_real_validation_errors(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
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

    def test_semantic_model_scope_treats_enum_severity_as_error(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
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

    def test_invalid_scope_returns_error(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime

        result = tool.validate_semantic(scope="unknown")

        assert result.success == 0
        assert "scope must be one of" in result.error

    def test_validate_semantic_schema_exposes_dosi_options(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        schema = trans_to_function_tool(tool.validate_semantic).params_json_schema

        assert set(schema["properties"]) == {"scope", "semantic_model_name"}

    def test_records_target_artifact_validation_evidence(self, semantic_tools_with_runtime, tmp_path):
        tool, _ = semantic_tools_with_runtime
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")
        evidence = GenerationEvidence()
        tool.generation_evidence = evidence

        class _Runtime:
            async def validate_semantic(self, scope="all", semantic_model_name=None):
                result = Mock()
                result.valid = True
                result.issues = []
                return result

        tool._runtime = _Runtime()
        with (
            patch.object(tool, "_reload_runtime", return_value=True),
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

    def test_resolves_target_artifact_validation_evidence(self, semantic_tools_with_runtime, tmp_path):
        tool, _ = semantic_tools_with_runtime
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")

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

    def test_dosi_resolves_osi_target_artifact_evidence(self, semantic_tools_with_runtime, tmp_path):
        tool, _ = semantic_tools_with_runtime
        artifact = tmp_path / "commerce.yml"
        artifact.write_text("semantic_model: commerce\n", encoding="utf-8")

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

    def test_exception_returns_failure(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("runtime crash")):
            result = tool.validate_semantic()

        assert result.success == 0
        assert "runtime crash" in result.error


class TestAttributionAnalyze:
    def test_tool_schema_exposes_drilldown_guardrail_parameters(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        schema = trans_to_function_tool(tool.attribution_analyze).params_json_schema

        assert {
            "where",
            "max_dimension_values",
            "time_dimension",
            "params",
        }.issubset(schema["properties"])
        assert "path" not in schema["properties"]
        assert "anomaly_context" not in schema["properties"]
        assert "exclusive" in schema["properties"]["baseline_end"]["description"].lower()
        assert "exclusive" in schema["properties"]["current_end"]["description"].lower()

    def test_tool_description_is_explicitly_non_causal(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        description = " ".join(trans_to_function_tool(tool.attribution_analyze).description.lower().split())

        assert "descriptive dimension analysis" in description
        assert "do not establish causation" in description
        assert "root cause analysis" not in description
        assert "failed, truncated, and non-additive dimensions are excluded" in description

    def test_success_builds_unified_request(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "metric": "revenue",
            "implementation": "dosi",
            "strategy": "term_wise",
            "dimension_ranking": [],
            "selected_dimensions": [],
            "top_dimension_values": [],
            "warnings": [{"code": "unequal_windows", "message": "not equal"}],
        }
        tool._attribute = AsyncMock(return_value=mock_result)

        result = tool.attribution_analyze(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_start="2024-01-01",
            baseline_end="2024-01-08",
            current_start="2024-01-08",
            current_end="2024-01-15",
            where="region = 'US'",
            max_dimension_values=25,
            time_dimension="orders.order_date",
            params={"currency": "USD"},
        )

        assert result.success == 1
        assert result.result["warnings"][0]["code"] == "unequal_windows"
        called_runtime, request = tool._attribute.await_args.args
        assert called_runtime is mock_runtime
        assert request.metric == "revenue"
        assert request.dimensions == ["region"]
        assert request.where_sql == "region = 'US'"
        assert request.max_values_per_dimension == 25
        assert request.time_dimension == "orders.order_date"
        assert request.params == {"currency": "USD"}
        mock_result.model_dump.assert_called_once_with(exclude_none=True)

    @pytest.mark.asyncio
    async def test_attribute_prefers_native_runtime(self, semantic_tools_with_runtime):
        tool, runtime = semantic_tools_with_runtime
        native_result = Mock()
        runtime.attribute = AsyncMock(return_value=native_result)
        request = AttributionRequest(
            metric="revenue",
            dimensions=["region"],
            baseline=AttributionWindow(start="2024-01-01", end="2024-01-08"),
            current=AttributionWindow(start="2024-01-08", end="2024-01-15"),
        )

        result = await tool._attribute(runtime, request)

        assert result is native_result

    def test_exception_returns_failure(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        tool._attribute = AsyncMock(side_effect=Exception("analysis failed"))

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

    def test_native_validation_exception_returns_runtime_payload(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        payload = Mock(
            error_type="semantic_validation_error",
            code="unknown_metric",
            message="Unknown metric 'revenues'.",
        )
        payload.model_dump.return_value = {
            "error_type": "semantic_validation_error",
            "code": "unknown_metric",
            "message": "Unknown metric 'revenues'.",
        }
        error = Exception(payload.message)
        error.payload = payload
        tool._attribute = AsyncMock(side_effect=error)

        result = tool.attribution_analyze(
            metric_name="revenues",
            candidate_dimensions=["region"],
            baseline_start="2024-01-01",
            baseline_end="2024-01-08",
            current_start="2024-01-08",
            current_end="2024-01-15",
        )

        assert result.success == 0
        assert result.error == "Unknown metric 'revenues'."
        assert result.result["code"] == "unknown_metric"


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


class TestReloadRuntime:
    def test_reload_success(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        new_runtime = Mock()
        # After clearing, the property should return a new runtime
        with patch.object(type(tool), "runtime", new_callable=lambda: property(lambda self: new_runtime)):
            result = tool._reload_runtime()
        assert result is True

    def test_reload_runtime_failure_returns_false(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime
        with patch("datus.tools.func_tool.semantic_tools.DosiRuntime", side_effect=RuntimeError("missing model")):
            assert tool._reload_runtime() is False


class TestCompressorModelName:
    """Verify that SemanticTools uses agent_config's model name for DataCompressor."""

    def test_compressor_uses_agent_config_model(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            config.active_model.return_value.model = "deepseek/deepseek-chat"
            tool = SemanticTools(agent_config=config)
            assert tool.compressor.model_name == "deepseek/deepseek-chat"

    def test_list_metrics_returns_envelope_without_compressor(self, semantic_tools_with_runtime):
        """list_metrics returns the canonical FuncToolListResult envelope.

        Regression: list_metrics used to wrap rows in DataCompressor output
        (``{original_rows, compressed_data, ...}``) regardless of size.
        After the envelope migration it returns ``{items, total, has_more,
        extra}`` with NO compressor artefacts — list_* never compresses.
        """
        tool, _ = semantic_tools_with_runtime
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


@pytest.mark.usefixtures("mock_runtime")
class TestQueryMetricsContextFilter:
    """``context_filter`` on the tool surface."""

    def test_context_filter_reaches_the_runtime(self, semantic_tools, mock_runtime):
        query_result = QueryResult(
            columns=["status", "item_rank"],
            data=[{"status": "paid", "item_rank": 1}],
            metadata={},
        )
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(
                metrics=["item_rank"],
                dimensions=["status"],
                where="status = 'paid'",
                context_filter="region = 'east'",
            )

        assert result.success == 1
        kwargs = mock_runtime.query_metrics.call_args.kwargs
        assert kwargs["where"] == "status = 'paid'"
        assert kwargs["context_filter"] == "region = 'east'"

    @pytest.mark.parametrize("context_filter", [None, "null", "", "  "])
    def test_an_unset_context_filter_is_not_passed_to_the_runtime(self, semantic_tools, mock_runtime, context_filter):
        query_result = QueryResult(columns=["revenue"], data=[{"revenue": 1}], metadata={})
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"], context_filter=context_filter)

        assert result.success == 1
        assert "context_filter" not in mock_runtime.query_metrics.call_args.kwargs

    def test_tool_schema_exposes_context_filter(self, semantic_tools):
        tool = trans_to_function_tool(semantic_tools.query_metrics)
        properties = tool.params_json_schema["properties"]
        assert "context_filter" in properties
        assert "context_filter" not in tool.params_json_schema.get("required", [])
        assert "before aggregation" in properties["context_filter"]["description"]
