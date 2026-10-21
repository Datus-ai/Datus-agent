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
from datus.tools.func_tool.semantic_tools import SemanticTools, _run_async, render_metric_list
from datus.tools.semantic_tools.models import (
    AttributionRequest,
    AttributionWindow,
    QueryResult,
)
from datus.utils.query_result_text import parse_query_result_text


def _text(result):
    """An executed query's result text, read back."""
    parsed = parse_query_result_text(result.result)
    assert parsed is not None, result.result
    return parsed


def _reported(result):
    """The lines a result reports beside its id and row count."""
    return [line for line in _text(result).lines if not line.startswith(("result_id: ", "rows: "))]


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
        assert parse_query_result_text(result["result"]).columns == ["revenue"]
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

    def test_query_metrics_returns_the_result_as_text(self, semantic_tools, mock_runtime):
        """An executed query reads as text: its id, its row count, then the rows as CSV."""
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

        result_id = _text(result).result_id
        assert result.result == (
            f"result_id: {result_id}\nrows: 2\n\ndate,revenue,orders\n2024-01-01,1000,50\n2024-01-02,1200,60"
        )
        cached = semantic_tools.get_cached_query_metrics_result(result_id)
        assert cached["row_count"] == 2
        # The text leaves the metadata out; the cached result keeps it for code that checks results.
        assert cached["metadata"] == {"execution_time": 0.5}

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

    def test_query_metrics_hints_in_list_values_missing_from_a_grouped_field(self, semantic_tools):
        calls = []

        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, **kwargs):
                calls.append(kwargs)
                return QueryResult(
                    columns=["segment", "customers"], data=[{"segment": 1251206313870885120, "customers": 4}]
                )

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["customers"],
            dimensions=["matches.segment"],
            where="matches.segment IN (1251206313870885120, 1281781308570207232)",
        )

        assert _reported(result) == [
            "note: "
            "matches.segment IN (...): 1 of 2 values matched no row in this query: 1281781308570207232. Check "
            "whether matches.segment holds these values at all before reading their absence as no data."
        ]
        assert len(calls) == 1

    def test_query_metrics_checks_an_in_list_on_an_ungrouped_field_by_grouping_it_too(self, semantic_tools):
        """A list of names checked against a name table that lacks some of them drops those names silently."""
        calls = []

        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, **kwargs):
                calls.append(kwargs)
                if "names.name" in kwargs["dimensions"]:
                    return QueryResult(columns=["name", "customers"], data=[{"name": "bridge", "customers": 3}])
                return QueryResult(
                    columns=["metric_time", "segment", "customers"], data=[{"segment": 7, "customers": 3}]
                )

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["customers"],
            dimensions=["matches.segment"],
            time_granularity="day",
            context_filter="names.name IN ('bridge', 'duel')",
            limit=10,
        )

        assert _reported(result) == [
            "note: "
            "names.name IN (...): 1 of 2 values matched no row in this query: duel. Check whether names.name "
            "holds these values at all before reading their absence as no data."
        ]
        # Grouped by the listed field alone: at most one row per listed value.
        assert calls[1]["dimensions"] == ["names.name"]
        assert calls[1]["time_granularity"] is None
        assert calls[1]["limit"] is None
        assert calls[1]["order_by"] is None
        assert calls[1]["context_filter"] == "names.name IN ('bridge', 'duel')"

    def test_query_metrics_rechecks_a_grouped_in_list_field_when_a_limit_cut_the_result(self, semantic_tools):
        """A limited result can drop listed values that do have rows; it cannot say which matched none."""
        calls = []

        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, **kwargs):
                calls.append(kwargs)
                if kwargs["limit"] is None:
                    return QueryResult(columns=["name", "customers"], data=[{"name": "bridge"}, {"name": "duel"}])
                return QueryResult(columns=["name", "customers"], data=[{"name": "bridge", "customers": 3}])

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["customers"],
            dimensions=["names.name"],
            where="names.name IN ('bridge', 'duel', 'arena')",
            order_by=["-customers"],
            limit=1,
        )

        assert _reported(result) == [
            "note: "
            "names.name IN (...): 1 of 3 values matched no row in this query: arena. Check whether names.name "
            "holds these values at all before reading their absence as no data."
        ]
        assert len(calls) == 2
        assert calls[1]["dimensions"] == ["names.name"]
        assert calls[1]["limit"] is None

    def test_query_metrics_skips_the_in_list_check_when_where_filters_on_a_selected_metric(self, semantic_tools):
        """A condition on a metric filters after aggregation, so a regrouped check would not mean the same."""
        calls = []

        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, **kwargs):
                calls.append(kwargs)
                return QueryResult(columns=["customers"], data=[{"customers": 3}])

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["customers"],
            where="names.name IN ('bridge', 'duel') AND customers > 2",
        )

        assert result.success == 1
        assert _reported(result) == []
        assert len(calls) == 1

    def test_filters_on_metric_matches_a_bare_metric_name_only(self):
        from datus.tools.func_tool.semantic_tools import _filters_on_metric

        assert _filters_on_metric("customers > 2", ["customers"]) is True
        assert _filters_on_metric("stats.customers > 2 AND customers_d7 > 0", ["customers"]) is False
        assert _filters_on_metric(None, ["customers"]) is False

    def test_result_column_for_prefers_the_qualified_name_and_refuses_an_ambiguous_short_one(self):
        from datus.tools.func_tool.semantic_tools import _result_column_for

        assert _result_column_for("names.name", ["matches.name", "names.name"]) == "names.name"
        assert _result_column_for("names.name", ["name", "customers"]) == "name"
        assert _result_column_for("names.name", ["name", "name", "customers"]) is None
        assert _result_column_for("names.name", ["customers"]) is None

    def test_query_metrics_drops_the_in_list_check_when_the_probe_fails(self, semantic_tools):
        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, **kwargs):
                if "names.name" in kwargs["dimensions"]:
                    raise RuntimeError("fan-out")
                return QueryResult(columns=["customers"], data=[{"customers": 3}])

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(metrics=["customers"], where="names.name IN ('bridge', 'duel')")

        assert result.success == 1
        assert _reported(result) == []

    def test_query_metrics_checks_each_in_list_field_once_and_skips_an_unreadable_probe(self, semantic_tools):
        probes = []

        class _Runtime:
            service_type = "dosi"

            async def query_metrics(self, **kwargs):
                if "names.name" in kwargs["dimensions"]:
                    probes.append(kwargs["dimensions"])
                    return QueryResult(columns=["something_else"], data=[{"something_else": 1}])
                return QueryResult(columns=["customers"], data=[{"customers": 3}])

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["customers"],
            where="names.name IN ('bridge') AND names.name IN ('duel')",
            context_filter="names.name IN ('arena')",
        )

        assert result.success == 1
        assert _reported(result) == []
        assert len(probes) == 1

    def test_in_list_conditions_reads_only_plain_lists_of_literals(self):
        from datus.tools.func_tool.semantic_tools import _in_list_conditions

        sql = (
            "a.mode IN (101, 102) AND b.name in ('it''s', 'x') AND c.id NOT IN (1, 2) "
            "AND d.id IN (SELECT id FROM t) AND e.code IN (f.code, 'y')"
        )

        assert _in_list_conditions(sql) == [("a.mode", ["101", "102"]), ("b.name", ["it's", "x"])]
        assert _in_list_conditions("NOT a.mode IN (101)") == []

    def test_query_metrics_says_how_to_query_a_metric_without_a_time_axis(self, semantic_tools):
        """The engine lists every time field of the model; the agent needs: no period here, the day in where."""
        from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
        from datus.tools.semantic_tools.models import SemanticValidationError

        snapshot = _detail_metric("snapshot_customers", {"base_kind": "aggregate", "datasets": ["customers_snapshot"]})

        class _Runtime:
            service_type = "dosi"

            async def list_metrics(self, path=None, limit=None, offset=0):
                return [snapshot] if offset == 0 else []

            async def query_metrics(self, metrics, **kwargs):
                raise SemanticValidationException(
                    SemanticValidationError(
                        code="no_primary_time_dimension",
                        metrics=["snapshot_customers__k_7"],
                        required_dimensions=["events.event_hour", "logins.stat_date"],
                        message='metric "snapshot_customers__k_7" has no primary time dimension '
                        "| candidates: events.event_hour, logins.stat_date",
                    )
                )

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(
            metrics=["snapshot_customers"], time_start="2024-01-01", time_end="2024-01-02", params={"k": [7]}
        )

        assert result.success == 0
        assert result.error.startswith("snapshot_customers has no time axis: the table (customers_snapshot) it reads")
        assert "without time_start/time_end" in result.error and "in `where`" in result.error
        assert "event_hour" not in result.error
        assert result.result["code"] == "no_primary_time_dimension"
        assert result.result["required_dimensions"] == []

    def test_the_no_time_axis_advice_stands_without_the_catalog(self, semantic_tools):
        from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
        from datus.tools.semantic_tools.models import SemanticValidationError

        class _Runtime:
            service_type = "dosi"

            async def list_metrics(self, path=None, limit=None, offset=0):
                raise RuntimeError("catalog unavailable")

            async def query_metrics(self, metrics, **kwargs):
                raise SemanticValidationException(
                    SemanticValidationError(
                        code="no_primary_time_dimension",
                        metrics=["snapshot_customers"],
                        message='metric "snapshot_customers" has no primary time dimension',
                    )
                )

        semantic_tools._runtime = _Runtime()
        result = semantic_tools.query_metrics(metrics=["snapshot_customers"], time_start="2024-01-01")

        assert result.success == 0
        assert result.error.startswith("snapshot_customers has no time axis: the table it reads")

    def test_query_metrics_small_result_shows_every_row(self, semantic_tools):
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

        assert _text(result).row_count == 2
        assert _text(result).csv == "id,value\n1,100\n2,200"

    def test_query_metrics_long_result_shows_its_first_and_last_rows(self, semantic_tools):
        """Past 20 rows the preview shows ten at each end around the count left out;
        the cache keeps every row."""
        rows = [{"id": i, "value": i * 100} for i in range(50)]
        query_result = QueryResult(
            columns=["id", "value"],
            data=rows,
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["value"])

        assert _text(result).row_count == 50
        assert _text(result).csv.splitlines() == (
            ["id,value"]
            + [f"{i},{i * 100}" for i in range(10)]
            + ["... 30 rows omitted"]
            + [f"{i},{i * 100}" for i in range(40, 50)]
        )

        cached_result = semantic_tools.get_cached_query_metrics_result(_text(result).result_id)
        assert cached_result["row_count"] == 50
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
        result_id = _text(self._query_fifty_rows(semantic_tools)).result_id

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
            result_id = _text(semantic_tools.query_metrics(metrics=["note"])).result_id

        page = semantic_tools.get_query_metrics_result(result_id, offset=1, limit=1)

        assert page.success == 1
        assert page.result["returned"] == 1
        assert big in page.result["csv"]

    def test_get_query_metrics_result_serializes_only_the_page(self, semantic_tools):
        """The rows before ``offset`` are sliced away, not parsed and discarded."""
        result_id = _text(self._query_fifty_rows(semantic_tools)).result_id
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
        result_id = _text(self._query_fifty_rows(semantic_tools)).result_id

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
        result_id = _text(self._query_fifty_rows(semantic_tools)).result_id

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

    def test_query_metrics_empty_result_keeps_its_header(self, semantic_tools):
        """No rows, but the columns the query would have returned are still named."""
        query_result = QueryResult(columns=["region", "value"], data=[], metadata={})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["value"])

        assert _text(result).row_count == 0
        assert _text(result).csv == "region,value"

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

    def test_query_metrics_result_reports_warnings_and_parameter_columns(self, semantic_tools):
        """An executed result reports the engine's warnings and the columns a
        parameter binding names; the SQL and dialect stay out."""
        warning = {"level": "warning", "code": "where_unread", "message": "where was not read", "hint": "move it"}
        bound = {"column": "active__k_h1", "metric": "active", "params": {"k": "1-7"}}
        query_result = QueryResult(
            columns=["metric_time__day", "revenue", "active__k_h1"],
            data=[{"metric_time__day": "2024-01-01", "revenue": 500, "active__k_h1": 3}],
            metadata={
                "sql": "SELECT " + "x, " * 5000 + "1",
                "dialect": "duckdb",
                "row_count": 1,
                "outputs": [{"column": "revenue", "metric": "revenue"}, bound],
                "warnings": [warning],
            },
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue", "active"], dimensions=["metric_time__day"])

        assert _text(result).columns == ["metric_time__day", "revenue", "active__k_h1"]
        assert _reported(result) == [
            "warning where_unread: where was not read; hint: move it",
            "column active__k_h1: active with k=1-7",
        ]
        assert "SELECT" not in result.result and "duckdb" not in result.result

    def test_query_metrics_result_lines_carry_what_a_warning_message_leaves_out(self, semantic_tools):
        warnings = [
            {"level": "info", "code": "filter_landed", "message": "region = 'N' filters orders", "part": "orders"},
            {"level": "warning", "code": "where_unread", "message": "a condition was not read", "part": "visits"},
            "plain text warning",
        ]
        outputs = [{"column": "share__k_h1", "metric": "share", "params": {"k": [1, 7], "region": "北区"}}]
        query_result = QueryResult(
            columns=["share__k_h1"],
            data=[{"share__k_h1": 0.5}],
            metadata={"warnings": warnings, "outputs": outputs},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["share"])

        assert _reported(result) == [
            "info filter_landed: region = 'N' filters orders",
            "warning where_unread: a condition was not read (part: visits)",
            "warning: plain text warning",
            "column share__k_h1: share with k=[1, 7], region=北区",
        ]

    def test_query_metrics_result_with_nothing_to_report_has_no_report_lines(self, semantic_tools):
        query_result = QueryResult(
            columns=["revenue"],
            data=[{"revenue": 500}],
            metadata={
                "sql": "SELECT " + "x, " * 5000 + "1",
                "dialect": "duckdb",
                "row_count": 1,
                "outputs": [{"column": "revenue", "metric": "revenue"}],
                "warnings": [],
            },
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"])

        assert _text(result).lines == [f"result_id: {_text(result).result_id}", "rows: 1"]

    def test_query_metrics_dry_run_keeps_the_compiled_sql_body(self, semantic_tools):
        """dry_run exists to return the SQL, and publish evidence hashes it."""
        query_result = QueryResult(columns=["sql"], data=[], metadata={"sql": "SELECT 1", "dry_run": True})

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"], dry_run=True)

        assert result.result["metadata"]["sql"] == "SELECT 1"
        assert "sql_sha256" not in result.result["metadata"]

    def test_query_metrics_keeps_every_column_over_the_budget(self, semantic_tools):
        """Every column of a metric result was asked for; the budget cuts rows instead.

        Driven with a deliberately tiny budget so the sample is over it whatever
        the production budget.
        """
        semantic_tools.compressor.max_tolerable_tokens = 200
        columns = [f"metric_{index}" for index in range(12)]
        query_result = QueryResult(
            columns=columns,
            data=[{name: 987654321 + index for index, name in enumerate(columns)} for _ in range(30)],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=columns)

        assert _text(result).columns == columns
        assert _text(result).shown_rows < 20, "expected fewer rows than the default ten at each end"
        assert f"... {30 - _text(result).shown_rows} rows omitted" in _text(result).csv

    def test_query_metrics_cuts_long_values_once_one_row_at_each_end_is_over(self, semantic_tools):
        semantic_tools.compressor.max_tolerable_tokens = 120
        query_result = QueryResult(
            columns=["id", "bitmap"], data=[{"id": i, "bitmap": "01" * 400} for i in range(5)], metadata={}
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["bitmap"])

        rows_line = _text(result).lines[1]
        assert rows_line.startswith("rows: 5; text values longer than ")
        assert _text(result).csv.splitlines()[1].endswith("…")
        assert _text(result).csv.splitlines()[2] == "... 3 rows omitted"

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

        assert _text(result).columns == columns
        assert _text(result).shown_rows == 20

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
            result = semantic_tools.query_metrics(metrics=["revenue"], dry_run=True)

        assert result.success == 1
        meta = result.result["metadata"]
        # A dry run keeps its metadata: non-serializable entries are dropped,
        # serializable ones pass through.
        assert "dataflow_plan" not in meta
        assert meta["sql"] == "SELECT 1"
        assert meta["count"] == 42

    def test_query_metrics_csv_header_names_every_column(self, semantic_tools):
        query_result = QueryResult(
            columns=["date", "revenue", "orders", "customers"],
            data=[
                {"date": "2024-01-01", "revenue": 1000, "orders": 50, "customers": 30},
            ],
            metadata={},
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=query_result):
            result = semantic_tools.query_metrics(metrics=["revenue"])

        assert _text(result).columns == ["date", "revenue", "orders", "customers"]

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
        assert names == [
            "list_metrics",
            "get_metric",
            "get_dataset",
            "query_metrics",
            "validate_semantic",
            "attribution_analyze",
        ]
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
        # One line per metric under its path: the knowledge-base path wins over
        # the runtime's own, and kind, time axis, measures and the always-empty
        # dimensions stay out — get_metric carries the per-metric detail.
        assert result.result == "Commerce / Orders:\n- orders -- Order count"
        mock_runtime.list_metrics.assert_called_once_with(path=None, limit=200, offset=0)

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

    def test_how_a_metric_derives_is_left_to_get_metric(self, semantic_tools_with_runtime):
        """The listing picks metrics; get_metric's ``expr`` says how each is
        computed, so the derivation keys a runtime publishes stay out of it."""
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

        assert result.result == "- area_score -- Weighted total\n- store_issue_num_rn -- Rank"

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
        # The whole filtered list fits the page: no ``more:`` line.
        assert result.result == "Finance:\n- m1\nFinance / Revenue:\n- m3"
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
        assert result.result == ("Metrics / flights:\n- m3\nmore: 1 of 3 metrics shown; call again with offset=2")
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
        assert result.result == "no metrics"
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
        assert result.result == "Metrics / flights:\n- flight_count -- Flights"

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
            assert result.result == "no metrics"
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

    def test_runtime_path_does_not_stand_in_for_the_kb_path(self, semantic_tools_with_runtime):
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
        # The KB knows no path for this metric, so the runtime's own path must
        # not leak in as a substitute: the metric is listed under no path.
        assert result.result == "- orders"

    def test_no_path_keeps_runtime_available_when_kb_read_fails(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime
        tool.metric_rag.search_all_metrics.side_effect = RuntimeError("KB unavailable")
        mock_metric = SimpleNamespace(name="orders", description="Order count", path=["Yaml", "Path"])

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        assert result.result == "- orders -- Order count"
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


def test_render_metric_list_puts_each_metric_on_one_line():
    rows = [
        {"name": "revenue", "description": "Net revenue,\n  after refunds", "unit": "USD", "path": ["Sales"]},
        {"name": "orders", "description": "Orders", "path": ["Sales"]},
        {"name": "margin", "path": ["Finance"]},
    ]
    # A runtime page has no total, only the offset to call again with.
    assert render_metric_list(rows, total=None, has_more=True, next_offset=3) == (
        "Sales:\n"
        "- revenue -- Net revenue, after refunds [unit USD]\n"
        "- orders -- Orders\n"
        "Finance:\n"
        "- margin\n"
        "more: 3 metrics shown; call again with offset=3"
    )
    assert render_metric_list([], total=0, has_more=False, next_offset=None) == "no metrics"


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


# What Dosi's Engine.metric / Engine.dataset return, trimmed to the keys the
# renderers read.
RETURNING_METRIC = {
    "name": "returning_customers",
    "description": "点击玩家中在对比窗口内又点过活动按钮的人数",
    "kind": "aggregate",
    "expr": "clicking_customers FILTER (WHERE visits.customer_rows_in_compare_window > 0)",
    "datasets": ["visits"],
    "dimension_datasets": [{"name": "visits"}, {"name": "customers", "fields": ["region", "level"]}],
    "compares_across": None,
    "window": None,
    "params": [
        {
            "name": "compare_start",
            "type": "string",
            "default": "19700101",
            "description": "对比窗口的第一天（YYYYMMDD，含）",
            "slots": ["filter.literal"],
            "declared_on": "click_rows_in_compare_window",
        },
        {"name": "k", "type": "int", "default": 7, "allowed": [1, 7, 30]},
        {"name": "days", "type": "int", "default": 3, "min": 1, "max": 30},
    ],
    "required_dimensions": [],
    "requires_time_axis": False,
    "time_granularities": ["day", "week", "month", "quarter", "year"],
}

VISITS_DATASET = {
    "name": "visits",
    "source": "db.fact_visits",
    "description": "活动按钮点击日志",
    "primary_time_dimension": "event_hour",
    "primary_key": ["event_id"],
    "fields": [
        {"name": "tdbank_imp_date", "description": "小时分区"},
        {
            "name": "event_hour",
            "expr": "tdbank_imp_date",
            "reads": ["tdbank_imp_date"],
            "is_time": True,
            "time_format": "%Y%m%d%H",
        },
        {"name": "event_day", "expr": "SUBSTR(tdbank_imp_date,\n  1, 8)", "reads": ["tdbank_imp_date"]},
        {"name": "tier", "expr": "CASE " + "WHEN x > 1 THEN 'a' " * 20 + "END", "reads": ["x"]},
        {"name": "uid", "is_dimension": False},
        {"name": "day_text", "time_format": "%Y%m%d"},
        {
            "name": "customer_spend",
            "description": "玩家付费",
            "expr": "fixed(pay_amount, customer_id)",
            "datasets": ["pay"],
            "time_axis": "pay.pay_date",
        },
    ],
    "relationships": [
        {"to": "customers", "columns": [["customer_id", "customer_id"]], "cardinality": "many_to_one"},
        {
            "name": "visit_amount",
            "to": "pay",
            "columns": [["customer_id", "customer_id"]],
            "cardinality": "many_to_many",
        },
    ],
}


class TestRenderMetricDetail:
    def test_one_line_per_fact(self):
        from datus.tools.func_tool.semantic_tools import render_metric_detail

        text = render_metric_detail(RETURNING_METRIC, ["砺刃使者", "活动与抽奖"])

        assert text.splitlines() == [
            "description: 点击玩家中在对比窗口内又点过活动按钮的人数",
            "path: 砺刃使者 / 活动与抽奖",
            "expr: clicking_customers FILTER (WHERE visits.customer_rows_in_compare_window > 0)",
            "datasets: visits",
            "dimension_datasets: visits, customers(region, level)",
            "params:",
            "- compare_start string = 19700101 -- 对比窗口的第一天（YYYYMMDD，含）"
            " [declared_on click_rows_in_compare_window]",
            "- k int = 7 [one of: 1, 7, 30]",
            "- days int = 3 [1..30]",
            "time: metric_time (day, week, month, quarter, year)",
        ]

    def test_states_what_a_query_must_add_and_a_snapshot(self):
        from datus.tools.func_tool.semantic_tools import render_metric_detail

        detail = {
            **RETURNING_METRIC,
            "kind": "ratio",
            "params": None,
            "required_dimensions": ["scores.item"],
            "requires_time_axis": True,
            "time_granularities": [],
            "window": {
                "family": "rank",
                "function": "rank",
                "order": {"by": "value", "direction": "desc"},
                "partition": {"mode": "query_dimensions_except", "exclude": ["scores.item"]},
            },
            "compares_across": ["scores.item"],
        }

        lines = render_metric_detail(detail).splitlines()

        assert "kind: ratio" in lines
        assert (
            "window: rank, rank, order by value desc, partition query_dimensions_except excluding scores.item" in lines
        )
        assert "compares_across: scores.item" in lines
        assert "required: scores.item" in lines
        assert "requires_time_axis: true" in lines
        assert lines[-1] == "time: none"
        assert not any(line.startswith(("path:", "params")) for line in lines)


class TestRenderDatasetDetail:
    def test_fields_markers_and_relationships(self):
        from datus.tools.func_tool.semantic_tools import render_dataset_detail

        lines = render_dataset_detail(VISITS_DATASET).splitlines()

        assert lines == [
            "活动按钮点击日志 | source: db.fact_visits | time: event_hour | pk: event_id",
            "- tdbank_imp_date -- 小时分区",
            "- tdbank_imp_date AS event_hour [time %Y%m%d%H]",
            "- SUBSTR(tdbank_imp_date, 1, 8) AS event_day",
            "- tier [from x]",
            "- uid [not a dimension]",
            "- day_text [format %Y%m%d]",
            "- fixed(pay_amount, customer_id) AS customer_spend [reads pay; time_axis pay.pay_date] -- 玩家付费",
            "relationships:",
            "- many_to_one customers on customer_id=customer_id",
            "- visit_amount: many_to_many pay on customer_id=customer_id",
        ]

    def test_a_bare_dataset_has_no_header(self):
        from datus.tools.func_tool.semantic_tools import render_dataset_detail

        text = render_dataset_detail({"name": "t", "source": "t", "fields": [{"name": "a"}]})

        assert text == "- a"


class TestGetMetric:
    @pytest.mark.parametrize("name", ["", "   ", None, "null"])
    def test_rejects_a_missing_name(self, semantic_tools_with_runtime, name):
        tool, _ = semantic_tools_with_runtime

        result = tool.get_metric(name=name)

        assert result.success == 0
        assert "requires a metric name" in result.error

    def test_renders_the_engine_description_with_the_kb_path(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", return_value=RETURNING_METRIC),
            patch.object(tool, "_metric_subject_paths", return_value={"returning_customers": ["A", "B"]}),
        ):
            result = tool.get_metric(name="returning_customers")

        assert result.success == 1
        assert result.result.startswith("description: ")
        assert "path: A / B" in result.result
        assert "dimension_datasets: visits, customers(region, level)" in result.result
        mock_runtime.describe_metric.assert_called_once_with("returning_customers")

    def test_unknown_name_points_at_list_metrics(self, semantic_tools_with_runtime):
        from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
        from datus.tools.semantic_tools.models import SemanticValidationError

        tool, _ = semantic_tools_with_runtime
        error = SemanticValidationException(SemanticValidationError(code="unknown_metric", message="unknown"))

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=error):
            result = tool.get_metric(name="no_such_metric")

        assert result.success == 0
        assert "no_such_metric" in result.error
        assert "selected model" in result.error

    def test_a_runtime_failure_is_reported(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=RuntimeError("boom")):
            result = tool.get_metric(name="revenue")

        assert result.success == 0
        assert "boom" in result.error

    def test_a_kb_path_failure_keeps_the_detail(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", return_value=RETURNING_METRIC),
            patch.object(tool, "_metric_subject_paths", side_effect=RuntimeError("kb down")),
        ):
            result = tool.get_metric(name="returning_customers")

        assert result.success == 1
        assert "path:" not in result.result


class TestMetricRecord:
    """The structured record metric artifacts and insight snapshots store."""

    def test_keeps_the_description_path_and_dimensions(self, semantic_tools_with_runtime):
        from datus.tools.semantic_tools.models import DimensionInfo

        tool, _ = semantic_tools_with_runtime
        detail = {"name": "revenue", "expr": "SUM(orders.amount)", "measures": ["orders_amount_sum"], "path": ["x"]}
        dimensions = [DimensionInfo(name="orders.region", description="Sales region")]

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=[detail, dimensions]),
            patch.object(tool, "_metric_subject_paths", return_value={"revenue": ["Sales", "Orders"]}),
        ):
            result = tool.metric_record(name="revenue", path=["Sales", "Orders"])

        assert result.success == 1
        record = result.result
        # The knowledge base owns the path; compiled measure names stay out.
        assert record["path"] == ["Sales", "Orders"]
        assert "measures" not in record
        assert record["expr"] == "SUM(orders.amount)"
        assert record["dimensions"] == [
            {key: value for key, value in dimensions[0].model_dump().items() if value is not None}
        ]

    def test_a_kb_path_failure_leaves_the_path_out(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        with (
            patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=[{"name": "revenue"}, []]),
            patch.object(tool, "_metric_subject_paths", side_effect=RuntimeError("kb down")),
        ):
            result = tool.metric_record(name="revenue")

        assert result.result == {"name": "revenue", "dimensions": []}

    def test_a_runtime_failure_is_a_failed_result(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=RuntimeError("engine down")):
            result = tool.metric_record(name="revenue")

        assert result.success == 0
        assert "engine down" in result.error


class TestGetDataset:
    def test_renders_the_engine_description(self, semantic_tools_with_runtime):
        tool, mock_runtime = semantic_tools_with_runtime

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=VISITS_DATASET):
            result = tool.get_dataset(name="visits")

        assert result.success == 1
        assert "- fixed(pay_amount, customer_id) AS customer_spend" in result.result
        mock_runtime.describe_dataset.assert_called_once_with("visits")

    def test_unknown_name_returns_the_candidates(self, semantic_tools_with_runtime):
        from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
        from datus.tools.semantic_tools.models import SemanticValidationError

        tool, _ = semantic_tools_with_runtime
        error = SemanticValidationException(
            SemanticValidationError(code="unknown_dataset", message="unknown dataset 'x' | candidates: visits")
        )

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=error):
            result = tool.get_dataset(name="x")

        assert result.success == 0
        assert "candidates: visits" in result.error

    def test_rejects_a_missing_name(self, semantic_tools_with_runtime):
        tool, _ = semantic_tools_with_runtime

        assert tool.get_dataset(name="").success == 0


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

    def test_hive_keeps_session_configuration_and_vendor_selection(self, semantic_tools):
        profile = Mock()
        profile.to_dict.return_value = {
            "type": "hive",
            "host": "localhost",
            "port": 10000,
            "extra": {
                "server_type": "inceptor",
                "auth": "KERBEROS",
                "kerberos_service_name": "hive",
                "configuration": {"hive.exec.dynamic.partition": "true"},
            },
        }
        semantic_tools.agent_config.current_db_config.return_value = profile
        result = semantic_tools._extract_db_config("warehouse")
        assert result["server_type"] == "inceptor"
        assert result["auth"] == "KERBEROS"
        assert result["kerberos_service_name"] == "hive"
        assert result["configuration"] == {"hive.exec.dynamic.partition": "true"}

    def test_extracts_and_filters_db_config(self, semantic_tools):
        """Preserve native connection values and exclude Agent-only fields."""
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
        assert result["port"] == 3306
        assert result["role"] == "ANALYST"
        assert result["private_key_file"] == "/tmp/rsa_key.p8"
        assert result["private_key_file_pwd"] == 1234
        assert "extra" not in result
        assert "path_pattern" not in result
        assert result["catalog"] == "skip"

    def test_preserves_hana_tls_options_and_explicit_values(self, semantic_tools):
        """HANA's native executor must receive booleans, including false."""
        db_config = Mock()
        db_config.to_dict.return_value = {
            "type": "hana",
            "port": 39041,
            "encrypt": True,
            "default": True,
            "extra": {
                "encrypt": False,
                "ssl_validate_certificate": False,
                "timeout_seconds": 30,
                "ssl_trust_store": "/tmp/hana-ca.pem",
                "userkey": None,
            },
        }
        semantic_tools.agent_config.current_db_config.return_value = db_config

        result = semantic_tools._extract_db_config("hana")

        assert result == {
            "type": "hana",
            "port": 39041,
            "encrypt": True,
            "ssl_validate_certificate": False,
            "timeout_seconds": 30,
            "ssl_trust_store": "/tmp/hana-ca.pem",
        }


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

    def test_list_metrics_returns_text_without_compressor(self, semantic_tools_with_runtime):
        """list_metrics answers with its own text page.

        Regression: list_metrics used to wrap rows in DataCompressor output
        (``{original_rows, compressed_data, ...}``) regardless of size; list_*
        never compresses.
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
        assert result.result == "- orders"


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


# ── Ontology tools ─────────────────────────────────────────────────


def _ontology_config(tmp_path=None):
    config = Mock()
    config.active_model.return_value.model = "gpt-4o"
    config.current_datasource = "flights_db"
    config.current_db_config.return_value = None
    config.runtime_db_context.return_value = {}
    config.path_manager.semantic_model_path.return_value = "/tmp/models/flights_db"
    config.path_manager.ontology_dir = "/tmp/subject/ontology"
    return config


def _validation_exception(**fields):
    from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
    from datus.tools.semantic_tools.models import SemanticValidationError

    return SemanticValidationException(SemanticValidationError(**fields))


class TestOntologyToolRegistration:
    def _names(self, tool):
        with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as convert:
            convert.side_effect = lambda function: SimpleNamespace(name=function.__name__)
            return [registered.name for registered in tool.available_tools()]

    def test_ontology_tools_hidden_by_default(self):
        with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
            tool = SemanticTools(agent_config=_ontology_config())
        assert not set(SemanticTools.ONTOLOGY_TOOL_NAMES) & set(self._names(tool))

    def test_ontology_tools_exposed_when_enabled(self):
        with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
            tool = SemanticTools(agent_config=_ontology_config(), ontology_enabled=True)
        names = self._names(tool)
        assert names[-3:] == ["describe_concept", "find_concept_paths", "select_records"]

    def test_all_tools_name_lists_ontology_tools(self):
        assert set(SemanticTools.ONTOLOGY_TOOL_NAMES) <= set(SemanticTools.all_tools_name())

    @staticmethod
    def _layout(tmp_path, *, plain_model: bool):
        """A datasource's model dir (with or without a plain model) and the project ontology dir with an inline mapping."""
        models_dir = tmp_path / "subject" / "semantic_models" / "flights_db"
        ontology_dir = tmp_path / "subject" / "ontology"
        models_dir.mkdir(parents=True)
        ontology_dir.mkdir(parents=True)
        if plain_model:
            (models_dir / "flights.yaml").write_text("version: '0.2.0.dev0'\nsemantic_model:\n  - name: flights\n")
        (ontology_dir / "flights.mapping.yaml").write_text(
            "version: 0.2.0.dev0\nname: flights_mapping\n"
            "ontology_ref: {name: Flights, document: {version: 0.2.0.dev0, name: Flights, ontology: []}}\n"
            "semantic_model_ref: {name: flights, document: {version: 0.2.0.dev0, name: flights, datasets: []}}\n"
            "concept_mappings: []\n"
        )
        config = _ontology_config()
        config.path_manager.semantic_model_path.return_value = str(models_dir)
        config.path_manager.ontology_dir = ontology_dir
        return config, ontology_dir

    def _runtime_config(self, config, **kwargs):
        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.DosiRuntime") as runtime,
        ):
            tool = SemanticTools(agent_config=config, **kwargs)
            assert tool.runtime is runtime.return_value
            return tool, runtime.call_args.args[0]

    def test_plain_model_datasource_loads_ontology_only_when_enabled(self, tmp_path):
        config, ontology_dir = self._layout(tmp_path, plain_model=True)
        _, plain = self._runtime_config(config)
        _, enabled = self._runtime_config(config, ontology_enabled=True)
        assert plain.ontology_dir is None
        assert enabled.ontology_dir == str(ontology_dir)

    def test_mapping_only_datasource_loads_its_mappings_without_the_ontology_tools(self, tmp_path):
        """Without a plain model the inline mappings are the datasource's only models, for every agent."""
        config, ontology_dir = self._layout(tmp_path, plain_model=False)
        tool, plain = self._runtime_config(config)
        assert plain.ontology_dir == str(ontology_dir)
        assert not set(SemanticTools.ONTOLOGY_TOOL_NAMES) & set(self._names(tool))


@pytest.fixture
def ontology_tools():
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        tool = SemanticTools(agent_config=_ontology_config(), ontology_enabled=True)
    runtime = Mock()
    tool._runtime = runtime
    return tool, runtime


class TestDescribeConcept:
    def test_catalog(self, ontology_tools):
        tool, runtime = ontology_tools
        runtime.describe = AsyncMock(return_value={"concepts": [{"name": "Flight"}]})
        result = tool.describe_concept()
        assert result.success == 1
        assert result.result == {"concepts": [{"name": "Flight"}]}
        runtime.describe.assert_awaited_once_with(None)

    @pytest.mark.parametrize("placeholder", ["", "null", "None"])
    def test_null_placeholders_mean_catalog(self, ontology_tools, placeholder):
        tool, runtime = ontology_tools
        runtime.describe = AsyncMock(return_value={})
        tool.describe_concept(placeholder)
        assert runtime.describe.await_args.args[0] in (None, "")

    def test_concept(self, ontology_tools):
        tool, runtime = ontology_tools
        runtime.describe = AsyncMock(return_value={"concept": "FlightAlert"})
        assert tool.describe_concept("FlightAlert").result == {"concept": "FlightAlert"}
        runtime.describe.assert_awaited_once_with("FlightAlert")

    def test_unknown_concept_returns_structured_candidates(self, ontology_tools):
        tool, runtime = ontology_tools
        runtime.describe = AsyncMock(
            side_effect=_validation_exception(
                code="unknown_concept", candidates=["Flight", "FlightAlert"], message="unknown concept 'Flght'"
            )
        )
        result = tool.describe_concept("Flght")
        assert result.success == 0
        assert result.result["code"] == "unknown_concept"
        assert result.result["candidates"] == ["Flight", "FlightAlert"]

    def test_infrastructure_failure_is_a_plain_failure(self, ontology_tools):
        tool, runtime = ontology_tools
        runtime.describe = AsyncMock(side_effect=RuntimeError("no ontology is loaded"))
        result = tool.describe_concept()
        assert result.success == 0
        assert "no ontology is loaded" in result.error
        assert result.result is None

    def test_runtime_unavailable(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
            patch("datus.tools.func_tool.semantic_tools.DosiRuntime", side_effect=RuntimeError("bad yaml")),
        ):
            tool = SemanticTools(agent_config=_ontology_config(), ontology_enabled=True)
            result = tool.describe_concept()
        assert result.success == 0
        assert "describe_concept requires" in result.error and "bad yaml" in result.error


class TestFindConceptPaths:
    def test_paths(self, ontology_tools):
        tool, runtime = ontology_tools
        payload = {"paths": [{"path": "Flight.arrives_at"}], "equivalences": [], "max_hops": 3}
        runtime.concept_paths = AsyncMock(return_value=payload)
        result = tool.find_concept_paths(" Flight ", "Airport")
        assert result.result == payload
        runtime.concept_paths.assert_awaited_once_with("Flight", "Airport")

    @pytest.mark.parametrize("source,target", [("", "Airport"), ("Flight", ""), (None, None)])
    def test_requires_both_concepts(self, ontology_tools, source, target):
        tool, runtime = ontology_tools
        runtime.concept_paths = AsyncMock()
        result = tool.find_concept_paths(source, target)
        assert result.success == 0
        runtime.concept_paths.assert_not_called()

    def test_invalid_path_is_structured(self, ontology_tools):
        tool, runtime = ontology_tools
        runtime.concept_paths = AsyncMock(
            side_effect=_validation_exception(code="invalid_concept_path", message="no path from Flight to Moon")
        )
        result = tool.find_concept_paths("Flight", "Moon")
        assert result.success == 0
        assert result.result["code"] == "invalid_concept_path"


class TestSelectRecords:
    @staticmethod
    def _result(rows):
        from datus.tools.semantic_tools.models import QueryResult

        return QueryResult(
            columns=["alert_id", "priority"],
            data=rows,
            metadata={
                "sql": "SELECT ...",
                "row_count": len(rows),
                "requested_columns": [{"requested": "FlightAlert.id", "column": "alert_id"}],
            },
        )

    def test_returns_the_result_text_and_caches_the_full_result(self, ontology_tools):
        tool, runtime = ontology_tools
        rows = [{"alert_id": "A007", "priority": "P1"}, {"alert_id": "A011", "priority": "P1"}]
        runtime.select = AsyncMock(return_value=self._result(rows))
        result = tool.select_records(
            from_concept="FlightAlert",
            fields=["FlightAlert.id", "FlightAlert.priority"],
            where="FlightAlert.status = 'OPEN'",
        )
        assert result.success == 1
        text = _text(result)
        assert text.columns == ["alert_id", "priority"]
        assert text.csv.splitlines()[1:] == ["A007,P1", "A011,P1"]
        # Nothing to report beside the rows: the SQL and column map stay out of the text.
        assert _reported(result) == []
        cached = tool.get_cached_query_metrics_result(text.result_id)
        assert cached["row_count"] == 2
        assert "A011" in cached["csv"]
        kwargs = runtime.select.await_args.kwargs
        assert kwargs["limit"] == SemanticTools.SELECT_RECORDS_DEFAULT_LIMIT
        assert kwargs["where"] == "FlightAlert.status = 'OPEN'"

    def test_a_full_default_page_says_there_may_be_more(self, ontology_tools, monkeypatch):
        tool, runtime = ontology_tools
        monkeypatch.setattr(SemanticTools, "SELECT_RECORDS_DEFAULT_LIMIT", 2)
        rows = [{"alert_id": "A007", "priority": "P1"}, {"alert_id": "A011", "priority": "P1"}]
        runtime.select = AsyncMock(return_value=self._result(rows))

        result = tool.select_records(from_concept="FlightAlert", fields=["FlightAlert.id"])

        assert _text(result).lines[1] == "rows: 2 (reached the 2-row cap; there may be more)"

    def test_a_limit_the_caller_asked_for_is_not_a_cap(self, ontology_tools):
        tool, runtime = ontology_tools
        rows = [{"alert_id": "A007", "priority": "P1"}, {"alert_id": "A011", "priority": "P1"}]
        runtime.select = AsyncMock(return_value=self._result(rows))

        result = tool.select_records(from_concept="FlightAlert", fields=["FlightAlert.id"], limit=2)

        assert _text(result).lines[1] == "rows: 2"

    def test_explicit_limit_and_order(self, ontology_tools):
        tool, runtime = ontology_tools
        runtime.select = AsyncMock(return_value=self._result([]))
        tool.select_records(
            from_concept="FlightAlert", fields="FlightAlert.id", order_by=["-FlightAlert.id"], limit="5"
        )
        kwargs = runtime.select.await_args.kwargs
        assert kwargs["fields"] == ["FlightAlert.id"]
        assert kwargs["order_by"] == ["-FlightAlert.id"]
        assert kwargs["limit"] == 5

    @pytest.mark.parametrize(
        "from_concept,fields", [("", ["FlightAlert.id"]), ("FlightAlert", []), ("FlightAlert", None)]
    )
    def test_requires_concept_and_fields(self, ontology_tools, from_concept, fields):
        tool, runtime = ontology_tools
        runtime.select = AsyncMock()
        result = tool.select_records(from_concept=from_concept, fields=fields)
        assert result.success == 0
        runtime.select.assert_not_called()

    def test_value_not_in_domain_returns_allowed_values(self, ontology_tools):
        tool, runtime = ontology_tools
        violations = [{"reference": "FlightAlert.status", "literal": "unresolved", "allowed": ["OPEN", "RESOLVED"]}]
        runtime.select = AsyncMock(
            side_effect=_validation_exception(
                code="value_not_in_domain",
                candidates=["OPEN", "RESOLVED"],
                detail={"violations": violations},
                message='"unresolved" is not a value of FlightAlert.status',
            )
        )
        result = tool.select_records(
            from_concept="FlightAlert", fields=["FlightAlert.id"], where="FlightAlert.status = 'unresolved'"
        )
        assert result.success == 0
        assert result.result["code"] == "value_not_in_domain"
        assert result.result["detail"]["violations"] == violations
