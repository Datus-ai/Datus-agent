# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for BIFuncTool - all CI-level tests (no external deps)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# ---- Minimal stubs for datus_bi_core (so tests run without the package) ----


class _AuthParam:
    def __init__(self, **kwargs):
        pass


class _ChartInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class _DashboardInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class _DatasetInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class MockListDashboardsMixin:
    def list_dashboards(self, search="", page_size=20):
        return [_DashboardInfo(id=1, name="Test Dashboard")]


class MockDashboardWriteMixin:
    def create_dashboard(self, spec):
        return _DashboardInfo(id=10, name=spec.title)

    def update_dashboard(self, dashboard_id, spec):
        return _DashboardInfo(id=dashboard_id, name=spec.title)

    def delete_dashboard(self, dashboard_id):
        return True


class MockChartWriteMixin:
    def create_chart(self, spec, dashboard_id=None):
        return _ChartInfo(id=5, name=spec.title, chart_type=spec.chart_type)

    def update_chart(self, chart_id, spec):
        return _ChartInfo(id=chart_id, name=spec.title, chart_type=spec.chart_type)

    def delete_chart(self, chart_id):
        return True

    def add_chart_to_dashboard(self, dashboard_id, chart_id):
        return True


class MockDatasetWriteMixin:
    def create_dataset(self, spec):
        return _DatasetInfo(id=3, name=spec.name, dialect="postgresql")

    def update_dataset(self, dataset_id, spec):
        return _DatasetInfo(id=dataset_id, name=spec.name, dialect="postgresql")

    def list_bi_databases(self):
        return [{"id": 1, "name": "PostgreSQL"}]


class FullMockAdaptor(MockListDashboardsMixin, MockDashboardWriteMixin, MockChartWriteMixin, MockDatasetWriteMixin):
    """Mock adaptor implementing all mixins."""

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Test", description="", chart_ids=[])

    def list_charts(self, dashboard_id):
        return [_ChartInfo(id=1, name="Chart 1", chart_type="bar")]

    def list_datasets(self, dashboard_id=""):
        return [_DatasetInfo(id=1, name="orders", dialect="postgresql")]

    def get_chart(self, chart_id, dashboard_id=None):
        return _ChartInfo(id=chart_id, name="Test Chart", chart_type="bar")


class ReadOnlyMockAdaptor:
    """Mock adaptor with only read operations."""

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Read Only Dashboard")

    def list_charts(self, dashboard_id):
        return []

    def list_datasets(self, dashboard_id=""):
        return []


# ---- Build a mock datus_bi_core module ----

_bi_core_mock = MagicMock()
_bi_core_mock.ListDashboardsMixin = MockListDashboardsMixin
_bi_core_mock.DashboardWriteMixin = MockDashboardWriteMixin
_bi_core_mock.ChartWriteMixin = MockChartWriteMixin
_bi_core_mock.DatasetWriteMixin = MockDatasetWriteMixin


class _MockChartSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _MockDatasetSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _MockDashboardSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_bi_core_mock.models.ChartSpec = _MockChartSpec
_bi_core_mock.models.DatasetSpec = _MockDatasetSpec
_bi_core_mock.models.DashboardSpec = _MockDashboardSpec


# ---- Tests ----


class TestBIFuncToolAvailableTools:
    def test_full_adaptor_all_tools(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            adaptor = FullMockAdaptor()
            tool = BIFuncTool(adaptor)
            tools = tool.available_tools()
            tool_names = {t.name for t in tools}
            # Base read tools
            assert "list_dashboards" in tool_names
            assert "get_dashboard" in tool_names
            assert "list_charts" in tool_names
            assert "list_datasets" in tool_names
            # Write tools
            assert "create_dashboard" in tool_names
            assert "create_chart" in tool_names
            assert "add_chart_to_dashboard" in tool_names
            assert "create_dataset" in tool_names
            assert "list_bi_databases" in tool_names

    def test_read_only_adaptor_limited_tools(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            adaptor = ReadOnlyMockAdaptor()
            tool = BIFuncTool(adaptor)
            tools = tool.available_tools()
            tool_names = {t.name for t in tools}
            # No list_dashboards method on this adaptor
            assert "list_dashboards" not in tool_names
            assert "get_dashboard" in tool_names
            # No write tools
            assert "create_dashboard" not in tool_names
            assert "create_chart" not in tool_names


class TestBIFuncToolReadOps:
    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            return BIFuncTool(FullMockAdaptor())

    def test_list_dashboards_success(self):
        tool = self._make_tool()
        result = tool.list_dashboards(search="Test")
        assert result.success == 1
        assert len(result.result) == 1

    def test_get_dashboard_success(self):
        tool = self._make_tool()
        result = tool.get_dashboard("1")
        assert result.success == 1
        assert result.result["id"] == "1"

    def test_list_charts_success(self):
        tool = self._make_tool()
        result = tool.list_charts("1")
        assert result.success == 1
        assert len(result.result) == 1


class TestBIFuncToolWriteOps:
    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            return BIFuncTool(FullMockAdaptor())

    def test_create_dashboard(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_dashboard("My Dashboard", description="Test")
        assert result.success == 1
        assert result.result["name"] == "My Dashboard"

    def test_create_chart_parses_metrics(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(
                chart_type="bar",
                title="Revenue Chart",
                dataset_id="1",
                metrics="revenue,count",
            )
        assert result.success == 1
        assert result.result["name"] == "Revenue Chart"

    def test_list_bi_databases(self):
        tool = self._make_tool()
        result = tool.list_bi_databases()
        assert result.success == 1
        assert result.result[0]["name"] == "PostgreSQL"

    def test_error_handling(self):
        tool = self._make_tool()
        tool.adaptor.list_dashboards = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("connection failed"))
        result = tool.list_dashboards()
        assert result.success == 0
        assert "connection failed" in result.error


class TestBIFuncToolWriteQuery:
    """Tests for write_query: source DB → dashboard DB materialisation."""

    def _make_tool_with_write_db(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            return BIFuncTool(
                FullMockAdaptor(),
                write_db_uri="postgresql+psycopg2://superset:superset@localhost:5432/superset",
                write_db_schema="public",
            )

    def test_write_query_no_write_db_uri_returns_error(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            tool = BIFuncTool(FullMockAdaptor())
            result = tool.write_query("SELECT 1", "my_table")
        assert result.success == 0
        assert "write_db" in result.error

    def test_write_query_no_read_connector_returns_error(self):
        tool = self._make_tool_with_write_db()
        # No _read_connector set
        result = tool.write_query("SELECT 1", "my_table")
        assert result.success == 0
        assert "connector" in result.error.lower()

    def test_write_query_success(self):
        import pandas as pd

        tool = self._make_tool_with_write_db()

        # Build a fake ExecuteSQLResult
        mock_execute_result = MagicMock()
        mock_execute_result.success = True
        mock_execute_result.sql_return = pd.DataFrame({"col": [1, 2, 3]})

        mock_connector = MagicMock()
        mock_connector.execute_query.return_value = mock_execute_result
        tool._read_connector = mock_connector

        mock_engine = MagicMock()
        tool._write_engine = mock_engine

        # Patch DataFrame.to_sql so no real DB is needed
        with patch.object(pd.DataFrame, "to_sql", return_value=None):
            result = tool.write_query("SELECT col FROM t", "my_materialized_table")

        assert result.success == 1
        assert result.result["table_name"] == "my_materialized_table"
        assert result.result["rows_written"] == 3
        assert result.result["schema"] == "public"
        mock_connector.execute_query.assert_called_once_with("SELECT col FROM t", result_format="pandas")

    def test_write_query_connector_failure_propagates(self):
        tool = self._make_tool_with_write_db()

        mock_execute_result = MagicMock()
        mock_execute_result.success = False
        mock_execute_result.error = "Table not found"

        mock_connector = MagicMock()
        mock_connector.execute_query.return_value = mock_execute_result
        tool._read_connector = mock_connector
        tool._write_engine = MagicMock()

        result = tool.write_query("SELECT * FROM nonexistent", "my_table")
        assert result.success == 0
        assert "Table not found" in result.error

    def test_write_query_appears_in_available_tools_when_write_db_set(self):
        tool = self._make_tool_with_write_db()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tools = tool.available_tools()
        tool_names = {t.name for t in tools}
        assert "write_query" in tool_names

    def test_write_query_absent_from_tools_when_no_write_db(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_func_tools import BIFuncTool

            tool = BIFuncTool(FullMockAdaptor())
            tools = tool.available_tools()
        tool_names = {t.name for t in tools}
        assert "write_query" not in tool_names
