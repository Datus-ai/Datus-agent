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


class _ChartDataResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


class _PaginatedResult:
    """Tiny stand-in for ``datus_bi_core.models.PaginatedResult[T]``.

    Real adapters return ``PaginatedResult`` (items + optional total); the
    stub mirrors that surface so BIFuncTool's envelope builder reads the
    same attributes either way.
    """

    def __init__(self, items, total=None):
        self.items = items
        self.total = total


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

    def delete_dataset(self, dataset_id):
        return True

    def list_bi_databases(self):
        return [{"id": 1, "name": "PostgreSQL"}]


class FullMockAdapter(MockDashboardWriteMixin, MockChartWriteMixin, MockDatasetWriteMixin):
    """Mock adapter implementing all mixins."""

    supports_chart_data = True

    def list_dashboards(self, search="", limit=50, offset=0):
        return _PaginatedResult(items=[_DashboardInfo(id=1, name="Test Dashboard")], total=1)

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Test", description="", chart_ids=[])

    def list_charts(self, dashboard_id, limit=50, offset=0):
        return _PaginatedResult(items=[_ChartInfo(id=1, name="Chart 1", chart_type="bar")], total=1)

    def list_datasets(self, dashboard_id="", limit=50, offset=0):
        return _PaginatedResult(items=[_DatasetInfo(id=1, name="orders", dialect="postgresql")], total=1)

    def get_chart(self, chart_id, dashboard_id=None):
        return _ChartInfo(id=chart_id, name="Test Chart", chart_type="bar")

    def get_chart_data(self, chart_id, dashboard_id=None, limit=None):
        rows = [
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
        ]
        if limit is not None:
            rows = rows[:limit]
        return _ChartDataResult(
            chart_id=chart_id,
            columns=["category", "value"],
            rows=rows,
            row_count=len(rows),
            sql="SELECT category, value FROM orders",
            extra={},
        )


class ReadOnlyMockAdapter:
    """Mock adapter with only read operations."""

    supports_chart_data = False

    def list_dashboards(self, search="", limit=50, offset=0):
        return _PaginatedResult(items=[], total=0)

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Read Only Dashboard")

    def list_charts(self, dashboard_id, limit=50, offset=0):
        return _PaginatedResult(items=[], total=0)

    def get_chart(self, chart_id, dashboard_id=None):
        return None

    def list_datasets(self, dashboard_id="", limit=50, offset=0):
        return _PaginatedResult(items=[], total=0)


class MethodOnlyChartDataAdapter:
    """Mock adapter that implements get_chart_data without any support flag."""

    def list_dashboards(self, search="", limit=50, offset=0):
        return _PaginatedResult(items=[_DashboardInfo(id=1, name="Test Dashboard")], total=1)

    def get_dashboard_info(self, dashboard_id):
        return _DashboardInfo(id=dashboard_id, name="Test")

    def list_charts(self, dashboard_id, limit=50, offset=0):
        return _PaginatedResult(items=[], total=0)

    def get_chart(self, chart_id, dashboard_id=None):
        return _ChartInfo(id=chart_id, name="Test Chart", chart_type="bar")

    def get_chart_data(self, chart_id, dashboard_id=None, limit=None):
        return _ChartDataResult(
            chart_id=chart_id,
            columns=["value"],
            rows=[{"value": 1}],
            row_count=1,
            sql="SELECT 1 AS value",
            extra={},
        )

    def list_datasets(self, dashboard_id="", limit=50, offset=0):
        return _PaginatedResult(items=[], total=0)


# ---- Build a mock datus_bi_core module ----

_bi_core_mock = MagicMock()
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
    def test_full_adapter_all_tools(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            adapter = FullMockAdapter()
            tool = BIFuncTool(adapter)
            tools = tool.available_tools()
            tool_names = {t.name for t in tools}
            # Base read tools
            assert "list_dashboards" in tool_names
            assert "get_dashboard" in tool_names
            assert "list_charts" in tool_names
            assert "get_chart" in tool_names
            assert "get_chart_data" in tool_names
            assert "list_datasets" in tool_names
            # Write tools
            assert "create_dashboard" in tool_names
            assert "create_chart" in tool_names
            assert "add_chart_to_dashboard" in tool_names
            assert "create_dataset" in tool_names
            assert "list_bi_databases" in tool_names
            # Delete tools
            assert "delete_dashboard" in tool_names
            assert "delete_chart" in tool_names
            assert "delete_dataset" in tool_names

    def test_read_only_adapter_limited_tools(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            adapter = ReadOnlyMockAdapter()
            tool = BIFuncTool(adapter)
            tools = tool.available_tools()
            tool_names = {t.name for t in tools}
            # list_dashboards is now part of BIAdapterBase — always present
            assert "list_dashboards" in tool_names
            assert "get_dashboard" in tool_names
            assert "get_chart" in tool_names
            # No write tools
            assert "create_dashboard" not in tool_names
            assert "create_chart" not in tool_names

    def test_method_override_enables_chart_data_tool_without_flag(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            tool = BIFuncTool(MethodOnlyChartDataAdapter())
            tool_names = {t.name for t in tool.available_tools()}

            assert "get_chart_data" in tool_names


class TestBIFuncToolReadOps:
    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            return BIFuncTool(FullMockAdapter())

    def test_list_dashboards_success(self):
        tool = self._make_tool()
        result = tool.list_dashboards(search="Test")
        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"id": 1, "name": "Test Dashboard"}]
        assert envelope["total"] == 1
        assert envelope["has_more"] is False
        assert envelope["extra"] is None

    def test_get_dashboard_success(self):
        tool = self._make_tool()
        result = tool.get_dashboard("1")
        assert result.success == 1
        assert result.result["id"] == "1"

    def test_list_charts_success(self):
        tool = self._make_tool()
        result = tool.list_charts("1")
        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"id": 1, "name": "Chart 1", "chart_type": "bar"}]
        assert envelope["total"] == 1
        assert envelope["has_more"] is False

    def test_get_chart_success(self):
        tool = self._make_tool()
        result = tool.get_chart("1")
        assert result.success == 1
        assert result.result["id"] == "1"
        assert result.result["name"] == "Test Chart"

    def test_get_chart_data_success(self):
        tool = self._make_tool()
        result = tool.get_chart_data("1", limit=1)
        assert result.success == 1
        assert result.result["chart_id"] == "1"
        assert result.result["columns"] == ["category", "value"]
        assert result.result["row_count"] == 1
        assert result.result["rows"] == [{"category": "A", "value": 10}]


class TestBIFuncToolWriteOps:
    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            return BIFuncTool(FullMockAdapter())

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

    def test_delete_dashboard(self):
        tool = self._make_tool()
        result = tool.delete_dashboard("10")
        assert result.success == 1
        assert result.result["deleted"] is True
        assert result.result["dashboard_id"] == "10"

    def test_delete_chart(self):
        tool = self._make_tool()
        result = tool.delete_chart("5")
        assert result.success == 1
        assert result.result["deleted"] is True
        assert result.result["chart_id"] == "5"

    def test_delete_dataset(self):
        tool = self._make_tool()
        result = tool.delete_dataset("3")
        assert result.success == 1
        assert result.result["deleted"] is True
        assert result.result["dataset_id"] == "3"

    def test_update_dashboard_not_found(self):
        tool = self._make_tool()
        tool.adapter.get_dashboard_info = lambda dashboard_id: None
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.update_dashboard("999", title="New Title")
        assert result.success == 0
        assert "not found" in result.error

    def test_update_chart_not_found(self):
        tool = self._make_tool()
        tool.adapter.get_chart = lambda chart_id, **kwargs: None
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.update_chart("999", title="New Title")
        assert result.success == 0
        assert "not found" in result.error

    def test_create_chart_with_sql_succeeds(self):
        """create_chart accepts sql parameter without dataset_id (Grafana path)."""
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(
                chart_type="line",
                title="SQL Chart",
                sql="SELECT date AS time, count FROM my_table ORDER BY date",
                dashboard_id="abc-123",
            )
        assert result.success == 1
        assert result.result["name"] == "SQL Chart"

    def test_create_chart_rejects_no_dataset_id_and_no_sql(self):
        """create_chart fails when neither dataset_id nor sql is provided."""
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(chart_type="bar", title="Test")
        assert result.success == 0
        assert "dataset_id" in result.error.lower() or "sql" in result.error.lower()

    def test_create_chart_sql_rejects_missing_dashboard_id(self):
        """Grafana path requires dashboard_id when sql is provided."""
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(chart_type="line", title="Test", sql="SELECT 1")
        assert result.success == 0
        assert "dashboard_id" in result.error.lower()

    def test_create_chart_rejects_zero_dataset_id(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(chart_type="bar", title="Test", dataset_id="0")
        assert result.success == 0
        assert "dataset_id" in result.error.lower()

    def test_create_chart_rejects_empty_dataset_id(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(chart_type="bar", title="Test", dataset_id="")
        assert result.success == 0
        assert "dataset_id" in result.error.lower()

    def test_create_dataset_rejects_non_numeric_database_id(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_dataset(name="test", database_id="abc")
        assert result.success == 0
        assert "database_id" in result.error.lower()

    def test_create_dataset_rejects_empty_database_id(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_dataset(name="test", database_id="")
        assert result.success == 0
        assert "database_id" in result.error.lower()

    def test_list_datasets_success(self):
        tool = self._make_tool()
        result = tool.list_datasets()
        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == [{"id": 1, "name": "orders", "dialect": "postgresql"}]
        assert envelope["total"] == 1
        assert envelope["has_more"] is False

    def test_update_dashboard_success(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.update_dashboard("1", title="Updated")
        assert result.success == 1
        assert result.result["name"] == "Updated"

    def test_update_chart_success(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.update_chart("1", title="New Title", chart_type="line")
        assert result.success == 1
        assert result.result["name"] == "New Title"

    def test_add_chart_to_dashboard_success(self):
        tool = self._make_tool()
        result = tool.add_chart_to_dashboard("5", "10")
        assert result.success == 1
        assert result.result["chart_id"] == "5"
        assert result.result["dashboard_id"] == "10"

    def test_create_dataset_success(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_dataset(name="my_ds", database_id="1", sql="SELECT * FROM t")
        assert result.success == 1
        assert result.result["name"] == "my_ds"

    def test_list_dashboards_read_only_adapter(self):
        """list_dashboards is now in BIAdapterBase and always available.

        The read-only adapter returns an empty page — exercises the
        envelope's empty-items / total=0 / has_more=False branch.
        """
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            tool = BIFuncTool(ReadOnlyMockAdapter())
        result = tool.list_dashboards()
        assert result.success == 1
        envelope = result.result
        assert envelope["items"] == []
        assert envelope["total"] == 0
        assert envelope["has_more"] is False

    def test_get_dashboard_not_found(self):
        tool = self._make_tool()
        tool.adapter.get_dashboard_info = lambda dashboard_id: None
        result = tool.get_dashboard("999")
        assert result.success == 0
        assert "not found" in result.error

    def test_get_chart_not_found(self):
        tool = self._make_tool()
        tool.adapter.get_chart = lambda chart_id, dashboard_id=None: None
        result = tool.get_chart("999")
        assert result.success == 0
        assert "not found" in result.error

    def test_get_chart_data_not_found(self):
        tool = self._make_tool()
        tool.adapter.get_chart_data = lambda chart_id, dashboard_id=None, limit=None: None
        result = tool.get_chart_data("999")
        assert result.success == 0
        assert "not found" in result.error

    def test_list_charts_exception(self):
        tool = self._make_tool()
        tool.adapter.list_charts = lambda dashboard_id: (_ for _ in ()).throw(RuntimeError("fail"))
        result = tool.list_charts("1")
        assert result.success == 0

    def test_get_chart_exception(self):
        tool = self._make_tool()
        tool.adapter.get_chart = lambda chart_id, dashboard_id=None: (_ for _ in ()).throw(RuntimeError("fail"))
        result = tool.get_chart("1")
        assert result.success == 0

    def test_get_chart_data_exception(self):
        tool = self._make_tool()
        tool.adapter.get_chart_data = lambda chart_id, dashboard_id=None, limit=None: (_ for _ in ()).throw(
            RuntimeError("fail")
        )
        result = tool.get_chart_data("1")
        assert result.success == 0

    def test_get_chart_data_unsupported(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            tool = BIFuncTool(ReadOnlyMockAdapter())
        result = tool.get_chart_data("1")
        assert result.success == 0
        assert "does not support" in result.error

    def test_list_datasets_exception(self):
        tool = self._make_tool()
        tool.adapter.list_datasets = lambda dashboard_id="": (_ for _ in ()).throw(RuntimeError("fail"))
        result = tool.list_datasets()
        assert result.success == 0

    def test_error_handling(self):
        tool = self._make_tool()
        tool.adapter.list_dashboards = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("connection failed"))
        result = tool.list_dashboards()
        assert result.success == 0
        assert "connection failed" in result.error


class TestBIFuncToolWriteQuery:
    """Tests for write_query: source DB → dashboard DB materialisation."""

    def _make_tool_with_dataset_db(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            return BIFuncTool(
                FullMockAdapter(),
                dataset_db_uri="postgresql+psycopg2://superset:superset@localhost:5432/superset",
                dataset_db_schema="public",
            )

    def test_write_query_no_dataset_db_uri_returns_error(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            tool = BIFuncTool(FullMockAdapter())
            result = tool.write_query("SELECT 1", "my_table")
        assert result.success == 0
        assert "dataset_db" in result.error

    def test_write_query_no_read_connector_returns_error(self):
        tool = self._make_tool_with_dataset_db()
        # No _read_connector set
        result = tool.write_query("SELECT 1", "my_table")
        assert result.success == 0
        assert "connector" in result.error.lower()

    def test_write_query_rejects_non_select_sql(self):
        tool = self._make_tool_with_dataset_db()
        result = tool.write_query("DROP TABLE users", "my_table")
        assert result.success == 0
        assert "SELECT" in result.error

    def test_write_query_rejects_multi_statement_sql(self):
        tool = self._make_tool_with_dataset_db()
        result = tool.write_query("SELECT 1; DROP TABLE users", "my_table")
        assert result.success == 0
        assert "Multi-statement" in result.error

    def test_write_query_allows_trailing_semicolon(self):
        """A single trailing semicolon is harmless and should be allowed."""
        tool = self._make_tool_with_dataset_db()
        tool._read_connector = None  # will fail at connector check, not SQL check
        result = tool.write_query("SELECT 1;", "my_table")
        # Should pass SQL validation but fail at connector check
        assert "connector" in result.error.lower()

    def test_write_query_rejects_invalid_table_name(self):
        tool = self._make_tool_with_dataset_db()
        result = tool.write_query("SELECT 1", "123-bad-name!")
        assert result.success == 0
        assert "table_name" in result.error.lower()

    def test_write_query_rejects_invalid_if_exists(self):
        tool = self._make_tool_with_dataset_db()
        result = tool.write_query("SELECT 1", "my_table", if_exists="drop")
        assert result.success == 0
        assert "if_exists" in result.error

    def test_write_query_success(self):
        import pandas as pd

        tool = self._make_tool_with_dataset_db()

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
        tool = self._make_tool_with_dataset_db()

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

    def test_write_query_appears_in_available_tools_when_dataset_db_set(self):
        tool = self._make_tool_with_dataset_db()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tools = tool.available_tools()
        tool_names = {t.name for t in tools}
        assert "write_query" in tool_names


class TestBIFuncToolResolveGrafanaDatasource:
    """Tests for _resolve_grafana_datasource_uid: lookup pre-configured datasources."""

    def test_returns_cached_uid(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            tool = BIFuncTool(FullMockAdapter())
            tool._grafana_ds_uid = "cached-uid-123"
            assert tool._resolve_grafana_datasource_uid() == "cached-uid-123"

    def test_returns_none_when_no_list_datasets(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            adapter = MagicMock(spec=[])  # no list_datasets
            tool = BIFuncTool(adapter)
            assert tool._resolve_grafana_datasource_uid() is None

    def test_matches_by_datasource_name(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            ds_info = MagicMock()
            ds_info.name = "My-PostgreSQL"
            ds_info.extra = {"grafana_ds": {"uid": "matched-uid"}}

            adapter = MagicMock()
            adapter.list_datasets.return_value = [ds_info]

            tool = BIFuncTool(adapter, datasource_name="My-PostgreSQL")
            uid = tool._resolve_grafana_datasource_uid()
            assert uid == "matched-uid"
            assert tool._grafana_ds_uid == "matched-uid"

    def test_fallback_matches_by_database_name(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            ds_info = MagicMock()
            ds_info.name = "some-other-name"
            ds_info.extra = {"grafana_ds": {"uid": "fallback-uid", "jsonData": {"database": "grafana_data"}}}

            adapter = MagicMock()
            adapter.list_datasets.return_value = [ds_info]

            tool = BIFuncTool(
                adapter,
                dataset_db_uri="postgresql+psycopg2://user:pass@localhost:5434/grafana_data",
            )
            uid = tool._resolve_grafana_datasource_uid()
            assert uid == "fallback-uid"

    def test_returns_none_when_no_match(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            ds_info = MagicMock()
            ds_info.name = "unrelated"
            ds_info.extra = {"grafana_ds": {"uid": "some-uid", "jsonData": {"database": "other_db"}}}

            adapter = MagicMock()
            adapter.list_datasets.return_value = [ds_info]

            tool = BIFuncTool(
                adapter,
                datasource_name="NonExistent",
                dataset_db_uri="postgresql+psycopg2://user:pass@localhost/my_db",
            )
            uid = tool._resolve_grafana_datasource_uid()
            assert uid is None


class TestBIFuncToolWriteQueryContinued:
    """Continuation of write_query tests (split to keep class sizes manageable)."""

    def _make_tool_with_dataset_db(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            return BIFuncTool(
                FullMockAdapter(),
                dataset_db_uri="postgresql+psycopg2://superset:superset@localhost:5432/superset",
                dataset_db_schema="public",
            )

    def test_write_query_absent_from_tools_when_no_dataset_db(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            from datus.tools.func_tool.bi_tools import BIFuncTool

            tool = BIFuncTool(FullMockAdapter())
            tools = tool.available_tools()
        tool_names = {t.name for t in tools}
        assert "write_query" not in tool_names
