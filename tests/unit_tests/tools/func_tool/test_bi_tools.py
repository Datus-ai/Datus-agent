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

    def get_dataset(self, dataset_id, dashboard_id=None):
        return _DatasetInfo(id=dataset_id, name="orders", dashboard_id=dashboard_id)


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

    def get_dataset(self, dataset_id, dashboard_id=None):
        return None


class DatasetErrorAdapter(FullMockAdapter):
    def get_dataset(self, dataset_id, dashboard_id=None):
        raise RuntimeError("adapter exploded")


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


# ---- Test helper ----


def _build_tool(
    adapter=None,
    *,
    dataset_db_uri: str = "",
    dataset_db_schema: str = "",
    datasource_name: str = "",
    dataset_db_config=None,
    bi_service: str = "test_bi",
    features=None,
    serving_datasource_name: str = "serving_pg",
    serving_db_config=None,
):
    """Construct ``BIFuncTool`` with a mock ``agent_config`` for tests.

    The new ``DatasetDbConfig`` model carries only ``datasource_ref`` +
    ``bi_database_name``; the actual DB connection lives under
    ``services.datasources.<ref>``. This helper accepts either:

    - ``dataset_db_config``: a ready-made DatasetDbConfig (ref-form)
    - legacy URI-form args (``dataset_db_uri`` / ``dataset_db_schema``) plus
      the new ``datasource_name`` alias for ``DatasetDbConfig.bi_database_name``

    The linked ``DbConfig`` is registered on the mock agent_config under
    ``services.datasources[serving_datasource_name]`` so ``serving_db_config``
    can resolve it.
    """
    from datus.configuration.agent_config import DatasetDbConfig, DbConfig, ServicesConfig
    from datus.tools.func_tool.bi_tools import BIFuncTool

    cfg = dataset_db_config
    db_cfg = serving_db_config
    if cfg is None:
        if dataset_db_uri or dataset_db_schema or datasource_name:
            kwargs = {}
            if dataset_db_uri:
                from sqlalchemy.engine.url import make_url

                url = make_url(dataset_db_uri)
                backend = url.get_backend_name()
                kwargs["type"] = "postgresql" if backend == "postgresql" else backend
                kwargs["host"] = url.host or ""
                kwargs["port"] = str(url.port) if url.port else ""
                kwargs["database"] = url.database or ""
                kwargs["username"] = url.username or ""
                kwargs["password"] = url.password or ""
            if dataset_db_schema:
                kwargs["schema"] = dataset_db_schema
            db_cfg = DbConfig(**kwargs)
            cfg = DatasetDbConfig(
                datasource_ref=serving_datasource_name,
                bi_database_name=datasource_name or None,
            )

    if cfg is not None and db_cfg is None:
        db_cfg = DbConfig(type="postgresql", host="127.0.0.1", port="5433", database="d", schema="s", username="u")

    dash_cfg = MagicMock()
    dash_cfg.api_base_url = ""
    dash_cfg.username = ""
    dash_cfg.password = ""
    dash_cfg.api_key = ""
    dash_cfg.extra = {}
    dash_cfg.dataset_db = cfg
    dash_cfg.features = features

    services = ServicesConfig()
    if cfg is not None and db_cfg is not None:
        services.datasources[cfg.datasource_ref] = db_cfg

    mock_cfg = MagicMock()
    mock_cfg.dashboard_config = {bi_service: dash_cfg}
    mock_cfg.services = services
    mock_cfg.datasource_configs = {}
    mock_cfg.current_datasource = ""

    return BIFuncTool(mock_cfg, bi_service=bi_service, adapter=adapter)


# ---- Tests ----


class TestBIFuncToolAvailableTools:
    def test_full_adapter_all_tools(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(adapter=FullMockAdapter())
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
            tool = _build_tool(adapter=ReadOnlyMockAdapter())
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
            tool = _build_tool(adapter=MethodOnlyChartDataAdapter())
            tool_names = {t.name for t in tool.available_tools()}

            assert "get_chart_data" in tool_names


class TestBIFuncToolReadOps:
    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            return _build_tool(adapter=FullMockAdapter())

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

    def test_get_chart_accepts_none_dashboard_id(self):
        """Layer A validation passes ChartTarget.dashboard_id through directly.
        Standalone Superset charts have ``dashboard_id=None``; that should be
        treated the same as an empty dashboard scope."""
        tool = self._make_tool()
        result = tool.get_chart("1", dashboard_id=None)
        assert result.success == 1
        assert result.result["id"] == "1"

    def test_get_chart_data_success(self):
        tool = self._make_tool()
        result = tool.get_chart_data("1", limit=1)
        assert result.success == 1
        assert result.result["chart_id"] == "1"
        assert result.result["columns"] == ["category", "value"]
        assert result.result["row_count"] == 1
        assert result.result["rows"] == [{"category": "A", "value": 10}]

    def test_get_chart_data_accepts_none_dashboard_id(self):
        tool = self._make_tool()
        result = tool.get_chart_data("1", dashboard_id=None, limit=1)
        assert result.success == 1
        assert result.result["chart_id"] == "1"


class TestBIFuncToolWriteOps:
    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            return _build_tool(adapter=FullMockAdapter())

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
            tool = _build_tool(adapter=ReadOnlyMockAdapter())
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
            tool = _build_tool(adapter=ReadOnlyMockAdapter())
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


class TestBIFuncToolResolveGrafanaDatasource:
    """``_resolve_grafana_datasource_uid`` looks up a pre-registered Grafana
    datasource and returns its UID. Used by Grafana adapter when building
    panels — the BI layer does not register datasources, only resolves them."""

    def test_returns_cached_uid(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(adapter=FullMockAdapter())
        tool._grafana_ds_uid = "cached-uid"
        assert tool._resolve_grafana_datasource_uid() == "cached-uid"

    def test_match_by_bi_database_name(self):
        ds_info = MagicMock()
        ds_info.name = "PostgreSQL"
        ds_info.extra = {"grafana_ds": {"uid": "primary-uid"}}
        adapter = MagicMock()
        adapter.list_datasets.return_value = [ds_info]

        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(
                adapter=adapter,
                datasource_name="PostgreSQL",
            )
        uid = tool._resolve_grafana_datasource_uid()
        assert uid == "primary-uid"

    def test_fallback_match_by_database(self):
        ds_info = MagicMock()
        ds_info.name = "unrelated"
        ds_info.extra = {"grafana_ds": {"uid": "fallback-uid", "jsonData": {"database": "superset_examples"}}}
        adapter = MagicMock()
        adapter.list_datasets.return_value = [ds_info]

        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(
                adapter=adapter,
                dataset_db_uri="postgresql+psycopg2://u:p@h/superset_examples",
            )
        uid = tool._resolve_grafana_datasource_uid()
        assert uid == "fallback-uid"

    def test_returns_none_when_no_match(self):
        ds_info = MagicMock()
        ds_info.name = "unrelated"
        ds_info.extra = {"grafana_ds": {"uid": "x", "jsonData": {"database": "other_db"}}}
        adapter = MagicMock()
        adapter.list_datasets.return_value = [ds_info]

        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(
                adapter=adapter,
                datasource_name="NonExistent",
                dataset_db_uri="postgresql+psycopg2://u:p@h/my_db",
            )
        uid = tool._resolve_grafana_datasource_uid()
        assert uid is None


class TestBIFuncToolDeliverableTarget:
    """The 5 mutating BI tools (create_dashboard / update_dashboard / create_chart
    / update_chart / create_dataset) must attach a DeliverableTarget to
    ``result.result`` so ValidationHook can see what was delivered."""

    def _make_tool(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            return _build_tool(adapter=FullMockAdapter(), bi_service="superset")

    def test_create_dashboard_attaches_dashboard_target(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_dashboard("My Dash", description="d")
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "dashboard"
        assert target["platform"] == "superset"
        assert target["dashboard_id"] == "10"
        assert target["dashboard_name"] == "My Dash"

    def test_update_dashboard_attaches_dashboard_target(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.update_dashboard("10", title="Renamed", description="x")
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "dashboard"
        assert target["dashboard_id"] == "10"

    def test_create_chart_attaches_chart_target(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_chart(
                chart_type="bar",
                title="Sales",
                dataset_id="1",
                metrics="revenue",
                dashboard_id="42",
            )
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "chart"
        assert target["platform"] == "superset"
        assert target["chart_id"] == "5"
        assert target["chart_name"] == "Sales"
        assert target["dashboard_id"] == "42"

    def test_update_chart_attaches_chart_target(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.update_chart("5", title="Renamed Chart")
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "chart"
        assert target["chart_id"] == "5"

    def test_create_dataset_attaches_dataset_target(self):
        tool = self._make_tool()
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock, "datus_bi_core.models": _bi_core_mock.models}):
            result = tool.create_dataset("events", database_id="1", sql="SELECT 1")
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "dataset"
        assert target["platform"] == "superset"
        assert target["dataset_id"] == "3"
        assert target["dataset_name"] == "events"

    def test_delete_dashboard_does_not_attach_target(self):
        """Deletes don't produce deliverables — no target should be attached."""
        tool = self._make_tool()
        result = tool.delete_dashboard("10")
        assert result.success == 1
        assert "deliverable_target" not in result.result

    def test_add_chart_to_dashboard_attaches_dashboard_scoped_chart_target(self):
        """Linking enriches the chart target with its dashboard scope."""
        tool = self._make_tool()
        result = tool.add_chart_to_dashboard("5", "42")
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "chart"
        assert target["platform"] == "superset"
        assert target["chart_id"] == "5"
        assert target["dashboard_id"] == "42"

    def test_get_dataset_exists(self):
        """BIFuncTool exposes get_dataset so Layer A can verify dataset existence."""
        tool = self._make_tool()
        assert hasattr(tool, "get_dataset")
        tool_names = {t.name for t in tool.available_tools()}
        assert "get_dataset" in tool_names

    def test_get_dataset_success(self):
        tool = self._make_tool()
        result = tool.get_dataset("3", dashboard_id=None)
        assert result.success == 1
        assert result.result["id"] == "3"
        assert result.result["dashboard_id"] is None

    def test_get_dataset_not_found(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(adapter=ReadOnlyMockAdapter())
        result = tool.get_dataset("missing")
        assert result.success == 0
        assert "not found" in result.error

    def test_get_dataset_adapter_exception(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(adapter=DatasetErrorAdapter())
        result = tool.get_dataset("3")
        assert result.success == 0
        assert "adapter exploded" in result.error


class TestGetBiServingTarget:
    """``get_bi_serving_target`` hands orchestrators the four fields they need
    to route a transfer job (gen_job) and a dashboard build (gen_dashboard)
    without each subagent re-reading agent.yml."""

    def _dataset_cfg(self):
        from datus.configuration.agent_config import DatasetDbConfig

        return DatasetDbConfig(datasource_ref="serving_pg", bi_database_name="analytics_pg")

    def _db_cfg(self):
        from datus.configuration.agent_config import DbConfig

        return DbConfig(
            type="postgresql",
            host="127.0.0.1",
            port="5433",
            database="superset_examples",
            schema="bi_public",
            username="datus_writer",
            password="pw",
        )

    def test_returns_serving_mapping(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(
                adapter=FullMockAdapter(),
                dataset_db_config=self._dataset_cfg(),
                serving_db_config=self._db_cfg(),
            )
        result = tool.get_bi_serving_target()
        assert result.success == 1
        payload = result.result
        assert payload["datus_datasource"] == "serving_pg"
        assert payload["database"] == "superset_examples"
        assert payload["schema"] == "bi_public"
        assert payload["bi_database_name"] == "analytics_pg"

    def test_error_when_not_configured(self):
        with patch.dict(sys.modules, {"datus_bi_core": _bi_core_mock}):
            tool = _build_tool(adapter=FullMockAdapter())
        result = tool.get_bi_serving_target()
        assert result.success == 0
        assert "dataset_db" in result.error
