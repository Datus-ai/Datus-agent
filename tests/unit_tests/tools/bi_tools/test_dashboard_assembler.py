# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for DashboardAssembler._can_qualify_table."""

from unittest.mock import MagicMock

import pytest

# datus-bi-core is a hard dependency (see pyproject.toml [project.dependencies]).
from datus_bi_core import ChartInfo, DashboardInfo, DatasetInfo, QuerySpec

from datus.tools.bi_tools.dashboard_assembler import (
    ChartSelection,
    DashboardAssembler,
    DashboardAssemblyResult,
    DashboardExtraction,
    SelectedSqlCandidate,
    parts_match,
    split_table_parts,
)
from datus.tools.db_tools import connector_registry  # noqa: E402
from datus.utils.constants import DBType  # noqa: E402


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register external dialect capabilities needed by tests."""
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("mysql", capabilities={"database"})
    connector_registry.register_handlers("snowflake", capabilities={"catalog", "database", "schema"})
    yield


class TestCanQualifyTable:
    """Test DashboardAssembler._can_qualify_table method."""

    @pytest.fixture
    def assembler(self):
        # _can_qualify_table doesn't use the adapter, so None is fine
        return DashboardAssembler(adapter=None)

    def test_sqlite_with_database(self, assembler):
        assert assembler._can_qualify_table(DBType.SQLITE, "", "main", "") is True

    def test_sqlite_without_database(self, assembler):
        assert assembler._can_qualify_table(DBType.SQLITE, "", "", "") is False

    def test_duckdb_with_database_and_schema(self, assembler):
        assert assembler._can_qualify_table(DBType.DUCKDB, "", "memory", "main") is True

    def test_duckdb_missing_schema(self, assembler):
        assert assembler._can_qualify_table(DBType.DUCKDB, "", "memory", "") is False

    def test_external_dialect_with_schema_support(self, assembler):
        # PostgreSQL supports schema — needs both database and schema
        assert assembler._can_qualify_table("postgresql", "", "mydb", "public") is True
        assert assembler._can_qualify_table("postgresql", "", "mydb", "") is False

    def test_external_dialect_database_only(self, assembler):
        # MySQL supports database but not schema
        assert assembler._can_qualify_table("mysql", "", "mydb", "") is True
        assert assembler._can_qualify_table("mysql", "", "", "") is False

    def test_unknown_dialect_fallback(self, assembler):
        # Unknown dialect: returns True if database or schema is present
        assert assembler._can_qualify_table("unknown_db", "", "db", "") is True
        assert assembler._can_qualify_table("unknown_db", "", "", "sch") is True
        assert assembler._can_qualify_table("unknown_db", "", "", "") is False

    def test_none_dialect(self, assembler):
        assert assembler._can_qualify_table(None, "", "db", "") is True


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _register_dialects():
    """Register dialect capabilities for tests."""
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("mysql", capabilities={"database"})
    yield


@pytest.fixture
def mock_adapter():
    return MagicMock()


@pytest.fixture
def assembler(mock_adapter):
    return DashboardAssembler(
        adapter=mock_adapter,
        default_dialect="sqlite",
        default_database="main",
        default_schema="public",
    )


def make_chart(chart_id=1, name="Chart1", sql=None, tables=None, payload=None, description=None):
    query = None
    if sql is not None or tables is not None or payload is not None:
        query = QuerySpec(
            sql=sql or [],
            tables=tables or [],
            payload=payload or {},
            kind="sql",
        )
    return ChartInfo(
        id=chart_id,
        name=name,
        description=description,
        chart_type="bar",
        query=query,
    )


def make_dataset(ds_id=1, name="sales", schema=None, database=None):
    extra = {}
    if schema:
        extra["schema"] = schema
    if database:
        extra["database"] = database
    return DatasetInfo(
        id=ds_id,
        name=name,
        dialect="sqlite",
        tables=[name],
        extra=extra,
    )


# =============================================================================
# Tests: SelectedSqlCandidate
# =============================================================================


class TestSelectedSqlCandidate:
    def test_question_name_and_desc(self):
        s = SelectedSqlCandidate(chart_id=1, chart_name="Revenue", description="Total rev", sql="SELECT 1", index=0)
        assert s.question() == "Revenue - Total rev"

    def test_question_name_only(self):
        s = SelectedSqlCandidate(chart_id=1, chart_name="Revenue", description=None, sql="SELECT 1", index=0)
        assert s.question() == "Revenue"

    def test_question_desc_only(self):
        s = SelectedSqlCandidate(chart_id=1, chart_name="", description="Some desc", sql="SELECT 1", index=0)
        assert s.question() == "Some desc"

    def test_question_empty(self):
        s = SelectedSqlCandidate(chart_id=1, chart_name="", description=None, sql="SELECT 1", index=0)
        assert s.question() == ""

    def test_to_payload(self):
        s = SelectedSqlCandidate(chart_id=42, chart_name="Sales", description="Desc", sql="SELECT *", index=2)
        p = s.to_payload()
        assert p["chart_id"] == 42
        assert p["chart_name"] == "Sales"
        assert p["sql"] == "SELECT *"
        assert p["index"] == 2
        assert p["question"] == "Sales - Desc"


# =============================================================================
# Tests: DashboardExtraction
# =============================================================================


class TestDashboardExtraction:
    def test_chart_map(self):
        charts = [make_chart(1, "A"), make_chart(2, "B")]
        datasets = [make_dataset(10, "sales")]
        dashboard = DashboardInfo(id=99, name="Dash", chart_ids=[1, 2])
        ext = DashboardExtraction(dashboard_id=99, dashboard=dashboard, charts=charts, datasets=datasets)
        m = ext.chart_map()
        assert m["1"].name == "A"
        assert m["2"].name == "B"

    def test_dataset_map(self):
        datasets = [make_dataset(10, "sales"), make_dataset(20, "orders")]
        dashboard = DashboardInfo(id=1, name="D")
        ext = DashboardExtraction(dashboard_id=1, dashboard=dashboard, charts=[], datasets=datasets)
        m = ext.dataset_map()
        assert "10" in m
        assert "20" in m


# =============================================================================
# Tests: split_table_parts / parts_match
# =============================================================================


class TestSplitTableParts:
    def test_simple_table(self):
        assert split_table_parts("sales") == ["sales"]

    def test_schema_qualified(self):
        assert split_table_parts("public.sales") == ["public", "sales"]

    def test_fully_qualified(self):
        assert split_table_parts("db.public.sales") == ["db", "public", "sales"]

    def test_backtick_quoted(self):
        assert split_table_parts("`sales`") == ["sales"]

    def test_bracket_quoted(self):
        assert split_table_parts("[dbo].[orders]") == ["dbo", "orders"]


class TestPartsMatch:
    def test_identical_parts(self):
        assert parts_match(["public", "sales"], ["public", "sales"]) is True

    def test_suffix_match(self):
        assert parts_match(["sales"], ["public", "sales"]) is True
        assert parts_match(["public", "sales"], ["sales"]) is True

    def test_no_match(self):
        assert parts_match(["orders"], ["sales"]) is False

    def test_empty_left(self):
        assert parts_match([], ["sales"]) is False

    def test_empty_right(self):
        assert parts_match(["sales"], []) is False


# =============================================================================
# Tests: DashboardAssembler._dedupe_tables
# =============================================================================


class TestDedupeTables:
    def test_removes_exact_duplicates(self, assembler):
        result = assembler._dedupe_tables(["sales", "orders", "sales"])
        assert result == ["sales", "orders"]

    def test_prefers_qualified_name(self, assembler):
        result = assembler._dedupe_tables(["sales", "public.sales"])
        assert result == ["public.sales"]

    def test_empty_list(self, assembler):
        assert assembler._dedupe_tables([]) == []

    def test_blank_entries_skipped(self, assembler):
        result = assembler._dedupe_tables(["", "sales", "  "])
        assert result == ["sales"]

    def test_preserves_distinct_tables(self, assembler):
        result = assembler._dedupe_tables(["orders", "customers", "products"])
        assert len(result) == 3


# =============================================================================
# Tests: DashboardAssembler._prefer_table_name
# =============================================================================


class TestPreferTableName:
    def test_right_wins_when_more_parts(self, assembler):
        assert assembler._prefer_table_name("sales", "public.sales") == "public.sales"

    def test_left_wins_when_more_parts(self, assembler):
        assert assembler._prefer_table_name("db.public.sales", "sales") == "db.public.sales"

    def test_equal_parts_left_wins(self, assembler):
        assert assembler._prefer_table_name("sales", "orders") == "sales"


# =============================================================================
# Tests: DashboardAssembler._extract_database_name
# =============================================================================


class TestExtractDatabaseName:
    def test_from_dict_database_name(self, assembler):
        assert assembler._extract_database_name({"database_name": "prod"}) == "prod"

    def test_from_dict_name(self, assembler):
        assert assembler._extract_database_name({"name": "staging"}) == "staging"

    def test_from_string(self, assembler):
        assert assembler._extract_database_name("  dev  ") == "dev"

    def test_from_none(self, assembler):
        assert assembler._extract_database_name(None) == ""

    def test_from_int(self, assembler):
        assert assembler._extract_database_name(42) == ""


# =============================================================================
# Tests: DashboardAssembler._resolve_table_context
# =============================================================================


class TestResolveTableContext:
    def test_defaults_when_no_dataset(self, assembler):
        catalog, db, schema = assembler._resolve_table_context(None)
        assert db == "main"
        assert schema == "public"

    def test_dataset_overrides_schema(self, assembler):
        ds = make_dataset(1, "sales", schema="myschema")
        _, _, schema = assembler._resolve_table_context(ds)
        assert schema == "myschema"

    def test_dataset_overrides_database(self, assembler):
        ds = make_dataset(1, "sales", database={"database_name": "warehouse"})
        _, db, _ = assembler._resolve_table_context(ds)
        assert db == "warehouse"

    def test_no_extra_uses_defaults(self, assembler):
        ds = DatasetInfo(id=1, name="sales", dialect="sqlite")
        catalog, db, schema = assembler._resolve_table_context(ds)
        assert db == "main"
        assert schema == "public"


# =============================================================================
# Tests: DashboardAssembler._can_qualify_table
# =============================================================================


class TestCanQualifyTableExtended:
    def test_empty_dialect_with_database(self, assembler):
        # No dialect, has database → can qualify
        result = assembler._can_qualify_table("", "", "db", "")
        assert result is True  # bool(database)

    def test_none_dialect_treated_as_empty(self, assembler):
        result = assembler._can_qualify_table(None, "", "db", "")
        assert result is True  # bool(database)


# =============================================================================
# Tests: DashboardAssembler._normalize_tables
# =============================================================================


class TestNormalizeTables:
    def test_already_qualified_passthrough(self, assembler):
        result = assembler._normalize_tables(["public.sales"])
        assert result == ["public.sales"]

    def test_empty_table_skipped(self, assembler):
        result = assembler._normalize_tables(["", "  "])
        assert result == []

    def test_plain_table_qualified_for_sqlite(self, assembler):
        # assembler has default_dialect="sqlite", database="main" → qualifies with database
        result = assembler._normalize_tables(["orders"])
        # Should be qualified with database
        assert len(result) == 1
        assert "orders" in result[0]


# =============================================================================
# Tests: DashboardAssembler._tables_from_sql
# =============================================================================


class TestTablesFromSql:
    def test_extracts_tables(self, assembler):
        sql = "SELECT * FROM sales JOIN customers ON sales.id = customers.id"
        tables = assembler._tables_from_sql(sql)
        assert isinstance(tables, list)
        assert len(tables) >= 1

    def test_empty_sql_returns_empty(self, assembler):
        assert assembler._tables_from_sql("") == []
        assert assembler._tables_from_sql(None) == []


# =============================================================================
# Tests: DashboardAssembler.assemble
# =============================================================================


class TestAssemble:
    def test_basic_assemble(self, assembler):
        dashboard = DashboardInfo(id=1, name="Sales Dashboard")
        chart = make_chart(1, "Revenue", sql=["SELECT SUM(amount) FROM sales"])
        selection = ChartSelection(chart=chart)
        datasets = [make_dataset(1, "sales")]

        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=[selection],
            chart_selections_metrics=[],
            datasets=datasets,
        )
        assert isinstance(result, DashboardAssemblyResult)
        assert result.dashboard is dashboard
        assert len(result.reference_sqls) == 1
        assert result.reference_sqls[0].sql == "SELECT SUM(amount) FROM sales"

    def test_assemble_dedupes_tables(self, assembler):
        dashboard = DashboardInfo(id=1, name="D")
        chart1 = make_chart(1, "A", sql=["SELECT * FROM sales"])
        chart2 = make_chart(2, "B", sql=["SELECT * FROM sales"])
        selections = [ChartSelection(chart=chart1), ChartSelection(chart=chart2)]
        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=selections,
            chart_selections_metrics=[],
            datasets=[],
        )
        # Duplicate "sales" should be deduped
        sales_count = sum(1 for t in result.tables if "sales" in t)
        assert sales_count == 1

    def test_assemble_sql_indices_filter(self, assembler):
        dashboard = DashboardInfo(id=1, name="D")
        chart = make_chart(1, "A", sql=["SELECT 1", "SELECT 2", "SELECT 3"])
        selection = ChartSelection(chart=chart, sql_indices=[0, 2])
        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=[selection],
            chart_selections_metrics=[],
            datasets=[],
        )
        assert len(result.reference_sqls) == 2
        assert result.reference_sqls[0].sql == "SELECT 1"
        assert result.reference_sqls[1].sql == "SELECT 3"

    def test_assemble_empty_selections(self, assembler):
        dashboard = DashboardInfo(id=1, name="D")
        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=[],
            chart_selections_metrics=[],
            datasets=[],
        )
        assert result.reference_sqls == []
        assert result.metric_sqls == []
        assert result.tables == []

    def test_assemble_with_query_tables(self, assembler):
        """query.tables should be added to tables list."""
        dashboard = DashboardInfo(id=1, name="D")
        chart = make_chart(1, "A", tables=["customers"])
        # no SQL, but tables from query
        selection = ChartSelection(chart=chart)
        result = assembler.assemble(
            dashboard=dashboard,
            chart_selections_ref=[selection],
            chart_selections_metrics=[],
            datasets=[],
        )
        assert any("customers" in t for t in result.tables)


# =============================================================================
# Tests: DashboardAssembler._resolve_chart_dataset
# =============================================================================


class TestResolveChartDataset:
    def test_resolves_by_datasource_id(self, assembler):
        ds = make_dataset(10, "sales")
        dataset_by_id = {"10": ds}
        chart = make_chart(1, "A", payload={"datasource": {"id": 10, "type": "table"}})
        result = assembler._resolve_chart_dataset(chart, dataset_by_id, {})
        assert result is ds

    def test_resolves_by_datasource_name(self, assembler):
        ds = make_dataset(10, "sales")
        dataset_by_name = {"sales": ds}
        chart = make_chart(1, "A", payload={"datasource": {"name": "sales"}})
        result = assembler._resolve_chart_dataset(chart, {}, dataset_by_name)
        assert result is ds

    def test_no_datasource_returns_none(self, assembler):
        chart = make_chart(1, "A")
        result = assembler._resolve_chart_dataset(chart, {}, {})
        assert result is None


# =============================================================================
# Tests: DashboardAssembler.hydrate_datasets
# =============================================================================


class TestHydrateDatasets:
    def test_hydrate_success(self, assembler, mock_adapter):
        ds = make_dataset(1, "sales")
        hydrated = make_dataset(1, "sales_hydrated")
        mock_adapter.get_dataset.return_value = hydrated
        result = assembler.hydrate_datasets([ds])
        assert result[0].name == "sales_hydrated"

    def test_hydrate_fallback_on_error(self, assembler, mock_adapter):
        ds = make_dataset(1, "sales")
        mock_adapter.get_dataset.side_effect = Exception("API error")
        result = assembler.hydrate_datasets([ds])
        # Should fall back to original
        assert result[0] is ds

    def test_hydrate_empty(self, assembler):
        result = assembler.hydrate_datasets([])
        assert result == []


# =============================================================================
# Tests: DashboardAssembler.extract_dashboard
# =============================================================================


class TestExtractDashboard:
    def test_extract_dashboard(self, assembler, mock_adapter):
        dashboard = DashboardInfo(id=1, name="D", chart_ids=[10])
        mock_adapter.parse_dashboard_id.return_value = 1
        mock_adapter.get_dashboard_info.return_value = dashboard
        mock_adapter.list_charts.return_value = [make_chart(10, "C")]
        mock_adapter.get_chart.return_value = make_chart(10, "C")
        mock_adapter.list_datasets.return_value = [make_dataset(1, "sales")]

        result = assembler.extract_dashboard("http://localhost/dashboard/1")
        assert result.dashboard_id == 1
        assert len(result.charts) == 1

    def test_extract_dashboard_not_found_raises(self, assembler, mock_adapter):
        mock_adapter.parse_dashboard_id.return_value = 999
        mock_adapter.get_dashboard_info.return_value = None
        with pytest.raises(ValueError, match="not found"):
            assembler.extract_dashboard("http://localhost/dashboard/999")

    def test_load_charts_fallback_to_meta(self, assembler, mock_adapter):
        """If get_chart fails, falls back to chart meta."""
        dashboard = DashboardInfo(id=1, name="D", chart_ids=[5])
        chart_meta = make_chart(5, "FallbackChart")
        mock_adapter.parse_dashboard_id.return_value = 1
        mock_adapter.get_dashboard_info.return_value = dashboard
        mock_adapter.list_charts.return_value = [chart_meta]
        mock_adapter.get_chart.side_effect = Exception("Not found")
        mock_adapter.list_datasets.return_value = []

        result = assembler.extract_dashboard("http://localhost/dashboard/1")
        assert len(result.charts) == 1
        assert result.charts[0].name == "FallbackChart"
