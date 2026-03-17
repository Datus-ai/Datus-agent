# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/schemas/node_models.py.

CI-level: zero external dependencies, pure Pydantic model validation.
"""

import pytest

from datus.schemas.node_models import (
    Context,
    ExecuteSQLInput,
    ExecuteSQLResult,
    GenerateSQLResult,
    Metric,
    OutputInput,
    ReferenceSql,
    ReflectionResult,
    SQLContext,
    SqlTask,
    StrategyType,
    TableSchema,
    TableValue,
)

# ---------------------------------------------------------------------------
# SqlTask
# ---------------------------------------------------------------------------


class TestSqlTask:
    def test_basic_creation(self):
        task = SqlTask(task="show all tables")
        assert task.task == "show all tables"

    def test_default_values(self):
        task = SqlTask(task="count rows")
        assert task.id == ""
        assert task.database_type == ""
        assert task.catalog_name == ""
        assert task.output_dir == "output"
        assert task.tables == []
        assert task.schema_linking_type == "table"
        assert task.current_date is None
        assert task.subject_path is None

    def test_empty_task_raises(self):
        with pytest.raises(ValueError):
            SqlTask(task="   ")

    def test_get_attribute(self):
        task = SqlTask(task="test", database_name="mydb")
        assert task.get("database_name") == "mydb"
        assert task.get("nonexistent", "default") == "default"

    def test_dict_access(self):
        task = SqlTask(task="query")
        assert task["task"] == "query"

    def test_to_dict_returns_dict(self):
        task = SqlTask(task="query", database_name="db1")
        d = task.to_dict()
        assert isinstance(d, dict)
        assert d["task"] == "query"
        assert d["database_name"] == "db1"

    def test_to_str_returns_json(self):
        task = SqlTask(task="query")
        s = task.to_str()
        assert "query" in s

    def test_from_dict_roundtrip(self):
        task = SqlTask(task="select 1", database_name="db", schema_linking_type="full")
        d = task.to_dict()
        task2 = SqlTask.from_dict(d)
        assert task2.task == task.task
        assert task2.database_name == task.database_name

    def test_from_str_roundtrip(self):
        task = SqlTask(task="select 1")
        s = task.to_str()
        task2 = SqlTask.from_str(s)
        assert task2.task == task.task

    def test_all_schema_linking_types(self):
        for t in ("table", "view", "mv", "full"):
            task = SqlTask(task="q", schema_linking_type=t)
            assert task.schema_linking_type == t

    def test_subject_path_accepted(self):
        task = SqlTask(task="q", subject_path=["Finance", "Revenue"])
        assert task.subject_path == ["Finance", "Revenue"]

    def test_current_date_accepted(self):
        task = SqlTask(task="q", current_date="2025-01-01")
        assert task.current_date == "2025-01-01"

    def test_external_knowledge_set(self):
        task = SqlTask(task="q", external_knowledge="some knowledge")
        assert task.external_knowledge == "some knowledge"


# ---------------------------------------------------------------------------
# TableSchema
# ---------------------------------------------------------------------------


class TestTableSchema:
    def _make(self, **kw):
        defaults = dict(table_name="users", database_name="mydb", definition="CREATE TABLE users (id INT)")
        defaults.update(kw)
        return TableSchema(**defaults)

    def test_basic_creation(self):
        schema = self._make()
        assert schema.table_name == "users"

    def test_to_prompt_simplifies_varchar(self):
        schema = self._make(definition="CREATE TABLE t (name VARCHAR(16777216))")
        prompt = schema.to_prompt()
        assert "VARCHAR(16777216)" not in prompt
        assert "VARCHAR" in prompt

    def test_to_prompt_removes_extra_whitespace(self):
        schema = self._make(definition="CREATE  TABLE   t   (id   INT)")
        prompt = schema.to_prompt()
        assert "  " not in prompt

    def test_list_to_prompt_joins_with_newlines(self):
        s1 = self._make(table_name="t1")
        s2 = self._make(table_name="t2")
        result = TableSchema.list_to_prompt([s1, s2])
        assert "\n\n" in result

    def test_list_to_prompt_empty_list_returns_empty(self):
        assert TableSchema.list_to_prompt([]) == ""

    def test_table_names_to_prompt(self):
        s1 = self._make(table_name="orders")
        s2 = self._make(table_name="customers")
        result = TableSchema.table_names_to_prompt([s1, s2])
        assert "orders" in result
        assert "customers" in result

    def test_table_names_to_prompt_empty_returns_empty(self):
        assert TableSchema.table_names_to_prompt([]) == ""

    def test_from_dict(self):
        data = {
            "table_name": "orders",
            "database_name": "shop",
            "definition": "CREATE TABLE orders (id INT)",
            "table_type": "table",
        }
        schema = TableSchema.from_dict(data)
        assert schema.table_name == "orders"
        assert schema.database_name == "shop"

    def test_to_dict(self):
        schema = self._make()
        d = schema.to_dict()
        assert isinstance(d, dict)
        assert d["table_name"] == "users"

    def test_default_table_type(self):
        schema = self._make()
        assert schema.table_type == "table"


# ---------------------------------------------------------------------------
# TableValue
# ---------------------------------------------------------------------------


class TestTableValue:
    def _make(self, **kw):
        defaults = dict(
            table_name="orders",
            database_name="shop",
            table_values="id,name\n1,Alice\n2,Bob",
            identifier="shop.orders",
        )
        defaults.update(kw)
        return TableValue(**defaults)

    def test_basic_creation(self):
        tv = self._make()
        assert tv.table_name == "orders"

    def test_to_prompt_includes_table_name(self):
        # For non-SQLite dialects, identifier is used; use sqlite dialect to use table_name
        tv = self._make()
        from datus.utils.constants import DBType

        prompt = tv.to_prompt(dialect=DBType.SQLITE)
        assert "orders" in prompt

    def test_to_prompt_truncates_long_values(self):
        long_values = "x" * 1000
        tv = self._make(table_values=long_values)
        prompt = tv.to_prompt(max_value_length=100)
        assert "truncated" in prompt
        assert len(prompt) < 200

    def test_from_dict_with_table_values_key(self):
        data = {
            "table_name": "t",
            "database_name": "db",
            "table_values": "a,b\n1,2",
        }
        tv = TableValue.from_dict(data)
        assert tv.table_values == "a,b\n1,2"

    def test_from_dict_with_sample_rows_key(self):
        data = {
            "table_name": "t",
            "database_name": "db",
            "sample_rows": "a,b\n1,2",
        }
        tv = TableValue.from_dict(data)
        assert tv.table_values == "a,b\n1,2"

    def test_to_dict(self):
        tv = self._make()
        d = tv.to_dict()
        assert "table_values" in d

    def test_process_text_columns_replaces_long_text(self):
        tv = self._make(
            table_name="t",
            table_values="id,name\n1,This is a very long text value that should be replaced",
        )
        result = tv._process_text_columns(
            tv.table_values,
            max_text_mark_length=10,
            table_schema={"id": "INT", "name": "TEXT"},
        )
        assert "<TEXT>" in result

    def test_parse_table_schema_extracts_columns(self):
        tv = self._make()
        processed_schemas = "orders: CREATE TABLE orders (`id` INT, `name` TEXT);"
        result = tv._parse_table_schema(processed_schemas)
        assert result == {"id": "INT", "name": "TEXT"}

    def test_parse_table_schema_returns_empty_when_not_found(self):
        tv = self._make()
        result = tv._parse_table_schema("unrelated text")
        assert result == {}


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class TestMetric:
    def test_basic_creation(self):
        m = Metric(name="revenue", description="Total revenue")
        assert m.name == "revenue"
        assert m.description == "Total revenue"

    def test_to_prompt_returns_description(self):
        m = Metric(name="revenue", description="Total revenue amount")
        assert m.to_prompt() == "Total revenue amount"

    def test_to_prompt_falls_back_to_name(self):
        m = Metric(name="revenue")
        assert "revenue" in m.to_prompt()

    def test_from_dict(self):
        m = Metric.from_dict({"name": "sales", "description": "Total sales"})
        assert m.name == "sales"

    def test_from_dict_missing_description(self):
        m = Metric.from_dict({"name": "sales"})
        assert m.description == ""


# ---------------------------------------------------------------------------
# ReferenceSql
# ---------------------------------------------------------------------------


class TestReferenceSql:
    def test_basic_creation(self):
        ref = ReferenceSql(name="monthly_sales", sql="SELECT * FROM sales")
        assert ref.name == "monthly_sales"
        assert ref.sql == "SELECT * FROM sales"

    def test_from_dict(self):
        data = {
            "name": "top_products",
            "sql": "SELECT id FROM products LIMIT 10",
            "comment": "top 10",
            "summary": "Gets top products",
        }
        ref = ReferenceSql.from_dict(data)
        assert ref.name == "top_products"
        assert ref.comment == "top 10"

    def test_defaults(self):
        ref = ReferenceSql(name="r", sql="SELECT 1")
        assert ref.comment == ""
        assert ref.summary == ""
        assert ref.tags == ""


# ---------------------------------------------------------------------------
# SQLContext
# ---------------------------------------------------------------------------


class TestSQLContext:
    def test_basic_creation(self):
        ctx = SQLContext(sql_query="SELECT 1")
        assert ctx.sql_query == "SELECT 1"

    def test_to_dict(self):
        ctx = SQLContext(sql_query="SELECT 1", row_count=5)
        d = ctx.to_dict()
        assert d["sql_query"] == "SELECT 1"
        assert d["row_count"] == 5

    def test_to_str_contains_sql(self):
        ctx = SQLContext(sql_query="SELECT * FROM t", sql_return="a,b\n1,2", row_count=1)
        s = ctx.to_str()
        assert "SELECT * FROM t" in s

    def test_to_str_truncates_long_return(self):
        ctx = SQLContext(sql_query="SELECT 1", sql_return="x" * 1000, row_count=1)
        s = ctx.to_str(max_sql_return_length=100)
        assert "truncated" in s

    def test_to_sample_str(self):
        ctx = SQLContext(sql_query="SELECT 1", sql_return="1")
        s = ctx.to_sample_str()
        assert "SELECT 1" in s

    def test_to_str_handles_dataframe_like(self):
        pass

        class FakeDF:
            empty = False

            def to_csv(self, **kw):
                return "col\n1\n"

        ctx = SQLContext(sql_query="SELECT 1", sql_return=FakeDF())
        s = ctx.to_str()
        assert "col" in s

    def test_to_str_handles_empty_dataframe(self):
        class FakeEmptyDF:
            empty = True

            def to_csv(self, **kw):
                return ""

        ctx = SQLContext(sql_query="SELECT 1", sql_return=FakeEmptyDF())
        s = ctx.to_str()
        assert "Empty result set" in s


# ---------------------------------------------------------------------------
# ExecuteSQLResult
# ---------------------------------------------------------------------------


class TestExecuteSQLResult:
    def test_basic_creation(self):
        result = ExecuteSQLResult(success=True)
        assert result.success is True

    def test_compact_result_with_string_return(self):
        result = ExecuteSQLResult(success=True, sql_return="col\n1\n2", row_count=2)
        compact = result.compact_result()
        assert "Rows: 2" in compact

    def test_compact_result_truncates_long_return(self):
        import os

        os.environ.setdefault("MAX_SQL_RESULT_LENGTH", "2000")
        result = ExecuteSQLResult(success=True, sql_return="x" * 5000, row_count=1)
        compact = result.compact_result()
        assert "..." in compact

    def test_compact_result_with_dataframe(self):
        class FakeDF:
            def to_csv(self, **kw):
                return "col\n1\n"

        result = ExecuteSQLResult(success=True, sql_return=FakeDF(), row_count=1)
        compact = result.compact_result()
        assert "col" in compact


# ---------------------------------------------------------------------------
# GenerateSQLResult
# ---------------------------------------------------------------------------


class TestGenerateSQLResult:
    def test_basic_creation(self):
        result = GenerateSQLResult(success=True, sql_query="SELECT 1")
        assert result.sql_query == "SELECT 1"
        assert result.tables == []

    def test_with_tables(self):
        result = GenerateSQLResult(success=True, sql_query="SELECT * FROM t", tables=["t"])
        assert result.tables == ["t"]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TestContext:
    def test_empty_creation(self):
        ctx = Context()
        assert ctx.sql_contexts == []
        assert ctx.table_schemas == []
        assert ctx.table_values == []
        assert ctx.metrics == []

    def test_update_schema_and_values(self):
        ctx = Context()
        schema = TableSchema(table_name="t", database_name="db", definition="CREATE TABLE t (id INT)")
        value = TableValue(table_name="t", database_name="db", table_values="id\n1")
        ctx.update_schema_and_values([schema], [value])
        assert len(ctx.table_schemas) == 1
        assert len(ctx.table_values) == 1

    def test_update_metrics(self):
        ctx = Context()
        m = Metric(name="rev")
        ctx.update_metrics([m])
        assert len(ctx.metrics) == 1

    def test_update_document_result(self):
        from unittest.mock import MagicMock

        ctx = Context()
        doc_result = MagicMock()
        ctx.update_document_result(doc_result)
        assert ctx.document_result is doc_result

    def test_update_parallel_results(self):
        ctx = Context()
        ctx.update_parallel_results({"task1": "result1"})
        assert ctx.parallel_results == {"task1": "result1"}

    def test_update_selection_result(self):
        ctx = Context()
        ctx.update_selection_result("chosen", {"score": 0.9})
        assert ctx.last_selected_result == "chosen"
        assert ctx.selection_metadata == {"score": 0.9}

    def test_to_dict_and_from_dict_roundtrip(self):
        ctx = Context()
        ctx.update_parallel_results({"x": 1})
        d = ctx.to_dict()
        ctx2 = Context.from_dict(d)
        assert ctx2.parallel_results == {"x": 1}

    def test_to_str(self):
        ctx = Context()
        sql_ctx = SQLContext(sql_query="SELECT 1", row_count=1)
        ctx.sql_contexts.append(sql_ctx)
        s = ctx.to_str()
        assert "SQL Contexts" in s

    def test_update_last_sql_context(self):
        ctx = Context()
        ctx.sql_contexts.append(SQLContext(sql_query="old"))
        new_ctx = SQLContext(sql_query="new")
        ctx.update_last_sql_context(new_ctx)
        assert ctx.sql_contexts[-1].sql_query == "new"

    def test_update_doc_search_keywords(self):
        ctx = Context()
        ctx.update_doc_search_keywords(["revenue", "sales"])
        assert ctx.doc_search_keywords == ["revenue", "sales"]


# ---------------------------------------------------------------------------
# StrategyType
# ---------------------------------------------------------------------------


class TestStrategyType:
    def test_all_values_exist(self):
        assert StrategyType.SUCCESS == "SUCCESS"
        assert StrategyType.DOC_SEARCH == "DOC_SEARCH"
        assert StrategyType.SIMPLE_REGENERATE == "SIMPLE_REGENERATE"
        assert StrategyType.SCHEMA_LINKING == "SCHEMA_LINKING"
        assert StrategyType.REASONING == "REASONING"
        assert StrategyType.COLUMN_EXPLORATION == "COLUMN_EXPLORATION"
        assert StrategyType.UNKNOWN == "UNKNOWN"


# ---------------------------------------------------------------------------
# ReflectionResult
# ---------------------------------------------------------------------------


class TestReflectionResult:
    def test_basic_creation(self):
        result = ReflectionResult(success=True, strategy=StrategyType.SUCCESS)
        assert result.strategy == "SUCCESS"

    def test_details_default_empty(self):
        result = ReflectionResult(success=True)
        assert result.details == {}

    def test_details_can_be_nested(self):
        result = ReflectionResult(
            success=True,
            strategy=StrategyType.SCHEMA_LINKING,
            details={"tables": ["t1", "t2"], "reason": "missing join"},
        )
        assert result.details["tables"] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# OutputInput
# ---------------------------------------------------------------------------


class TestOutputInput:
    def test_basic_creation(self):
        oi = OutputInput(
            task_id="t1",
            task="count rows",
            database_name="mydb",
            output_dir="/tmp",
            gen_sql="SELECT COUNT(*) FROM t",
        )
        assert oi.task_id == "t1"
        assert oi.finished is True

    def test_file_type_default(self):
        oi = OutputInput(
            task_id="t1",
            task="q",
            database_name="db",
            output_dir="/tmp",
            gen_sql="SELECT 1",
        )
        assert oi.file_type == "all"


# ---------------------------------------------------------------------------
# ExecuteSQLInput
# ---------------------------------------------------------------------------


class TestExecuteSQLInput:
    def test_basic_creation(self):
        ei = ExecuteSQLInput(sql_query="SELECT 1")
        assert ei.sql_query == "SELECT 1"
        assert ei.result_format == "csv"

    def test_database_name_default(self):
        ei = ExecuteSQLInput(sql_query="SELECT 1")
        assert ei.database_name == ""
