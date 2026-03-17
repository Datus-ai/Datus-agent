# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/cli_context.py — CliContext.

Tests cover all public methods:
- add_table / add_tables
- add_metric / add_metrics
- add_sql_context
- get_recent_* / get_last_*
- update_database_context
- set_current_sql_task / get_or_create_sql_task
- clear_* methods
- to_dict / get_context_summary
"""

from unittest.mock import MagicMock

import pytest

from datus.cli.cli_context import CliContext
from datus.schemas.node_models import Metric, SQLContext, SqlTask, TableSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(catalog="cat", db="db", name="table1") -> TableSchema:
    return TableSchema(
        catalog_name=catalog,
        database_name=db,
        table_name=name,
        definition="CREATE TABLE table1 (id INTEGER)",
    )


def _make_metric(name="revenue") -> Metric:
    return Metric(name=name, description="Revenue metric")


def _make_sql_context(sql="SELECT 1") -> SQLContext:
    return SQLContext(sql_query=sql)


def _make_sql_task(task="find top 10") -> SqlTask:
    return SqlTask(
        id="test-id",
        database_type="sqlite",
        task=task,
        database_name="test_db",
        output_dir="output",
        external_knowledge="",
    )


# ---------------------------------------------------------------------------
# Tests: add_table / get_recent_tables
# ---------------------------------------------------------------------------


class TestAddTable:
    def test_add_single_table(self):
        ctx = CliContext()
        t = _make_table()
        ctx.add_table(t)
        assert t in ctx.get_recent_tables()

    def test_add_duplicate_moves_to_front(self):
        ctx = CliContext()
        t1 = _make_table(name="t1")
        t2 = _make_table(name="t2")
        ctx.add_table(t1)
        ctx.add_table(t2)
        ctx.add_table(t1)  # re-add t1 -> should move to front
        tables = ctx.get_recent_tables()
        assert tables[0] is t1

    def test_add_tables_multiple(self):
        ctx = CliContext()
        tables = [_make_table(name=f"t{i}") for i in range(3)]
        ctx.add_tables(tables)
        assert len(ctx.get_recent_tables()) == 3

    def test_maxlen_respected(self):
        ctx = CliContext()
        for i in range(25):
            ctx.add_table(_make_table(name=f"t{i}"))
        assert len(ctx.get_recent_tables()) == 20


# ---------------------------------------------------------------------------
# Tests: add_metric / get_recent_metrics
# ---------------------------------------------------------------------------


class TestAddMetric:
    def test_add_single_metric(self):
        ctx = CliContext()
        m = _make_metric()
        ctx.add_metric(m)
        assert m in ctx.get_recent_metrics()

    def test_duplicate_metric_moves_to_front(self):
        ctx = CliContext()
        m1 = _make_metric("m1")
        m2 = _make_metric("m2")
        ctx.add_metric(m1)
        ctx.add_metric(m2)
        ctx.add_metric(m1)
        assert ctx.get_recent_metrics()[0] is m1

    def test_add_metrics_multiple(self):
        ctx = CliContext()
        ctx.add_metrics([_make_metric("a"), _make_metric("b")])
        assert len(ctx.get_recent_metrics()) == 2


# ---------------------------------------------------------------------------
# Tests: add_sql_context / get_last_sql
# ---------------------------------------------------------------------------


class TestAddSqlContext:
    def test_add_sql_context(self):
        ctx = CliContext()
        sc = _make_sql_context("SELECT 1")
        ctx.add_sql_context(sc)
        assert ctx.get_last_sql_context() is sc

    def test_get_last_sql(self):
        ctx = CliContext()
        ctx.add_sql_context(_make_sql_context("SELECT 42"))
        assert ctx.get_last_sql() == "SELECT 42"

    def test_get_last_sql_empty(self):
        ctx = CliContext()
        assert ctx.get_last_sql() is None

    def test_get_last_sql_context_empty(self):
        ctx = CliContext()
        assert ctx.get_last_sql_context() is None


# ---------------------------------------------------------------------------
# Tests: update_database_context
# ---------------------------------------------------------------------------


class TestUpdateDatabaseContext:
    def test_update_db_name(self):
        ctx = CliContext()
        ctx.update_database_context(db_name="mydb")
        assert ctx.current_db_name == "mydb"

    def test_update_catalog(self):
        ctx = CliContext()
        ctx.update_database_context(catalog="mycat")
        assert ctx.current_catalog == "mycat"

    def test_update_schema(self):
        ctx = CliContext()
        ctx.update_database_context(schema="myschema")
        assert ctx.current_schema == "myschema"

    def test_update_logic_name(self):
        ctx = CliContext()
        ctx.update_database_context(db_logic_name="logic_db")
        assert ctx.current_logic_db_name == "logic_db"

    def test_none_values_not_applied(self):
        ctx = CliContext()
        ctx.current_db_name = "existing"
        ctx.update_database_context()  # all None
        assert ctx.current_db_name == "existing"


# ---------------------------------------------------------------------------
# Tests: set_current_sql_task / get_or_create_sql_task
# ---------------------------------------------------------------------------


class TestSqlTask:
    def test_set_current_sql_task(self):
        ctx = CliContext()
        task = _make_sql_task()
        ctx.set_current_sql_task(task)
        assert ctx.current_sql_task is task

    def test_get_or_create_returns_existing_when_no_text(self):
        ctx = CliContext()
        task = _make_sql_task()
        ctx.current_sql_task = task
        result = ctx.get_or_create_sql_task()
        assert result is task

    def test_get_or_create_new_with_text(self):
        ctx = CliContext()
        result = ctx.get_or_create_sql_task(task_text="analyze sales", database_type="sqlite")
        assert result.task == "analyze sales"
        assert result.database_type == "sqlite"

    def test_get_or_create_raises_without_text(self):
        ctx = CliContext()
        with pytest.raises(ValueError, match="required"):
            ctx.get_or_create_sql_task()

    def test_get_or_create_uses_prompt_callback(self):
        ctx = CliContext()
        callback = MagicMock(return_value="from callback")
        result = ctx.get_or_create_sql_task(prompt_callback=callback)
        assert result.task == "from callback"
        callback.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: clear methods
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_history(self):
        ctx = CliContext()
        ctx.add_table(_make_table())
        ctx.add_metric(_make_metric())
        ctx.add_sql_context(_make_sql_context())
        ctx.clear_history()
        assert len(ctx.get_recent_tables()) == 0
        assert len(ctx.get_recent_metrics()) == 0
        assert len(ctx.get_recent_sql_contexts()) == 0

    def test_clear_tables(self):
        ctx = CliContext()
        ctx.add_table(_make_table())
        ctx.add_metric(_make_metric())
        ctx.clear_tables()
        assert len(ctx.get_recent_tables()) == 0
        assert len(ctx.get_recent_metrics()) == 1

    def test_clear_metrics(self):
        ctx = CliContext()
        ctx.add_metric(_make_metric())
        ctx.add_table(_make_table())
        ctx.clear_metrics()
        assert len(ctx.get_recent_metrics()) == 0
        assert len(ctx.get_recent_tables()) == 1

    def test_clear_sql_contexts(self):
        ctx = CliContext()
        ctx.add_sql_context(_make_sql_context())
        ctx.clear_sql_contexts()
        assert len(ctx.get_recent_sql_contexts()) == 0


# ---------------------------------------------------------------------------
# Tests: to_dict / get_context_summary
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_structure(self):
        ctx = CliContext()
        ctx.update_database_context(db_name="mydb", catalog="mycat", schema="myschema")
        d = ctx.to_dict()
        assert d["current_db_name"] == "mydb"
        assert d["current_catalog"] == "mycat"
        assert d["current_schema"] == "myschema"
        assert "recent_tables_count" in d
        assert "recent_metrics_count" in d

    def test_get_context_summary_empty(self):
        ctx = CliContext()
        summary = ctx.get_context_summary()
        assert summary == "No context available"

    def test_get_context_summary_with_db(self):
        ctx = CliContext()
        ctx.update_database_context(db_name="mydb", catalog="mycat")
        summary = ctx.get_context_summary()
        assert "mydb" in summary
        assert "mycat" in summary

    def test_get_context_summary_with_task(self):
        ctx = CliContext()
        ctx.set_current_sql_task(_make_sql_task("find top 10 customers"))
        summary = ctx.get_context_summary()
        assert "Task:" in summary

    def test_get_context_summary_long_task_truncated(self):
        ctx = CliContext()
        long_task = "x" * 100
        ctx.set_current_sql_task(_make_sql_task(long_task))
        summary = ctx.get_context_summary()
        assert "..." in summary

    def test_get_context_summary_with_counts(self):
        ctx = CliContext()
        ctx.add_table(_make_table())
        ctx.add_metric(_make_metric())
        ctx.add_sql_context(_make_sql_context())
        summary = ctx.get_context_summary()
        assert "tables" in summary
        assert "metrics" in summary
        assert "SQL" in summary
