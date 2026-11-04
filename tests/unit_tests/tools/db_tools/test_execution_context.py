# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Immutable namespace bindings isolate shared native and pooled connectors."""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import duckdb
import pytest

from datus.configuration.agent_config import DbConfig
from datus.schemas.database_context import DatabaseExecutionContext
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.func_tool.database import DBFuncTool


def test_same_datasource_argument_keeps_the_turn_database():
    manager = Mock(spec=DBManager)
    manager.get_conn.side_effect = lambda ds, database: SimpleNamespace(database_name=database)
    tool = DBFuncTool.__new__(DBFuncTool)
    tool._db_manager = manager
    tool._default_datasource = "warehouse"
    tool._default_database = "selected"
    tool._connector_cache = OrderedDict()
    tool._connector_cache_size = 8
    assert tool._get_connector("warehouse").database_name == "selected"
    assert tool._get_connector("other").database_name == ""


def test_two_namespace_bindings_do_not_mutate_the_shared_config_or_connection():
    configured = DbConfig(type="postgresql", database="configured", schema="public")
    manager = DBManager({"warehouse": configured})
    manager._build_conn = lambda cfg: SimpleNamespace(database=cfg.database, schema=cfg.schema, catalog=cfg.catalog)
    a = DatabaseExecutionContext(datasource="warehouse", database="analytics", schema="a")
    b = DatabaseExecutionContext(datasource="warehouse", database="analytics", schema="b")
    first = manager.get_conn("warehouse", "analytics", context=a)
    second = manager.get_conn("warehouse", "analytics", context=b)
    assert (first.database, first.schema) == ("analytics", "a")
    assert (second.database, second.schema) == ("analytics", "b")
    assert first is manager.get_conn("warehouse", "analytics", context=a)
    assert first is not second
    assert (configured.database, configured.schema) == ("configured", "public")


def test_changed_catalog_does_not_reintroduce_the_old_catalog_database_default():
    manager = DBManager({"warehouse": DbConfig(type="starrocks", catalog="cat_a", database="analytics")})
    context = DatabaseExecutionContext(datasource="warehouse", catalog="cat_b")
    with pytest.raises(ValueError, match="Select a database explicitly"):
        manager.get_conn("warehouse", context=context)


@pytest.mark.parametrize("engine", ["postgresql", "mysql", "starrocks", "sqlite", "duckdb"])
def test_resolved_empty_database_cannot_reuse_a_configured_or_cached_database(engine):
    configured = DbConfig(type=engine, catalog="cat_a", database="analytics", schema="tenant_a")
    manager = DBManager({"warehouse": configured})
    manager._build_conn = Mock(return_value=SimpleNamespace(database="analytics", connect=lambda: None))
    assert manager.get_conn("warehouse").database == "analytics"
    context = DatabaseExecutionContext(datasource="warehouse", catalog="cat_a")
    with pytest.raises(ValueError, match="Select a database explicitly"):
        manager.get_conn("warehouse", context=context)
    manager._build_conn.assert_called_once()
    assert configured.database == "analytics"


def test_resolved_database_is_authoritative_even_when_the_positional_argument_is_omitted():
    manager = DBManager({"warehouse": DbConfig(type="mysql", database="configured")})
    manager._build_conn = lambda cfg: SimpleNamespace(database=cfg.database)
    context = DatabaseExecutionContext(datasource="warehouse", database="selected")
    assert manager.get_conn("warehouse", context=context).database == "selected"


def test_resolved_empty_database_cannot_select_the_first_file_of_a_glob(tmp_path):
    import sqlite3

    with sqlite3.connect(tmp_path / "analytics.sqlite"):
        pass
    manager = DBManager({"warehouse": DbConfig(type="sqlite", path_pattern=str(tmp_path / "*.sqlite"))})
    with pytest.raises(ValueError, match="Select a database explicitly"):
        manager.get_conn("warehouse", context=DatabaseExecutionContext(datasource="warehouse"))


def test_persisted_artifact_binding_keeps_original_database_after_the_turn_switches(tmp_path):
    import sqlite3

    for name, marker in [("a", "original"), ("b", "current")]:
        with sqlite3.connect(tmp_path / f"{name}.sqlite") as conn:
            conn.execute("CREATE TABLE marker(value TEXT)")
            conn.execute("INSERT INTO marker VALUES (?)", (marker,))
    configured = DbConfig(type="sqlite", path_pattern=str(tmp_path / "*.sqlite"))
    manager = DBManager({"warehouse": configured})
    tool = DBFuncTool.__new__(DBFuncTool)
    tool._db_manager = manager
    tool._default_datasource = "warehouse"
    tool._default_database = "b"
    tool._execution_context = {"datasource": "warehouse", "database": "b"}
    tool._connector_cache = OrderedDict()
    tool._connector_cache_size = 8
    current = tool._get_connector("warehouse")
    assert current.execute_query("SELECT value FROM marker", result_format="list").sql_return[0]["value"] == "current"
    persisted = tool.connector_for_execution_context({"datasource": "warehouse", "database": "a"})
    assert (
        persisted.execute_query("SELECT value FROM marker", result_format="list").sql_return[0]["value"] == "original"
    )
    assert tool._get_connector("warehouse") is current
    manager.close()


def test_duckdb_schema_is_bound_before_caching_and_replayed_in_another_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(path)) as conn:
        for schema in ["main", "tenant_a", 'tenant_"b']:
            quoted = '"' + schema.replace('"', '""') + '"'
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")
            conn.execute(f"CREATE TABLE {quoted}.marker AS SELECT ? AS value", [schema])
    manager = DBManager({"warehouse": DbConfig(type="duckdb", uri=str(path))})
    try:
        a = DatabaseExecutionContext(datasource="warehouse", database="warehouse", schema="tenant_a")
        b = DatabaseExecutionContext(datasource="warehouse", database="warehouse", schema='tenant_"b')
        first = manager.get_conn("warehouse", "warehouse", context=a)
        second = manager.get_conn("warehouse", "warehouse", context=b)
        with ThreadPoolExecutor(max_workers=1) as pool:
            for connector, expected in [(first, a.schema), (second, b.schema), (first, a.schema)]:
                result = pool.submit(connector.execute_query, "SELECT value FROM marker", "list").result()
                assert result.sql_return == [{"value": expected}]
                assert connector.schema_name == expected
        invalid = DatabaseExecutionContext(datasource="warehouse", database="warehouse", schema="missing")
        with pytest.raises(Exception, match="missing"):
            manager.get_conn("warehouse", "warehouse", context=invalid)
        assert ("warehouse", "", "missing") not in manager._conn_dict["warehouse"]
    finally:
        manager.close()
