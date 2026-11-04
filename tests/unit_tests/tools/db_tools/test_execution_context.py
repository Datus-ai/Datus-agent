# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Immutable namespace bindings isolate shared native and pooled connectors."""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

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
