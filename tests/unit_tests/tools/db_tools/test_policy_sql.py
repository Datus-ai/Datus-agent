# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import sqlglot
from sqlglot import exp

from datus.tools.db_tools.policy_sql import resolve_policy_sql
from datus.utils.exceptions import DatusException


def test_namespace_identifiers_are_quoted_without_changing_table_aliases():
    connector = SimpleNamespace(
        dialect="starrocks",
        get_current_context=lambda: {"catalog_name": "catalog-with-dash", "database_name": "shop.with.dot"},
    )
    sql = resolve_policy_sql("SELECT o.id FROM `orders.with.dot` AS o", connector)
    table = sqlglot.parse_one(sql, dialect="starrocks").find(exp.Table)
    assert (table.catalog, table.db, table.name, table.alias) == (
        "catalog-with-dash",
        "shop.with.dot",
        "orders.with.dot",
        "o",
    )


@pytest.mark.parametrize("context", [None, {}, {"catalog_name": "", "database_name": "shop"}])
def test_missing_context_cannot_silently_bypass_a_qualified_policy(context):
    connector = SimpleNamespace(dialect="starrocks", get_current_context=lambda: context)
    with pytest.raises(DatusException, match="Cannot resolve physical table names"):
        resolve_policy_sql("SELECT * FROM orders", connector)


@pytest.mark.parametrize(
    "sql", ["SELECT 1", "WITH result AS (SELECT 1) SELECT * FROM result", "SELECT * FROM catalog.db.orders"]
)
def test_queries_without_unresolved_physical_tables_do_not_need_defaults(sql):
    connector = SimpleNamespace(dialect="starrocks", get_current_context=Mock(side_effect=AssertionError("unexpected")))
    assert resolve_policy_sql(sql, connector) == sql
    connector.get_current_context.assert_not_called()


def test_connector_errors_do_not_expose_connection_details():
    connector = SimpleNamespace(
        dialect="starrocks", get_current_context=Mock(side_effect=ValueError("private-connection-marker"))
    )
    with pytest.raises(DatusException) as error:
        resolve_policy_sql("SELECT * FROM orders", connector)
    assert "private-connection-marker" not in str(error.value)


def test_other_dialects_keep_their_existing_namespace_semantics():
    connector = SimpleNamespace(dialect="sqlite")
    assert resolve_policy_sql("SELECT * FROM orders", connector) == "SELECT * FROM orders"
