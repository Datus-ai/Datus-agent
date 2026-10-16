"""Companion adapter integration, requiring the optional local datus-hana package."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

hana = pytest.importorskip("datus_hana", reason="Optional datus-hana package is not installed")

from datus.configuration.agent_config import DbConfig  # noqa: E402
from datus.tools.db_tools import connector_registry  # noqa: E402
from datus.tools.db_tools.db_manager import DBManager  # noqa: E402
from datus.tools.func_tool.database import DBFuncTool  # noqa: E402
from datus.utils.sql_utils import extract_table_names, parse_table_name_parts, table_name_field_order  # noqa: E402


@pytest.fixture(autouse=True)
def register_hana():
    attrs = ("_connectors", "_factories", "_metadata", "_capabilities", "_uri_builders", "_context_resolvers")
    saved = {name: dict(getattr(connector_registry, name)) for name in attrs}
    hana.register()
    yield
    for name, values in saved.items():
        getattr(connector_registry, name).clear()
        getattr(connector_registry, name).update(values)


@pytest.fixture
def connector():
    return hana.HanaConnector(hana.HanaConfig(host="h", username="U", password="runtime-secret", schema="Sales"))


def test_dbmanager_transmits_sdk_fields_and_keeps_tenant_out_of_sql_namespace():
    config = DbConfig.filter_kwargs(
        DbConfig,
        {
            "type": "hana",
            "host": "h",
            "port": 443,
            "username": "U",
            "password": "runtime-secret",
            "database": "TENANT",
            "schema": "Sales",
            "encrypt": True,
            "ssl_trust_store": "ca.pem",
        },
    )
    manager = DBManager.__new__(DBManager)
    connector = manager._build_conn(config)
    assert isinstance(connector, hana.HanaConnector)
    assert connector.config.ssl_trust_store == "ca.pem"
    assert connector.config.port == 443
    assert connector.config.database == "TENANT"
    assert connector.get_current_context() == {"catalog_name": "", "database_name": "", "schema_name": "Sales"}


def test_scoped_query_executes_original_sql_through_official_sdk(connector):
    sql = 'SELECT * FROM "Sales"."Orders" LIMIT 2'
    tool = DBFuncTool(connector, scoped_tables=['"Sales"."Orders"'], read_only=True)
    with patch("datus_hana.connector.dbapi.connect") as connect:
        cursor = connect.return_value.cursor.return_value
        cursor.description = [("id",)]
        cursor.fetchmany.side_effect = [[(1,)], []]
        result = tool.execute_read_enforced(sql, connector, result_format="list")
    assert result.success is True, result.error
    assert result.sql_return == [{"id": 1}]
    cursor.execute.assert_called_once_with(sql)


@pytest.mark.parametrize(
    "sql",
    [
        'SELECT TOP 1 * FROM "Sales"."Orders"',
        'SELECT * FROM "Sales"."Orders"; DELETE FROM "Sales"."Orders"',
        'SELECT * FROM "Sales"."Orders" FOR UPDATE',
        'SELECT * FROM "Sales"."Calculation"(PLACEHOLDER."$$p$$" => \'x\')',
        "DO BEGIN SELECT * FROM t; END",
    ],
)
def test_incomplete_analysis_rejected_before_sdk_even_without_scope(connector, sql):
    tool = DBFuncTool(connector, read_only=True)
    with patch("datus_hana.connector.dbapi.connect") as connect:
        result = tool.execute_read_enforced(sql, connector, result_format="list")
    assert not result.success
    assert "SQL analysis" in result.error
    connect.assert_not_called()


@pytest.mark.parametrize("sql", ['SELECT TOP 1 * FROM "Sales"."Orders"', 'SELECT * FROM "Sales"."Orders" FOR UPDATE'])
def test_writable_unified_dispatch_rejects_unverified_hana_before_sdk(connector, sql):
    tool = DBFuncTool(connector, read_only=False)
    with patch("datus_hana.connector.dbapi.connect") as connect:
        result = tool.execute_sql(sql)
    assert result.success == 0
    assert "SQL analysis" in result.error
    connect.assert_not_called()


def test_physical_table_scope_and_quoted_case(connector):
    sql = 'WITH x AS (SELECT * FROM "Sales"."Orders") SELECT * FROM x'
    assert extract_table_names(sql, "hana") == ['"Sales"."Orders"']
    assert parse_table_name_parts('"Sales"."Order.Item"', "hana")["table_name"] == "Order.Item"
    assert table_name_field_order("hana") == ["schema_name", "table_name"]
    tool = DBFuncTool(connector, scoped_tables=['"Sales"."Orders"'], read_only=True)
    with patch("datus_hana.connector.dbapi.connect") as connect:
        result = tool.execute_read_enforced('SELECT * FROM "Private"."Orders"', connector)
    assert not result.success
    assert "outside scoped context" in result.error
    connect.assert_not_called()


def test_policy_cannot_execute_unverified_generator_output(connector):
    sql = 'SELECT * FROM "Sales"."Orders"'
    tool = DBFuncTool(connector, read_only=True)
    decision = SimpleNamespace(allowed=True, sql=sql + " LIMIT 1", applied_policies=[])
    with (
        patch("datus.tools.policy_runtime.PolicyRuntime.before_sql_read", return_value=decision),
        patch("datus_hana.connector.dbapi.connect") as connect,
    ):
        result = tool.execute_read_enforced(sql, connector)
    assert not result.success
    assert "rewriting is unsupported" in result.error
    connect.assert_not_called()


def test_userkey_unresolved_default_schema_cannot_bypass_explicit_scope():
    connector = hana.HanaConnector(hana.HanaConfig(userkey="KEY"))
    tool = DBFuncTool(connector, scoped_tables=["PRIVATE.ORDERS"], read_only=True)
    with patch("datus_hana.connector.dbapi.connect") as connect:
        result = tool.execute_read_enforced("SELECT * FROM orders", connector)
    assert not result.success
    assert "Unresolved schema" in result.error
    connect.assert_not_called()


def test_lineage_does_not_use_compatible_parser_as_hana_grammar(tmp_path):
    from datus.tools.func_tool.lineage_tools import LineageTools

    tool = LineageTools(root_path=str(tmp_path))
    result = tool.upsert_lineage(sql="SELECT * FROM S.T", dialect="hana")
    assert not result.success
    assert "lineage is unsupported" in result.error
