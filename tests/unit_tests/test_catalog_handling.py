from unittest.mock import MagicMock, call

import pandas as pd

from datus.tools.db_tools.mysql_connector import MySQLConnector
from datus.tools.db_tools.starrocks_connector import StarRocksConnector


def _make_starrocks_connector() -> StarRocksConnector:
    return StarRocksConnector(
        host="localhost",
        port=9030,
        user="user",
        password="password",
        catalog="default_catalog",
        database="",
    )


def _make_mysql_connector() -> MySQLConnector:
    return MySQLConnector(
        host="localhost",
        port=3306,
        user="user",
        password="password",
        database="test_db",
    )


def test_starrocks_do_switch_context_sets_catalog_and_database():
    connector = _make_starrocks_connector()
    execute_mock = MagicMock()
    connector._execute = execute_mock

    connector.do_switch_context(catalog_name="internal", database_name="analytics")

    execute_mock.assert_has_calls([call("SET CATALOG `internal`"), call("USE `analytics`")])
    assert execute_mock.call_count == 2

    execute_mock.reset_mock()
    connector.do_switch_context(catalog_name="def")
    execute_mock.assert_called_once_with("SET CATALOG `default_catalog`")


def test_mysql_do_switch_context_only_uses_database():
    connector = _make_mysql_connector()
    execute_mock = MagicMock()
    connector._execute = execute_mock

    connector.do_switch_context(database_name="analytics`zone")

    execute_mock.assert_called_once_with("USE `analytics``zone`")


def test_starrocks_materialized_views_respect_catalog_filter():
    connector = _make_starrocks_connector()
    mv_rows = pd.DataFrame(
        {
            "TABLE_CATALOG": ["def"],
            "TABLE_SCHEMA": ["db_one"],
            "TABLE_NAME": ["mv_sales"],
            "MATERIALIZED_VIEW_DEFINITION": ["CREATE MATERIALIZED VIEW mv_sales AS SELECT 1"],
        }
    )
    execute_pandas_mock = MagicMock(return_value=mv_rows)

    def switch_side_effect(catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        if catalog_name:
            connector.catalog_name = catalog_name

    connector.connect = MagicMock()
    connector.switch_context = MagicMock(side_effect=switch_side_effect)
    connector._execute_pandas = execute_pandas_mock

    views = connector.get_materialized_views_with_ddl(catalog_name="test_catalog")

    connector.switch_context.assert_called_once_with(catalog_name="test_catalog")
    query_sql = execute_pandas_mock.call_args[0][0]
    assert "TABLE_CATALOG IN" not in query_sql.upper()
    assert len(views) == 1
    view = views[0]
    assert view["catalog_name"] == "test_catalog"
    assert view["identifier"].split(".")[0] == "test_catalog"


def test_starrocks_get_tables_switches_catalog():
    connector = _make_starrocks_connector()
    tables_df = pd.DataFrame(
        {
            "TABLE_CATALOG": ["def"],
            "TABLE_SCHEMA": ["db_two"],
            "TABLE_NAME": ["orders"],
        }
    )
    execute_pandas_mock = MagicMock(return_value=tables_df)

    def switch_side_effect(catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        if catalog_name:
            connector.catalog_name = catalog_name

    connector.connect = MagicMock()
    connector.switch_context = MagicMock(side_effect=switch_side_effect)
    connector._execute_pandas = execute_pandas_mock

    tables = connector.get_tables(catalog_name="analytics_catalog")

    connector.switch_context.assert_called_once_with(catalog_name="analytics_catalog")
    assert tables == ["orders"]
