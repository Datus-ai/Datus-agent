import os
from random import Random
from typing import List

import pandas as pd
import pyarrow
import pytest

from datus.tools.db_tools import SnowflakeConnector
from datus.utils.exceptions import DatusException


@pytest.fixture
def connector() -> SnowflakeConnector:
    connector = SnowflakeConnector(
        account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
        user=os.getenv("SNOWFLAKE_USER", "") or os.getenv("SNOWFLAKE_USERNAME", ""),
        password=os.getenv("SNOWFLAKE_PASSWORD", ""),
        warehouse="COMPUTE_WH_PARTICIPANT",
    )
    return connector


def test_query(connector: SnowflakeConnector):
    connector.test_connection()

    res = connector.execute(
        {
            "sql_query": """select * from BBC.BBC_NEWS.FULLTEXT limit 10""",
        },
        result_format="arrow",
    )
    assert res.success
    assert isinstance(res.sql_return, pyarrow.Table)
    res = connector.execute(
        {
            "sql_query": """select * from PATENTS_GOOGLE.PATENTS_GOOGLE.DISCLOSURES_13
                            where "record_id" = 'REC00003' limit 10""",
        },
        result_format="pandas",
    )
    assert res.success
    assert isinstance(res.sql_return, pd.DataFrame)

    res = connector.execute(
        {
            "sql_query": """select * from PATENTS_GOOGLE.PATENTS_GOOGLE.DISCLOSURES_13
                            where "record_id" = 'REC00003' limit 10""",
        },
        result_format="csv",
    )
    assert res.success
    assert isinstance(res.sql_return, str)


def test_content_switch(connector: SnowflakeConnector):
    # connector.execute({"sql_query": "use PATENTS_GOOGLE"})
    # connector.switch_context(database_name="PATENTS_GOOGLE")
    with pytest.raises(DatusException):
        connector.switch_context(database_name="NONEXISTENT_DB")


def _do_get_databases(connector: SnowflakeConnector) -> List[str]:
    return connector.get_databases()


def test_get_databases(connector: SnowflakeConnector):
    database_names = _do_get_databases(connector)
    assert len(database_names) > 0

    res = connector.execute({"sql_query": "show databases"}, result_format="arrow")
    assert res.success
    assert isinstance(res.sql_return, pyarrow.Table)
    assert res.row_count == res.sql_return.num_rows

    res = connector.execute({"sql_query": "show databases"}, result_format="pandas")
    assert res.success
    assert isinstance(res.sql_return, pd.DataFrame)
    assert res.row_count == len(res.sql_return)

    res = connector.execute({"sql_query": "show databases"}, result_format="list")
    assert res.success
    assert isinstance(res.sql_return, list)
    assert res.row_count == len(res.sql_return)

    res = connector.execute({"sql_query": "show databases"}, result_format="csv")
    assert res.success
    assert res.row_count > 0
    assert isinstance(res.sql_return, str)


def test_get_schemas(connector: SnowflakeConnector):
    database_names = _do_get_databases(connector)
    rd = Random()
    for _ in range(3):
        index = rd.randint(0, len(database_names) - 1)
        db_name = database_names[index]
        connector.switch_context(database_name=db_name)
        print("switch database:", db_name)
        assert len(connector.get_schemas(database_name=db_name)) > 0
        assert len(connector.get_schemas()) > 0


def test_get_tables(connector: SnowflakeConnector):
    connector.switch_context(database_name="BBC", schema_name="BBC_NEWS")
    assert len(connector.get_tables()) > 0
    connector.switch_context(database_name="HACKER_NEWS", schema_name="HACKER_NEWS")
    assert len(connector.get_tables()) > 0


def test_get_tables_with_ddl(connector: SnowflakeConnector):
    connector.switch_context(database_name="BBC", schema_name="BBC_NEWS")
    tables = connector.get_tables_with_ddl(database_name="BBC", schema_name="BBC_NEWS")

    if not tables:
        pytest.skip("No tables available in BBC.BBC_NEWS to verify DDL retrieval")
    for table in tables:
        assert table["table_name"]
        assert table["definition"].strip()
        assert table["table_type"] == "table"
        assert table["database_name"] == "BBC"
        assert table["schema_name"] == "BBC_NEWS"
        assert table["catalog_name"] == ""
        identifier_parts = table["identifier"].split(".")
        assert len(identifier_parts) >= 3
        assert identifier_parts[0] == "BBC"
        assert identifier_parts[1] == "BBC_NEWS"
        assert identifier_parts[2] == table["table_name"]

    sample_table = tables[0]["table_name"]
    filtered = connector.get_tables_with_ddl(
        database_name="BBC",
        schema_name="BBC_NEWS",
        tables=[sample_table],
    )

    assert len(filtered) == 1
    assert filtered[0]["table_name"] == sample_table


def test_get_sample_rows(connector: SnowflakeConnector):
    sample_rows = connector.get_sample_rows(catalog_name="", database_name="BBC", table_type="table")
    assert len(sample_rows) > 0

    sample_rows = connector.get_sample_rows(database_name="BBC", table_type="table")
    assert len(sample_rows) > 0
    sample_rows = connector.get_sample_rows(
        catalog_name="",
        database_name="BBC",
        schema_name="BBC_NEWS",
    )
    assert len(sample_rows) > 0

    sample_rows = connector.get_sample_rows(
        catalog_name="", database_name="BBC", schema_name="BBC_NEWS", table_type="table", tables=["FULLTEXT"]
    )
    assert len(sample_rows) == 1

    first_item = sample_rows[0]
    assert first_item["database_name"]
    assert first_item["table_name"]
    assert not first_item["catalog_name"]
    assert first_item["schema_name"]
    assert first_item["identifier"]
    assert len(first_item["identifier"].split(".")) == 3


def test_get_views(connector: SnowflakeConnector):
    connector.switch_context(database_name="BBC", schema_name="BBC_NEWS")
    views = connector.get_views(database_name="BBC", schema_name="BBC_NEWS")
    assert isinstance(views, list)
    if not views:
        pytest.skip("No views available in BBC.BBC_NEWS to verify metadata retrieval")
    for view_name in views:
        assert isinstance(view_name, str)
        assert view_name.strip()


def test_get_views_with_ddl(connector: SnowflakeConnector):
    connector.switch_context(database_name="BBC", schema_name="BBC_NEWS")
    views = connector.get_views_with_ddl(database_name="BBC", schema_name="BBC_NEWS")
    if not views:
        pytest.skip("No views available in BBC.BBC_NEWS to verify DDL retrieval")
    for view in views:
        assert view["table_name"]
        assert view["definition"].strip()
        assert view["table_type"] == "view"
        assert view["database_name"] == "BBC"
        assert view["schema_name"] == "BBC_NEWS"
        assert view["catalog_name"] == ""
        identifier_parts = view["identifier"].split(".")
        assert len(identifier_parts) >= 3
        assert identifier_parts[0] == "BBC"
        assert identifier_parts[1] == "BBC_NEWS"
        assert identifier_parts[2] == view["table_name"]


def test_get_materialized_views(connector: SnowflakeConnector):
    connector.switch_context(database_name="BBC", schema_name="BBC_NEWS")
    materialized_views = connector.get_materialized_views(database_name="BBC", schema_name="BBC_NEWS")
    assert isinstance(materialized_views, list)
    if not materialized_views:
        pytest.skip("No materialized views available in BBC.BBC_NEWS to verify metadata retrieval")
    for mv_name in materialized_views:
        assert isinstance(mv_name, str)
        assert mv_name.strip()


def test_get_materialized_views_with_ddl(connector: SnowflakeConnector):
    connector.switch_context(database_name="BBC", schema_name="BBC_NEWS")
    materialized_views = connector.get_materialized_views_with_ddl(database_name="BBC", schema_name="BBC_NEWS")
    if not materialized_views:
        pytest.skip("No materialized views available in BBC.BBC_NEWS to verify DDL retrieval")
    for mv in materialized_views:
        assert mv["table_name"]
        assert mv["definition"].strip()
        assert mv["table_type"] == "mv"
        assert mv["database_name"] == "BBC"
        assert mv["schema_name"] == "BBC_NEWS"
        assert mv["catalog_name"] == ""
        identifier_parts = mv["identifier"].split(".")
        assert len(identifier_parts) >= 3
        assert identifier_parts[0] == "BBC"
        assert identifier_parts[1] == "BBC_NEWS"
        assert identifier_parts[2] == mv["table_name"]


class TestExceptions:
    def test_syntax(self, connector: SnowflakeConnector):
        res = connector.execute({"sql_query": "SELECT * FROM_ PATENTS_GOOGLE.PATENTS_GOOGLE.DISCLOSURES_13"})
        assert res.success
        assert "syntax" in res.error

        res = connector.execute({"sql_query": "SELECT * FROM PATENTS_GOOGLE.PATENTS_GOOGLE.NOEXIST_TABLE"})
        assert res.success
