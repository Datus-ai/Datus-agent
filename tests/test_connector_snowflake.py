import os
from random import Random
from typing import List

import pytest

from datus.tools.db_tools import SnowflakeConnector


@pytest.fixture
def connector() -> SnowflakeConnector:
    connector = SnowflakeConnector(
        account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
        user=os.getenv("SNOWFLAKE_USER", "") or os.getenv("SNOWFLAKE_USERNAME", ""),
        password=os.getenv("SNOWFLAKE_PASSWORD", ""),
        warehouse="COMPUTE_WH_PARTICIPANT",
    )
    return connector


def test_snowflake_connector(connector: SnowflakeConnector):
    connector.test_connection()

    res = connector.execute(
        {
            "sql_query": """select * from PATENTS_GOOGLE.PATENTS_GOOGLE.DISCLOSURES_13
            where "record_id" = 'REC00003' limit 10""",
        }
    )
    print(res.to_dict())


def _do_get_databases(connector: SnowflakeConnector) -> List[str]:
    return connector.get_databases()


def test_get_databases(connector: SnowflakeConnector):
    database_names = _do_get_databases(connector)
    assert len(database_names) > 0


def test_get_schemas(connector: SnowflakeConnector):
    database_names = _do_get_databases(connector)
    rd = Random()
    for _ in range(3):
        index = rd.randint(0, len(database_names))
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
