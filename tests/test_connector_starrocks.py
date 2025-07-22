import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.db_tools.starrocks_connector import StarRocksConnector
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture
def agent_config(namespace: str = "starrocks") -> AgentConfig:
    return load_agent_config(namespace=namespace)


@pytest.fixture
def connector(agent_config: AgentConfig) -> StarRocksConnector:
    db_manager = db_manager_instance(agent_config.namespaces)
    connector = db_manager.get_conn(agent_config.current_namespace, agent_config.db_type)
    assert isinstance(connector, StarRocksConnector)
    return connector


def test_connector(connector: StarRocksConnector):
    assert connector.test_connection()
    assert len(connector.get_tables(database_name="ssb")) > 0


def test_get_tables_with_ddl(connector: StarRocksConnector):
    tables = connector.get_tables_with_ddl(database_name="ssb")
    assert len(tables) > 0
    for table in tables:
        assert table["table_name"]
        assert table["definition"]
        assert table["table_type"] == "table"
        assert table["database_name"]
        assert table["schema_name"] == ""
        assert table["catalog_name"] == "default_catalog"
        assert table["identifier"]
        assert len(table["identifier"].split(".")) == 3


def test_get_views_with_ddl(connector: StarRocksConnector):
    views = connector.get_views_with_ddl(database_name="ssb")
    assert len(views) >= 0
    for view in views:
        assert view["table_name"]
        assert view["definition"]
        assert view["table_type"] == "view"
        assert view["database_name"] == "ssb"
        assert view["schema_name"] == ""
        assert view["catalog_name"]
        full_array = view["identifier"].split(".")
        assert len(full_array) == 3
        assert full_array[0] == "default_catalog"
        assert full_array[1] == "ssb"
        assert full_array[2] == view["table_name"]


def test_get_materialized_views_with_ddl(connector: StarRocksConnector):
    views = connector.get_materialized_views_with_ddl(database_name="ssb")
    assert len(views) >= 0
    for view in views:
        assert view["table_name"]
        assert view["definition"]
        assert view["table_type"] == "mv"
        assert view["database_name"] == "ssb"
        assert view["schema_name"] == ""
        assert view["catalog_name"] == "default_catalog"
        full_array = view["identifier"].split(".")
        assert len(full_array) == 3


def test_exceptions(connector: StarRocksConnector):
    with pytest.raises(DatusException, match=ErrorCode.TOOL_DB_EXECUTE_QUERY_FAILED.code):
        connector.get_sample_rows(database_name="ssb", tables=["nonexistent_table"])


def test_get_databases(connector: StarRocksConnector):
    databases = connector.get_databases()
    assert len(databases) > 0
