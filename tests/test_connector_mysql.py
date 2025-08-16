import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.db_tools.mysql_connector import MySQLConnector
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture
def agent_config(namespace: str = "local_mysql") -> AgentConfig:
    return load_agent_config(namespace=namespace)


@pytest.fixture
def connector(agent_config: AgentConfig) -> MySQLConnector:
    db_manager = db_manager_instance(agent_config.namespaces)
    connector = db_manager.get_conn(agent_config.current_namespace)
    assert isinstance(connector, MySQLConnector)
    return connector


def test_mysql_connector(connector: MySQLConnector):
    assert connector.test_connection()
    assert len(connector.get_tables()) > 0


def test_get_tables_with_ddl(connector: MySQLConnector):
    tables = connector.get_tables_with_ddl()
    assert len(tables) > 0
    for table in tables:
        assert table["table_name"]
        assert table["definition"]
        assert table["table_type"] == "table"
        assert table["database_name"]
        assert table["schema_name"] == ""
        assert table["catalog_name"] == ""
        assert table["identifier"]
        assert len(table["identifier"].split(".")) == 2


def test_get_views_with_ddl(connector: MySQLConnector):
    views = connector.get_views_with_ddl()
    assert len(views) >= 0
    for view in views:
        assert view["table_name"]
        assert view["definition"]
        assert view["table_type"] == "view"
        assert view["database_name"]
        assert view["schema_name"] == ""


def test_exceptions(connector: MySQLConnector):
    with pytest.raises(DatusException, match=ErrorCode.DB_EXECUTION_ERROR.code):
        connector.get_sample_rows(database_name="test", tables=["nonexistent_table"])


def test_get_databases(connector: MySQLConnector):
    databases = connector.get_databases()
    assert len(databases) > 0
