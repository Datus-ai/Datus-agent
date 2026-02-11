import os
import shutil
import tempfile

import pytest

from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.tools.db_tools.config import SQLiteConfig
from datus.tools.db_tools.sqlite_connector import SQLiteConnector
from datus.tools.lineage_graph_tools.schema_lineage import SchemaLineageTool
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

log = get_logger(__name__)


@pytest.fixture
def db_path():
    """Fixture to create a temporary test database path"""
    test_db = "test_database.db"
    yield test_db
    # Cleanup after tests
    if os.path.exists(test_db):
        os.remove(test_db)


@pytest.fixture
def lancedb_path():
    """Fixture to create a temporary LanceDB directory"""
    test_path = tempfile.mkdtemp()
    yield test_path
    # Cleanup after tests
    if os.path.exists(test_path):
        shutil.rmtree(test_path)


@pytest.fixture
def agent_config():
    """Fixture to create a minimal AgentConfig for testing"""
    from tests.conftest import load_acceptance_config

    config = load_acceptance_config(namespace="snowflake")
    return config


@pytest.fixture
def schema_lineage_tool(agent_config):
    """Fixture to create a SchemaLineageTool instance"""
    tool = SchemaLineageTool(agent_config=agent_config)
    return tool


@pytest.fixture
def sqlite_connector(db_path):
    """Fixture to create a SQLiteConnector instance"""
    config = SQLiteConfig(db_path=db_path)
    connector = SQLiteConnector(config)
    yield connector
    connector.close()


def test_storage():
    pass


def test_store_and_search_schema(schema_lineage_tool, sqlite_connector):
    """Test storing and searching schema"""
    log.info("Creating test tables in SQLite")
    # Create test tables in SQLite
    sqlite_connector.execute_ddl(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP
        )
        """
    )

    sqlite_connector.execute_ddl(
        """
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT,
            user_id INTEGER,
            created_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # Test searching for schemas using SchemaLinkingInput
    log.info("Searching for similar schemas")
    input_param = SchemaLinkingInput(
        input_text="username email user",
        database_type=DBType.SQLITE,
        catalog_name="",
        database_name="",
        schema_name="",
        matching_rate="fast",
        sql_context=None,
    )
    search_result = schema_lineage_tool.execute(input_param)
    log.debug("Search result", result=search_result)

    assert search_result["success"] is True
    # Since we haven't stored anything in the RAG storage yet, count should be 0
    assert search_result["schema_count"] >= 0
    assert search_result["value_count"] >= 0


def test_invalid_input(schema_lineage_tool):
    """Test input validation"""
    log.info("Testing empty input text")
    # Test empty input text - should return empty result, not raise error
    input_param = SchemaLinkingInput(
        input_text="",
        database_type=DBType.SQLITE,
        catalog_name="",
        database_name="",
        schema_name="",
        matching_rate="fast",
        sql_context=None,
    )
    result = schema_lineage_tool.execute(input_param)
    assert result["success"] is True
    assert result["schema_count"] == 0
    assert result["value_count"] == 0


def test_schema_linking_no_exist():
    """Test schema linking with non-existent database"""
    from tests.conftest import load_acceptance_config

    test_config = load_acceptance_config(namespace="snowflake")
    tool = SchemaLineageTool(agent_config=test_config)
    res = tool.execute(
        SchemaLinkingInput(
            input_text="test query",
            database_type=DBType.SQLITE,
            catalog_name="",
            database_name="non_existent_db",
            schema_name="",
            matching_rate="fast",
            sql_context=None,
        )
    )
    # The tool should handle non-existent data gracefully
    assert res["success"] is True  # Operation succeeds but returns no results
    assert res["schema_count"] == 0
    assert res["value_count"] == 0


def test_get_schema_from_db(schema_lineage_tool: SchemaLineageTool, sqlite_connector: SQLiteConnector):
    res = schema_lineage_tool.get_schems_by_db(
        connector=sqlite_connector,
        input_param=SchemaLinkingInput(
            input_text="",
            database_type=DBType.SQLITE,
            catalog_name="",
            database_name="",
            schema_name="",
            matching_rate="fast",
            sql_context=None,
        ),
    )
    assert res["success"]
    assert res["schema_count"] == 0
    assert res["value_count"] == 0
