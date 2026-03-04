import os

import pytest
import structlog
from pandas import DataFrame

from datus.tools.db_tools.config import SQLiteConfig
from datus.tools.db_tools.sqlite_connector import SQLiteConnector

log = structlog.get_logger()


@pytest.fixture
def db_path():
    """Fixture to create a temporary test database path"""
    test_db = "test_database.db"
    yield test_db
    # Cleanup after tests
    if os.path.exists(test_db):
        os.remove(test_db)


@pytest.fixture
def sqlite_connector(db_path):
    """Fixture to create a SQLiteConnector instance"""
    config = SQLiteConfig(db_path=db_path)
    connector = SQLiteConnector(config)
    yield connector
    connector.close()


def test_connection(sqlite_connector: SQLiteConnector, db_path):
    """Test database connection and close operations"""
    # Test connection
    sqlite_connector.connect()
    assert sqlite_connector.connection is not None
    assert os.path.exists(db_path)

    # Test close
    sqlite_connector.close()
    assert sqlite_connector.connection is None


def test_test_connection(tmp_path):
    db_path = tmp_path / "test.db"
    config = SQLiteConfig(db_path=str(db_path))
    connector = SQLiteConnector(config)
    connector.connect()

    # Test connection validation
    assert connector.test_connection() is True

    # Verify PRAGMA database_list contains main database
    result = connector.execute({"sql_query": "PRAGMA database_list;"}, result_format="list")
    assert result.success is True
    assert any(db["name"] == "main" for db in result.sql_return)


def test_execute_select(sqlite_connector: SQLiteConnector):
    """Test SELECT query execution"""
    # Create a test table and insert data
    sqlite_connector.execute_ddl("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    sqlite_connector.execute_insert("INSERT INTO test (name) VALUES ('test1')")
    sqlite_connector.execute_insert("INSERT INTO test (name) VALUES ('test2')")

    # Test SELECT query
    result = sqlite_connector.execute({"sql_query": "SELECT * FROM test"}, result_format="list")
    print("result", result)
    assert result["success"] is True
    assert result["row_count"] == 2
    assert len(result.sql_return) == 2
    assert result.sql_return[0]["name"] == "test1"
    assert result.sql_return[1]["name"] == "test2"

    result = sqlite_connector.execute_pandas("select * from test")
    assert isinstance(result.sql_return, DataFrame)
    assert result.row_count == 2

    sample_values = sqlite_connector.get_sample_rows(tables=["test"], top_n=5)
    assert len(sample_values) == 1
    assert sample_values[0]["table_name"] == "test"


def test_execute_insert(sqlite_connector):
    """Test INSERT query execution"""
    # Create test table
    sqlite_connector.execute_ddl("CREATE TABLE test_insert (id INTEGER PRIMARY KEY, value TEXT)")

    # Test INSERT
    result = sqlite_connector.execute_insert("INSERT INTO test_insert (value) VALUES ('test_value')")
    assert result.success is True
    assert result.row_count == 1

    # Verify insertion
    verify = sqlite_connector.execute({"sql_query": "SELECT * FROM test_insert"}, result_format="list")
    assert verify.sql_return[0]["value"] == "test_value"


def test_execute_update(sqlite_connector):
    """Test UPDATE query execution"""
    # Setup test data
    sqlite_connector.execute_ddl("CREATE TABLE test_update (id INTEGER PRIMARY KEY, value TEXT)")
    sqlite_connector.execute_insert("INSERT INTO test_update (value) VALUES ('old_value')")

    # Test UPDATE
    result = sqlite_connector.execute_update("UPDATE test_update SET value = 'new_value' WHERE value = 'old_value'")
    assert result["success"] is True
    assert result["row_count"] == 1

    # Verify update
    verify = sqlite_connector.execute({"sql_query": "SELECT * FROM test_update"}, result_format="list")
    assert verify.sql_return[0]["value"] == "new_value"


def test_execute_delete(sqlite_connector):
    """Test DELETE query execution"""
    # Setup test data
    sqlite_connector.execute_ddl("CREATE TABLE test_delete (id INTEGER PRIMARY KEY, value TEXT)")
    sqlite_connector.execute_insert("INSERT INTO test_delete (value) VALUES ('to_delete')")

    # Test DELETE
    result = sqlite_connector.execute_delete("DELETE FROM test_delete WHERE value = 'to_delete'")
    assert result["success"] is True
    assert result["row_count"] == 1

    # Verify deletion
    verify = sqlite_connector.execute({"sql_query": "SELECT * FROM test_delete"}, result_format="list")
    assert verify["row_count"] == 0


def test_execute_error(sqlite_connector: SQLiteConnector):
    """Test error handling in query execution"""
    result = sqlite_connector.execute({"sql_query": "SELECT * FROM non_existent_table"}, result_format="list")
    assert result.error is not None


def test_input_validation(sqlite_connector):
    """Test input parameter validation"""
    with pytest.raises(ValueError, match="'sql_query' parameter is required"):
        sqlite_connector.execute({}, result_format="list")

    with pytest.raises(ValueError, match="'sql_query' must be a string"):
        sqlite_connector.execute({"sql_query": 123}, result_format="list")


def test_get_schema(sqlite_connector):
    """Test schema retrieval"""
    # Create test tables
    sqlite_connector.execute(
        {
            "sql_query": """
        CREATE TABLE test_table1 (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value INTEGER
        )
        """
        }
    )
    sqlite_connector.execute(
        {
            "sql_query": """
        CREATE TABLE test_table2 (
            id INTEGER PRIMARY KEY,
            description TEXT
        )
        """
        }
    )

    schema_list = sqlite_connector.get_tables_with_ddl()
    log.debug("schema_list", schema_list=schema_list)
    assert isinstance(schema_list, list)
    assert len(schema_list) == 2

    # Verify first table schema
    table1 = next(s for s in schema_list if s["table_name"] == "test_table1")
    assert isinstance(table1, dict)
    assert "table_name" in table1
    assert "definition" in table1
    assert table1["table_name"] == "test_table1"
    assert "id INTEGER PRIMARY KEY" in table1["definition"]
    assert "name TEXT NOT NULL" in table1["definition"]
    assert "value INTEGER" in table1["definition"]

    # Verify second table schema
    table2 = next(s for s in schema_list if s["table_name"] == "test_table2")
    assert isinstance(table2, dict)
    assert "table_name" in table2
    assert "definition" in table2
    assert table2["table_name"] == "test_table2"
    assert "id INTEGER PRIMARY KEY" in table2["definition"]
    assert "description TEXT" in table2["definition"]
