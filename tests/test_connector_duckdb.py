import uuid

import pytest

from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture
def duckdb_connector() -> DuckdbConnector:
    config = DuckDBConfig(db_path="sample_data/duckdb-demo.duckdb")
    return DuckdbConnector(config)


@pytest.fixture
def duckdb_memory_connector() -> DuckdbConnector:
    config = DuckDBConfig(db_path="duckdb:///:memory:", database_name="quickstart")
    return DuckdbConnector(config)


@pytest.mark.acceptance
def test_get_table_with_ddl(duckdb_connector: DuckdbConnector):
    tables = duckdb_connector.get_tables_with_ddl()
    assert len(tables) > 0
    for t in tables:
        values = duckdb_connector.get_sample_rows(
            tables=[t["table_name"]],
            top_n=5,
            database_name=t["database_name"],
            schema_name=t["schema_name"],
        )
        assert len(values) > 0
        assert values[0]["table_name"] == t["table_name"]
        assert values[0]["database_name"] == t["database_name"]
        assert values[0]["schema_name"] == t["schema_name"]


@pytest.mark.acceptance
def test_get_views_with_ddl(duckdb_connector: DuckdbConnector):
    views = duckdb_connector.get_views_with_ddl()
    print(views)


@pytest.mark.acceptance
def test_get_table_schema(duckdb_connector: DuckdbConnector):
    schema = duckdb_connector.get_schema(table_name="search_trends")
    assert len(schema) > 0

    assert len(duckdb_connector.get_schema()) == 0

    with pytest.raises(DatusException, match=ErrorCode.DB_QUERY_METADATA_FAILED.code):
        duckdb_connector.get_schema(table_name="unexist_table")


@pytest.mark.acceptance
def test_execute_query(duckdb_connector: DuckdbConnector):
    duckdb_connector.test_connection()

    res = duckdb_connector.execute_query("select * from bank_failures limit 10", result_format="list")
    assert res.success
    assert len(res.sql_return) > 0
    assert isinstance(res.sql_return[0], dict)

    res = duckdb_connector.execute_query("select * from unexist_table")
    assert not res.success
    assert ErrorCode.DB_TABLE_NOT_EXISTS.code in res.error


@pytest.mark.acceptance
def test_get_schemas(duckdb_connector: DuckdbConnector):
    schemas = duckdb_connector.get_schemas()
    assert len(schemas) > 0
    schemas = duckdb_connector.get_schemas(database_name="nonexists")
    assert len(schemas) == 0


def test_insert_round_trip(duckdb_memory_connector: DuckdbConnector):
    suffix = uuid.uuid4().hex[:8]
    table_name = f"datus_insert_test_{suffix}"

    duckdb_memory_connector.switch_context(database_name="quickstart")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGINT PRIMARY KEY,
        name VARCHAR(64)
    );
    """
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"

    create_result = duckdb_memory_connector.execute_ddl(create_sql)
    if not create_result.success:
        pytest.skip(f"Unable to create test table for INSERT: {create_result.error}")

    try:
        # Insert two rows
        insert_res = duckdb_memory_connector.execute_insert(
            f"INSERT INTO {table_name} (id, name) VALUES (1, 'Alice'), (2, 'Bob')"
        )
        assert insert_res.success

        # Verify rows
        final_query = duckdb_memory_connector.execute(
            {
                "sql_query": f"SELECT id, name FROM {table_name} ORDER BY id",
            },
            result_format="list",
        )
        assert final_query.success is True
        rows = final_query.sql_return
        assert [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}] == rows
    finally:
        duckdb_memory_connector.execute_ddl(drop_sql)


def test_update_round_trip(duckdb_memory_connector: DuckdbConnector):
    suffix = uuid.uuid4().hex[:8]
    table_name = f"datus_update_test_{suffix}"

    duckdb_memory_connector.switch_context(database_name="quickstart")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGINT PRIMARY KEY,
        name VARCHAR(64)
    );
    """
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"

    create_result = duckdb_memory_connector.execute_ddl(create_sql)
    if not create_result.success:
        pytest.skip(f"Unable to create test table for UPDATE: {create_result.error}")

    try:
        duckdb_memory_connector.execute_insert(f"INSERT INTO {table_name} (id, name) VALUES (1, 'Alice'), (2, 'Bob')")

        # Perform UPDATE
        update_res = duckdb_memory_connector.execute(
            {"sql_query": f"UPDATE {table_name} SET name = 'Alice Updated' WHERE id = 1"},
            result_format="list",
        )
        assert update_res.success is True

        # Verify
        final_query = duckdb_memory_connector.execute(
            {"sql_query": f"SELECT id, name FROM {table_name} ORDER BY id"},
            result_format="list",
        )
        rows = final_query.sql_return
        assert rows[0]["id"] == 1 and rows[0]["name"] == "Alice Updated"
        assert rows[1]["id"] == 2 and rows[1]["name"] == "Bob"
    finally:
        duckdb_memory_connector.execute_ddl(drop_sql)


def test_delete_round_trip(duckdb_memory_connector: DuckdbConnector):
    suffix = uuid.uuid4().hex[:8]
    table_name = f"datus_delete_test_{suffix}"

    duckdb_memory_connector.switch_context(database_name="quickstart")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGINT PRIMARY KEY,
        name VARCHAR(64)
    );
    """
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"

    create_result = duckdb_memory_connector.execute_ddl(create_sql)
    if not create_result.success:
        pytest.skip(f"Unable to create test table for DELETE: {create_result.error}")

    try:
        duckdb_memory_connector.execute_insert(f"INSERT INTO {table_name} (id, name) VALUES (1, 'Alice'), (2, 'Bob')")

        # Perform DELETE
        delete_res = duckdb_memory_connector.execute(
            {"sql_query": f"DELETE FROM {table_name} WHERE id = 2"},
            result_format="list",
        )
        assert delete_res.success is True

        # Verify only id=1 remains
        final_query = duckdb_memory_connector.execute(
            {"sql_query": f"SELECT id, name FROM {table_name} ORDER BY id"},
            result_format="list",
        )
        rows = final_query.sql_return
        assert rows == [{"id": 1, "name": "Alice"}]
    finally:
        duckdb_memory_connector.execute_ddl(drop_sql)
