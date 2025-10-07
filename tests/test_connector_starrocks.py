import uuid

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
    connector = db_manager.get_conn(agent_config.current_namespace)
    assert isinstance(connector, StarRocksConnector)
    return connector


def test_connector(connector: StarRocksConnector):
    assert connector.test_connection()
    assert len(connector.get_tables()) > 0


def test_get_tables_with_ddl(connector: StarRocksConnector):
    tables = connector.get_tables_with_ddl(catalog_name="default_catalog")
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
    views = connector.get_views_with_ddl(catalog_name="default_catalog")
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
    views = connector.get_materialized_views_with_ddl(catalog_name="default_catalog")
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
    with pytest.raises(DatusException, match=ErrorCode.DB_EXECUTION_ERROR.code):
        connector.get_sample_rows(catalog_name="default_catalog", tables=["nonexistent_table"])


def test_get_databases(connector: StarRocksConnector):
    databases = connector.get_databases()
    assert len(databases) > 0


def test_get_catalogs(connector: StarRocksConnector):
    catalogs = connector.get_catalogs()
    assert len(catalogs) > 0
    assert "default_catalog" in catalogs


def test_execute_merge_returns_database_error(connector: StarRocksConnector):
    merge_sql = (
        "MERGE INTO nonexistent_target AS t USING nonexistent_source AS s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET t.value = s.value "
        "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.value)"
    )

    result = connector.execute({"sql_query": merge_sql})

    assert result.sql_query == merge_sql
    assert result.row_count is not None
    assert "Unknown type of SQL" not in (result.error or "")


def test_execute_explain_returns_plan(connector: StarRocksConnector):
    tables = connector.get_tables_with_ddl(catalog_name=connector.catalog_name)
    if not tables:
        pytest.skip("No tables available to run EXPLAIN")

    table = tables[0]
    full_table_name = connector.full_name(
        catalog_name=table["catalog_name"],
        database_name=table["database_name"],
        table_name=table["table_name"],
    )

    explain_sql = f"EXPLAIN SELECT * FROM {full_table_name} LIMIT 1"

    result = connector.execute({"sql_query": explain_sql})
    assert result.sql_query == explain_sql
    assert not result.error
    assert result.row_count is not None
    assert str(result.sql_return).strip()


def test_execute_insert_round_trip(connector: StarRocksConnector):
    suffix = uuid.uuid4().hex[:8]
    table_name = f"datus_insert_test_{suffix}"

    connector.switch_context(database_name="quickstart")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS  {table_name} (
        `id` BIGINT NOT NULL,
        `name` VARCHAR(64)
    ) ENGINE=OLAP
    PRIMARY KEY (`id`)
    DISTRIBUTED BY HASH(`id`) BUCKETS 1
    PROPERTIES (
        "replication_num" = "1"
    );
    """

    drop_sql = f"DROP TABLE IF EXISTS {table_name}"

    create_result = connector.execute_ddl(create_sql)
    if not create_result.success:
        pytest.skip(f"Unable to create test table for INSERT: {create_result.error}")

    try:
        # Insert two rows
        insert_result = connector.execute_insert(f"INSERT INTO {table_name} (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        assert insert_result.success

        # Verify rows
        final_query = connector.execute(
            {"sql_query": f"SELECT id, name FROM {table_name} ORDER BY id"},
            result_format="list",
        )
        assert final_query.success is True
        assert not final_query.error
        assert final_query.sql_return == [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    finally:
        connector.execute_ddl(drop_sql)


def test_execute_update_round_trip(connector: StarRocksConnector):
    suffix = uuid.uuid4().hex[:8]
    table_name = f"datus_update_test_{suffix}"

    connector.switch_context(database_name="quickstart")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS  {table_name} (
        `id` BIGINT NOT NULL,
        `name` VARCHAR(64)
    ) ENGINE=OLAP
    PRIMARY KEY (`id`)
    DISTRIBUTED BY HASH(`id`) BUCKETS 1
    PROPERTIES (
        "replication_num" = "1"
    );
    """

    drop_sql = f"DROP TABLE IF EXISTS {table_name}"

    create_result = connector.execute_ddl(create_sql)
    if not create_result.success:
        pytest.skip(f"Unable to create test table for UPDATE: {create_result.error}")

    try:
        # Seed data
        seed = connector.execute_insert(f"INSERT INTO {table_name} (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        assert seed.success

        # Perform UPDATE via generic execute
        update_result = connector.execute(
            {"sql_query": f"UPDATE {table_name} SET name = 'Alice Updated' WHERE id = 1"},
            result_format="list",
        )
        assert update_result.success is True
        assert not update_result.error

        # Verify
        final_query = connector.execute(
            {"sql_query": f"SELECT id, name FROM {table_name} ORDER BY id"},
            result_format="list",
        )
        assert final_query.success is True
        rows = final_query.sql_return
        assert rows == [
            {"id": 1, "name": "Alice Updated"},
            {"id": 2, "name": "Bob"},
        ]
    finally:
        connector.execute_ddl(drop_sql)


def test_execute_delete_round_trip(connector: StarRocksConnector):
    suffix = uuid.uuid4().hex[:8]
    table_name = f"datus_delete_test_{suffix}"

    connector.switch_context(database_name="quickstart")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS  {table_name} (
        `id` BIGINT NOT NULL,
        `name` VARCHAR(64)
    ) ENGINE=OLAP
    PRIMARY KEY (`id`)
    DISTRIBUTED BY HASH(`id`) BUCKETS 1
    PROPERTIES (
        "replication_num" = "1"
    );
    """

    drop_sql = f"DROP TABLE IF EXISTS {table_name}"

    create_result = connector.execute_ddl(create_sql)
    if not create_result.success:
        pytest.skip(f"Unable to create test table for DELETE: {create_result.error}")

    try:
        # Seed data
        seed = connector.execute_insert(f"INSERT INTO {table_name} (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        assert seed.success

        # Perform DELETE via generic execute
        delete_result = connector.execute(
            {"sql_query": f"DELETE FROM {table_name} WHERE id = 2"},
            result_format="list",
        )
        assert delete_result.success is True
        assert not delete_result.error

        # Verify only id=1 remains
        final_query = connector.execute(
            {"sql_query": f"SELECT id, name FROM {table_name} ORDER BY id"},
            result_format="list",
        )
        assert final_query.success is True
        assert final_query.sql_return == [
            {"id": 1, "name": "Alice"},
        ]
    finally:
        connector.execute_ddl(drop_sql)


def test_get_sample_rows(connector: StarRocksConnector):
    sample_rows = connector.get_sample_rows(catalog_name="default_catalog", database_name="quickstart")
    assert len(sample_rows) > 0

    sample_rows = connector.get_sample_rows(database_name="quickstart")
    assert len(sample_rows) > 0

    sample_rows = connector.get_sample_rows()
    assert len(sample_rows) > 0

    sample_rows = connector.get_sample_rows(
        catalog_name="default_catalog",
        database_name="quickstart",
        table_type="table",
        tables=["crashdata"],
    )
    assert len(sample_rows) > 0
    print(sample_rows)
    assert len(sample_rows) == 1

    sample_rows = connector.get_sample_rows(
        catalog_name="default_catalog",
        database_name="quickstart",
        table_type="table",
        tables=["crashdata", "weatherdata"],
    )
    assert len(sample_rows) > 0
    print(sample_rows)
    assert len(sample_rows) == 2

    first_item = sample_rows[0]
    assert first_item["database_name"]
    assert first_item["table_name"]
    assert first_item["catalog_name"]
    assert not first_item["schema_name"]
    assert first_item["identifier"]
    assert len(first_item["identifier"].split(".")) == 3
