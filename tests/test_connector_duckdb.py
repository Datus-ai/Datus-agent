import pytest

from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture
def duckdb_connector() -> DuckdbConnector:
    return DuckdbConnector(db_path="tests/duckdb-demo.duckdb")


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
    assert ErrorCode.DB_EXECUTION_ERROR.code in res.error


@pytest.mark.acceptance
def test_get_schemas(duckdb_connector: DuckdbConnector):
    assert len(duckdb_connector.get_schemas()) > 0
