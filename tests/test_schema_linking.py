import os

import lancedb
import pytest

from datus.schemas.schema_linking_node_models import SchemaLinkingInput
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
    test_path = "test_lancedb"
    yield test_path
    # Cleanup after tests
    if os.path.exists(test_path):
        import shutil

        shutil.rmtree(test_path)


@pytest.fixture
def schema_lineage_tool(lancedb_path):
    """Fixture to create a SchemaLineageTool instance"""
    tool = SchemaLineageTool(lancedb_path)
    return tool


@pytest.fixture
def sqlite_connector(db_path):
    """Fixture to create a SQLiteConnector instance"""
    connector = SQLiteConnector(db_path)
    yield connector
    connector.close()


def test_storage():
    pass


def test_store_and_search_schema(schema_lineage_tool, sqlite_connector):
    """Test storing and searching schema"""
    log.info("Creating test tables in SQLite")
    # Create test tables in SQLite
    sqlite_connector.execute(
        {
            "sql_query": """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP
        )
        """
        }
    )

    sqlite_connector.execute(
        {
            "sql_query": """
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT,
            user_id INTEGER,
            created_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        }
    )

    # Get schema from SQLite
    log.info("Retrieving schema from SQLite")
    schema = sqlite_connector.get_schema()
    log.debug("Retrieved schema", schema=schema)

    # Store schema in lineage tool
    log.info("Storing schema in lineage tool")
    result = schema_lineage_tool.execute({"schema_text": schema, "database_name": "test_db"})
    log.debug("Store schema result", result=result)
    assert result["success"] is True

    # Search for similar schemas
    log.info("Searching for similar schemas")
    search_result = schema_lineage_tool.execute({"schema_text": "username", "top_n": 1})
    log.debug("Search result", result=search_result)

    assert search_result["success"] is True
    assert search_result["count"] > 0
    assert len(search_result["similar_schemas"]) == 1


def test_invalid_input(schema_lineage_tool):
    """Test input validation"""
    log.info("Testing missing schema_text parameter")
    # Test missing schema_text
    with pytest.raises(ValueError, match="'schema_text' parameter is required"):
        schema_lineage_tool.execute({})

    log.info("Testing invalid schema_text type")
    # Test invalid schema_text type
    with pytest.raises(ValueError, match="'schema_text' must be a string"):
        schema_lineage_tool.execute({"schema_text": 123})

    log.info("Testing invalid top_n parameter")
    # Test invalid top_n
    with pytest.raises(ValueError, match="'top_n' must be a positive integer"):
        schema_lineage_tool.execute({"schema_text": "CREATE TABLE test (id INTEGER)", "top_n": -1})


def test_query_nyc_trees_from_lancedb():
    """Test querying NYC trees data directly from LanceDB"""
    db = lancedb.connect("data/datus_db_spider2")

    # Get table references
    schema_table = db.open_table("schema_lineage")
    values_table = db.open_table("schema_value")

    # Check all available records
    all_schemas = schema_table.to_pandas()
    log.info(
        "Available schemas:",
        table_names=all_schemas["table_name"].unique().tolist(),
        database_names=all_schemas["database_name"].unique().tolist(),
    )

    tables = [
        "NEW_YORK.NEW_YORK.TREE_CENSUS_1995",
        "NEW_YORK.NEW_YORK.TREE_CENSUS_2015",
        "NEW_YORK.NEW_YORK.TREE_SPECIES",
    ]

    table_schemas = []
    table_values = []
    from datus.schemas.node_models import TableSchema, TableValue
    from datus.schemas.schema_linking_node_models import SchemaLinkingResult

    for table in tables:
        schema_results = schema_table.search().where(f"table_name = '{table.split('.')[-1]}'").limit(1).to_list()

        value_results = values_table.search().where(f"table_name = '{table.split('.')[-1]}'").limit(1).to_list()

        for schema, value in zip(schema_results, value_results):
            table_schemas.append(
                TableSchema(
                    table_name=table.split(".")[-1],
                    database_name=table.split(".")[0],
                    schema_name=table.split(".")[0],
                    schema_text=schema["schema_text"],
                )
            )
            table_values.append(
                TableValue(
                    table_name=table.split(".")[-1],
                    database_name=table.split(".")[0],
                    schema_name=table.split(".")[0],
                    table_values=value["sample_rows"],
                )
            )

    result = SchemaLinkingResult(
        success=True,
        error=None,
        table_schemas=table_schemas,
        schema_count=len(table_schemas),
        table_values=table_values,
        value_count=len(table_values),
    )
    log.info("Found all schemas", result=result.model_dump())
    log.info("Found all schemas", result.schema_count, result.value_count)


def test_schema_linking_no_exist():
    tool = SchemaLineageTool(db_path="data/datus_db_no_exist")
    res = tool.execute(
        SchemaLinkingInput(
            input_text="",
            database_type=DBType.SQLITE,
            catalog_name="",
            database_name="",
            schema_name="",
            matching_rate="fast",
            sql_context=None,
        )
    )
    assert res["success"] is False
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
