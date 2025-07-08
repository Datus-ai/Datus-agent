from typing import Dict, Set

from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def exists_table_value(
    storage: SchemaWithValueRAG,
    database_name: str = "",
    catalog_name: str = "",
    schema_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
) -> tuple[Dict[str, str], Set[str]]:
    """
    Get the existing tables and values from the storage.
    Return:
        all_schema_tables: Dict[str,  str]] identifier -> definition
        all_value_tables: Set[str]
    """
    all_schema_tables: Dict[str, str] = {}
    all_value_tables: Set[str] = set()
    if build_mode == "overwrite":
        return all_schema_tables, all_value_tables

    schemas = storage.search_all_schemas(
        catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
    )
    for schema in schemas:
        all_schema_tables[schema["identifier"]] = schema["definition"]

    values = storage.search_all_value(catalog_name=catalog_name, database_name=database_name, table_type=table_type)
    for value in values:
        all_value_tables.add(value["identifier"])
    return all_schema_tables, all_value_tables
