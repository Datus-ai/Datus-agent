# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict, Set

from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.exceptions import DatusException, ErrorCode
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

    try:
        schemas = storage.search_all_schemas(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )
        if schemas.num_rows > 0:
            identifier_idx = schemas.schema.get_field_index("identifier")
            definition_idx = schemas.schema.get_field_index("definition")

            batch_size = 500
            for i in range(0, schemas.num_rows, batch_size):
                batch = schemas.slice(i, min(batch_size, schemas.num_rows - i))
                identifiers = batch.column(identifier_idx).to_pylist()
                definitions = batch.column(definition_idx).to_pylist()

                all_schema_tables.update(zip(identifiers, definitions))

        values = storage.search_all_value(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )
        if values.num_rows > 0:
            identifier_idx = values.schema.get_field_index("identifier")

            batch_size = 500
            for i in range(0, values.num_rows, batch_size):
                batch = values.slice(i, min(batch_size, values.num_rows - i))
                identifiers = batch.column(identifier_idx).to_pylist()

                all_value_tables.update(identifiers)

    except Exception as e:
        raise DatusException(
            ErrorCode.COMMON_UNKNOWN, message=f"Failed to load already existing metadata, reason: {str(e)}"
        ) from e

    return all_schema_tables, all_value_tables
