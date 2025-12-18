# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# init schema for Bird-Bench
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

from datus.storage.schema_metadata.init_utils import exists_table_value
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.csv_utils import read_csv_and_clean_text
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
# Get the current directory path
current_path = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach project root
dir_path = os.path.dirname(os.path.dirname(current_path))


# def init_critic_schema(rag_base_path: str):
#     """
#     Initialize the schema for Bird-Bench.
#     Args:
#         base_db_path: the path to the base database
#         input_jsonl_path: the path to the input jsonl file, modify this path if necessary
#     FIXME: there has some error in postgresql parsing
#     """

#     schema_dict = load_jsonl_dict()
#     bird_tasks = load_dataset("birdsql/bird-critic-1.0-open")["open"]
#     # dialect -> db_id -> table_name -> table_schema, table_data
#     db_table_dict = {}
#     for task in bird_tasks:
#         dialect = task["dialect"]
#         if dialect not in db_table_dict:
#             db_table_dict[dialect] = {}
#         db_id = task["db_id"]
#         if db_id not in db_table_dict[dialect]:
#             db_table_dict[dialect][db_id] = {}
#         table_dict = db_table_dict[dialect][db_id]
#         instance_id = task["instance_id"]
#         if instance_id not in schema_dict:
#             logger.warning(f"Instance ID {instance_id} not found in schema_dict")
#             continue
#         schema_data = schema_dict[instance_id]
#         # Get schema string from either preprocess_schema or original_schema
#         schema_str = schema_data.get("preprocess_schema") or schema_data.get("original_schema")

#         if not schema_str:
#             logger.warning(f"Schema string for instance ID {instance_id} is empty")
#             continue
#         # Parse the schema string into table schemas
#         for schema_data in schema_str.split("\n...\n"):
#             schema_data = schema_data.strip()
#             if not schema_data:
#                 continue
#             split_index = schema_data.find(";\n")
#             ddl = schema_data[:split_index].strip()
#             table_data = schema_data[split_index + 2 :].strip()
#             if table_data.startswith("First 3 rows:"):
#                 table_data = table_data[len("First 3 rows:") :]
#             table_metadata = parse_metadata(ddl, dialect=dialect)

#             table_name = table_metadata["table"]["name"]
#             if table_name in table_dict:
#                 continue
#             table_dict[table_name] = {"definition": ddl, "sample_rows": table_data}

#         # Add each table schema to the db_table_dict
#         for dialect, db_table in db_table_dict.items():
#             table_schemas = []
#             table_values = []
#             for db_id, table_dict in db_table.items():
#                 for tb_name, table_meta in table_dict.items():
#                     table_schemas.append(
#                         {
#                             "database_name": db_id,
#                             "schema_name": "",
#                             "table_name": tb_name,
#                             "definition": table_meta["definition"],
#                         }
#                     )
#                     table_values.append(
#                         {
#                             "database_name": db_id,
#                             "schema_name": "",
#                             "table_name": tb_name,
#                             "sample_rows": table_meta["sample_rows"],
#                         }
#                     )
#             logger.info(f"{dialect} {db_id} {len(table_schemas)} {len(table_values)}")
#             break


def init_dev_schema(
    rag: SchemaWithValueRAG,
    db_manager: DBManager,
    namespace: str,
    bird_path: str = "benchmark/bird/dev_20240627",
    build_mode: str = "overwrite",
    pool_size: int = 4,
    database_names: list = None,
):
    """
    Initialize the schema for Bird-Bench.
    Args:
        rag: SchemaWithValueRAG instance
        db_manager: DBManager instance
        namespace: Database namespace
        bird_path: Path to the bird benchmark directory
        build_mode: Build mode for the schema
    """
    if database_names is None:
        database_names = []

    db_table_keys = load_table_keys(f"{bird_path}/dev_tables.json")
    databases_path = f"{bird_path}/dev_databases"
    all_schema_tables, all_value_tables = exists_table_value(rag, build_mode=build_mode)

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        all_databases = os.listdir(databases_path)
        # Filter by database_names if provided
        if database_names:
            all_databases = [db for db in all_databases if db in database_names]

        futures = [
            executor.submit(
                init_dev_schema_by_db,
                rag,
                db_manager,
                namespace,
                database_name,
                db_table_keys,
                databases_path,
                set(all_schema_tables.keys()),
                all_value_tables,
            )
            for database_name in all_databases
        ]
        for future in as_completed(futures):
            future.result()
    rag.after_init()


def init_dev_schema_by_db(
    rag: SchemaWithValueRAG,
    db_manager: DBManager,
    namespace: str,
    database_name: str,
    table_keys: Dict[str, Any],
    databases_path: str,
    all_schema_tables: Set[str],
    all_value_tables: Set[str],
):
    if database_name == ".DS_Store":
        return
    logger.info(f"start init {database_name}")
    schema_result, value_result = init_db(
        db_manager,
        namespace,
        database_name,
        table_keys[database_name],
        databases_path,
        all_schema_tables,
        all_value_tables,
    )
    rag.store_batch(schema_result, value_result)
    logger.info(f"finish init {database_name}, tables size: {len(schema_result)}, " f"value size: {len(value_result)}")


def init_db(
    db_manager: DBManager,
    namespace: str,
    database_name: str,
    table_keys: Dict[str, Any],
    databases_path: str,
    all_schema_tables: Set[str],
    all_value_tables: Set[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    db_path = os.path.join(databases_path, database_name)
    if not os.path.isdir(db_path):
        return [], []
    desc_path = os.path.join(db_path, "database_description")
    schema_result = []
    value_result = []
    sql_conn = db_manager.get_conn(namespace, database_name)
    for table_csv in os.listdir(desc_path):
        if not table_csv.endswith(".csv"):
            continue
        logger.info(f"start init {table_csv}")
        table_metadata = generate_sql_by_desc_file(database_name, table_keys, os.path.join(desc_path, table_csv))
        table_name = table_metadata["table_name"]
        identifier = sql_conn.identifier(
            database_name=database_name,
            table_name=table_name,
        )
        table_metadata["identifier"] = identifier
        table_metadata["database_name"] = database_name
        table_metadata["catalog_name"] = ""
        table_metadata["schema_name"] = ""
        table_metadata["table_type"] = "table"
        full_table_name = f"{database_name}.{table_name}"
        if full_table_name not in all_schema_tables:
            schema_result.append(table_metadata)
        if full_table_name in all_value_tables:
            continue
        all_value_tables.add(full_table_name)
        sample_rows = sql_conn.get_sample_rows(tables=[table_name], top_n=5)
        if not sample_rows:
            logger.warning(f"No sample rows for {table_name}")
            continue
        sample_value = sample_rows[0]
        sample_value["identifier"] = identifier
        sample_value["database_name"] = database_name
        sample_value["schema_name"] = ""
        sample_value["table_type"] = "table"
        value_result.append(sample_value)
        logger.info(f"finish init {table_csv}")

    return schema_result, value_result


def load_table_keys(
    input_json_path: str = "benchmark/bird/dev_20240627/dev_tables.json",
) -> Dict[str, Dict[str, Any]]:
    """
    Load the primary keys for each table in the database from bird-bench dev_tables.json.
    Args:
        input_json_path (str, optional): the path to the dev_tables.json file.
        Defaults to "benchmark/bird/dev_20240627/dev_tables.json".

    Returns:
        Dict[str, Dict[str, Dict[str, List[str]]]]: db_id -> table_name ->
                {'primary_keys': [column_name], 'foreign_keys': [column_name]}
    """
    with open(input_json_path, "r", encoding="utf-8-sig") as f:
        dev_tables = json.load(f)
    table_primary_keys = {}
    for table in dev_tables:
        db_id = table["db_id"]
        table_primary_keys[db_id] = {}
        for tb_name in table["table_names_original"]:
            table_primary_keys[db_id][tb_name] = {}

        table_names = table["table_names_original"]
        columns = table["column_names_original"]
        for pk_index in table["primary_keys"]:
            if isinstance(pk_index, int):
                column = columns[pk_index]
                table_name = table_names[column[0]]
                table_primary_keys[db_id][table_name] = {"primary_keys": [f"`{column[1]}`"]}
            elif isinstance(pk_index, list):
                table_name = table_names[columns[pk_index[0]][0]]
                columns_names = [f"`{columns[i][1]}`" for i in pk_index]
                table_primary_keys[db_id][table_name] = {"primary_keys": columns_names}

        for fk_index in table["foreign_keys"]:
            src_col_i, target_col_i = fk_index[0], fk_index[1]
            src_col = columns[src_col_i]
            target_col = columns[target_col_i]
            src_table_name = table_names[src_col[0]]
            target_table_name = table_names[target_col[0]]
            if "foreign_keys" not in table_primary_keys[db_id][src_table_name]:
                table_primary_keys[db_id][src_table_name]["foreign_keys"] = []
            table_primary_keys[db_id][src_table_name]["foreign_keys"].append(
                {
                    "column": f"`{src_col[1]}`",
                    "target_table": f"`{target_table_name}`",
                    "target_column": f"`{target_col[1]}`",
                }
            )

    return table_primary_keys


def generate_sql_by_desc_file(database_name, table_infos, csv_path) -> Dict[str, str]:
    """
    Generate SQLite CREATE TABLE statement with field descriptions as comments
    from a CSV file containing column descriptions.

    Args:
        database_name (str): Name of the database
        table_infos (Dict): Table information including primary keys and foreign keys
        csv_path (str): Path to the CSV file containing column descriptions
    """

    # Get table name from CSV filename
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    table_info = table_infos[table_name]

    # Generate SQL with comments
    sql_lines = []
    sql_lines.append(f"CREATE TABLE `{table_name}` (")

    has_primary = table_info and "primary_keys" in table_info and table_info["primary_keys"]
    has_foreign = table_info and "foreign_keys" in table_info and table_info["foreign_keys"]

    # Process each column
    # Read the CSV file with error handling for encoding
    columns = read_csv_and_clean_text(csv_path)
    for i, col in enumerate(columns):
        # Determine SQL type based on data_format
        sql_type = "TEXT"  # default type
        if "data_format" not in col:
            logger.warning(f"data_format not in {col}")
        if col["data_format"] == "integer":
            sql_type = "INTEGER"
        elif col["data_format"] == "real":
            sql_type = "REAL"

        # Add column definition
        line = f"    `{col['original_column_name']}` {sql_type}"

        # Add comma if not the last column or if there are constraints
        if i < len(columns) - 1 or has_primary or has_foreign:
            line += ","

        # Combine column description and value description
        comment_parts = []
        if col["column_description"]:
            comment_parts.append(col["column_description"])
        if col["value_description"]:
            comment_parts.append(f"Values: {col['value_description']}")

        # Add comment with combined description
        if comment_parts:
            line += f"  /* {' | '.join(comment_parts)} */"

        sql_lines.append(line)

    # Add primary key constraint if exists
    if has_primary:
        primary_key_cols = table_info["primary_keys"]
        sql_lines.append(f"    PRIMARY KEY ({', '.join(primary_key_cols)})")
        if has_foreign:
            sql_lines[-1] += ","

    # Add foreign key constraints if exist
    if has_foreign:
        for i, fk in enumerate(table_info["foreign_keys"]):
            line = f"    FOREIGN KEY ({fk['column']}) REFERENCES " f"{fk['target_table']}({fk['target_column']})"
            if i < len(table_info["foreign_keys"]) - 1:
                line += ","
            sql_lines.append(line)

    sql_lines.append(");")
    return {
        "catalog_name": "",
        "database_name": database_name,
        "table_name": table_name,
        "schema_name": "",
        "definition": "\n".join(sql_lines),
        "table_type": "table",
    }
