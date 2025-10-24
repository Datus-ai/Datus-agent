# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Set

import pandas as pd

from datus.storage.schema_metadata.init_utils import exists_table_value
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.utils.json_utils import json2csv
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import metadata_identifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = get_logger(__name__)
# init spider2 schemas


def init_snowflake_schema(
    storage: SchemaWithValueRAG,
    benchmark_path: str = "benchmark/spider2/spider2-snow",
    build_mode: str = "overwrite",
    pool_size: int = 4,
    instance_ids: list = None,
):
    """Initialize the schema store for Snowflake."""
    if instance_ids is None:
        instance_ids = []

    all_schema_tables, all_value_tables = exists_table_value(storage, build_mode=build_mode)

    # Join with benchmark path
    db_ids = []

    with (
        open(os.path.join(benchmark_path, "spider2-snow.jsonl"), "r") as f,
        ThreadPoolExecutor(max_workers=pool_size) as executor,
    ):
        for line in f:
            json_data = json.loads(line)

            # Filter by instance_ids if provided
            if instance_ids and json_data.get("instance_id") not in instance_ids:
                continue

            db_id = json_data["db_id"]
            if db_id in db_ids:
                continue
            db_ids.append(db_id)
            executor.submit(
                process_line,
                storage,
                json_data,
                benchmark_path,
                set(all_schema_tables.keys()),
                all_value_tables,
            )

    storage.after_init()


def process_line(
    storage: SchemaWithValueRAG,
    item: dict,
    benchmark_path: str,
    all_schema_tables: Set[str],
    all_value_tables: Set[str],
):
    try:
        do_process_by_database(storage, item["db_id"], benchmark_path, all_schema_tables, all_value_tables)
    except Exception:
        logger.error(f"Error processing line: {traceback.format_exc()}")


def do_process_by_database(
    storage: SchemaWithValueRAG,
    db_id: str,
    benchmark_path: str,
    all_schema_tables: Set[str],
    all_value_tables: Set[str],
):
    databases_dir = os.path.join(benchmark_path, "resource/databases")

    datasource_dir = f"{databases_dir}/{db_id}"

    if not os.path.exists(datasource_dir):
        return
    process_data_size = 0
    for schema_name in os.listdir(datasource_dir):
        dir_path = os.path.join(datasource_dir, schema_name)
        logger.info(f"Process {db_id}.{schema_name} begin")
        if not os.path.isdir(dir_path):
            continue
        batch_records = []
        batch_value_records = []

        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if not file.endswith(".csv"):
                continue
            df = pd.read_csv(file_path)
            df.dropna(axis=0, how="all", inplace=True)
            for _, row in df.iterrows():
                if row.empty or "DDL" not in row or pd.isna(row["DDL"]) or not row["DDL"]:
                    continue
                data_file = f'{row["table_name"].split(".")[-1]}.json'
                json_data = {}
                if os.path.exists(f"{dir_path}/{data_file}"):
                    with open(f"{dir_path}/{data_file}", "r") as f:
                        try:
                            json_data = json.load(f)
                        except Exception:
                            logger.error(f"Error loading json data: {traceback.format_exc()}")
                # fix table name form
                table_name = row["table_name"].split(".")[-1]
                description = "" if pd.isna(row["description"]) else row["description"]
                full_tb_name = f"{db_id}.{schema_name}.{table_name}"
                # Remove unnecessary whitespace and newlines from DDL
                if full_tb_name not in all_schema_tables:
                    all_schema_tables.add(full_tb_name)
                    ddl = " ".join(row["DDL"].strip().split())
                    if description:
                        if ddl.endswith(";"):
                            ddl = f"""{ddl[:-1]} COMMENT = '{description}';"""
                        else:
                            ddl = f"""{ddl} COMMENT = '{description}';"""
                    batch_records.append(
                        {
                            "identifier": metadata_identifier(
                                dialect="snowflake",
                                catalog_name="",
                                database_name=db_id,
                                schema_name=schema_name,
                                table_name=table_name,
                            ),
                            "catalog_name": "",
                            "database_name": db_id,
                            "schema_name": schema_name,
                            "table_name": table_name,
                            "definition": ddl,
                            "table_type": "table",
                        }
                    )
                if full_tb_name not in all_value_tables and (sample_rows := json_data.get("sample_rows")):
                    batch_value_records.append(
                        {
                            "identifier": metadata_identifier(
                                dialect="snowflake",
                                catalog_name="",
                                database_name=db_id,
                                schema_name=schema_name,
                                table_name=table_name,
                            ),
                            "catalog_name": "",
                            "database_name": db_id,
                            "schema_name": schema_name,
                            "table_name": table_name,
                            "sample_rows": json2csv(sample_rows),
                            "table_type": "table",
                        }
                    )
                    all_value_tables.add(full_tb_name)
                if len(batch_records) == 500 or len(batch_value_records) == 500:
                    storage.store_batch(batch_records, batch_value_records)
                    batch_records = []
                    batch_value_records = []
        if len(batch_records) > 0 or len(batch_value_records) > 0:
            storage.store_batch(batch_records, batch_value_records)
            process_data_size += len(batch_records)
            batch_records.clear()
            batch_value_records.clear()
        logger.info(f"Processe {db_id}.{schema_name} end")
