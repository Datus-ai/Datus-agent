# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/storage/schema_metadata/benchmark_init_bird.py

CI-level: zero external deps, all file I/O mocked or tmp_path-based.
"""

import csv
import json
from unittest.mock import MagicMock

import pytest

from datus.storage.schema_metadata.benchmark_init_bird import (
    generate_sql_by_desc_file,
    init_db,
    init_dev_schema_by_db,
    load_table_keys,
)

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# load_table_keys
# ---------------------------------------------------------------------------


class TestLoadTableKeys:
    def _write_dev_tables(self, path, data):
        with open(path, "w", encoding="utf-8-sig") as f:
            json.dump(data, f)

    def test_load_basic_table_keys(self, tmp_path):
        data = [
            {
                "db_id": "mydb",
                "table_names_original": ["users", "orders"],
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [1, "order_id"]],
                "primary_keys": [1],
                "foreign_keys": [],
            }
        ]
        json_path = tmp_path / "dev_tables.json"
        self._write_dev_tables(str(json_path), data)

        result = load_table_keys(str(json_path))
        assert "mydb" in result
        assert "users" in result["mydb"]
        assert "orders" in result["mydb"]

    def test_primary_key_int_index(self, tmp_path):
        data = [
            {
                "db_id": "db1",
                "table_names_original": ["employees"],
                "column_names_original": [[-1, "*"], [0, "emp_id"], [0, "name"]],
                "primary_keys": [1],
                "foreign_keys": [],
            }
        ]
        json_path = tmp_path / "dev_tables.json"
        self._write_dev_tables(str(json_path), data)

        result = load_table_keys(str(json_path))
        assert "primary_keys" in result["db1"]["employees"]
        assert "`emp_id`" in result["db1"]["employees"]["primary_keys"]

    def test_primary_key_list_index(self, tmp_path):
        data = [
            {
                "db_id": "db1",
                "table_names_original": ["orders"],
                "column_names_original": [[-1, "*"], [0, "order_id"], [0, "item_id"]],
                "primary_keys": [[1, 2]],
                "foreign_keys": [],
            }
        ]
        json_path = tmp_path / "dev_tables.json"
        self._write_dev_tables(str(json_path), data)

        result = load_table_keys(str(json_path))
        assert "`order_id`" in result["db1"]["orders"]["primary_keys"]
        assert "`item_id`" in result["db1"]["orders"]["primary_keys"]

    def test_foreign_key_parsed(self, tmp_path):
        data = [
            {
                "db_id": "db2",
                "table_names_original": ["orders", "customers"],
                "column_names_original": [[-1, "*"], [0, "customer_id"], [1, "id"]],
                "primary_keys": [],
                "foreign_keys": [[1, 2]],
            }
        ]
        json_path = tmp_path / "dev_tables.json"
        self._write_dev_tables(str(json_path), data)

        result = load_table_keys(str(json_path))
        assert "foreign_keys" in result["db2"]["orders"]
        fk = result["db2"]["orders"]["foreign_keys"][0]
        assert fk["column"] == "`customer_id`"
        assert fk["target_table"] == "`customers`"
        assert fk["target_column"] == "`id`"

    def test_multiple_databases(self, tmp_path):
        data = [
            {
                "db_id": "db_a",
                "table_names_original": ["t1"],
                "column_names_original": [[-1, "*"], [0, "id"]],
                "primary_keys": [],
                "foreign_keys": [],
            },
            {
                "db_id": "db_b",
                "table_names_original": ["t2"],
                "column_names_original": [[-1, "*"], [0, "bid"]],
                "primary_keys": [],
                "foreign_keys": [],
            },
        ]
        json_path = tmp_path / "dev_tables.json"
        self._write_dev_tables(str(json_path), data)

        result = load_table_keys(str(json_path))
        assert "db_a" in result
        assert "db_b" in result


# ---------------------------------------------------------------------------
# generate_sql_by_desc_file
# ---------------------------------------------------------------------------


def _write_desc_csv(path, rows):
    """Helper to write a CSV file with bird-bench description format."""
    fieldnames = ["original_column_name", "data_format", "column_description", "value_description"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestGenerateSqlByDescFile:
    def test_basic_table_creation(self, tmp_path):
        csv_path = tmp_path / "users.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "id",
                    "data_format": "integer",
                    "column_description": "Primary key",
                    "value_description": "",
                },
                {
                    "original_column_name": "name",
                    "data_format": "text",
                    "column_description": "User name",
                    "value_description": "",
                },
            ],
        )
        table_infos = {"users": {}}
        result = generate_sql_by_desc_file("mydb", table_infos, str(csv_path))
        assert result["table_name"] == "users"
        assert result["database_name"] == "mydb"
        assert "CREATE TABLE" in result["definition"]
        assert "`id` INTEGER" in result["definition"]
        assert "`name` TEXT" in result["definition"]

    def test_integer_type_mapped(self, tmp_path):
        csv_path = tmp_path / "scores.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "score",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )
        result = generate_sql_by_desc_file("db", {"scores": {}}, str(csv_path))
        assert "INTEGER" in result["definition"]

    def test_real_type_mapped(self, tmp_path):
        csv_path = tmp_path / "prices.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "price",
                    "data_format": "real",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )
        result = generate_sql_by_desc_file("db", {"prices": {}}, str(csv_path))
        assert "REAL" in result["definition"]

    def test_text_type_default(self, tmp_path):
        csv_path = tmp_path / "items.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "label",
                    "data_format": "text",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )
        result = generate_sql_by_desc_file("db", {"items": {}}, str(csv_path))
        assert "TEXT" in result["definition"]

    def test_column_description_as_comment(self, tmp_path):
        csv_path = tmp_path / "orders.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "status",
                    "data_format": "text",
                    "column_description": "Order status",
                    "value_description": "",
                }
            ],
        )
        result = generate_sql_by_desc_file("db", {"orders": {}}, str(csv_path))
        assert "Order status" in result["definition"]

    def test_value_description_appended(self, tmp_path):
        csv_path = tmp_path / "products.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "type",
                    "data_format": "text",
                    "column_description": "Product type",
                    "value_description": "A=1, B=2",
                }
            ],
        )
        result = generate_sql_by_desc_file("db", {"products": {}}, str(csv_path))
        assert "Values: A=1, B=2" in result["definition"]

    def test_primary_key_added(self, tmp_path):
        csv_path = tmp_path / "users.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "id",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )
        table_infos = {"users": {"primary_keys": ["`id`"]}}
        result = generate_sql_by_desc_file("db", table_infos, str(csv_path))
        assert "PRIMARY KEY" in result["definition"]

    def test_foreign_key_added(self, tmp_path):
        csv_path = tmp_path / "orders.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "user_id",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )
        table_infos = {
            "orders": {"foreign_keys": [{"column": "`user_id`", "target_table": "`users`", "target_column": "`id`"}]}
        }
        result = generate_sql_by_desc_file("db", table_infos, str(csv_path))
        assert "FOREIGN KEY" in result["definition"]
        assert "REFERENCES `users`" in result["definition"]

    def test_result_metadata_fields(self, tmp_path):
        csv_path = tmp_path / "tables.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "x",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )
        result = generate_sql_by_desc_file("mydb", {"tables": {}}, str(csv_path))
        assert result["catalog_name"] == ""
        assert result["schema_name"] == ""
        assert result["table_type"] == "table"
        assert result["definition"].endswith(");")


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_nonexistent_db_path_returns_empty(self, tmp_path):
        mock_db_manager = MagicMock()
        schema_result, value_result = init_db(
            db_manager=mock_db_manager,
            namespace="ns",
            database_name="nonexistent_db",
            table_keys={},
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert schema_result == []
        assert value_result == []

    def test_processes_csv_files(self, tmp_path):
        # Create database directory structure
        db_dir = tmp_path / "mydb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        # Create a CSV description file
        csv_path = desc_dir / "users.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "id",
                    "data_format": "integer",
                    "column_description": "ID",
                    "value_description": "",
                }
            ],
        )

        mock_db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "mydb.users"
        mock_conn.get_sample_rows.return_value = [{"table_name": "users", "sample_rows": "id\n1\n2"}]
        mock_db_manager.get_conn.return_value = mock_conn

        # table_keys is {database_name: {table_name: {pk/fk info}}}
        table_keys = {"mydb": {"users": {}}}

        schema_result, value_result = init_db(
            db_manager=mock_db_manager,
            namespace="ns",
            database_name="mydb",
            table_keys=table_keys["mydb"],  # init_db expects flat {table_name: {}} dict
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert len(schema_result) == 1
        assert schema_result[0]["table_name"] == "users"
        assert schema_result[0]["database_name"] == "mydb"

    def test_skips_non_csv_files(self, tmp_path):
        db_dir = tmp_path / "mydb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        # Create a non-CSV file
        (desc_dir / "readme.txt").write_text("not a csv")

        mock_db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_db_manager.get_conn.return_value = mock_conn

        schema_result, value_result = init_db(
            db_manager=mock_db_manager,
            namespace="ns",
            database_name="mydb",
            table_keys={},
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert schema_result == []
        assert value_result == []

    def test_skips_existing_schema_tables(self, tmp_path):
        db_dir = tmp_path / "mydb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        csv_path = desc_dir / "users.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "id",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )

        mock_db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "mydb.users"
        mock_db_manager.get_conn.return_value = mock_conn

        schema_result, value_result = init_db(
            db_manager=mock_db_manager,
            namespace="ns",
            database_name="mydb",
            table_keys={"users": {}},
            databases_path=str(tmp_path),
            all_schema_tables={"mydb.users"},  # already exists
            all_value_tables=set(),
        )
        assert schema_result == []

    def test_no_sample_rows_logged(self, tmp_path):
        db_dir = tmp_path / "mydb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        csv_path = desc_dir / "users.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "id",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )

        mock_db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "mydb.users"
        mock_conn.get_sample_rows.return_value = []  # No sample rows
        mock_db_manager.get_conn.return_value = mock_conn

        schema_result, value_result = init_db(
            db_manager=mock_db_manager,
            namespace="ns",
            database_name="mydb",
            table_keys={"users": {}},
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        assert len(schema_result) == 1
        assert len(value_result) == 0


# ---------------------------------------------------------------------------
# init_dev_schema_by_db
# ---------------------------------------------------------------------------


class TestInitDevSchemaByDb:
    def test_skips_ds_store(self):
        mock_rag = MagicMock()
        mock_db_manager = MagicMock()
        init_dev_schema_by_db(
            rag=mock_rag,
            db_manager=mock_db_manager,
            namespace="ns",
            database_name=".DS_Store",
            table_keys={},
            databases_path="/fake",
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        mock_rag.store_batch.assert_not_called()

    def test_calls_store_batch(self, tmp_path):
        db_dir = tmp_path / "mydb"
        desc_dir = db_dir / "database_description"
        desc_dir.mkdir(parents=True)

        csv_path = desc_dir / "products.csv"
        _write_desc_csv(
            str(csv_path),
            [
                {
                    "original_column_name": "id",
                    "data_format": "integer",
                    "column_description": "",
                    "value_description": "",
                }
            ],
        )

        mock_rag = MagicMock()
        mock_db_manager = MagicMock()
        mock_conn = MagicMock()
        mock_conn.identifier.return_value = "mydb.products"
        mock_conn.get_sample_rows.return_value = [{"table_name": "products", "sample_rows": "id\n1"}]
        mock_db_manager.get_conn.return_value = mock_conn

        init_dev_schema_by_db(
            rag=mock_rag,
            db_manager=mock_db_manager,
            namespace="ns",
            database_name="mydb",
            table_keys={"mydb": {"products": {}}},  # init_dev_schema_by_db looks up by db_name internally
            databases_path=str(tmp_path),
            all_schema_tables=set(),
            all_value_tables=set(),
        )
        mock_rag.store_batch.assert_called_once()
