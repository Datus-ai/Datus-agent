# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.schema_metadata.benchmark_init."""

import json
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# process_line
# ---------------------------------------------------------------------------


class TestProcessLine:
    """Tests for process_line error handling."""

    def test_process_line_catches_exceptions(self):
        """process_line should catch and log exceptions without raising."""
        from datus.storage.schema_metadata.benchmark_init import process_line

        mock_storage = MagicMock()
        # Invalid item that will cause an error in do_process_by_database
        item = {"db_id": "nonexistent_db"}

        # Should not raise
        process_line(mock_storage, item, "/nonexistent/path", set(), set())


# ---------------------------------------------------------------------------
# do_process_by_database
# ---------------------------------------------------------------------------


class TestDoProcessByDatabase:
    """Tests for do_process_by_database."""

    def test_skips_nonexistent_directory(self, tmp_path):
        """Should return early when the database directory does not exist."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)
        os.makedirs(os.path.join(benchmark_path, "resource/databases"), exist_ok=True)

        # This should not raise or call store_batch
        do_process_by_database(mock_storage, "nonexistent_db", benchmark_path, set(), set())
        mock_storage.store_batch.assert_not_called()

    def test_skips_non_csv_files(self, tmp_path):
        """Should skip non-CSV files in schema directories."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/test_schema")
        os.makedirs(db_dir, exist_ok=True)

        # Create a non-CSV file
        with open(os.path.join(db_dir, "data.json"), "w") as f:
            json.dump({"key": "value"}, f)

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())
        mock_storage.store_batch.assert_not_called()

    def test_skips_non_directory_entries(self, tmp_path):
        """Should skip files (not directories) at the schema level."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db")
        os.makedirs(db_dir, exist_ok=True)

        # Create a file (not a directory) at the schema level
        with open(os.path.join(db_dir, "readme.txt"), "w") as f:
            f.write("Not a schema dir")

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())
        mock_storage.store_batch.assert_not_called()

    def test_processes_csv_with_ddl(self, tmp_path):
        """Should process CSV files with DDL entries and store to batch."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        # Capture args at call time since the code calls .clear() after store_batch
        captured_calls = []

        def capture_store_batch(batch_records, batch_value_records):
            captured_calls.append((list(batch_records), list(batch_value_records)))

        mock_storage = MagicMock()
        mock_storage.store_batch.side_effect = capture_store_batch
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        # Create a CSV file with DDL
        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.users"],
                "DDL": ["CREATE TABLE users (id INT, name VARCHAR)"],
                "description": ["Users table"],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        all_schema_tables = set()
        all_value_tables = set()

        do_process_by_database(mock_storage, "test_db", benchmark_path, all_schema_tables, all_value_tables)

        # Should have called store_batch
        assert len(captured_calls) == 1
        batch_records, _ = captured_calls[0]
        assert len(batch_records) == 1
        assert batch_records[0]["table_name"] == "users"
        assert batch_records[0]["database_name"] == "test_db"
        assert batch_records[0]["schema_name"] == "public"

    def test_skips_rows_without_ddl(self, tmp_path):
        """Should skip rows with missing or empty DDL."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        # Row without DDL
        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.empty_table"],
                "DDL": [None],
                "description": ["No DDL"],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())
        mock_storage.store_batch.assert_not_called()

    def test_skips_already_existing_tables(self, tmp_path):
        """Tables already in all_schema_tables should not be added again."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.users"],
                "DDL": ["CREATE TABLE users (id INT)"],
                "description": ["Users"],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        # Already existing
        all_schema_tables = {"test_db.public.users"}
        all_value_tables = set()

        do_process_by_database(mock_storage, "test_db", benchmark_path, all_schema_tables, all_value_tables)

        # store_batch should not be called since table already exists
        mock_storage.store_batch.assert_not_called()

    def test_ddl_comment_appended_with_description(self, tmp_path):
        """When description is present, COMMENT should be appended to DDL."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        captured_calls = []

        def capture_store_batch(batch_records, batch_value_records):
            captured_calls.append((list(batch_records), list(batch_value_records)))

        mock_storage = MagicMock()
        mock_storage.store_batch.side_effect = capture_store_batch
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.orders"],
                "DDL": ["CREATE TABLE orders (id INT);"],
                "description": ["Order records"],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())

        batch_records = captured_calls[0][0]
        ddl = batch_records[0]["definition"]
        assert "COMMENT = 'Order records'" in ddl
        assert ddl.endswith(";")

    def test_ddl_comment_for_no_trailing_semicolon(self, tmp_path):
        """DDL without trailing semicolon should have COMMENT appended differently."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        captured_calls = []

        def capture_store_batch(batch_records, batch_value_records):
            captured_calls.append((list(batch_records), list(batch_value_records)))

        mock_storage = MagicMock()
        mock_storage.store_batch.side_effect = capture_store_batch
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.items"],
                "DDL": ["CREATE TABLE items (id INT)"],
                "description": ["Items table"],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())

        batch_records = captured_calls[0][0]
        ddl = batch_records[0]["definition"]
        assert "COMMENT = 'Items table'" in ddl
        assert ddl.endswith(";")

    def test_sample_rows_stored_as_value_records(self, tmp_path):
        """Tables with sample_rows in JSON data should have value records."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        captured_calls = []

        def capture_store_batch(batch_records, batch_value_records):
            captured_calls.append((list(batch_records), list(batch_value_records)))

        mock_storage = MagicMock()
        mock_storage.store_batch.side_effect = capture_store_batch
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.products"],
                "DDL": ["CREATE TABLE products (id INT, name VARCHAR)"],
                "description": ["Products"],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        # Create JSON data file with sample_rows
        json_data = {"sample_rows": [{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}]}
        with open(os.path.join(db_dir, "products.json"), "w") as f:
            json.dump(json_data, f)

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())

        batch_value_records = captured_calls[0][1]
        assert len(batch_value_records) == 1
        assert batch_value_records[0]["table_name"] == "products"

    def test_skips_value_records_for_existing_value_tables(self, tmp_path):
        """Tables already in all_value_tables should not have value records stored again."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        captured_calls = []

        def capture_store_batch(batch_records, batch_value_records):
            captured_calls.append((list(batch_records), list(batch_value_records)))

        mock_storage = MagicMock()
        mock_storage.store_batch.side_effect = capture_store_batch
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.products"],
                "DDL": ["CREATE TABLE products (id INT)"],
                "description": [""],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        json_data = {"sample_rows": [{"id": 1}]}
        with open(os.path.join(db_dir, "products.json"), "w") as f:
            json.dump(json_data, f)

        all_value_tables = {"test_db.public.products"}

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), all_value_tables)

        batch_value_records = captured_calls[0][1]
        assert len(batch_value_records) == 0

    def test_empty_description_no_comment(self, tmp_path):
        """Empty description should not add COMMENT to DDL."""
        from datus.storage.schema_metadata.benchmark_init import do_process_by_database

        captured_calls = []

        def capture_store_batch(batch_records, batch_value_records):
            captured_calls.append((list(batch_records), list(batch_value_records)))

        mock_storage = MagicMock()
        mock_storage.store_batch.side_effect = capture_store_batch
        benchmark_path = str(tmp_path)

        db_dir = os.path.join(benchmark_path, "resource/databases/test_db/public")
        os.makedirs(db_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "table_name": ["test_db.public.logs"],
                "DDL": ["CREATE TABLE logs (id INT)"],
                "description": [""],
            }
        )
        df.to_csv(os.path.join(db_dir, "tables.csv"), index=False)

        do_process_by_database(mock_storage, "test_db", benchmark_path, set(), set())

        batch_records = captured_calls[0][0]
        ddl = batch_records[0]["definition"]
        assert "COMMENT" not in ddl


# ---------------------------------------------------------------------------
# init_snowflake_schema
# ---------------------------------------------------------------------------


class TestInitSnowflakeSchema:
    """Tests for init_snowflake_schema."""

    def test_filters_by_instance_ids(self, tmp_path):
        """Should filter JSONL entries by instance_ids when provided."""
        from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        # Create JSONL file
        jsonl_path = os.path.join(benchmark_path, "spider2-snow.jsonl")
        lines = [
            json.dumps({"instance_id": "inst1", "db_id": "db1"}),
            json.dumps({"instance_id": "inst2", "db_id": "db2"}),
        ]
        with open(jsonl_path, "w") as f:
            f.write("\n".join(lines))

        # Create db directories (but they won't have real data)
        for db_id in ["db1", "db2"]:
            os.makedirs(os.path.join(benchmark_path, f"resource/databases/{db_id}"), exist_ok=True)

        init_snowflake_schema(
            mock_storage,
            benchmark_path=benchmark_path,
            build_mode="overwrite",
            pool_size=1,
            instance_ids=["inst1"],
        )

        # after_init should always be called
        mock_storage.after_init.assert_called_once()

    def test_deduplicates_db_ids(self, tmp_path):
        """Should process each db_id only once."""
        from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        # Create JSONL with duplicate db_ids
        jsonl_path = os.path.join(benchmark_path, "spider2-snow.jsonl")
        lines = [
            json.dumps({"instance_id": "inst1", "db_id": "db1"}),
            json.dumps({"instance_id": "inst2", "db_id": "db1"}),  # Same db_id
        ]
        with open(jsonl_path, "w") as f:
            f.write("\n".join(lines))

        os.makedirs(os.path.join(benchmark_path, "resource/databases/db1"), exist_ok=True)

        init_snowflake_schema(
            mock_storage,
            benchmark_path=benchmark_path,
            build_mode="overwrite",
            pool_size=1,
        )

        mock_storage.after_init.assert_called_once()

    def test_calls_after_init(self, tmp_path):
        """after_init should be called on storage."""
        from datus.storage.schema_metadata.benchmark_init import init_snowflake_schema

        mock_storage = MagicMock()
        benchmark_path = str(tmp_path)

        # Create empty JSONL
        jsonl_path = os.path.join(benchmark_path, "spider2-snow.jsonl")
        with open(jsonl_path, "w") as _f:  # noqa: F841
            pass  # Empty file

        init_snowflake_schema(
            mock_storage,
            benchmark_path=benchmark_path,
            build_mode="overwrite",
            pool_size=1,
        )

        mock_storage.after_init.assert_called_once()
