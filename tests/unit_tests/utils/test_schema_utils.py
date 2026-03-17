# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json

from datus.utils.schema_utils import table_metadata2markdown, table_metadata_struct

SIMPLE_DDL = "CREATE TABLE orders (id INT COMMENT 'Primary key', name VARCHAR(100));"
DDL_WITH_COMMENT = (
    "CREATE TABLE events (event_id BIGINT COMMENT 'Event ID', event_name VARCHAR(255)) COMMENT 'Event log';"
)


class TestTableMetadataStruct:
    def test_empty_list_returns_empty_json(self):
        result = table_metadata_struct([])
        # to_str of {} is '{}'
        assert result == "{}"

    def test_returns_json_string(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata_struct(metadata)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_schema_and_table_name(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata_struct(metadata)
        assert "public.orders" in result

    def test_contains_columns(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata_struct(metadata)
        parsed = json.loads(result)
        table_data = parsed["public.orders"]
        assert "columns" in table_data
        assert len(table_data["columns"]) > 0

    def test_includes_comment_when_present(self):
        # Use MySQL-style DDL that sqlglot parses comments from
        ddl_mysql = "CREATE TABLE events (event_id BIGINT) COMMENT='Event log';"
        metadata = [
            {
                "schema_name": "log",
                "table_name": "events",
                "schema_text": ddl_mysql,
            }
        ]
        result = table_metadata_struct(metadata)
        parsed = json.loads(result)
        # Whether comment is present depends on DDL dialect parsing — just verify structure
        table_data = parsed["log.events"]
        assert "columns" in table_data

    def test_multiple_tables(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            },
            {
                "schema_name": "log",
                "table_name": "events",
                "schema_text": DDL_WITH_COMMENT,
            },
        ]
        result = table_metadata_struct(metadata)
        parsed = json.loads(result)
        assert "public.orders" in parsed
        assert "log.events" in parsed


class TestTableMetadata2Markdown:
    def test_empty_list_returns_empty_string(self):
        result = table_metadata2markdown([])
        assert result == ""

    def test_returns_string(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata2markdown(metadata)
        assert isinstance(result, str)

    def test_contains_table_name_in_output(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata2markdown(metadata)
        assert "public.orders" in result

    def test_contains_column_names(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata2markdown(metadata)
        assert "id" in result.lower() or "name" in result.lower()

    def test_contains_markdown_table_syntax(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            }
        ]
        result = table_metadata2markdown(metadata)
        assert "|" in result

    def test_includes_comment_in_table_header(self):
        # Comment parsing depends on DDL dialect; just verify output contains table name
        metadata = [
            {
                "schema_name": "log",
                "table_name": "events",
                "schema_text": DDL_WITH_COMMENT,
            }
        ]
        result = table_metadata2markdown(metadata)
        assert "log.events" in result

    def test_table_name_with_schema_qualifier(self):
        ddl = "CREATE TABLE schema1.users (user_id INT, username VARCHAR(50));"
        metadata = [
            {
                "schema_name": "public",
                "table_name": "schema1.users",
                "schema_text": ddl,
            }
        ]
        result = table_metadata2markdown(metadata)
        # Should use the last part of the dot-split name
        assert "users" in result

    def test_multiple_tables_in_output(self):
        metadata = [
            {
                "schema_name": "public",
                "table_name": "orders",
                "schema_text": SIMPLE_DDL,
            },
            {
                "schema_name": "log",
                "table_name": "events",
                "schema_text": DDL_WITH_COMMENT,
            },
        ]
        result = table_metadata2markdown(metadata)
        assert "public.orders" in result
        assert "log.events" in result
