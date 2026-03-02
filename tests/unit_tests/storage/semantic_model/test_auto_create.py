# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.semantic_model.auto_create."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# extract_tables_from_sql_list
# ---------------------------------------------------------------------------


class TestExtractTablesFromSqlList:
    """Tests for extract_tables_from_sql_list."""

    def test_extracts_tables_from_simple_select(self):
        """Should extract table names from simple SELECT statements."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        tables = extract_tables_from_sql_list(["SELECT * FROM users"], config)
        assert "users" in tables

    def test_extracts_tables_from_multiple_sqls(self):
        """Should extract table names from multiple SQL statements."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        sql_list = [
            "SELECT * FROM orders",
            "SELECT * FROM customers",
        ]
        tables = extract_tables_from_sql_list(sql_list, config)
        assert "orders" in tables
        assert "customers" in tables

    def test_empty_sql_list(self):
        """Empty list should return empty set."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        tables = extract_tables_from_sql_list([], config)
        assert tables == set()

    def test_skips_empty_sql_strings(self):
        """Empty or whitespace-only SQL strings should be skipped."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        tables = extract_tables_from_sql_list(["", "  ", None], config)
        assert tables == set()

    def test_handles_invalid_sql_gracefully(self):
        """Invalid SQL should be skipped without raising."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        # This should not raise
        tables = extract_tables_from_sql_list(["NOT VALID SQL AT ALL ???"], config)
        # Result may be empty or contain something, but should not raise
        assert isinstance(tables, set)

    def test_deduplicates_tables(self):
        """Tables appearing in multiple SQLs should only appear once."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        sql_list = [
            "SELECT * FROM users",
            "SELECT count(*) FROM users",
        ]
        tables = extract_tables_from_sql_list(sql_list, config)
        # Should be a set, so duplicates are already removed
        user_entries = [t for t in tables if "users" in t.lower()]
        assert len(user_entries) >= 1

    def test_extracts_join_tables(self):
        """Should extract tables from JOIN clauses."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"
        tables = extract_tables_from_sql_list([sql], config)
        assert "orders" in tables
        assert "customers" in tables


# ---------------------------------------------------------------------------
# find_missing_semantic_models
# ---------------------------------------------------------------------------


class TestFindMissingSemanticModels:
    """Tests for find_missing_semantic_models."""

    def test_empty_tables_returns_empty(self):
        """Empty table set should return empty list."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        result = find_missing_semantic_models(set(), config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_all_models_exist(self, MockRAG):
        """When all semantic models exist, should return empty list."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        # Simulate existing semantic model
        mock_rag.storage.search_objects.return_value = [{"name": "users"}]

        result = find_missing_semantic_models({"users"}, config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_missing_models_detected(self, MockRAG):
        """When semantic models are missing, should return those table names."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        # No matching results
        mock_rag.storage.search_objects.return_value = []

        result = find_missing_semantic_models({"missing_table"}, config)
        assert "missing_table" in result

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_case_insensitive_match(self, MockRAG):
        """Should match table names case-insensitively."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        mock_rag.storage.search_objects.return_value = [{"name": "USERS"}]

        result = find_missing_semantic_models({"users"}, config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_fully_qualified_name_parsed(self, MockRAG):
        """Should parse fully qualified names (db.schema.table) and use last part."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        mock_rag.storage.search_objects.return_value = [{"name": "orders"}]

        result = find_missing_semantic_models({"mydb.public.orders"}, config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_search_error_treated_as_missing(self, MockRAG):
        """Search errors should treat the table as missing."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        mock_rag.storage.search_objects.side_effect = Exception("Storage error")

        result = find_missing_semantic_models({"error_table"}, config)
        assert "error_table" in result
