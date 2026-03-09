# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.storage.reference_sql.init_utils module."""

import hashlib
import tempfile

import pytest

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.reference_sql.init_utils import exists_reference_sql, gen_reference_sql_id
from datus.storage.reference_sql.store import ReferenceSqlStorage

# ============================================================
# Tests for gen_reference_sql_id
# ============================================================


class TestGenReferenceSqlId:
    """Tests for gen_reference_sql_id function."""

    def test_basic_id_generation(self):
        """Basic SQL string produces deterministic MD5 hash."""
        sql = "SELECT * FROM users"
        result = gen_reference_sql_id(sql)
        expected = hashlib.md5(sql.encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 32  # MD5 hex digest is always 32 chars

    def test_deterministic_same_input(self):
        """Same SQL always produces the same ID."""
        sql = "SELECT id, name FROM orders WHERE status = 'active'"
        id1 = gen_reference_sql_id(sql)
        id2 = gen_reference_sql_id(sql)
        assert id1 == id2

    def test_different_sql_different_ids(self):
        """Different SQL strings produce different IDs."""
        id1 = gen_reference_sql_id("SELECT 1")
        id2 = gen_reference_sql_id("SELECT 2")
        assert id1 != id2

    def test_empty_string(self):
        """Empty string produces valid MD5 hash."""
        result = gen_reference_sql_id("")
        expected = hashlib.md5(b"").hexdigest()
        assert result == expected
        assert len(result) == 32

    def test_unicode_sql(self):
        """Unicode SQL produces valid MD5 hash."""
        sql = "SELECT * FROM users WHERE name = '\u5f20\u4e09'"
        result = gen_reference_sql_id(sql)
        expected = hashlib.md5(sql.encode("utf-8")).hexdigest()
        assert result == expected

    def test_whitespace_sensitivity(self):
        """Whitespace differences produce different IDs."""
        id1 = gen_reference_sql_id("SELECT  1")
        id2 = gen_reference_sql_id("SELECT 1")
        assert id1 != id2

    def test_case_sensitivity(self):
        """Case differences produce different IDs."""
        id1 = gen_reference_sql_id("select 1")
        id2 = gen_reference_sql_id("SELECT 1")
        assert id1 != id2

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "SELECT * FROM t WHERE id > 0",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "SELECT DISTINCT col FROM table_name ORDER BY col",
        ],
    )
    def test_parametrized_valid_hex_output(self, sql):
        """All generated IDs are valid 32-char hex strings."""
        result = gen_reference_sql_id(sql)
        assert len(result) == 32
        # Ensure it's a valid hex string
        int(result, 16)


# ============================================================
# Tests for exists_reference_sql
# ============================================================


class TestExistsReferenceSql:
    """Tests for exists_reference_sql function."""

    def _make_storage(self, scope_dir):
        """Create a ReferenceSqlStorage in a temporary directory."""
        return ReferenceSqlStorage(embedding_model=get_db_embedding_model())

    def test_overwrite_mode_returns_empty_set(self):
        """Overwrite mode always returns an empty set regardless of stored data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = self._make_storage(tmp_dir)
            storage.batch_store_sql(
                [
                    {
                        "subject_path": ["Finance", "Revenue"],
                        "name": "quarterly_sales",
                        "id": gen_reference_sql_id("SELECT * FROM sales"),
                        "sql": "SELECT * FROM sales",
                        "comment": "quarterly sales",
                        "summary": "Get quarterly sales data",
                        "search_text": "quarterly sales data",
                        "filepath": "/tmp/test.sql",
                        "tags": "",
                    }
                ]
            )
            result = exists_reference_sql(storage, build_mode="overwrite")
            assert result == set()
            assert isinstance(result, set)

    def test_overwrite_mode_empty_store(self):
        """Overwrite mode on empty store returns empty set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = self._make_storage(tmp_dir)
            result = exists_reference_sql(storage, build_mode="overwrite")
            assert result == set()

    def test_incremental_mode_returns_existing_ids(self):
        """Incremental mode returns IDs of all stored reference SQL entries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = self._make_storage(tmp_dir)
            sql1 = "SELECT * FROM sales"
            sql2 = "SELECT * FROM orders"
            id1 = gen_reference_sql_id(sql1)
            id2 = gen_reference_sql_id(sql2)
            storage.batch_store_sql(
                [
                    {
                        "subject_path": ["Finance", "Revenue"],
                        "name": "quarterly_sales",
                        "id": id1,
                        "sql": sql1,
                        "comment": "sales query",
                        "summary": "Get sales data",
                        "search_text": "quarterly sales data",
                        "filepath": "/tmp/a.sql",
                        "tags": "",
                    },
                    {
                        "subject_path": ["Logistics", "Orders"],
                        "name": "all_orders",
                        "id": id2,
                        "sql": sql2,
                        "comment": "orders query",
                        "summary": "Get orders data",
                        "search_text": "all orders data",
                        "filepath": "/tmp/b.sql",
                        "tags": "",
                    },
                ]
            )
            result = exists_reference_sql(storage, build_mode="incremental")
            assert len(result) == 2
            assert isinstance(result, set)
            assert id1 in result
            assert id2 in result

    def test_incremental_mode_empty_store(self):
        """Incremental mode on empty store returns empty set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = self._make_storage(tmp_dir)
            result = exists_reference_sql(storage, build_mode="incremental")
            assert result == set()

    def test_default_build_mode_is_overwrite(self):
        """Default build_mode parameter is 'overwrite'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = self._make_storage(tmp_dir)
            # Default build_mode
            result = exists_reference_sql(storage)
            assert result == set()
