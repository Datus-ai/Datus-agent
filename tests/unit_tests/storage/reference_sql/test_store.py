# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/reference_sql/store.py -- ReferenceSqlStorage."""

import hashlib

import pytest

from datus.storage.embedding_models import get_db_embedding_model
from datus.storage.reference_sql.store import ReferenceSqlStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_id(sql: str) -> str:
    """Generate a deterministic MD5 ID from a SQL string."""
    return hashlib.md5(sql.encode("utf-8")).hexdigest()


def _make_sql_item(
    idx: int,
    subject_path: list | None = None,
    sql: str = "",
    name: str = "",
) -> dict:
    """Build a single reference SQL item with required fields."""
    actual_sql = sql or f"SELECT col_{idx} FROM table_{idx} WHERE id > 0"
    actual_name = name or f"query_{idx}"
    return {
        "subject_path": subject_path or ["Analytics", "Reports"],
        "id": _gen_id(actual_sql),
        "name": actual_name,
        "sql": actual_sql,
        "comment": f"Comment for query {idx}",
        "summary": f"Summary of query {idx} for retrieving data",
        "search_text": f"Search text for query {idx} about data retrieval",
        "filepath": f"/queries/query_{idx}.sql",
        "tags": f"tag_{idx}",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_sql_storage(tmp_path) -> ReferenceSqlStorage:
    """Create a ReferenceSqlStorage with real vector backend."""
    return ReferenceSqlStorage(embedding_model=get_db_embedding_model())


# ============================================================
# ReferenceSqlStorage.__init__
# ============================================================


class TestReferenceSqlStorageInit:
    """Tests for ReferenceSqlStorage initialization."""

    def test_table_name(self, ref_sql_storage):
        """Table name should be 'reference_sql'."""
        assert ref_sql_storage.table_name == "reference_sql"

    def test_vector_source_name(self, ref_sql_storage):
        """Vector source should be 'search_text'."""
        assert ref_sql_storage.vector_source_name == "search_text"

    def test_vector_column_name(self, ref_sql_storage):
        """Vector column should be 'vector'."""
        assert ref_sql_storage.vector_column_name == "vector"

    def test_schema_has_expected_fields(self, ref_sql_storage):
        """Schema should contain all required reference SQL fields."""
        expected_fields = {
            "id",
            "name",
            "sql",
            "comment",
            "summary",
            "search_text",
            "filepath",
            "tags",
            "vector",
            "subject_node_id",
            "created_at",
        }
        schema_names = set(ref_sql_storage._schema.names)
        for field in expected_fields:
            assert field in schema_names, f"Field '{field}' missing from schema"

    def test_subject_tree_initialized(self, ref_sql_storage):
        """Subject tree should be initialized."""
        assert ref_sql_storage.subject_tree is not None


# ============================================================
# ReferenceSqlStorage.batch_store_sql
# ============================================================


class TestBatchStoreSql:
    """Tests for batch_store_sql with field validation."""

    def test_batch_store_sql_empty(self, ref_sql_storage):
        """Storing empty list should be a no-op."""
        ref_sql_storage.batch_store_sql([])
        results = ref_sql_storage.search_all_reference_sql()
        assert results == []

    def test_batch_store_sql_single(self, ref_sql_storage):
        """Storing a single SQL item should be retrievable."""
        item = _make_sql_item(1)
        ref_sql_storage.batch_store_sql([item])
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 1
        assert results[0]["name"] == "query_1"

    def test_batch_store_sql_multiple(self, ref_sql_storage):
        """Storing multiple SQL items should all be retrievable."""
        items = [_make_sql_item(i) for i in range(3)]
        ref_sql_storage.batch_store_sql(items)
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 3

    def test_batch_store_sql_skips_missing_subject_path(self, ref_sql_storage):
        """Items with empty subject_path should be skipped."""
        items = [
            _make_sql_item(1),
            {
                "subject_path": [],
                "id": "bad_id",
                "name": "bad_query",
                "sql": "SELECT 1",
                "comment": "",
                "summary": "Bad summary",
                "search_text": "bad search",
                "filepath": "/bad.sql",
                "tags": "",
            },
        ]
        ref_sql_storage.batch_store_sql(items)
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 1

    def test_batch_store_sql_skips_missing_required_fields(self, ref_sql_storage):
        """Items missing required fields (name, sql, summary, search_text) should be skipped."""
        items = [
            _make_sql_item(1),
            {
                "subject_path": ["Analytics"],
                "id": "incomplete_id",
                "name": "",
                "sql": "",
                "comment": "",
                "summary": "",
                "search_text": "",
                "filepath": "",
                "tags": "",
            },
        ]
        ref_sql_storage.batch_store_sql(items)
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 1
        assert results[0]["name"] == "query_1"

    def test_batch_store_sql_with_different_subject_paths(self, ref_sql_storage):
        """Items with different subject paths should be stored under correct paths."""
        items = [
            _make_sql_item(1, subject_path=["Finance", "Revenue"]),
            _make_sql_item(2, subject_path=["Operations", "Logistics"]),
        ]
        ref_sql_storage.batch_store_sql(items)

        finance_results = ref_sql_storage.search_all_reference_sql(subject_path=["Finance", "Revenue"])
        assert len(finance_results) == 1
        assert finance_results[0]["name"] == "query_1"

        ops_results = ref_sql_storage.search_all_reference_sql(subject_path=["Operations", "Logistics"])
        assert len(ops_results) == 1
        assert ops_results[0]["name"] == "query_2"


# ============================================================
# ReferenceSqlStorage.batch_upsert_sql
# ============================================================


class TestBatchUpsertSql:
    """Tests for batch_upsert_sql with validation."""

    def test_batch_upsert_sql_empty(self, ref_sql_storage):
        """Upserting empty list should be a no-op."""
        ref_sql_storage.batch_upsert_sql([])

    def test_batch_upsert_sql_insert(self, ref_sql_storage):
        """Upserting new items should insert them."""
        item = _make_sql_item(1)
        ref_sql_storage.batch_upsert_sql([item])
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 1

    def test_batch_upsert_sql_update(self, ref_sql_storage):
        """Upserting existing items should update them."""
        item = _make_sql_item(1)
        ref_sql_storage.batch_store_sql([item])

        # Update the same item with new content
        updated_item = _make_sql_item(1)
        updated_item["summary"] = "Updated summary for query 1"
        ref_sql_storage.batch_upsert_sql([updated_item])

        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 1
        assert results[0]["summary"] == "Updated summary for query 1"

    def test_batch_upsert_sql_missing_subject_path_raises(self, ref_sql_storage):
        """Missing subject_path should raise ValueError."""
        bad_item = {
            "id": "bad_id",
            "name": "bad_query",
            "sql": "SELECT 1",
            "summary": "Bad summary",
            "search_text": "bad search",
        }
        with pytest.raises(ValueError, match="subject_path is required"):
            ref_sql_storage.batch_upsert_sql([bad_item])

    def test_batch_upsert_sql_empty_subject_path_raises(self, ref_sql_storage):
        """Empty subject_path list should raise ValueError."""
        bad_item = {
            "subject_path": [],
            "id": "bad_id",
            "name": "bad_query",
            "sql": "SELECT 1",
            "summary": "Bad summary",
            "search_text": "bad search",
        }
        with pytest.raises(ValueError, match="subject_path is required"):
            ref_sql_storage.batch_upsert_sql([bad_item])


# ============================================================
# ReferenceSqlStorage.search_reference_sql
# ============================================================


class TestSearchReferenceSql:
    """Tests for search_reference_sql with vector search and subject filtering."""

    @pytest.fixture(autouse=True)
    def _populate(self, ref_sql_storage):
        """Populate storage with test data."""
        self.storage = ref_sql_storage
        items = [
            _make_sql_item(1, subject_path=["Finance", "Revenue"]),
            _make_sql_item(2, subject_path=["Finance", "Revenue"]),
            _make_sql_item(3, subject_path=["Operations", "Logistics"]),
        ]
        ref_sql_storage.batch_store_sql(items)

    def test_search_by_query_text(self):
        """Vector search with query text returns relevant results."""
        results = self.storage.search_reference_sql(query_text="data retrieval", top_n=5)
        assert len(results) > 0

    def test_search_by_subject_path(self):
        """Filtering by subject_path returns only matching entries."""
        results = self.storage.search_reference_sql(
            query_text="data retrieval",
            subject_path=["Finance", "Revenue"],
            top_n=10,
        )
        for r in results:
            assert r["subject_path"][0] == "Finance"

    def test_search_with_top_n_limit(self):
        """top_n should limit the number of results."""
        results = self.storage.search_reference_sql(query_text="data retrieval", top_n=1)
        assert len(results) <= 1

    def test_search_with_selected_fields(self):
        """selected_fields should filter returned fields."""
        results = self.storage.search_reference_sql(
            query_text="data retrieval",
            selected_fields=["name", "sql"],
            top_n=5,
        )
        assert len(results) > 0
        for r in results:
            assert "name" in r
            assert "sql" in r


# ============================================================
# ReferenceSqlStorage.search_all_reference_sql
# ============================================================


class TestSearchAllReferenceSql:
    """Tests for search_all_reference_sql."""

    def test_search_all_empty(self, ref_sql_storage):
        """Empty storage should return empty list."""
        results = ref_sql_storage.search_all_reference_sql()
        assert results == []

    def test_search_all_no_filter(self, ref_sql_storage):
        """Without filter, returns all entries."""
        items = [_make_sql_item(i) for i in range(3)]
        ref_sql_storage.batch_store_sql(items)
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 3

    def test_search_all_with_subject_filter(self, ref_sql_storage):
        """With subject_path filter, returns only matching entries."""
        items = [
            _make_sql_item(1, subject_path=["Finance", "Revenue"]),
            _make_sql_item(2, subject_path=["Operations", "Logistics"]),
        ]
        ref_sql_storage.batch_store_sql(items)

        results = ref_sql_storage.search_all_reference_sql(subject_path=["Finance", "Revenue"])
        assert len(results) == 1
        assert results[0]["name"] == "query_1"

    def test_search_all_with_select_fields(self, ref_sql_storage):
        """select_fields should limit returned fields."""
        items = [_make_sql_item(1)]
        ref_sql_storage.batch_store_sql(items)
        results = ref_sql_storage.search_all_reference_sql(select_fields=["name", "sql"])
        assert len(results) == 1
        for r in results:
            assert "name" in r
            assert "sql" in r


# ============================================================
# ReferenceSqlStorage.delete_reference_sql
# ============================================================


class TestDeleteReferenceSql:
    """Tests for delete_reference_sql."""

    def test_delete_existing_entry(self, ref_sql_storage):
        """Deleting an existing entry should return True."""
        items = [_make_sql_item(1, subject_path=["Finance", "Revenue"])]
        ref_sql_storage.batch_store_sql(items)

        result = ref_sql_storage.delete_reference_sql(subject_path=["Finance", "Revenue"], name="query_1")
        assert result is True

        remaining = ref_sql_storage.search_all_reference_sql()
        assert len(remaining) == 0

    def test_delete_nonexistent_entry(self, ref_sql_storage):
        """Deleting a non-existent entry should return False."""
        items = [_make_sql_item(1, subject_path=["Finance", "Revenue"])]
        ref_sql_storage.batch_store_sql(items)

        result = ref_sql_storage.delete_reference_sql(subject_path=["Finance", "Revenue"], name="nonexistent_query")
        assert result is False

    def test_delete_preserves_other_entries(self, ref_sql_storage):
        """Deleting one entry should not affect others."""
        items = [
            _make_sql_item(1, subject_path=["Finance", "Revenue"]),
            _make_sql_item(2, subject_path=["Finance", "Revenue"]),
        ]
        ref_sql_storage.batch_store_sql(items)

        ref_sql_storage.delete_reference_sql(subject_path=["Finance", "Revenue"], name="query_1")

        remaining = ref_sql_storage.search_all_reference_sql(subject_path=["Finance", "Revenue"])
        assert len(remaining) == 1
        assert remaining[0]["name"] == "query_2"


# ============================================================
# ReferenceSqlStorage.create_indices
# ============================================================


class TestReferenceSqlStorageCreateIndices:
    """Tests for create_indices."""

    def test_create_indices_after_data(self, ref_sql_storage):
        """Creating indices after storing data should not raise."""
        items = [_make_sql_item(i) for i in range(3)]
        ref_sql_storage.batch_store_sql(items)
        ref_sql_storage.create_indices()
        # Verify search still works after index creation
        results = ref_sql_storage.search_all_reference_sql()
        assert len(results) == 3
