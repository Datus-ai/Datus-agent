# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/reference_sql/store.py -- ReferenceSqlStorage."""

import hashlib
import os
import tempfile

import pytest
import yaml

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


# ============================================================
# ReferenceSqlStorage.update_entry (YAML sync)
# ============================================================


class TestUpdateReferenceSqlYaml:
    """Tests for update_entry YAML sync in ReferenceSqlStorage."""

    def test_update_reference_sql_syncs_to_yaml_file(self, ref_sql_storage):
        """update_entry should write changed fields back to the source YAML file."""
        original_doc = {
            "id": "abc123",
            "name": "Daily Sales",
            "sql": "SELECT * FROM sales",
            "comment": "Original comment",
            "summary": "Original summary",
            "search_text": "daily sales revenue",
            "filepath": "",
            "subject_tree": "finance/revenue",
            "tags": "finance, sales",
        }

        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump(original_doc, tmp_file, allow_unicode=True, sort_keys=False)
            tmp_file.close()

            item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            ref_sql_storage.update_entry(
                subject_path=["Finance", "Revenue"],
                name="query_1",
                update_values={"sql": "SELECT 1 FROM new_table", "summary": "Updated summary"},
            )

            with open(tmp_file.name, encoding="utf-8") as f:
                updated_doc = yaml.safe_load(f)

            assert updated_doc["sql"] == "SELECT 1 FROM new_table"
            assert updated_doc["summary"] == "Updated summary"
        finally:
            os.unlink(tmp_file.name)

    def test_update_reference_sql_no_filepath_still_succeeds(self, ref_sql_storage):
        """update_entry should return True even when filepath is empty."""
        item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
        item["filepath"] = ""
        ref_sql_storage.batch_store_sql([item])

        result = ref_sql_storage.update_entry(
            subject_path=["Finance", "Revenue"],
            name="query_1",
            update_values={"summary": "New summary"},
        )
        assert result is True

    def test_update_reference_sql_nonexistent_file_still_succeeds(self, ref_sql_storage):
        """update_entry should return True even when the YAML file does not exist."""
        item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
        item["filepath"] = "/nonexistent/path/query.yaml"
        ref_sql_storage.batch_store_sql([item])

        result = ref_sql_storage.update_entry(
            subject_path=["Finance", "Revenue"],
            name="query_1",
            update_values={"summary": "New summary"},
        )
        assert result is True

    def test_update_reference_sql_preserves_other_yaml_fields(self, ref_sql_storage):
        """update_entry should only modify the specified syncable fields and leave others intact."""
        original_doc = {
            "id": "abc123",
            "name": "Daily Sales",
            "sql": "SELECT * FROM sales",
            "comment": "Keep this comment",
            "summary": "Original summary",
            "search_text": "daily sales revenue",
            "filepath": "",
            "subject_tree": "finance/revenue",
            "tags": "finance, sales",
        }

        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump(original_doc, tmp_file, allow_unicode=True, sort_keys=False)
            tmp_file.close()

            item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            ref_sql_storage.update_entry(
                subject_path=["Finance", "Revenue"],
                name="query_1",
                update_values={"summary": "New summary"},
            )

            with open(tmp_file.name, encoding="utf-8") as f:
                updated_doc = yaml.safe_load(f)

            assert updated_doc["summary"] == "New summary"
            assert updated_doc["comment"] == "Keep this comment"
            assert updated_doc["subject_tree"] == "finance/revenue"
            assert updated_doc["name"] == "Daily Sales"
            assert updated_doc["sql"] == "SELECT * FROM sales"
        finally:
            os.unlink(tmp_file.name)

    def test_update_reference_sql_non_dict_yaml_is_noop(self, ref_sql_storage):
        """_sync_reference_sql_update_to_yaml returns when YAML is not a dict."""
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            tmp_file.write("- just a list\n- not a dict\n")
            tmp_file.close()

            item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            result = ref_sql_storage.update_entry(
                subject_path=["Finance", "Revenue"],
                name="query_1",
                update_values={"sql": "SELECT 1"},
            )
            assert result is True

            with open(tmp_file.name, encoding="utf-8") as f:
                content = f.read()
            assert "just a list" in content
        finally:
            os.unlink(tmp_file.name)

    def test_update_reference_sql_no_syncable_fields(self, ref_sql_storage):
        """update_entry with non-syncable fields should not rewrite the YAML file."""
        original_doc = {"name": "q", "sql": "SELECT 1", "subject_tree": "a/b"}
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump(original_doc, tmp_file, allow_unicode=True, sort_keys=False)
            tmp_file.close()

            item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            ref_sql_storage.update_entry(
                subject_path=["Finance", "Revenue"],
                name="query_1",
                update_values={"id": "new_id_value"},  # not in _SYNCABLE_FIELDS
            )

            with open(tmp_file.name, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            assert doc["sql"] == "SELECT 1"
        finally:
            os.unlink(tmp_file.name)

    def test_update_reference_sql_corrupt_yaml_does_not_raise(self, ref_sql_storage):
        """_sync_reference_sql_update_to_yaml catches exceptions on corrupt YAML files."""
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            tmp_file.write("{{invalid yaml")
            tmp_file.close()

            item = _make_sql_item(1, subject_path=["Finance", "Revenue"])
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            result = ref_sql_storage.update_entry(
                subject_path=["Finance", "Revenue"],
                name="query_1",
                update_values={"sql": "SELECT 1"},
            )
            assert result is True
        finally:
            os.unlink(tmp_file.name)

    # ---------------------------------------------------------------------------
    def test_update_reference_sql_entry_not_found_does_not_touch_yaml(self, ref_sql_storage):
        """update_entry on a non-existent entry should raise and leave the YAML file untouched."""
        original_doc = {
            "id": "abc123",
            "name": "Daily Sales",
            "sql": "SELECT * FROM sales",
            "summary": "Original summary",
        }
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump(original_doc, tmp_file, allow_unicode=True, sort_keys=False)
            tmp_file.close()

            with pytest.raises(ValueError):
                ref_sql_storage.update_entry(
                    subject_path=["Finance", "Revenue"],
                    name="nonexistent_query",
                    update_values={"sql": "SELECT 1"},
                )

            with open(tmp_file.name, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            assert doc["sql"] == "SELECT * FROM sales"
        finally:
            os.unlink(tmp_file.name)


# YAML subject_tree sync on rename
# ---------------------------------------------------------------------------


class TestRenameReferenceSqlSubjectTreeYaml:
    """Tests for YAML subject_tree sync in rename."""

    def test_rename_move_rewrites_subject_tree(self, ref_sql_storage):
        """Moving an entry to a new parent should rewrite the top-level subject_tree field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_file:
            yaml.dump(
                {
                    "name": "move_me",
                    "sql": "SELECT 1",
                    "summary": "summary",
                    "search_text": "search",
                    "tags": "tagA",
                    "subject_tree": "Analytics/Reports",
                    "comment": "untouched",
                },
                tmp_file,
            )

        try:
            # Pre-create the target subject path so rename() can resolve it
            ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Dashboards"])

            item = _make_sql_item(1, subject_path=["Analytics", "Reports"], name="move_me")
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            result = ref_sql_storage.rename(
                ["Analytics", "Reports", "move_me"],
                ["Analytics", "Dashboards", "move_me"],
            )
            assert result is True

            with open(tmp_file.name, encoding="utf-8") as f:
                updated_doc = yaml.safe_load(f)
            assert updated_doc["subject_tree"] == "Analytics/Dashboards"
            # Unrelated fields preserved
            assert updated_doc["comment"] == "untouched"
            assert updated_doc["sql"] == "SELECT 1"
        finally:
            os.unlink(tmp_file.name)

    def test_rename_only_syncs_name_to_yaml(self, ref_sql_storage):
        """Renaming without moving should update the YAML ``name`` field but leave subject_tree alone."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_file:
            yaml.dump(
                {
                    "name": "old_name",
                    "sql": "SELECT 1",
                    "summary": "summary",
                    "search_text": "search",
                    "tags": "tagA",
                    "subject_tree": "Analytics/Reports",
                },
                tmp_file,
            )

        try:
            item = _make_sql_item(2, subject_path=["Analytics", "Reports"], name="old_name")
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            result = ref_sql_storage.rename(
                ["Analytics", "Reports", "old_name"],
                ["Analytics", "Reports", "new_name"],
            )
            assert result is True

            with open(tmp_file.name, encoding="utf-8") as f:
                updated_doc = yaml.safe_load(f)
            # YAML subject_tree untouched when only the entry name changes
            assert updated_doc["subject_tree"] == "Analytics/Reports"
            # YAML name must reflect the new entry name
            assert updated_doc["name"] == "new_name"
            # Other fields preserved
            assert updated_doc["sql"] == "SELECT 1"
        finally:
            os.unlink(tmp_file.name)

    def test_rename_only_skips_yaml_when_name_does_not_match(self, ref_sql_storage):
        """If the YAML's ``name`` does not match old_name, the file is not rewritten (safety guard)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_file:
            yaml.dump(
                {
                    "name": "different_name",  # intentionally NOT old_name
                    "sql": "SELECT 1",
                    "summary": "summary",
                    "search_text": "search",
                    "subject_tree": "Analytics/Reports",
                },
                tmp_file,
            )

        try:
            item = _make_sql_item(7, subject_path=["Analytics", "Reports"], name="old_name")
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            result = ref_sql_storage.rename(
                ["Analytics", "Reports", "old_name"],
                ["Analytics", "Reports", "new_name"],
            )
            assert result is True

            with open(tmp_file.name, encoding="utf-8") as f:
                updated_doc = yaml.safe_load(f)
            # Mismatched name must not be clobbered
            assert updated_doc["name"] == "different_name"
        finally:
            os.unlink(tmp_file.name)

    def test_rename_move_and_rename_syncs_both_subject_tree_and_name(self, ref_sql_storage):
        """A combined move + rename should update both subject_tree and name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp_file:
            yaml.dump(
                {
                    "name": "old_name",
                    "sql": "SELECT 1",
                    "summary": "summary",
                    "search_text": "search",
                    "subject_tree": "Analytics/Reports",
                },
                tmp_file,
            )

        try:
            ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Dashboards"])

            item = _make_sql_item(8, subject_path=["Analytics", "Reports"], name="old_name")
            item["filepath"] = tmp_file.name
            ref_sql_storage.batch_store_sql([item])

            result = ref_sql_storage.rename(
                ["Analytics", "Reports", "old_name"],
                ["Analytics", "Dashboards", "new_name"],
            )
            assert result is True

            with open(tmp_file.name, encoding="utf-8") as f:
                updated_doc = yaml.safe_load(f)
            assert updated_doc["subject_tree"] == "Analytics/Dashboards"
            assert updated_doc["name"] == "new_name"
        finally:
            os.unlink(tmp_file.name)

    def test_rename_move_without_filepath_still_succeeds(self, ref_sql_storage):
        """Moving an entry with no filepath should still return True."""
        ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Dashboards"])

        item = _make_sql_item(3, subject_path=["Analytics", "Reports"], name="no_filepath")
        item["filepath"] = ""
        ref_sql_storage.batch_store_sql([item])

        result = ref_sql_storage.rename(
            ["Analytics", "Reports", "no_filepath"],
            ["Analytics", "Dashboards", "no_filepath"],
        )
        assert result is True


# ---------------------------------------------------------------------------
# YAML subject_tree sync when a subject_node is renamed/moved
# ---------------------------------------------------------------------------


class TestSyncYamlSubjectTreeForSubtreeRefSql:
    """Tests for sync_yaml_subject_tree_for_subtree — triggered after a
    subject_tree node is renamed/moved, to sync YAML files of all descendants.
    """

    def _write_ref_sql_yaml(self, tmp_path, name: str, subject_tree_value: str, sql: str) -> str:
        filepath = os.path.join(tmp_path, f"{name}.yaml")
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "name": name,
                    "sql": sql,
                    "summary": f"summary for {name}",
                    "search_text": f"search for {name}",
                    "tags": "tagA",
                    "subject_tree": subject_tree_value,
                    "comment": "keep_me",
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )
        return filepath

    def test_rename_node_syncs_yaml_for_direct_and_descendant_ref_sql(self, ref_sql_storage, tmp_path):
        """Renaming an intermediate subject_node should update YAML subject_tree for
        reference SQL entries under that node AND under descendant nodes."""
        tmp_path_str = str(tmp_path)

        # Create subject tree: Analytics / Reports / Q1 / Detail
        ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Reports", "Q1"])
        ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Reports", "Q1", "Detail"])

        top_fp = self._write_ref_sql_yaml(tmp_path_str, "top_query", "Analytics/Reports/Q1", "SELECT 1")
        deep_fp = self._write_ref_sql_yaml(tmp_path_str, "deep_query", "Analytics/Reports/Q1/Detail", "SELECT 2")

        top_item = _make_sql_item(10, subject_path=["Analytics", "Reports", "Q1"], name="top_query", sql="SELECT 1")
        top_item["filepath"] = top_fp
        deep_item = _make_sql_item(
            11, subject_path=["Analytics", "Reports", "Q1", "Detail"], name="deep_query", sql="SELECT 2"
        )
        deep_item["filepath"] = deep_fp
        ref_sql_storage.batch_store_sql([top_item, deep_item])

        q1_node = ref_sql_storage.subject_tree.get_node_by_path(["Analytics", "Reports", "Q1"])
        assert q1_node is not None
        root_id = q1_node["node_id"]

        # Rename Q1 -> Quarter1 in place
        ref_sql_storage.subject_tree.rename(
            ["Analytics", "Reports", "Q1"],
            ["Analytics", "Reports", "Quarter1"],
        )

        ref_sql_storage.sync_yaml_subject_tree_for_subtree(root_id)

        with open(top_fp, encoding="utf-8") as f:
            top_doc = yaml.safe_load(f)
        assert top_doc["subject_tree"] == "Analytics/Reports/Quarter1"
        assert top_doc["comment"] == "keep_me"

        with open(deep_fp, encoding="utf-8") as f:
            deep_doc = yaml.safe_load(f)
        assert deep_doc["subject_tree"] == "Analytics/Reports/Quarter1/Detail"
        assert deep_doc["comment"] == "keep_me"

    def test_move_node_syncs_yaml_for_descendant_ref_sql(self, ref_sql_storage, tmp_path):
        """Moving a subject_node to a different parent should update YAML subject_tree
        for all descendant reference SQL entries."""
        tmp_path_str = str(tmp_path)

        ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Reports", "Q1"])
        ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Dashboards"])

        fp = self._write_ref_sql_yaml(tmp_path_str, "movable", "Analytics/Reports/Q1", "SELECT 1")

        item = _make_sql_item(12, subject_path=["Analytics", "Reports", "Q1"], name="movable", sql="SELECT 1")
        item["filepath"] = fp
        ref_sql_storage.batch_store_sql([item])

        q1_node = ref_sql_storage.subject_tree.get_node_by_path(["Analytics", "Reports", "Q1"])
        assert q1_node is not None
        root_id = q1_node["node_id"]

        # Move Q1 from Analytics/Reports to Analytics/Dashboards
        ref_sql_storage.subject_tree.rename(
            ["Analytics", "Reports", "Q1"],
            ["Analytics", "Dashboards", "Q1"],
        )

        ref_sql_storage.sync_yaml_subject_tree_for_subtree(root_id)

        with open(fp, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        assert doc["subject_tree"] == "Analytics/Dashboards/Q1"

    def test_sync_skips_entries_without_filepath(self, ref_sql_storage, tmp_path):
        """Reference SQL entries with an empty filepath should be silently skipped."""
        ref_sql_storage.subject_tree.find_or_create_path(["Analytics", "Reports", "Q1"])

        item = _make_sql_item(13, subject_path=["Analytics", "Reports", "Q1"], name="no_fp")
        item["filepath"] = ""
        ref_sql_storage.batch_store_sql([item])

        q1_node = ref_sql_storage.subject_tree.get_node_by_path(["Analytics", "Reports", "Q1"])
        root_id = q1_node["node_id"]

        # Should not raise
        ref_sql_storage.sync_yaml_subject_tree_for_subtree(root_id)

    def test_sync_handles_subtree_with_no_ref_sql(self, ref_sql_storage):
        """A subtree with no reference SQL entries should be a no-op."""
        ref_sql_storage.subject_tree.find_or_create_path(["Empty", "Branch"])
        node = ref_sql_storage.subject_tree.get_node_by_path(["Empty", "Branch"])
        assert node is not None

        # Should not raise
        ref_sql_storage.sync_yaml_subject_tree_for_subtree(node["node_id"])
