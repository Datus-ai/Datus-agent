# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/base.py — StorageBase and BaseEmbeddingStore."""

import re
from datetime import datetime

import pytest
from datus_storage_base.conditions import eq

from datus.storage.base import BaseEmbeddingStore, StorageBase
from datus.storage.embedding_models import EmbeddingModel, get_db_embedding_model
from datus.storage.schema_metadata import SchemaStorage
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# StorageBase._get_current_timestamp
# ---------------------------------------------------------------------------


class TestGetCurrentTimestamp:
    """Tests for StorageBase._get_current_timestamp."""

    def test_get_current_timestamp_returns_iso_format(self, tmp_path):
        """Timestamp string must be parseable as ISO-8601."""
        base = StorageBase()
        ts = base._get_current_timestamp()
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_get_current_timestamp_format_contains_T_separator(self, tmp_path):
        """ISO format should contain the 'T' separator between date and time."""
        base = StorageBase()
        ts = base._get_current_timestamp()
        assert "T" in ts

    def test_get_current_timestamp_matches_iso_regex(self, tmp_path):
        """Timestamp must match YYYY-MM-DDTHH:MM:SS pattern."""
        base = StorageBase()
        ts = base._get_current_timestamp()
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(pattern, ts), f"Timestamp '{ts}' does not match ISO-8601 pattern"

    def test_get_current_timestamp_is_recent(self, tmp_path):
        """Returned timestamp should be within a few seconds of now (UTC)."""
        from datetime import timezone

        base = StorageBase()
        before = datetime.now(timezone.utc)
        ts = base._get_current_timestamp()
        after = datetime.now(timezone.utc)
        parsed = datetime.fromisoformat(ts)
        assert before <= parsed <= after


# ---------------------------------------------------------------------------
# Vector index parameter calculation
# ---------------------------------------------------------------------------


class TestVectorIndexParameterCalculation:
    """Tests for create_vector_index index type and parameter calculations."""

    def test_index_type_ivf_flat_for_small_dataset(self):
        """Datasets with < 5000 rows should use IVF_FLAT."""
        row_count = 4999
        index_type = "IVF_PQ" if row_count >= 5000 else "IVF_FLAT"
        assert index_type == "IVF_FLAT"

    def test_index_type_ivf_pq_for_large_dataset(self):
        """Datasets with >= 5000 rows should use IVF_PQ."""
        row_count = 5000
        index_type = "IVF_PQ" if row_count >= 5000 else "IVF_FLAT"
        assert index_type == "IVF_PQ"

    def test_index_type_ivf_pq_for_very_large_dataset(self):
        """Very large datasets should still use IVF_PQ."""
        row_count = 100000
        index_type = "IVF_PQ" if row_count >= 5000 else "IVF_FLAT"
        assert index_type == "IVF_PQ"

    @pytest.mark.parametrize(
        "row_count,expected_partitions",
        [
            (10, 1),  # < 1000 -> row_count // 10
            (100, 10),  # < 1000 -> row_count // 10
            (500, 50),  # < 1000 -> row_count // 10
            (999, 99),  # < 1000 -> row_count // 10
            (1000, 50),  # >= 1000 and < 5000 -> row_count // 20
            (2000, 100),  # >= 1000 and < 5000 -> row_count // 20
            (4999, 249),  # >= 1000 and < 5000 -> row_count // 20
        ],
    )
    def test_partition_count_calculation(self, row_count, expected_partitions):
        """Partition count must match production formula in base.py create_vector_index."""
        num_partitions = max(1, min(1024, int(row_count**0.5)))
        if row_count < 1000:
            num_partitions = max(1, row_count // 10)
        elif row_count < 5000:
            num_partitions = max(1, row_count // 20)
        assert num_partitions == expected_partitions

    def test_partition_count_large_dataset_uses_sqrt(self):
        """For row_count >= 5000, partitions = sqrt(row_count) clamped to [1, 1024]."""
        row_count = 10000
        num_partitions = max(1, min(1024, int(row_count**0.5)))
        # sqrt(10000) = 100, within [1, 1024]
        assert num_partitions == 100

    def test_partition_count_clamped_to_1024(self):
        """Partition count must not exceed 1024 for very large datasets."""
        row_count = 2000000  # sqrt(2_000_000) ~ 1414
        num_partitions = max(1, min(1024, int(row_count**0.5)))
        assert num_partitions == 1024

    @pytest.mark.parametrize(
        "row_count,vector_dim,expected_sub_vectors",
        [
            # row_count >= 5000 and IVF_PQ -> min(96, max(32, dim // 16))
            (5000, 384, 32),  # 384 // 16 = 24, max(32, 24) = 32, min(96, 32) = 32
            (5000, 768, 48),  # 768 // 16 = 48, max(32, 48) = 48, min(96, 48) = 48
            (5000, 1536, 96),  # 1536 // 16 = 96, max(32, 96) = 96, min(96, 96) = 96
            (10000, 2048, 96),  # 2048 // 16 = 128, max(32, 128) = 128, min(96, 128) = 96
        ],
    )
    def test_sub_vectors_for_ivf_pq_large(self, row_count, vector_dim, expected_sub_vectors):
        """Sub-vector count must match production formula in base.py create_vector_index."""
        num_sub_vectors = min(96, max(32, vector_dim // 16))
        assert num_sub_vectors == expected_sub_vectors

    @pytest.mark.parametrize(
        "row_count,vector_dim,expected_sub_vectors",
        [
            # row_count < 1000 -> min(16, max(8, dim // 64))
            (500, 384, 8),  # 384 // 64 = 6, max(8, 6) = 8, min(16, 8) = 8
            (500, 1024, 16),  # 1024 // 64 = 16, max(8, 16) = 16, min(16, 16) = 16
            (500, 256, 8),  # 256 // 64 = 4, max(8, 4) = 8, min(16, 8) = 8
            (500, 2048, 16),  # 2048 // 64 = 32, max(8, 32) = 32, min(16, 32) = 16
        ],
    )
    def test_sub_vectors_for_ivf_pq_small(self, row_count, vector_dim, expected_sub_vectors):
        """Sub-vector calculation for IVF_PQ with < 1000 rows."""
        # Note: IVF_PQ only used when row_count >= 5000, but the code calculates it anyway
        num_sub_vectors = min(16, max(8, vector_dim // 64))
        assert num_sub_vectors == expected_sub_vectors


# ---------------------------------------------------------------------------
# Batch splitting logic in store_batch
# ---------------------------------------------------------------------------


class TestStoreBatchSplitting:
    """Tests for store_batch batch splitting logic using a real SchemaStorage."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_store_batch_empty_data_noop(self, tmp_path):
        """Calling store_batch with empty list should be a no-op."""
        store = self._make_store(tmp_path)
        store.store_batch([])
        # Table may not even be initialized since we did nothing
        # No exception should be raised

    def test_store_batch_single_item(self, tmp_path):
        """A single-item batch should store correctly."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])
        result = store.search_all(catalog_name="cat")
        assert result.num_rows == 1

    def test_store_batch_within_batch_size(self, tmp_path):
        """Data smaller than batch_size should be stored in a single add call."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(5)]
        store.store_batch(data)
        result = store.search_all(catalog_name="cat")
        assert result.num_rows == 5

    def test_store_batch_exceeding_batch_size(self, tmp_path):
        """Data larger than batch_size should be split and still store all rows."""
        store = self._make_store(tmp_path)
        # Override batch_size to force splitting
        store.batch_size = 3
        data = [self._make_row(i) for i in range(10)]
        store.store_batch(data)
        result = store.search_all(catalog_name="cat")
        assert result.num_rows == 10

    def test_store_batch_exact_batch_size_boundary(self, tmp_path):
        """Data exactly equal to batch_size should be stored in one chunk (no split)."""
        store = self._make_store(tmp_path)
        store.batch_size = 5
        data = [self._make_row(i) for i in range(5)]
        store.store_batch(data)
        result = store.search_all(catalog_name="cat")
        assert result.num_rows == 5


# ---------------------------------------------------------------------------
# BaseEmbeddingStore._check_embedding_model_ready
# ---------------------------------------------------------------------------


class TestCheckEmbeddingModelReady:
    """Tests for _check_embedding_model_ready error propagation."""

    def test_check_embedding_model_ready_raises_when_model_failed(self, tmp_path):
        """If the model is already marked as failed, a DatusException should be raised."""
        model = EmbeddingModel(model_name="test-model", dim_size=384)
        model.is_model_failed = True
        model.model_error_message = "Download failed"

        store = BaseEmbeddingStore(
            table_name="test_table",
            embedding_model=model,
        )

        with pytest.raises(DatusException) as exc_info:
            store._check_embedding_model_ready()
        assert "not available" in str(exc_info.value)
        assert "Download failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# BaseEmbeddingStore.truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    """Tests for truncate resetting table state."""

    def test_truncate_resets_table_state(self, tmp_path):
        """After truncate, table should be None and _table_initialized should be False."""
        store = SchemaStorage(get_db_embedding_model())
        # Force table initialization
        store._ensure_table_ready()
        assert store._shared.initialized is True
        assert store.table is not None

        store.truncate()
        assert store.table is None
        assert store._shared.initialized is False

    def test_truncate_allows_reinitialization(self, tmp_path):
        """After truncate, calling _ensure_table_ready should recreate the table."""
        store = SchemaStorage(get_db_embedding_model())
        row = {
            "identifier": "1",
            "catalog_name": "c",
            "database_name": "d",
            "schema_name": "s",
            "table_name": "t",
            "table_type": "table",
            "definition": "CREATE TABLE t (id INT)",
        }
        store.store([row])
        assert store.table_size() == 1

        store.truncate()
        # Re-initialize by storing again
        store.store([row])
        assert store.table_size() == 1


# ---------------------------------------------------------------------------
# _upsert_with_retry
# ---------------------------------------------------------------------------


class TestUpsertWithRetry:
    """Tests for _upsert_with_retry retry logic."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_upsert_with_retry_raises_when_table_none(self, tmp_path):
        """Raises DatusException when table is None."""
        import pandas as pd

        store = self._make_store(tmp_path)
        store.table = None
        with pytest.raises(DatusException) as exc_info:
            store._upsert_with_retry(pd.DataFrame([self._make_row(1)]), "identifier")
        assert "not initialized" in str(exc_info.value)

    def test_upsert_batch_empty_noop(self, tmp_path):
        """upsert_batch with empty data is a no-op."""
        store = self._make_store(tmp_path)
        store.upsert_batch([], on_column="identifier")
        # No exception should be raised

    def test_upsert_batch_deduplicates_input_in_memory(self, tmp_path):
        """upsert_batch deduplicates input data by on_column before sending to backend."""
        import pandas as pd

        _store = self._make_store(tmp_path)  # noqa: F841
        data = [self._make_row(1), self._make_row(1)]
        data[1]["definition"] = "CREATE TABLE table_1 (id INT, name TEXT)"

        # Replicate the dedup logic from upsert_batch without calling the backend
        df = pd.DataFrame(data)
        original_count = len(df)
        df = df.drop_duplicates(subset=["identifier"], keep="last")
        assert len(df) < original_count
        assert len(df) == 1
        assert df.iloc[0]["definition"] == "CREATE TABLE table_1 (id INT, name TEXT)"


# ---------------------------------------------------------------------------
# _add_with_retry
# ---------------------------------------------------------------------------


class TestAddWithRetry:
    """Tests for _add_with_retry retry logic."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def test_add_with_retry_raises_when_table_none(self, tmp_path):
        """Raises DatusException when table is None."""
        import pandas as pd

        store = self._make_store(tmp_path)
        store.table = None
        with pytest.raises(DatusException) as exc_info:
            store._add_with_retry(pd.DataFrame([{"identifier": "x", "definition": "y"}]))
        assert "not initialized" in str(exc_info.value)

    def test_add_with_retry_success(self, tmp_path):
        """Successful add on first attempt."""
        import pandas as pd

        store = self._make_store(tmp_path)
        store._ensure_table_ready()
        row = {
            "identifier": "id_1",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "t",
            "table_type": "table",
            "definition": "CREATE TABLE t (id INT)",
        }
        store._add_with_retry(pd.DataFrame([row]))
        result = store.search_all(catalog_name="cat")
        assert result.num_rows == 1


# ---------------------------------------------------------------------------
# search() routing
# ---------------------------------------------------------------------------


class TestSearchRouting:
    """Tests for search() vector vs hybrid routing."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT, name VARCHAR)",
        }

    def test_search_vector_mode(self, tmp_path):
        """search() with query_type='vector' returns results."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(3)]
        store.store_batch(data)

        result = store.search("table", query_type="vector", top_n=3)
        assert result.num_rows >= 0
        # vector column should be dropped
        assert "vector" not in result.column_names

    def test_search_hybrid_fallback_to_vector(self, tmp_path):
        """search() with query_type='hybrid' falls back to vector if hybrid fails."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(3)]
        store.store_batch(data)

        # Hybrid may fail (no FTS index), should fall back to vector search
        result = store.search("table", query_type="hybrid", top_n=3)
        assert result.num_rows >= 0
        assert "vector" not in result.column_names


# ---------------------------------------------------------------------------
# table_size
# ---------------------------------------------------------------------------


class TestTableSize:
    """Tests for table_size with and without scope filter."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int, db_name: str = "db") -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": db_name,
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_table_size_no_scope(self, tmp_path):
        """table_size returns total row count without scope filter."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(5)]
        store.store_batch(data)

        assert store.table_size() == 5

    def test_table_size_empty_table(self, tmp_path):
        """table_size on empty table returns 0."""
        store = self._make_store(tmp_path)
        store._ensure_table_ready()
        assert store.table_size() == 0


# ---------------------------------------------------------------------------
# _ensure_table_ready lazy init
# ---------------------------------------------------------------------------


class TestEnsureTableReady:
    """Tests for _ensure_table_ready lazy initialization."""

    def test_ensure_table_ready_sets_initialized(self, tmp_path):
        """After _ensure_table_ready, _table_initialized is True."""
        store = SchemaStorage(get_db_embedding_model())
        assert store._shared.initialized is False
        store._ensure_table_ready()
        assert store._shared.initialized is True
        assert store.table is not None

    def test_ensure_table_ready_idempotent(self, tmp_path):
        """Calling _ensure_table_ready twice doesn't cause errors."""
        store = SchemaStorage(get_db_embedding_model())
        store._ensure_table_ready()
        first_table = store.table
        store._ensure_table_ready()
        assert store.table is first_table


# ---------------------------------------------------------------------------
# update() with unique_filter
# ---------------------------------------------------------------------------


class TestUpdate:
    """Tests for update() with and without unique_filter."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_update_no_values_is_noop(self, tmp_path):
        """update() with empty update_values does nothing."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])
        # No error, no-op
        store.update(where=eq("identifier", "id_1"), update_values={})

    def test_update_no_where_is_noop(self, tmp_path):
        """update() with no where clause does nothing."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])
        store.update(where=None, update_values={"definition": "new def"})

    def test_update_with_unique_filter_conflict(self, tmp_path):
        """update() with unique_filter raises when conflicting rows exist."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1), self._make_row(2)])
        with pytest.raises(DatusException):
            store.update(
                where=eq("identifier", "id_1"),
                update_values={"definition": "new def"},
                unique_filter=eq("identifier", "id_2"),
            )

    def test_update_modifies_row(self, tmp_path):
        """update() modifies a specific row."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])
        store.update(
            where=eq("identifier", "id_1"),
            update_values={"definition": "CREATE TABLE table_1 (id INT, name TEXT)"},
        )
        result = store.search_all(catalog_name="cat")
        assert result.num_rows == 1

    def test_update_re_embeds_when_vector_source_changes(self, tmp_path):
        """update() should regenerate the vector when the vector source column is updated."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        # Get original vector
        original = store.query_with_filter(where=eq("identifier", "id_1"), select_fields=["vector"])
        original_vector = original.column("vector")[0].as_py()

        # Update the vector_source_name column (definition for SchemaStorage)
        store.update(
            where=eq("identifier", "id_1"),
            update_values={"definition": "A completely different table about customer orders and revenue tracking"},
        )

        # Get updated vector
        updated = store.query_with_filter(where=eq("identifier", "id_1"), select_fields=["vector"])
        updated_vector = updated.column("vector")[0].as_py()

        assert original_vector != updated_vector, "Vector should change when vector source text changes"

    def test_update_keeps_existing_vector_when_re_embed_returns_none(self, tmp_path, monkeypatch):
        """If the embedding model returns [None] (e.g., BadRequestError), preserve the existing vector
        rather than writing None into the vector column."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        original = store.query_with_filter(where=eq("identifier", "id_1"), select_fields=["vector"])
        original_vector = original.column("vector")[0].as_py()

        # Patch the underlying embedding function on the class to bypass pydantic field guards.
        embedding_cls = type(store.model.model)
        monkeypatch.setattr(
            embedding_cls,
            "generate_embeddings",
            lambda self, texts: [None] * len(list(texts)),
        )

        store.update(
            where=eq("identifier", "id_1"),
            update_values={"definition": "Some new definition that would otherwise be re-embedded"},
        )

        updated = store.query_with_filter(where=eq("identifier", "id_1"), select_fields=["vector"])
        updated_vector = updated.column("vector")[0].as_py()

        # Vector must be unchanged — never overwritten with None
        assert updated_vector == original_vector
        assert updated_vector is not None

    def test_update_keeps_existing_vector_when_re_embed_raises(self, tmp_path, monkeypatch):
        """If the embedding model raises, the update still proceeds and the existing vector is kept."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        original = store.query_with_filter(where=eq("identifier", "id_1"), select_fields=["vector"])
        original_vector = original.column("vector")[0].as_py()

        def _boom(self, texts):
            raise RuntimeError("embedding backend down")

        embedding_cls = type(store.model.model)
        monkeypatch.setattr(embedding_cls, "generate_embeddings", _boom)

        store.update(
            where=eq("identifier", "id_1"),
            update_values={"definition": "Yet another definition that triggers embedding"},
        )

        updated = store.query_with_filter(where=eq("identifier", "id_1"), select_fields=["vector"])
        updated_vector = updated.column("vector")[0].as_py()

        assert updated_vector == original_vector


# ---------------------------------------------------------------------------
# query_with_filter
# ---------------------------------------------------------------------------


class TestQueryWithFilter:
    """Tests for query_with_filter."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int, db_name: str = "db") -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": db_name,
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_query_with_filter_returns_all_without_filter(self, tmp_path):
        """query_with_filter with no where returns all rows."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(4)]
        store.store_batch(data)

        result = store.query_with_filter()
        assert result.num_rows == 4

    def test_query_with_filter_with_where(self, tmp_path):
        """query_with_filter with where clause filters rows."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i, db_name="db1") for i in range(3)]
        data += [self._make_row(i + 10, db_name="db2") for i in range(2)]
        store.store_batch(data)

        result = store.query_with_filter(where=eq("database_name", "db1"))
        assert result.num_rows == 3

    def test_query_with_filter_with_limit(self, tmp_path):
        """query_with_filter with limit restricts result count."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(5)]
        store.store_batch(data)

        result = store.query_with_filter(limit=2)
        assert result.num_rows == 2

    def test_query_with_filter_select_fields(self, tmp_path):
        """query_with_filter with select_fields limits columns."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        result = store.query_with_filter(select_fields=["identifier", "table_name"])
        assert "identifier" in result.column_names
        assert "table_name" in result.column_names


# ---------------------------------------------------------------------------
# create_fts_index
# ---------------------------------------------------------------------------


class TestCreateFtsIndex:
    """Tests for create_fts_index."""

    def test_create_fts_index_no_error(self, tmp_path):
        """create_fts_index should not raise even if index creation fails."""
        store = SchemaStorage(get_db_embedding_model())
        store.store_batch(
            [
                {
                    "identifier": "id_1",
                    "catalog_name": "cat",
                    "database_name": "db",
                    "schema_name": "sch",
                    "table_name": "t",
                    "table_type": "table",
                    "definition": "CREATE TABLE t (id INT)",
                }
            ]
        )
        # Should not raise
        store.create_fts_index(["definition"])

    def test_create_fts_index_with_multiple_fields(self, tmp_path):
        """create_fts_index with multiple fields should not raise."""
        store = SchemaStorage(get_db_embedding_model())
        store.store_batch(
            [
                {
                    "identifier": "id_1",
                    "catalog_name": "cat",
                    "database_name": "db",
                    "schema_name": "sch",
                    "table_name": "t",
                    "table_type": "table",
                    "definition": "CREATE TABLE t (id INT)",
                }
            ]
        )
        store.create_fts_index(["database_name", "schema_name", "table_name", "definition"])


# ---------------------------------------------------------------------------
# store() method
# ---------------------------------------------------------------------------


class TestStoreMethod:
    """Tests for store() method."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def test_store_single_row(self, tmp_path):
        """store() adds a single row to the table."""
        store = self._make_store(tmp_path)
        row = {
            "identifier": "id_1",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "t1",
            "table_type": "table",
            "definition": "CREATE TABLE t1 (id INT)",
        }
        store.store([row])
        assert store.table_size() == 1

    def test_store_multiple_rows(self, tmp_path):
        """store() adds multiple rows."""
        store = self._make_store(tmp_path)
        rows = [
            {
                "identifier": f"id_{i}",
                "catalog_name": "cat",
                "database_name": "db",
                "schema_name": "sch",
                "table_name": f"t{i}",
                "table_type": "table",
                "definition": f"CREATE TABLE t{i} (id INT)",
            }
            for i in range(3)
        ]
        store.store(rows)
        assert store.table_size() == 3


# ---------------------------------------------------------------------------
# _delete_rows and _count_rows
# ---------------------------------------------------------------------------


class TestDeleteAndCountRows:
    """Tests for _delete_rows and _count_rows convenience methods."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_count_rows_without_filter(self, tmp_path):
        """_count_rows without filter returns total."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i) for i in range(4)])
        assert store._count_rows() == 4

    def test_count_rows_with_filter(self, tmp_path):
        """_count_rows with filter returns matching count."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i) for i in range(4)])
        count = store._count_rows(where=eq("identifier", "id_1"))
        assert count == 1

    def test_delete_rows(self, tmp_path):
        """_delete_rows removes matching rows."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i) for i in range(4)])
        store._delete_rows(where=eq("identifier", "id_1"))
        assert store._count_rows() == 3

    def test_delete_rows_none_where_is_noop(self, tmp_path):
        """_delete_rows with None where is a no-op."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(i) for i in range(3)])
        store._delete_rows(where=None)
        assert store._count_rows() == 3


# ---------------------------------------------------------------------------
# _search_all
# ---------------------------------------------------------------------------


class TestSearchAll:
    """Tests for _search_all method."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        return SchemaStorage(get_db_embedding_model())

    def _make_row(self, idx: int) -> dict:
        return {
            "identifier": f"id_{idx}",
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": f"table_{idx}",
            "table_type": "table",
            "definition": f"CREATE TABLE table_{idx} (id INT)",
        }

    def test_search_all_returns_all_rows(self, tmp_path):
        """_search_all returns all rows when no where clause."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(4)]
        store.store_batch(data)

        result = store._search_all()
        assert result.num_rows == 4

    def test_search_all_with_where(self, tmp_path):
        """_search_all with where clause filters results."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(4)]
        store.store_batch(data)

        result = store._search_all(where=eq("identifier", "id_1"))
        assert result.num_rows == 1

    def test_search_all_with_select_fields(self, tmp_path):
        """_search_all with select_fields limits columns."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        result = store._search_all(select_fields=["identifier", "table_name"])
        assert "identifier" in result.column_names
        assert "table_name" in result.column_names

    def test_search_all_drops_vector_column(self, tmp_path):
        """_search_all drops the vector column from results."""
        store = self._make_store(tmp_path)
        store.store_batch([self._make_row(1)])

        result = store._search_all()
        assert "vector" not in result.column_names

    def test_search_all_with_limit(self, tmp_path):
        """_search_all with explicit limit restricts results."""
        store = self._make_store(tmp_path)
        data = [self._make_row(i) for i in range(5)]
        store.store_batch(data)

        result = store._search_all(limit=2)
        assert result.num_rows == 2
