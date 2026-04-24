# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/document/store.py -- DocumentStore."""

from unittest.mock import patch

import pytest

from datus.storage.document.schemas import PlatformDocChunk
from datus.storage.document.store import (
    DocumentStore,
    document_store,
    get_platform_doc_schema,
    list_indexed_platforms,
)
from datus.storage.embedding_models import get_document_embedding_model
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    doc_path: str = "docs/guide/intro.md",
    chunk_index: int = 0,
    version: str = "v1.0.0",
    chunk_text: str = "This is a test chunk about SQL syntax and data loading.",
    title: str = "Introduction",
    source_url: str = "https://example.com/docs/intro",
) -> PlatformDocChunk:
    """Build a PlatformDocChunk for testing."""
    chunk_id = PlatformDocChunk.generate_chunk_id(doc_path, chunk_index, version)
    return PlatformDocChunk(
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        title=title,
        titles=[title],
        nav_path=["Guides", "User Guide"],
        group_name="Guides",
        hierarchy=f"Guides > User Guide > {title}",
        version=version,
        source_type="github",
        source_url=source_url,
        doc_path=doc_path,
        keywords=["sql", "data"],
        language="en",
        content_hash="abc123",
    )


def _make_chunks(count: int, version: str = "v1.0.0", doc_path: str = "docs/guide/intro.md") -> list:
    """Build multiple test chunks from a single doc."""
    return [
        _make_chunk(
            doc_path=doc_path,
            chunk_index=i,
            version=version,
            chunk_text=f"Chunk {i} content about topic {i} and database operations.",
            title=f"Section {i}",
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def doc_store(tmp_path) -> DocumentStore:
    """Create a DocumentStore with real vector backend."""
    return DocumentStore(embedding_model=get_document_embedding_model())


# ============================================================
# get_platform_doc_schema
# ============================================================


class TestGetPlatformDocSchema:
    """Tests for the get_platform_doc_schema function."""

    def test_default_embedding_dim(self):
        """Default schema should use 384-dim embedding."""
        schema = get_platform_doc_schema()
        vec_field = schema.field("vector")
        assert vec_field is not None
        assert vec_field.type.list_size == 384

    def test_custom_embedding_dim(self):
        """Schema should accept a custom embedding dimension."""
        schema = get_platform_doc_schema(embedding_dim=768)
        vec_field = schema.field("vector")
        assert vec_field.type.list_size == 768

    def test_schema_has_required_fields(self):
        """Schema should include all required document fields."""
        schema = get_platform_doc_schema()
        expected_fields = {
            "chunk_id",
            "chunk_text",
            "chunk_index",
            "title",
            "titles",
            "nav_path",
            "group_name",
            "hierarchy",
            "version",
            "source_type",
            "source_url",
            "doc_path",
            "keywords",
            "language",
            "created_at",
            "updated_at",
            "content_hash",
            "vector",
        }
        schema_names = set(schema.names)
        for field in expected_fields:
            assert field in schema_names, f"Field '{field}' missing from schema"


# ============================================================
# DocumentStore.__init__
# ============================================================


class TestDocumentStoreInit:
    """Tests for DocumentStore initialization."""

    def test_table_name(self, doc_store):
        """Table name should be 'document'."""
        assert doc_store.table_name == "document"
        assert doc_store.TABLE_NAME == "document"

    def test_vector_source_name(self, doc_store):
        """Vector source should be 'chunk_text'."""
        assert doc_store.vector_source_name == "chunk_text"

    def test_vector_column_name(self, doc_store):
        """Vector column should be 'vector'."""
        assert doc_store.vector_column_name == "vector"

    def test_on_duplicate_columns(self, doc_store):
        """Deduplication column should be 'chunk_id'."""
        assert doc_store.on_duplicate_columns == "chunk_id"


# ============================================================
# DocumentStore.store_chunks
# ============================================================


class TestDocumentStoreStoreChunks:
    """Tests for store_chunks with deduplication."""

    def test_store_chunks_empty(self, doc_store):
        """Storing empty list should return 0."""
        count = doc_store.store_chunks([])
        assert count == 0

    def test_store_chunks_single(self, doc_store):
        """Storing a single chunk should return 1."""
        chunk = _make_chunk()
        count = doc_store.store_chunks([chunk])
        assert count == 1

    def test_store_chunks_multiple(self, doc_store):
        """Storing multiple chunks should return the correct count."""
        chunks = _make_chunks(5)
        count = doc_store.store_chunks(chunks)
        assert count == 5

    def test_store_chunks_deduplication(self, doc_store):
        """Storing same chunks twice should not create duplicates."""
        chunks = _make_chunks(3)
        doc_store.store_chunks(chunks)

        # Store the same chunks again -- dedup should handle it
        doc_store.store_chunks(chunks)

        # Verify total count is still 3 (not 6)
        stats = doc_store.get_stats()
        assert stats["total_chunks"] == 3

    def test_store_chunks_different_versions(self, doc_store):
        """Chunks from different versions should be stored separately."""
        chunks_v1 = _make_chunks(2, version="v1.0.0")
        chunks_v2 = _make_chunks(2, version="v2.0.0")

        doc_store.store_chunks(chunks_v1)
        doc_store.store_chunks(chunks_v2)

        stats = doc_store.get_stats()
        assert stats["total_chunks"] == 4
        assert set(stats["versions"]) == {"v1.0.0", "v2.0.0"}


# ============================================================
# DocumentStore.search_docs
# ============================================================


class TestDocumentStoreSearchDocs:
    """Tests for search_docs with version filtering."""

    @pytest.fixture(autouse=True)
    def _populate(self, doc_store):
        """Populate store with test data across two versions."""
        self.store = doc_store
        chunks_v1 = _make_chunks(3, version="v1.0.0")
        chunks_v2 = _make_chunks(2, version="v2.0.0", doc_path="docs/api/ref.md")
        doc_store.store_chunks(chunks_v1)
        doc_store.store_chunks(chunks_v2)

    def test_search_docs_no_filter(self):
        """Search without version filter returns results from all versions."""
        results = self.store.search_docs("database operations", top_n=10)
        assert len(results) > 0

    def test_search_docs_with_version_filter(self):
        """Search with version filter returns only matching version."""
        results = self.store.search_docs("database operations", version="v1.0.0", top_n=10)
        for r in results:
            assert r["version"] == "v1.0.0"

    def test_search_docs_with_select_fields(self):
        """Search with select_fields returns only requested fields."""
        results = self.store.search_docs(
            "database operations",
            top_n=5,
            select_fields=["chunk_text", "title", "version"],
        )
        assert len(results) > 0
        for r in results:
            assert "chunk_text" in r
            assert "title" in r
            assert "version" in r

    def test_search_docs_top_n_limit(self):
        """top_n should limit the number of results."""
        results = self.store.search_docs("database operations", top_n=2)
        assert len(results) <= 2


# ============================================================
# DocumentStore.list_versions
# ============================================================


class TestDocumentStoreListVersions:
    """Tests for list_versions."""

    def test_list_versions_empty(self, doc_store):
        """Empty store should return empty list."""
        versions = doc_store.list_versions()
        assert versions == []

    def test_list_versions_single(self, doc_store):
        """Store with one version should return it."""
        doc_store.store_chunks(_make_chunks(2, version="v1.0.0"))
        versions = doc_store.list_versions()
        assert len(versions) == 1
        assert versions[0]["version"] == "v1.0.0"
        assert versions[0]["chunk_count"] == 2

    def test_list_versions_multiple(self, doc_store):
        """Store with multiple versions should return all sorted."""
        doc_store.store_chunks(_make_chunks(3, version="v1.0.0"))
        doc_store.store_chunks(_make_chunks(2, version="v2.0.0", doc_path="docs/api/ref.md"))
        versions = doc_store.list_versions()
        assert len(versions) == 2
        version_names = [v["version"] for v in versions]
        assert version_names == sorted(version_names)


# ============================================================
# DocumentStore.get_stats / get_stats_by_version
# ============================================================


class TestDocumentStoreStats:
    """Tests for get_stats and get_stats_by_version."""

    def test_get_stats_empty(self, doc_store):
        """Empty store should return zero stats."""
        stats = doc_store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["versions"] == []
        assert stats["doc_count"] == 0

    def test_get_stats_with_data(self, doc_store):
        """Stats should reflect stored data accurately."""
        doc_store.store_chunks(_make_chunks(3, version="v1.0.0"))
        doc_store.store_chunks(_make_chunks(2, version="v2.0.0", doc_path="docs/api/ref.md"))
        stats = doc_store.get_stats()
        assert stats["total_chunks"] == 5
        assert set(stats["versions"]) == {"v1.0.0", "v2.0.0"}
        assert stats["doc_count"] == 2
        assert stats["latest_update"] is not None

    def test_get_stats_by_version(self, doc_store):
        """Version-specific stats should match filtered data."""
        doc_store.store_chunks(_make_chunks(3, version="v1.0.0"))
        doc_store.store_chunks(_make_chunks(2, version="v2.0.0", doc_path="docs/api/ref.md"))

        stats_v1 = doc_store.get_stats_by_version("v1.0.0")
        assert stats_v1["total_chunks"] == 3
        assert stats_v1["doc_count"] == 1

        stats_v2 = doc_store.get_stats_by_version("v2.0.0")
        assert stats_v2["total_chunks"] == 2
        assert stats_v2["doc_count"] == 1

    def test_get_stats_by_version_nonexistent(self, doc_store):
        """Querying non-existent version should return zero stats."""
        doc_store.store_chunks(_make_chunks(2, version="v1.0.0"))
        stats = doc_store.get_stats_by_version("v99.0.0")
        assert stats["total_chunks"] == 0
        assert stats["doc_count"] == 0


# ============================================================
# DocumentStore._validate_identifier
# ============================================================


class TestValidateIdentifier:
    """Tests for _validate_identifier SQL injection prevention."""

    def test_valid_simple_string(self):
        """Simple alphanumeric string should pass."""
        DocumentStore._validate_identifier("v1.0.0", "version")

    def test_valid_with_hyphens(self):
        """String with hyphens should pass."""
        DocumentStore._validate_identifier("v1-beta-2", "version")

    def test_valid_with_underscores(self):
        """String with underscores should pass."""
        DocumentStore._validate_identifier("version_1_0", "version")

    def test_valid_with_spaces(self):
        """String with spaces should pass."""
        DocumentStore._validate_identifier("version 1", "version")

    def test_valid_with_dots(self):
        """String with dots should pass."""
        DocumentStore._validate_identifier("v1.2.3", "version")

    def test_invalid_with_semicolon(self):
        """String with semicolon should raise DatusException."""
        with pytest.raises(DatusException, match="Invalid version"):
            DocumentStore._validate_identifier("v1; DROP TABLE", "version")

    def test_invalid_with_quotes(self):
        """String with quotes should raise DatusException."""
        with pytest.raises(DatusException, match="Invalid version"):
            DocumentStore._validate_identifier("v1' OR '1'='1", "version")

    def test_invalid_with_parentheses(self):
        """String with parentheses should raise DatusException."""
        with pytest.raises(DatusException, match="Invalid version"):
            DocumentStore._validate_identifier("v1()", "version")


# ============================================================
# DocumentStore.delete_docs
# ============================================================


class TestDocumentStoreDeleteDocs:
    """Tests for delete_docs with version-specific and full deletion."""

    def test_delete_docs_empty_store(self, doc_store):
        """Deleting from empty store should return 0."""
        count = doc_store.delete_docs(version="v1.0.0")
        assert count == 0

    def test_delete_docs_by_version(self, doc_store):
        """Deleting a specific version should remove only those chunks."""
        doc_store.store_chunks(_make_chunks(3, version="v1.0.0"))
        doc_store.store_chunks(_make_chunks(2, version="v2.0.0", doc_path="docs/api/ref.md"))

        deleted = doc_store.delete_docs(version="v1.0.0")
        assert deleted == 3

        stats = doc_store.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["versions"] == ["v2.0.0"]

    def test_delete_docs_invalid_version_raises(self, doc_store):
        """Deleting with unsafe version string should raise DatusException."""
        doc_store.store_chunks(_make_chunks(1, version="v1.0.0"))
        with pytest.raises(DatusException, match="Invalid version"):
            doc_store.delete_docs(version="v1; DROP TABLE docs")

    def test_delete_docs_nonexistent_version(self, doc_store):
        """Deleting non-existent version should return 0 deleted."""
        doc_store.store_chunks(_make_chunks(2, version="v1.0.0"))
        deleted = doc_store.delete_docs(version="v99.0.0")
        assert deleted == 0
        assert doc_store.get_stats()["total_chunks"] == 2


# ============================================================
# DocumentStore.create_indices
# ============================================================


class TestDocumentStoreCreateIndices:
    """Tests for create_indices."""

    def test_create_indices_after_data(self, doc_store):
        """Creating indices after storing data should not raise."""
        doc_store.store_chunks(_make_chunks(3))
        doc_store.create_indices()
        # Verify search still works
        results = doc_store.search_docs("database operations", top_n=2)
        assert len(results) > 0

    def test_create_indices_calls_vector_index(self, doc_store):
        """create_indices must delegate to the backend's create_vector_index."""
        doc_store.store_chunks(_make_chunks(3))
        with patch.object(doc_store.table, "create_vector_index", wraps=doc_store.table.create_vector_index) as mock_vi:
            doc_store.create_indices()
            mock_vi.assert_called_once()
            args, kwargs = mock_vi.call_args
            # First positional arg is the vector column name
            assert args[0] == "vector"
            assert kwargs.get("metric") == "cosine"
            assert kwargs.get("replace") is True

    def test_create_indices_calls_fts_index(self, doc_store):
        """create_indices must delegate to the backend's create_fts_index."""
        doc_store.store_chunks(_make_chunks(3))
        with patch.object(doc_store.table, "create_fts_index", wraps=doc_store.table.create_fts_index) as mock_fts:
            doc_store.create_indices()
            mock_fts.assert_called_once()
            args, kwargs = mock_fts.call_args
            # field_names is passed positionally from base.create_fts_index
            field_names = kwargs.get("field_names") or args[0]
            assert set(field_names) == {"chunk_text", "title", "hierarchy"}


# ============================================================
# DocumentStore.get_all_rows
# ============================================================


class TestDocumentStoreGetAllRows:
    """Tests for get_all_rows."""

    def test_get_all_rows_empty(self, doc_store):
        """Empty store should return empty list."""
        results = doc_store.get_all_rows()
        assert results == []

    def test_get_all_rows_with_data(self, doc_store):
        """get_all_rows should return all stored chunks."""
        doc_store.store_chunks(_make_chunks(3))
        results = doc_store.get_all_rows()
        assert len(results) == 3

    def test_get_all_rows_with_select_fields(self, doc_store):
        """get_all_rows should respect select_fields."""
        doc_store.store_chunks(_make_chunks(2))
        results = doc_store.get_all_rows(select_fields=["chunk_text", "version"])
        assert len(results) == 2
        for r in results:
            assert "chunk_text" in r
            assert "version" in r


# ============================================================
# DocumentStore.has_data
# ============================================================


class TestDocumentStoreHasData:
    """Tests for has_data method."""

    def test_has_data_empty_store(self, doc_store):
        """Empty store should return False."""
        assert doc_store.has_data() is False

    def test_has_data_with_data(self, doc_store):
        """Store with chunks should return True."""
        doc_store.store_chunks(_make_chunks(2))
        assert doc_store.has_data() is True

    def test_has_data_exception_returns_false(self, doc_store):
        """has_data should return False when an exception occurs."""
        with patch.object(doc_store, "db") as mock_db:
            mock_db.table_exists.side_effect = RuntimeError("connection lost")
            assert doc_store.has_data() is False

    def test_has_data_count_rows_exception_returns_false(self, doc_store):
        """has_data should return False when table.count_rows raises."""
        doc_store.store_chunks(_make_chunks(1))
        with patch.object(doc_store.table, "count_rows", side_effect=RuntimeError("broken")):
            assert doc_store.has_data() is False


# ============================================================
# document_store factory
# ============================================================


class TestDocumentStoreFactory:
    """Tests for the document_store() factory function."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Clear lru_cache before each test."""
        document_store.cache_clear()
        yield
        document_store.cache_clear()

    def test_valid_platform(self):
        """Valid platform name should return a DocumentStore."""
        store = document_store("test_factory_valid")
        assert isinstance(store, DocumentStore)

    def test_invalid_platform_empty(self):
        """Empty platform should raise DatusException."""
        with pytest.raises(DatusException, match="Invalid platform name"):
            document_store("")

    def test_invalid_platform_special_chars(self):
        """Platform with spaces/dots should raise DatusException."""
        with pytest.raises(DatusException, match="Invalid platform name"):
            document_store("my platform.v1")

    def test_invalid_platform_sql_injection(self):
        """Platform with SQL injection attempt should raise DatusException."""
        with pytest.raises(DatusException, match="Invalid platform name"):
            document_store("test; DROP TABLE")


class TestListIndexedPlatforms:
    """Filesystem-scan based enumeration used by PlatformDocSearchTool.available_tools."""

    @staticmethod
    def _patch_environment(project_name, data_dir):
        path_manager = type("PM", (), {"project_name": project_name})()
        return (
            patch("datus.utils.path_manager.get_path_manager", return_value=path_manager),
            patch("datus.storage.backend_holder._data_dir", data_dir),
        )

    def test_empty_when_no_project_name(self, tmp_path):
        pm_patch, dir_patch = self._patch_environment("", str(tmp_path))
        with pm_patch, dir_patch:
            assert list_indexed_platforms() == []

    def test_empty_when_data_dir_blank(self):
        pm_patch, dir_patch = self._patch_environment("myproj", "")
        with pm_patch, dir_patch:
            assert list_indexed_platforms() == []

    def test_empty_when_data_dir_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        pm_patch, dir_patch = self._patch_environment("myproj", str(missing))
        with pm_patch, dir_patch:
            assert list_indexed_platforms() == []

    def test_returns_platforms_from_matching_dirs(self, tmp_path):
        (tmp_path / "myproj__docstore__duckdb").mkdir()
        (tmp_path / "myproj__docstore__snowflake").mkdir()
        (tmp_path / "other_project__docstore__pg").mkdir()
        (tmp_path / "myproj__something_else").mkdir()
        (tmp_path / "myproj__docstore__").mkdir()

        pm_patch, dir_patch = self._patch_environment("myproj", str(tmp_path))
        with pm_patch, dir_patch:
            platforms = list_indexed_platforms()

        assert platforms == ["duckdb", "snowflake"]

    def test_empty_when_no_matching_dirs(self, tmp_path):
        (tmp_path / "unrelated_dir").mkdir()
        pm_patch, dir_patch = self._patch_environment("myproj", str(tmp_path))
        with pm_patch, dir_patch:
            assert list_indexed_platforms() == []
