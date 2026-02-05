# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Storage Module

Provides vector storage for documents using LanceDB with full-featured schema:
- Version tracking
- Navigation path (nav_path, group_name, hierarchy)
- Titles and keywords extraction
- Deduplication via chunk_id
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.storage.base import BaseEmbeddingStore
from datus.storage.document.schemas import PlatformDocChunk
from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.storage.lancedb_conditions import And, Condition, WhereExpr, eq
from datus.utils.loggings import get_logger

# Validation pattern for version strings to prevent SQL injection
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-. ]+$")

logger = get_logger(__name__)

# =============================================================================
# LanceDB Schema
# =============================================================================


def get_platform_doc_schema(embedding_dim: int = 384) -> pa.Schema:
    """Get PyArrow schema for platform documentation table.

    Args:
        embedding_dim: Dimension of the embedding vector

    Returns:
        PyArrow schema for the table
    """
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("chunk_text", pa.string()),  # Source field for embedding
            pa.field("chunk_index", pa.int32()),
            pa.field("title", pa.string()),
            pa.field("titles", pa.list_(pa.string())),  # Page-internal headings
            pa.field("nav_path", pa.list_(pa.string())),  # Site navigation path
            pa.field("group_name", pa.string()),  # Top-level group
            pa.field("hierarchy", pa.string()),  # Full combined path
            pa.field("version", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("doc_path", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("language", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=embedding_dim)),
        ]
    )


class DocumentStore(BaseEmbeddingStore):
    """Vector store for documentation with full-featured schema.

    Each platform has its own DocumentStore instance (one LanceDB per platform).

    Features:
    - Semantic search with vector embeddings
    - Filtering by version
    - Full-text search on chunk_text and keywords
    - Upsert with deduplication on chunk_id
    - Navigation tracking (titles, nav_path, group_name, hierarchy)

    Example:
        >>> store = DocumentStore(db_path, embedding_model)
        >>> store.store_chunks(chunks)
        >>> results = store.search_docs("CREATE TABLE syntax")
    """

    TABLE_NAME = "document"

    def __init__(
        self,
        db_path: str,
        embedding_model: EmbeddingModel,
    ):
        """Initialize the document store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model for vectorization
        """
        schema = get_platform_doc_schema(embedding_model.dim_size)
        super().__init__(
            db_path=db_path,
            table_name=self.TABLE_NAME,
            embedding_model=embedding_model,
            vector_source_name="chunk_text",
            vector_column_name="vector",
            on_duplicate_columns="chunk_id",
            schema=schema,
        )

    def store_chunks(self, chunks: List[PlatformDocChunk]) -> int:
        """Store documentation chunks with automatic embedding.

        Uses delete-then-add instead of merge_insert to avoid lance 0.22.0
        merge_insert panics. Deduplication is handled by removing existing
        chunks with matching chunk_ids before inserting.

        Args:
            chunks: List of PlatformDocChunk objects to store

        Returns:
            Number of chunks stored
        """
        if not chunks:
            return 0

        data = [chunk.to_dict() for chunk in chunks]

        # Delete existing chunks with matching chunk_ids to handle deduplication,
        # then use store_batch (table.add) which is stable in lance 0.22.0.
        # This avoids merge_insert which has known Rust-level panics.
        self._ensure_table_ready()
        if self.table:
            try:
                row_count = self.table.count_rows()
            except Exception:
                row_count = 0

            if row_count > 0:
                chunk_ids = [c.chunk_id for c in chunks]
                # Delete in batches to avoid overly long WHERE clauses
                batch_size = 500
                for i in range(0, len(chunk_ids), batch_size):
                    batch_ids = chunk_ids[i : i + batch_size]
                    id_list = ", ".join(f"'{cid}'" for cid in batch_ids)
                    try:
                        self.table.delete(f"chunk_id IN ({id_list})")
                    except Exception:
                        pass  # Ignore if chunks don't exist yet

        self.store_batch(data)

        logger.info(f"Stored {len(chunks)} chunks, version '{chunks[0].version}'")
        return len(chunks)

    def search_docs(
        self,
        query: str,
        version: Optional[str] = None,
        top_n: int = 10,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documentation by semantic similarity.

        Args:
            query: Search query text
            version: Filter by version (e.g., "v1.2.3")
            top_n: Maximum number of results to return
            select_fields: Fields to include in results (default: all)

        Returns:
            List of matching chunks as dictionaries
        """
        conditions: List[Condition] = []

        if version:
            conditions.append(eq("version", version))

        where: WhereExpr = None
        if len(conditions) > 1:
            where = And(conditions)
        elif len(conditions) == 1:
            where = conditions[0]

        results = self.search(
            query_txt=query,
            top_n=top_n,
            where=where,
            select_fields=select_fields,
        )

        return results.to_pylist()

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all indexed versions with chunk counts.

        Returns:
            List of dicts with version and chunk_count
        """
        self._ensure_table_ready()

        all_data = self._search_all(
            select_fields=["version"],
        )

        version_counts: Dict[str, int] = {}
        for row in all_data.to_pylist():
            version = row["version"]
            version_counts[version] = version_counts.get(version, 0) + 1

        return [{"version": ver, "chunk_count": count} for ver, count in sorted(version_counts.items())]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this document store.

        Returns:
            Dict with versions, total_chunks, doc_count, etc.
        """
        self._ensure_table_ready()

        all_data = self._search_all(
            select_fields=["version", "doc_path", "created_at"],
        )

        rows = all_data.to_pylist()

        if not rows:
            return {
                "total_chunks": 0,
                "versions": [],
                "doc_count": 0,
            }

        versions = set()
        doc_paths = set()
        latest_update = None

        for row in rows:
            versions.add(row["version"])
            doc_paths.add(row["doc_path"])
            created = row.get("created_at")
            if created and (latest_update is None or created > latest_update):
                latest_update = created

        return {
            "total_chunks": len(rows),
            "versions": sorted(versions),
            "doc_count": len(doc_paths),
            "latest_update": latest_update,
        }

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        """Validate a string to prevent SQL injection.

        Args:
            value: String to validate
            name: Parameter name for error messages

        Raises:
            ValueError: If the string contains unsafe characters
        """
        if not _SAFE_IDENTIFIER_RE.match(value):
            raise ValueError(
                f"Invalid {name}: '{value}'. "
                f"Only alphanumeric characters, underscores, hyphens, dots, and spaces are allowed."
            )

    def delete_docs(
        self,
        version: Optional[str] = None,
    ) -> int:
        """Delete documentation chunks with physical file cleanup.

        Args:
            version: If specified, only delete this version (with compaction
                     to reclaim disk space); otherwise physically remove the
                     entire LanceDB directory and reinitialize.

        Returns:
            Number of chunks deleted

        Raises:
            ValueError: If version contains unsafe characters
        """
        self._ensure_table_ready()

        count_before = self.table.count_rows()
        if count_before == 0:
            logger.info(f"No chunks exists for version '{version or 'all'}'")
            return 0

        if version:
            self._validate_identifier(version, "version")
            where_clause = f"version = '{version}'"
            self.table.delete(where_clause)
            # Compact and remove old data files to reclaim disk space
            try:
                self.table.compact_files()
                self.table.cleanup_old_versions()
            except Exception as e:
                logger.warning(f"Post-delete cleanup failed (non-fatal): {e}")
            # Calculate actual deleted count
            count_after = self.table.count_rows()
            deleted_count = count_before - count_after
        else:
            # Physically remove the entire lance directory and reinitialize.
            # This is more thorough than drop_table which leaves orphan files.
            import shutil

            import lancedb

            shutil.rmtree(self.db_path, ignore_errors=True)
            self.db = lancedb.connect(self.db_path)
            self._table_initialized = False
            self._ensure_table_ready()
            deleted_count = count_before

        logger.info(f"Deleted {deleted_count} chunks for version '{version or 'all'}'")
        return deleted_count

    def get_all_rows(
        self,
        where: WhereExpr = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all rows matching a condition.

        Public wrapper around _search_all for external consumers.

        Args:
            where: Filter condition (tuple or list of tuples)
            select_fields: Fields to include in results

        Returns:
            List of matching rows as dictionaries
        """
        self._ensure_table_ready()
        results = self._search_all(where=where, select_fields=select_fields)
        return results.to_pylist()

    def create_indices(self):
        """Create optimized indices for the table.

        Creates:
        - Vector index for semantic search
        - FTS index for keyword search
        """
        self._ensure_table_ready()

        self.create_vector_index(metric="cosine")
        self.create_fts_index(field_names=["chunk_text", "title", "hierarchy"])

        logger.info(f"Created indices for table '{self.TABLE_NAME}'")


# =============================================================================
# Factory functions
# =============================================================================


@lru_cache(maxsize=8)
def document_store(storage_path: str) -> DocumentStore:
    """Get a cached DocumentStore instance.

    Args:
        storage_path: Path to LanceDB database

    Returns:
        Cached DocumentStore instance
    """
    return DocumentStore(storage_path, get_document_embedding_model())
