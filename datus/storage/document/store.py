# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Storage Module

Provides vector storage for documents with full-featured schema:
- Version tracking
- Navigation path (nav_path, group_name, hierarchy)
- Titles and keywords extraction
- Deduplication via chunk_id
"""

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa

if TYPE_CHECKING:
    from datus_storage_base.vector.base import VectorDatabase

from datus_storage_base.conditions import And, Condition, WhereExpr, eq, in_

from datus.storage.base import BaseEmbeddingStore
from datus.storage.document.schemas import PlatformDocChunk
from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

# Validation pattern for version strings to prevent SQL injection
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-. ]+$")

logger = get_logger(__name__)

# =============================================================================
# Vector Store Schema
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

    Each platform has its own DocumentStore instance.

    Features:
    - Semantic search with vector embeddings
    - Filtering by version
    - Full-text search on chunk_text and keywords
    - Upsert with deduplication on chunk_id
    - Navigation tracking (titles, nav_path, group_name, hierarchy)

    Example:
        >>> store = DocumentStore(embedding_model)
        >>> store.store_chunks(chunks)
        >>> results = store.search_docs("CREATE TABLE syntax")
    """

    TABLE_NAME = "document"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        db: Optional["VectorDatabase"] = None,
    ):
        """Initialize the document store.

        Args:
            embedding_model: Embedding model for vectorization
            db: Optional pre-created VectorDatabase for per-platform isolation.
        """
        schema = get_platform_doc_schema(embedding_model.dim_size)
        super().__init__(
            table_name=self.TABLE_NAME,
            embedding_model=embedding_model,
            vector_source_name="chunk_text",
            vector_column_name="vector",
            on_duplicate_columns="chunk_id",
            schema=schema,
            db=db,
        )

    def has_data(self) -> bool:
        """Return True if the document table exists and contains rows.

        This method avoids booting the embedding model by opening the
        table directly through the vector backend instead of calling
        ``_count_rows()`` (which triggers ``_ensure_table_ready()``).
        """
        try:
            if not self.db.table_exists(self.TABLE_NAME):
                return False
            table = self.table or self.db.open_table(self.TABLE_NAME)
            return table.count_rows() > 0
        except Exception as e:
            logger.debug(f"has_data() check failed for table '{self.TABLE_NAME}': {e}")
            return False

    def store_chunks(self, chunks: List[PlatformDocChunk]) -> int:
        """Store documentation chunks with automatic embedding.

        Uses delete-then-add instead of merge_insert to avoid known
        merge_insert issues. Deduplication is handled by removing existing
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
        # then use store_batch (table.add) which is stable.
        # This avoids merge_insert which has known issues.
        self._ensure_table_ready()
        if self.table:
            try:
                row_count = self._count_rows()
            except Exception:
                row_count = 0

            if row_count > 0:
                chunk_ids = [c.chunk_id for c in chunks]
                # Delete in batches to avoid overly long WHERE clauses
                batch_size = 500
                for i in range(0, len(chunk_ids), batch_size):
                    batch_ids = chunk_ids[i : i + batch_size]
                    try:
                        self.table.delete(in_("chunk_id", batch_ids))
                    except Exception as e:
                        raise DatusException(
                            ErrorCode.STORAGE_FAILED,
                            message=f"Failed to delete existing chunks during deduplication: {e}",
                        ) from e

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

    def get_stats_by_version(self, version: str) -> Dict[str, Any]:
        """Get statistics for a specific version.

        Args:
            version: Version string to filter by

        Returns:
            Dict with total_chunks, doc_count for this version
        """
        self._ensure_table_ready()
        self._validate_identifier(version, "version")

        all_data = self._search_all(
            where=eq("version", version),
            select_fields=["doc_path"],
        )

        rows = all_data.to_pylist()
        doc_paths = {row["doc_path"] for row in rows}

        return {
            "total_chunks": len(rows),
            "doc_count": len(doc_paths),
        }

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        """Validate a string to prevent injection.

        Args:
            value: String to validate
            name: Parameter name for error messages

        Raises:
            DatusException: If the string contains unsafe characters
        """
        if not _SAFE_IDENTIFIER_RE.match(value):
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Invalid {name}: '{value}'. "
                f"Only alphanumeric characters, underscores, hyphens, dots, and spaces are allowed.",
            )

    def delete_docs(
        self,
        version: Optional[str] = None,
    ) -> int:
        """Delete documentation chunks with physical file cleanup.

        Args:
            version: If specified, only delete this version (with compaction
                     to reclaim disk space); otherwise physically remove the
                     entire storage directory and reinitialize.

        Returns:
            Number of chunks deleted

        Raises:
            ValueError: If version contains unsafe characters
        """
        self._ensure_table_ready()

        count_before = self._count_rows()
        if count_before == 0:
            logger.info(f"No chunks exists for version '{version or 'all'}'")
            return 0

        if version:
            self._validate_identifier(version, "version")
            self.table.delete(eq("version", version))
            # Compact and remove old data files to reclaim disk space
            try:
                self.table.compact_files()
                self.table.cleanup_old_versions()
            except Exception as e:
                logger.warning(f"Post-delete cleanup failed (non-fatal): {e}")
            # Calculate actual deleted count
            count_after = self._count_rows()
            deleted_count = count_before - count_after
        else:
            # Drop the table via backend abstraction and reinitialize
            self.db.drop_table(self.table_name, ignore_missing=True)
            self.table = None
            self._shared.initialized = False
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


_DOCUMENT_NS_PREFIX = "docstore__"


@lru_cache(maxsize=8)
def document_store(platform: str) -> DocumentStore:
    """Get a cached DocumentStore instance for a platform.

    Each unique *platform* produces an isolated vector database via a
    dedicated namespace (``docstore__{platform}``), so different platforms
    never share the same table.  This is backend-agnostic — LanceDB
    creates a separate directory, pgvector would use a separate schema.

    Args:
        platform: Platform name (e.g. ``polaris``, ``snowflake``).
            Must contain only alphanumeric characters, underscores,
            and hyphens.

    Returns:
        Cached DocumentStore instance for the given platform.

    Raises:
        DatusException: If *platform* is empty or contains invalid characters.
    """
    _PLATFORM_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
    if not platform or not _PLATFORM_NAME_RE.match(platform):
        raise DatusException(
            ErrorCode.COMMON_VALIDATION_FAILED,
            message=f"Invalid platform name: '{platform}'. "
            f"Only alphanumeric characters, underscores, and hyphens are allowed.",
        )

    from datus.storage.backend_holder import create_vector_connection
    from datus.utils.path_manager import get_path_manager

    # Compose a project identifier for per-platform document isolation within
    # the active project: e.g. ``my_project__document__snowflake``.
    # _SEGMENT_RE (backend path validator) accepts the double-underscore form.
    active_project = get_path_manager().project_name
    if not active_project:
        raise DatusException(
            ErrorCode.STORAGE_FAILED,
            message="get_document_store requires an active project_name on the current path manager.",
        )
    doc_project = f"{active_project}__{_DOCUMENT_NS_PREFIX}{platform}"
    db = create_vector_connection(doc_project)
    return DocumentStore(embedding_model=get_document_embedding_model(), db=db)
