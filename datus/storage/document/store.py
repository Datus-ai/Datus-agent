# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Storage Module

Provides vector storage for documents using LanceDB with full-featured schema:
- Platform/version tracking
- Navigation path (nav_path, group_name, hierarchy)
- Titles and keywords extraction
- Deduplication via chunk_id
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from datus.storage.base import BaseEmbeddingStore
from datus.storage.document.schemas import PlatformDocChunk, get_platform_doc_schema
from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.storage.lancedb_conditions import And, Condition, WhereExpr, eq, like
from datus.utils.loggings import get_logger

# Validation pattern for platform/version strings to prevent SQL injection
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-. ]+$")

logger = get_logger(__name__)


class DocumentStore(BaseEmbeddingStore):
    """Vector store for documentation with full-featured schema.

    Features:
    - Semantic search with vector embeddings
    - Filtering by platform, version
    - Full-text search on chunk_text and keywords
    - Upsert with deduplication on chunk_id
    - Navigation tracking (titles, nav_path, group_name, hierarchy)

    Example:
        >>> store = DocumentStore(db_path, embedding_model)
        >>> store.store_chunks(chunks)
        >>> results = store.search_docs("CREATE TABLE syntax", platform="snowflake")
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

        logger.info(
            f"Stored {len(chunks)} chunks for platform '{chunks[0].platform}' " f"version '{chunks[0].version}'"
        )
        return len(chunks)

    def search_docs(
        self,
        query: str,
        platform: Optional[str] = None,
        version: Optional[str] = None,
        top_n: int = 10,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documentation by semantic similarity.

        Args:
            query: Search query text
            platform: Filter by platform (e.g., "snowflake")
            version: Filter by version (e.g., "v1.2.3")
            top_n: Maximum number of results to return
            select_fields: Fields to include in results (default: all)

        Returns:
            List of matching chunks as dictionaries
        """
        conditions: List[Condition] = []

        if platform:
            conditions.append(eq("platform", platform))

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

    def search_by_hierarchy(
        self,
        query: str,
        hierarchy_prefix: str,
        platform: Optional[str] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search within a specific documentation hierarchy.

        Args:
            query: Search query text
            hierarchy_prefix: Hierarchy prefix to filter (e.g., "SQL Reference > DDL")
            platform: Filter by platform
            top_n: Maximum number of results

        Returns:
            List of matching chunks
        """
        conditions: List[Condition] = [like("hierarchy", f"{hierarchy_prefix}%")]

        if platform:
            conditions.append(eq("platform", platform))

        where: WhereExpr = And(conditions) if len(conditions) > 1 else conditions[0]

        results = self.search(
            query_txt=query,
            top_n=top_n,
            where=where,
        )

        return results.to_pylist()

    def list_platforms(self) -> List[Dict[str, Any]]:
        """List all indexed platforms with their versions.

        Returns:
            List of dicts with platform, version, and chunk_count
        """
        self._ensure_table_ready()

        all_data = self._search_all(
            select_fields=["platform", "version"],
        )

        platform_versions: Dict[str, Dict[str, int]] = {}
        for row in all_data.to_pylist():
            platform = row["platform"]
            version = row["version"]

            if platform not in platform_versions:
                platform_versions[platform] = {}

            if version not in platform_versions[platform]:
                platform_versions[platform][version] = 0

            platform_versions[platform][version] += 1

        result = []
        for platform, versions in sorted(platform_versions.items()):
            for version, count in sorted(versions.items()):
                result.append(
                    {
                        "platform": platform,
                        "version": version,
                        "chunk_count": count,
                    }
                )

        return result

    def get_platform_stats(self, platform: str) -> Dict[str, Any]:
        """Get statistics for a specific platform.

        Args:
            platform: Platform name

        Returns:
            Dict with versions, total_chunks, doc_paths, etc.
        """
        self._ensure_table_ready()

        all_data = self._search_all(
            where=eq("platform", platform),
            select_fields=["version", "doc_path", "created_at"],
        )

        rows = all_data.to_pylist()

        if not rows:
            return {
                "platform": platform,
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
            "platform": platform,
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

    def delete_by_platform(
        self,
        platform: str,
        version: Optional[str] = None,
    ) -> int:
        """Delete documentation for a platform.

        Args:
            platform: Platform to delete
            version: If specified, only delete this version

        Returns:
            Number of chunks deleted

        Raises:
            ValueError: If platform or version contains unsafe characters
        """
        self._ensure_table_ready()

        self._validate_identifier(platform, "platform")
        if version:
            self._validate_identifier(version, "version")

        where_clause = f"platform = '{platform}'"
        if version:
            where_clause += f" AND version = '{version}'"

        count_before = self.table.count_rows(where_clause)

        if count_before == 0:
            logger.info(f"No chunks found for platform '{platform}' version '{version or 'all'}'")
            return 0

        self.table.delete(where_clause)

        logger.info(f"Deleted {count_before} chunks for platform '{platform}' " f"version '{version or 'all'}'")
        return count_before

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

    def get_chunks_by_doc_path(
        self,
        doc_path: str,
        platform: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document.

        Args:
            doc_path: Document path to filter
            platform: Filter by platform
            version: Filter by version

        Returns:
            List of chunks ordered by chunk_index
        """
        self._ensure_table_ready()

        conditions: List[Condition] = [eq("doc_path", doc_path)]

        if platform:
            conditions.append(eq("platform", platform))
        if version:
            conditions.append(eq("version", version))

        where: WhereExpr = And(conditions) if len(conditions) > 1 else conditions[0]

        results = self._search_all(where=where)
        chunks = results.to_pylist()

        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        return chunks


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
