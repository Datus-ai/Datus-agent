# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import EmbeddingModel
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns

logger = logging.getLogger(__file__)


class ReferenceSqlStorage(BaseSubjectEmbeddingStore):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the reference SQL store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model for vector search
        """
        super().__init__(
            db_path=db_path,
            table_name="reference_sql",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()
                + [
                    pa.field("id", pa.string()),
                    pa.field("sql", pa.string()),
                    pa.field("comment", pa.string()),
                    pa.field("summary", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("filepath", pa.string()),
                    pa.field("tags", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="search_text",
        )
        self.reranker = None

    def create_indices(self):
        """Create scalar and full-text search indices."""
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # Create scalar indices
        self.table.create_scalar_index("id", replace=True)
        self.table.create_scalar_index("name", replace=True)
        self.table.create_scalar_index("filepath", replace=True)

        # Use base class method for subject index
        self.create_subject_index()

        # Create FTS index for reference SQL-specific fields
        self.create_fts_index(["sql", "name", "comment", "summary", "tags", "search_text"])

    def batch_store_sql(self, sql_items: List[Dict[str, Any]], subject_path_field: str = "subject_path") -> None:
        """Store multiple reference SQL items in batch with subject path processing.

        Args:
            sql_items: List of SQL item dictionaries, each containing:
                - name: str - SQL name/title (required)
                - sql: str - SQL query content (required)
                - comment: str - Optional comment
                - summary: str - Summary for embedding (required)
                - search_text: str - Text for vector search (required, used for embedding generation)
                - filepath: str - File path where SQL is stored
                - subject_path: List[str] - Subject hierarchy path (required, e.g., ['Finance', 'Revenue'])
                - tags: str - Optional tags
                - created_at: str - Creation timestamp (optional, will auto-generate if not provided)
            subject_path_field: Field name containing subject_path in each item
        """
        if not sql_items:
            return

        # Validate required fields
        valid_items = []
        for item in sql_items:
            subject_path = item.get(subject_path_field, [])
            name = item.get("name", "")
            sql = item.get("sql", "")
            summary = item.get("summary", "")
            search_text = item.get("search_text", "")

            # Validate required fields including search_text (used for embedding generation)
            if not all([subject_path, name, sql, summary, search_text]):
                logger.warning(
                    f"Skipping SQL item with missing required fields "
                    f"(subject_path, name, sql, summary, search_text): {item.get('name', 'unknown')}"
                )
                continue

            valid_items.append(item)

        # Use base class batch_store method
        self.batch_store(valid_items)

    def search_reference_sql(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search reference SQL by query text with optional subject path filtering.

        Args:
            query_text: Query text to search for (optional, if None returns all matching subject entries)
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])
            top_n: Number of results to return

        Returns:
            List of matching reference SQL entries with subject_path enriched
        """
        return self.search_with_subject_filter(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            selected_fields=selected_fields,
        )

    def search_all_reference_sql(
        self,
        subject_path: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search all reference SQL entries with optional subject path filtering.

        Args:
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])

        Returns:
            List of matching reference SQL entries
        """
        return self.search_with_subject_filter(
            subject_path=subject_path,
        )


class ReferenceSqlRAG:
    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        from datus.storage.cache import get_storage_cache_instance

        self.reference_sql_storage = get_storage_cache_instance(agent_config).reference_sql_storage(sub_agent_name)

    def store_batch(self, reference_sql_items: List[Dict[str, Any]]):
        """Store batch of reference SQL items."""
        logger.info(f"store reference SQL items: {len(reference_sql_items)} items")
        self.reference_sql_storage.batch_store_sql(reference_sql_items)

    def search_all_reference_sql(self, subject_path: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search all reference SQL items.

        Args:
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])

        Returns:
            List of matching reference SQL entries
        """
        return self.reference_sql_storage.search_all_reference_sql(subject_path)

    def after_init(self):
        """Initialize indices after data loading."""
        self.reference_sql_storage.create_indices()

    def get_reference_sql_size(self):
        """Get total number of reference SQL entries."""
        return self.reference_sql_storage.table_size()

    def search_reference_sql(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search reference SQL by summary using vector search.

        Args:
            query_text: Query text to search for
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])
            top_n: Number of results to return
            selected_fields: Optional list of fields to return

        Returns:
            List of matching reference SQL entries with selected fields
        """
        return self.reference_sql_storage.search_reference_sql(
            query_text=query_text, subject_path=subject_path, top_n=top_n, selected_fields=selected_fields
        )

    def get_reference_sql_detail(self, subject_path: List[str], name: str) -> List[Dict[str, Any]]:
        """Get reference SQL detail by subject path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: Reference SQL name

        Returns:
            List containing the matching reference SQL entry details
        """
        full_path = list(subject_path) + [name]
        return self.reference_sql_storage.search_all_reference_sql(full_path)
