# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.storage.embedding_models import EmbeddingModel, get_document_embedding_model
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExtKnowledgeStore(BaseSubjectEmbeddingStore):
    """Store and manage external business knowledge in LanceDB."""

    def __init__(self, db_path: str, embedding_model: Optional[EmbeddingModel] = None):
        """Initialize the external knowledge store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model to use, defaults to document embedding model
        """
        if embedding_model is None:
            embedding_model = get_document_embedding_model()

        super().__init__(
            db_path=db_path,
            table_name="ext_knowledge",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()
                + [
                    pa.field("terminology", pa.string()),
                    pa.field("explanation", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="explanation",
        )

    def create_indices(self):
        """Create scalar and FTS indices for better search performance."""
        # Use base class method for subject index
        self.create_subject_index()

        # Create FTS index for knowledge-specific fields
        self._ensure_table_ready()
        self.create_fts_index(["terminology", "explanation"])

    def batch_store_knowledge(
        self,
        knowledge_entries: List[Dict],
    ) -> None:
        """Store multiple knowledge entries in batch for better performance.

        Args:
            knowledge_entries: List of knowledge entry dictionaries, each containing:
                - subject_path: List[str] - subject hierarchy path components
                - terminology: str - business terminology/concept
                - explanation: str - detailed explanation
                - name: str - name for the knowledge entry
                - created_at: str - creation timestamp (optional)
        """
        if not knowledge_entries:
            return

        # Validate and filter entries
        valid_entries = []
        for entry in knowledge_entries:
            subject_path = entry.get("subject_path", [])
            name = entry.get("name")
            terminology = entry.get("terminology", "")
            explanation = entry.get("explanation", "")

            # Validate required fields
            if not all([subject_path, name, terminology, explanation]):
                logger.warning(f"Skipping entry with missing required fields: {entry}")
                continue

            valid_entries.append(entry)

        # Use base class batch_store method
        self.batch_store(valid_entries)

    def store_knowledge(
        self,
        subject_path: List[str],
        name: str,
        terminology: str,
        explanation: str,
    ):
        """Store a single knowledge entry.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            terminology: Business terminology/concept
            explanation: Detailed explanation
            name: Name for the knowledge entry (defaults to terminology if not provided)
        """
        # Find or create the subject tree path to get node_id
        subject_node_id = self.subject_tree.find_or_create_path(subject_path)

        data = [
            {
                "subject_node_id": subject_node_id,
                "name": name,
                "terminology": terminology,
                "explanation": explanation,
                "created_at": self._get_current_timestamp(),
            }
        ]
        self.store_batch(data)

    def search_knowledge(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar knowledge entries.

        Args:
            query_text: Query text to search for
            subject_path: Filter by subject path (e.g., ['Finance', 'Revenue']) (optional)
            top_n: Number of results to return

        Returns:
            List of matching knowledge entries
        """
        # Use base class method with knowledge-specific field selection
        return self.search_with_subject_filter(
            query_text=query_text,
            subject_path=subject_path,
            selected_fields=select_fields,
            top_n=top_n,
            name_field="terminology",
        )

    def search_all_knowledge(
        self,
        subject_path: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all knowledge entries with optional filtering.

        Args:
            subject_path: Filter by subject path (e.g., ['Finance', 'Revenue']) (optional)

        Returns:
            List of all matching knowledge entries
        """
        return self.search_knowledge(query_text=None, subject_path=subject_path, top_n=None)

    def after_init(self):
        """After initialization, create indices for the table."""
        self.create_indices()
