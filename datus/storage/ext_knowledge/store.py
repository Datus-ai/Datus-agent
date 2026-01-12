# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import EmbeddingModel
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExtKnowledgeStore(BaseSubjectEmbeddingStore):
    """Store and manage external business knowledge in LanceDB."""

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the external knowledge store.

        Args:
            db_path: Path to the LanceDB database directory
            embedding_model: Embedding model for vector search
        """
        super().__init__(
            db_path=db_path,
            table_name="ext_knowledge",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()
                + [
                    pa.field("search_text", pa.string()),
                    pa.field("explanation", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="search_text",
        )

    def create_indices(self):
        """Create scalar and FTS indices for better search performance."""
        # Use base class method for subject index
        self.create_subject_index()

        # Create FTS index for knowledge-specific fields
        self._ensure_table_ready()
        self.create_fts_index(["search_text", "explanation"])

    def batch_store_knowledge(
        self,
        knowledge_entries: List[Dict],
    ) -> None:
        """Store multiple knowledge entries in batch for better performance.

        Args:
            knowledge_entries: List of knowledge entry dictionaries, each containing:
                - subject_path: List[str] - subject hierarchy path components
                - search_text: str - business search_text/concept
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
            search_text = entry.get("search_text", "")
            explanation = entry.get("explanation", "")

            # Validate required fields
            if not all([subject_path, name, search_text, explanation]):
                logger.warning(f"Skipping entry with missing required fields: {entry}")
                continue

            valid_entries.append(entry)

        # Use base class batch_store method
        self.batch_store(valid_entries)

    def store_knowledge(
        self,
        subject_path: List[str],
        name: str,
        search_text: str,
        explanation: str,
    ):
        """Store a single knowledge entry.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            search_text: Business search_text/concept
            explanation: Detailed explanation
            name: Name for the knowledge entry (defaults to search_text if not provided)
        """
        # Find or create the subject tree path to get node_id
        subject_node_id = self.subject_tree.find_or_create_path(subject_path)

        data = [
            {
                "subject_node_id": subject_node_id,
                "name": name,
                "search_text": search_text,
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
            name_field="name",
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


class ExtKnowledgeRAG:
    """RAG interface for external knowledge with CRUD operations suitable for LLM tools.

    This class provides a simple, tool-friendly interface for managing external knowledge entries.
    All methods return structured dictionaries with success/failure status and messages.
    """

    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        from datus.storage.cache import get_storage_cache_instance

        self.store = get_storage_cache_instance(agent_config).ext_knowledge_storage(sub_agent_name)

    def _parse_subject_path(self, subject_path) -> List[str]:
        """Parse subject_path from string or list format.

        Args:
            subject_path: Either a string like "Finance/Revenue" or a list like ["Finance", "Revenue"]

        Returns:
            List of path components
        """
        if isinstance(subject_path, str):
            return [part.strip() for part in subject_path.split("/") if part.strip()]
        elif isinstance(subject_path, list):
            return subject_path
        else:
            raise ValueError(f"subject_path must be string or list, got {type(subject_path)}")

    def get_knowledge_size(self):
        return self.store.table_size()

    def query_knowledge(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        # Perform search
        return self.store.search_knowledge(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
        )

    def get_knowledge_detail(self, subject_path: List[str], name: str) -> List[Dict[str, Any]]:
        """Get knowledge detail by subject path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: Knowledge entry name

        Returns:
            List containing the matching knowledge entry details
        """
        full_path = subject_path.copy()
        full_path.append(name)
        return self.store.search_all_knowledge(subject_path=full_path)
