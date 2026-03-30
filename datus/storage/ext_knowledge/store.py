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
    """Store and manage external business knowledge."""

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        """Initialize the external knowledge store.

        Args:
            embedding_model: Embedding model for vector search
        """
        super().__init__(
            table_name="ext_knowledge",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()
                + [
                    pa.field("id", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("explanation", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="search_text",
            unique_columns=["id"],
            **kwargs,
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

        # Validate and filter entries, add id field
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

            # Generate id from subject_path + name
            entry_with_id = entry.copy()
            entry_with_id["id"] = gen_subject_item_id(subject_path, name)
            valid_entries.append(entry_with_id)

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

        # Generate id from subject_path + name
        knowledge_id = gen_subject_item_id(subject_path, name)

        data = [
            {
                "id": knowledge_id,
                "subject_node_id": subject_node_id,
                "name": name,
                "search_text": search_text,
                "explanation": explanation,
                "created_at": self._get_current_timestamp(),
            }
        ]
        self.store_batch(data)

    def upsert_knowledge(
        self,
        subject_path: List[str],
        name: str,
        search_text: str,
        explanation: str,
    ):
        """Upsert a knowledge entry (update if exists, insert if not).

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: Name for the knowledge entry
            search_text: Business search_text/concept
            explanation: Detailed explanation
        """

        # Generate id from subject_path + name
        knowledge_id = gen_subject_item_id(subject_path, name)

        data = [
            {
                "id": knowledge_id,
                "subject_path": subject_path,
                "name": name,
                "search_text": search_text,
                "explanation": explanation,
                "created_at": self._get_current_timestamp(),
            }
        ]
        self.batch_upsert(data, on_column="id")

    def batch_upsert_knowledge(
        self,
        knowledge_entries: List[Dict],
    ) -> List[str]:
        """Upsert multiple knowledge entries in batch.

        Uses id (subject_path/name) for deduplication,
        so entries with the same subject path and name will be updated.

        Args:
            knowledge_entries: List of knowledge entry dictionaries, each containing:
                - subject_path: List[str] - subject hierarchy path components
                - name: str - name for the knowledge entry
                - search_text: str - business search_text/concept
                - explanation: str - detailed explanation

        Returns:
            List[str]: List of ids of upserted knowledge entries
        """
        if not knowledge_entries:
            return []

        data = []
        upserted_ids = []

        for entry in knowledge_entries:
            subject_path = entry.get("subject_path", [])
            name = entry.get("name")
            search_text = entry.get("search_text", "")
            explanation = entry.get("explanation", "")

            # Validate required fields
            if not all([subject_path, name, search_text, explanation]):
                logger.warning(f"Skipping entry with missing required fields: {entry}")
                continue

            # Generate id from subject_path + name
            knowledge_id = gen_subject_item_id(subject_path, name)
            upserted_ids.append(knowledge_id)

            data.append(
                {
                    "id": knowledge_id,
                    "subject_path": subject_path,
                    "name": name,
                    "search_text": search_text,
                    "explanation": explanation,
                    "created_at": self._get_current_timestamp(),
                }
            )

        if data:
            # Use id for deduplication
            self.batch_upsert(data, on_column="id")

        return upserted_ids

    def search_knowledge(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar knowledge entries.

        Args:
            query_text: Query text to search for
            subject_path: Filter by subject path (e.g., ['Finance', 'Revenue']) (optional)
            top_n: Number of results to return
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)
            datasource_id: Datasource identifier for tenant isolation

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
            additional_conditions=extra_conditions,
        )

    def search_all_knowledge(
        self,
        subject_path: Optional[List[str]] = None,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Get all knowledge entries with optional filtering.

        Args:
            subject_path: Filter by subject path (e.g., ['Finance', 'Revenue']) (optional)
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)
            datasource_id: Datasource identifier for tenant isolation

        Returns:
            List of all matching knowledge entries
        """
        return self.search_knowledge(
            query_text=None,
            subject_path=subject_path,
            top_n=None,
            extra_conditions=extra_conditions,
        )

    def after_init(self):
        """After initialization, create indices for the table."""
        self.create_indices()

    def delete_knowledge(
        self, subject_path: List[str], name: str, extra_conditions: Optional[List] = None, datasource_id: str = ""
    ) -> bool:
        """Delete knowledge entry by subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Business', 'Terms'])
            name: Name of the knowledge entry to delete
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)
            datasource_id: Datasource identifier for tenant isolation

        Returns:
            True if deleted successfully, False if entry not found
        """
        return self.delete_entry(subject_path, name, extra_conditions=extra_conditions)


def gen_subject_item_id(subject_path: List[str], name: str) -> str:
    """Generate a unique ID from subject_path and name.

    Args:
        subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
        name: Item name

    Returns:
        ID string in format: "path/component1/component2/.../name"
    """
    parts = list(subject_path) + [name]
    return "/".join(parts)


class ExtKnowledgeRAG:
    """RAG interface for external knowledge with CRUD operations.

    Handles datasource_id filtering on reads and field injection on writes.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        datasource_id: Optional[str] = None,
    ):
        from datus.storage.rag_scope import _build_sub_agent_filter
        from datus.storage.registry import get_storage

        self.datasource_id = datasource_id or agent_config.current_namespace or ""
        self.store = get_storage(ExtKnowledgeStore, "ext_knowledge", namespace=self.datasource_id)
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.store, "ext_knowledge")

    def _sub_agent_conditions(self) -> List:
        """Build sub-agent filter conditions (datasource_id handled by backend)."""
        conditions = []
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def truncate(self) -> None:
        """Delete all ext_knowledge data for this datasource."""
        self.store.truncate_scoped()

    def _parse_subject_path(self, subject_path) -> List[str]:
        if isinstance(subject_path, str):
            return [part.strip() for part in subject_path.split("/") if part.strip()]
        elif isinstance(subject_path, list):
            return subject_path
        else:
            raise ValueError(f"subject_path must be string or list, got {type(subject_path)}")

    def get_knowledge_size(self):
        from datus_storage_base.conditions import and_

        conditions = self._sub_agent_conditions()
        if not conditions:
            return self.store._count_rows()
        where = conditions[0] if len(conditions) == 1 else and_(*conditions)
        return self.store._count_rows(where=where)

    def query_knowledge(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.store.search_knowledge(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            extra_conditions=self._sub_agent_conditions(),
        )

    def get_knowledge_detail(self, subject_path: List[str], name: str) -> List[Dict[str, Any]]:
        full_path = subject_path.copy()
        full_path.append(name)
        return self.store.search_all_knowledge(
            subject_path=full_path,
            extra_conditions=self._sub_agent_conditions(),
        )

    def delete_knowledge(self, subject_path: List[str], name: str) -> bool:
        return self.store.delete_knowledge(
            subject_path,
            name,
            extra_conditions=self._sub_agent_conditions(),
        )

    def get_knowledge_batch(self, paths: List[List[str]]) -> List[Dict[str, Any]]:
        results = []
        for path in paths:
            if not path:
                continue
            entries = self.store.search_all_knowledge(
                subject_path=path,
                extra_conditions=self._sub_agent_conditions(),
            )
            results.extend(entries)
        return results

    def batch_upsert_knowledge(self, knowledge_entries: List[Dict]) -> List[str]:
        """Upsert multiple knowledge entries."""
        return self.store.batch_upsert_knowledge(knowledge_entries)
