# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import EmbeddingModel
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ReferenceTemplateStorage(BaseSubjectEmbeddingStore):
    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        """Initialize the reference template store.

        Args:
            embedding_model: Embedding model for vector search
        """
        super().__init__(
            table_name="reference_template",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()
                + [
                    pa.field("id", pa.string()),
                    pa.field("template", pa.string()),
                    pa.field("parameters", pa.string()),
                    pa.field("comment", pa.string()),
                    pa.field("summary", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("filepath", pa.string()),
                    pa.field("tags", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="search_text",
            unique_columns=["id"],
            **kwargs,
        )

    def create_indices(self):
        """Create scalar and full-text search indices."""
        self._ensure_table_ready()

        self._create_scalar_index("id")
        self._create_scalar_index("name")
        self._create_scalar_index("filepath")

        self.create_subject_index()
        self.create_fts_index(["template", "name", "summary", "tags", "search_text"])

    def batch_store_templates(
        self, template_items: List[Dict[str, Any]], subject_path_field: str = "subject_path"
    ) -> None:
        """Store multiple reference template items in batch with subject path processing.

        Args:
            template_items: List of template item dictionaries, each containing:
                - name: str - Template name/title (required)
                - template: str - Raw J2 template content (required)
                - parameters: str - JSON string of parameter definitions (required)
                - comment: str - Optional comment
                - summary: str - Summary for embedding (required)
                - search_text: str - Text for vector search (required)
                - filepath: str - File path where template is stored
                - subject_path: List[str] - Subject hierarchy path (required)
                - tags: str - Optional tags
            subject_path_field: Field name containing subject_path in each item
        """
        if not template_items:
            return

        valid_items = []
        for item in template_items:
            subject_path = item.get(subject_path_field, [])
            name = item.get("name", "")
            template = item.get("template", "")
            summary = item.get("summary", "")
            search_text = item.get("search_text", "")

            if not all([subject_path, name, template, summary, search_text]):
                logger.warning(
                    f"Skipping template item with missing required fields "
                    f"(subject_path, name, template, summary, search_text): {item.get('name', 'unknown')}"
                )
                continue

            valid_items.append(item)

        self.batch_store(valid_items)

    def batch_upsert_templates(self, template_items: List[Dict[str, Any]]) -> None:
        """Upsert multiple reference template items (update if id exists, insert if not).

        Args:
            template_items: List of template item dictionaries with required fields:
                - id: str - Unique identifier for the template item
                - subject_path: List[str] - Subject hierarchy path
                - Other fields same as batch_store_templates
        """
        if not template_items:
            return

        for item in template_items:
            subject_path = item.get("subject_path")
            if not subject_path:
                raise DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message_args={"field_name": "subject_path"})

        self.batch_upsert(template_items, on_column="id")

    def search_reference_templates(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
        selected_fields: Optional[List[str]] = None,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search reference templates by query text with optional subject path filtering.

        Args:
            query_text: Query text to search for
            subject_path: Optional subject hierarchy path
            top_n: Number of results to return
            selected_fields: Fields to include in results
            extra_conditions: Additional filter conditions

        Returns:
            List of matching reference template entries with subject_path enriched
        """
        return self.search_with_subject_filter(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            selected_fields=selected_fields,
            additional_conditions=extra_conditions,
        )

    def search_all_reference_templates(
        self,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search all reference template entries with optional subject path filtering.

        Args:
            subject_path: Optional subject hierarchy path
            select_fields: Fields to include in results
            extra_conditions: Additional filter conditions

        Returns:
            List of matching reference template entries
        """
        return self.search_with_subject_filter(
            subject_path=subject_path,
            selected_fields=select_fields,
            additional_conditions=extra_conditions,
        )

    def delete_reference_template(
        self, subject_path: List[str], name: str, extra_conditions: Optional[List] = None
    ) -> bool:
        """Delete reference template by subject_path and name.

        Args:
            subject_path: Subject hierarchy path
            name: Name of the reference template to delete
            extra_conditions: Additional filter conditions

        Returns:
            True if deleted successfully, False if entry not found
        """
        return self.delete_entry(subject_path, name, extra_conditions=extra_conditions)


class ReferenceTemplateRAG:
    """RAG interface for reference template operations.

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

        self.datasource_id = datasource_id or agent_config.current_datasource or ""
        self.reference_template_storage = get_storage(
            ReferenceTemplateStorage, "reference_template", project=agent_config.project_name
        )
        self._sub_agent_filter = _build_sub_agent_filter(
            agent_config, sub_agent_name, self.reference_template_storage, "templates"
        )

    def _sub_agent_conditions(self) -> List:
        """Build sub-agent filter conditions."""
        conditions = []
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def truncate(self) -> None:
        """Delete all reference template data for this datasource."""
        self.reference_template_storage.truncate_scoped()

    def store_batch(self, reference_template_items: List[Dict[str, Any]]):
        """Store batch of reference template items."""
        logger.info(f"store reference template items: {len(reference_template_items)} items")
        self.reference_template_storage.batch_store_templates(reference_template_items)

    def upsert_batch(self, reference_template_items: List[Dict[str, Any]]):
        """Upsert batch of reference template items (update if id exists, insert if not)."""
        logger.info(f"upsert reference template items: {len(reference_template_items)} items")
        self.reference_template_storage.batch_upsert_templates(reference_template_items)

    def search_all_reference_templates(
        self,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.reference_template_storage.search_all_reference_templates(
            subject_path,
            select_fields=select_fields,
            extra_conditions=self._sub_agent_conditions(),
        )

    def after_init(self):
        """Initialize indices after data loading."""
        self.reference_template_storage.create_indices()

    def get_reference_template_size(self):
        from datus_storage_base.conditions import and_

        conditions = self._sub_agent_conditions()
        if not conditions:
            return self.reference_template_storage._count_rows()
        where = conditions[0] if len(conditions) == 1 else and_(*conditions)
        return self.reference_template_storage._count_rows(where=where)

    def search_reference_templates(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.reference_template_storage.search_reference_templates(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            selected_fields=selected_fields,
            extra_conditions=self._sub_agent_conditions(),
        )

    def get_reference_template_detail(
        self,
        subject_path: List[str],
        name: str,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        full_path = list(subject_path) + [name]
        return self.reference_template_storage.search_all_reference_templates(
            full_path,
            select_fields=selected_fields,
            extra_conditions=self._sub_agent_conditions(),
        )

    def delete_reference_template(self, subject_path: List[str], name: str) -> bool:
        return self.reference_template_storage.delete_reference_template(
            subject_path,
            name,
            extra_conditions=self._sub_agent_conditions(),
        )
