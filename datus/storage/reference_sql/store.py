# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any, Dict, List, Optional

import pyarrow as pa
import yaml

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import EmbeddingModel
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ReferenceSqlStorage(BaseSubjectEmbeddingStore):
    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        """Initialize the reference SQL store.

        Args:
            embedding_model: Embedding model for vector search
        """
        super().__init__(
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
        self.create_fts_index(["sql", "name", "summary", "tags", "search_text"])

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

    def batch_upsert_sql(self, sql_items: List[Dict[str, Any]]) -> None:
        """Upsert multiple reference SQL items (update if id exists, insert if not).

        Args:
            sql_items: List of SQL item dictionaries with required fields:
                - id: str - Unique identifier for the SQL item
                - subject_path: List[str] - Subject hierarchy path
                - Other fields same as batch_store_sql
        """
        if not sql_items:
            return

        # Validate all items have required subject_path
        for item in sql_items:
            subject_path = item.get("subject_path")
            if not subject_path:
                raise ValueError("subject_path is required in SQL item data")

        # Use base class batch_upsert method
        self.batch_upsert(sql_items, on_column="id")

    def search_reference_sql(
        self,
        query_text: Optional[str] = None,
        subject_path: Optional[List[str]] = None,
        top_n: Optional[int] = 5,
        selected_fields: Optional[List[str]] = None,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search reference SQL by query text with optional subject path filtering.

        Args:
            query_text: Query text to search for (optional, if None returns all matching subject entries)
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])
            top_n: Number of results to return
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)
            datasource_id: Datasource identifier for tenant isolation

        Returns:
            List of matching reference SQL entries with subject_path enriched
        """
        return self.search_with_subject_filter(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            selected_fields=selected_fields,
            additional_conditions=extra_conditions,
        )

    def search_all_reference_sql(
        self,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search all reference SQL entries with optional subject path filtering.

        Args:
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)
            datasource_id: Datasource identifier for tenant isolation

        Returns:
            List of matching reference SQL entries
        """
        return self.search_with_subject_filter(
            subject_path=subject_path,
            selected_fields=select_fields,
            additional_conditions=extra_conditions,
        )

    def delete_reference_sql(
        self, subject_path: List[str], name: str, extra_conditions: Optional[List] = None, datasource_id: str = ""
    ) -> bool:
        """Delete reference SQL by subject_path and name.

        Only deletes from vector store, does not modify any files.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Analytics', 'Reports'])
            name: Name of the reference SQL to delete
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)

        Returns:
            True if deleted successfully, False if entry not found

        Examples:
            deleted = storage.delete_reference_sql(
                subject_path=['Analytics', 'Reports'],
                name='daily_sales_query'
            )
        """
        return self.delete_entry(subject_path, name, extra_conditions=extra_conditions)

    def update_entry(
        self,
        subject_path: List[str],
        name: str,
        update_values: Dict[str, Any],
        extra_conditions: Optional[List] = None,
    ) -> bool:
        """Update a reference SQL entry in the vector DB and sync changes to the source YAML file.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Analytics', 'Reports'])
            name: Name of the reference SQL entry to update
            update_values: Dictionary of fields to update
            extra_conditions: Additional filter conditions

        Returns:
            True if updated successfully, False if entry not found
        """
        full_path = list(subject_path) + [name]
        entries = self.search_all_reference_sql(
            subject_path=full_path,
            select_fields=["name", "filepath"],
            extra_conditions=extra_conditions,
        )
        filepaths = list({e.get("filepath") for e in entries if e.get("filepath")})

        result = super().update_entry(subject_path, name, update_values, extra_conditions)
        if not result:
            return False

        for filepath in filepaths:
            self._sync_reference_sql_update_to_yaml(filepath, update_values)

        return result

    # ``comment`` (the YAML data key, not ``#`` annotations) is intentionally excluded:
    # it is an internal reserved field, and dropping it here means an incoming
    # update_values["comment"] is ignored so the existing ``comment:`` key in the
    # file is not overwritten. This says nothing about ``#`` annotations — see the
    # sync method's docstring for that caveat.
    _SYNCABLE_FIELDS = {"sql", "summary", "search_text", "tags"}

    def _sync_reference_sql_update_to_yaml(self, filepath: str, update_values: Dict[str, Any]) -> None:
        """Sync update_values to the source YAML file for a reference SQL entry.

        Only keys in _SYNCABLE_FIELDS are written back to the YAML file; all other
        keys in ``update_values`` (including the reserved ``comment`` data key) are
        ignored, leaving those YAML keys untouched.

        Caveat — ``#`` annotations: this helper goes through ``yaml.safe_load`` /
        ``yaml.safe_dump``, which do not preserve hand-authored ``#`` comments or
        blank lines. Any such annotations in the source file are lost whenever an
        update actually rewrites the file. Preserving them would require a
        round-trip loader (e.g. ``ruamel.yaml(typ="rt")``); that is out of scope.

        Args:
            filepath: Path to the YAML file to update
            update_values: Dictionary of fields to sync
        """
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, encoding="utf-8") as f:
                doc = yaml.safe_load(f)

            if not isinstance(doc, dict):
                return

            updated = False
            for key, value in update_values.items():
                if key in self._SYNCABLE_FIELDS:
                    doc[key] = value
                    updated = True

            if not updated:
                return

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)

            logger.info(f"Updated reference SQL in yaml file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to update yaml file {filepath}: {e}")

    def rename(self, old_path: List[str], new_path: List[str]) -> bool:
        """Rename or move a reference SQL entry and sync subject_tree to YAML.

        When the parent subject path changes, update the top-level
        ``subject_tree`` field in the YAML file to reflect the new path.

        Args:
            old_path: Current full path (subject_path + name)
            new_path: Target full path (subject_path + name)

        Returns:
            True on successful rename.
        """
        # Pre-query filepaths BEFORE the rename, using the old path
        filepaths: List[str] = []
        if len(old_path) >= 2:
            try:
                entries = self.search_all_reference_sql(
                    subject_path=old_path,
                    select_fields=["name", "filepath"],
                )
                filepaths = list({e.get("filepath") for e in entries if e.get("filepath")})
            except Exception as e:
                logger.warning(f"Failed to query filepath before reference sql rename: {e}")

        result = super().rename(old_path, new_path)

        # Sync subject_tree to YAML only when the parent path actually changes;
        # also sync the top-level ``name`` field when the entry basename changes.
        old_parent = old_path[:-1] if len(old_path) >= 2 else []
        new_parent = new_path[:-1] if len(new_path) >= 2 else []
        old_name = old_path[-1] if old_path else ""
        new_name = new_path[-1] if new_path else ""
        if result and filepaths:
            parent_changed = old_parent != new_parent
            name_changed = old_name != new_name
            for filepath in filepaths:
                if parent_changed:
                    self._sync_reference_sql_subject_tree_to_yaml(filepath, new_parent)
                if name_changed:
                    self._sync_reference_sql_name_to_yaml(filepath, old_name, new_name)

        return result

    def _sync_reference_sql_subject_tree_to_yaml(self, filepath: str, new_parent_path: List[str]) -> None:
        """Update the top-level ``subject_tree`` field in a reference SQL YAML file.

        Args:
            filepath: Path to the YAML file
            new_parent_path: New subject path components (excluding the entry name)
        """
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, encoding="utf-8") as f:
                doc = yaml.safe_load(f)

            if not isinstance(doc, dict):
                return

            doc["subject_tree"] = "/".join(new_parent_path)

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)

            logger.info(f"Updated subject_tree in reference SQL yaml file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to sync subject_tree to yaml {filepath}: {e}")

    def _sync_reference_sql_name_to_yaml(self, filepath: str, old_name: str, new_name: str) -> None:
        """Update the top-level ``name`` field in a reference SQL YAML file after a rename.

        Only rewrites the file when the existing ``name`` matches ``old_name``; this avoids
        clobbering files that contain a different entry but happen to share a filepath.

        Args:
            filepath: Path to the YAML file
            old_name: The previous entry name (for safety check)
            new_name: The new entry name
        """
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, encoding="utf-8") as f:
                doc = yaml.safe_load(f)

            if not isinstance(doc, dict):
                return
            if doc.get("name") != old_name:
                return

            doc["name"] = new_name

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)

            logger.info(f"Updated name '{old_name}' -> '{new_name}' in reference SQL yaml file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to sync name to yaml {filepath}: {e}")

    def sync_yaml_subject_tree_for_subtree(self, root_node_id: int) -> None:
        """Sync the ``subject_tree`` field in reference SQL YAML files for a subtree.

        Intended to be called AFTER a subject_tree node has been renamed or moved
        (via ``SubjectTreeStore.rename``). Walks ``root_node_id`` and all descendant
        nodes, re-computes each node's full path from the (already updated)
        subject_tree, and rewrites the top-level ``subject_tree`` field for every
        reference SQL whose ``subject_node_id`` matches.

        Vector DB rows are not touched here -- only the YAML files on disk.

        Args:
            root_node_id: ID of the renamed/moved subject node.
        """
        try:
            descendants = self.subject_tree.get_descendants(root_node_id)
        except Exception as e:
            logger.warning(f"Failed to enumerate descendants of node {root_node_id}: {e}")
            return

        node_ids = [root_node_id] + [d["node_id"] for d in descendants]

        for node_id in node_ids:
            try:
                new_parent_path = self.subject_tree.get_full_path(node_id)
            except Exception as e:
                logger.warning(f"Failed to compute full path for node {node_id}: {e}")
                continue

            if not new_parent_path:
                continue

            try:
                entries = self.list_entries(node_id)
            except Exception as e:
                logger.warning(f"Failed to list reference SQL entries under node {node_id}: {e}")
                continue

            seen: set = set()
            for entry in entries:
                filepath = entry.get("filepath")
                if filepath and filepath not in seen:
                    seen.add(filepath)
                    self._sync_reference_sql_subject_tree_to_yaml(filepath, new_parent_path)


class ReferenceSqlRAG:
    """RAG interface for reference SQL operations.

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

        self.datasource_id = datasource_id or agent_config.current_database or ""
        self.reference_sql_storage = get_storage(ReferenceSqlStorage, "reference_sql", namespace=self.datasource_id)
        self._sub_agent_filter = _build_sub_agent_filter(
            agent_config, sub_agent_name, self.reference_sql_storage, "sqls"
        )

    def _sub_agent_conditions(self) -> List:
        """Build sub-agent filter conditions (datasource_id handled by backend)."""
        conditions = []
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def truncate(self) -> None:
        """Delete all reference SQL data for this datasource."""
        self.reference_sql_storage.truncate_scoped()

    def store_batch(self, reference_sql_items: List[Dict[str, Any]]):
        """Store batch of reference SQL items."""
        logger.info(f"store reference SQL items: {len(reference_sql_items)} items")
        self.reference_sql_storage.batch_store_sql(reference_sql_items)

    def upsert_batch(self, reference_sql_items: List[Dict[str, Any]]):
        """Upsert batch of reference SQL items (update if id exists, insert if not)."""
        logger.info(f"upsert reference SQL items: {len(reference_sql_items)} items")
        self.reference_sql_storage.batch_upsert_sql(reference_sql_items)

    def search_all_reference_sql(
        self,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.reference_sql_storage.search_all_reference_sql(
            subject_path,
            select_fields=select_fields,
            extra_conditions=self._sub_agent_conditions(),
        )

    def after_init(self):
        """Initialize indices after data loading."""
        self.reference_sql_storage.create_indices()

    def get_reference_sql_size(self):
        from datus_storage_base.conditions import and_

        conditions = self._sub_agent_conditions()
        if not conditions:
            return self.reference_sql_storage._count_rows()
        where = conditions[0] if len(conditions) == 1 else and_(*conditions)
        return self.reference_sql_storage._count_rows(where=where)

    def search_reference_sql(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.reference_sql_storage.search_reference_sql(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            selected_fields=selected_fields,
            extra_conditions=self._sub_agent_conditions(),
        )

    def get_reference_sql_detail(
        self,
        subject_path: List[str],
        name: str,
        selected_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        full_path = list(subject_path) + [name]
        return self.reference_sql_storage.search_all_reference_sql(
            full_path,
            select_fields=selected_fields,
            extra_conditions=self._sub_agent_conditions(),
        )

    def delete_reference_sql(self, subject_path: List[str], name: str) -> bool:
        return self.reference_sql_storage.delete_reference_sql(
            subject_path,
            name,
            extra_conditions=self._sub_agent_conditions(),
        )
