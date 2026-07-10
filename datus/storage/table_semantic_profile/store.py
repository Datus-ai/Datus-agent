# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa
from datus_storage_base.conditions import And, eq, in_, not_

from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.datasource_scope import add_datasource_scope_to_rows, datasource_condition, resolve_datasource_id
from datus.storage.fts import FtsField, FtsSpec
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)

_PROFILE_TABLE_FIELDS = ["catalog_name", "database_name", "schema_name", "table_name"]


class TableSemanticProfileStorage(BaseEmbeddingStore):
    """Table-level semantic profile projection used by database tools.

    This store is intentionally separate from SemanticModelRAG/MetricRAG. It
    keeps a physical-table keyed projection of authoring formats such as
    MetricFlow data_source or OSI dataset without changing existing schemas.
    """

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
            table_name="table_semantic_profile",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("format", pa.string()),
                    pa.field("physical_table_fq_name", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("semantic_model_name", pa.string()),
                    pa.field("dataset_name", pa.string()),
                    pa.field("data_source_name", pa.string()),
                    pa.field("description", pa.string()),
                    pa.field("ai_context_json", pa.string()),
                    pa.field("columns_json", pa.string()),
                    pa.field("relationships_json", pa.string()),
                    pa.field("custom_extensions_json", pa.string()),
                    pa.field("yaml_path", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="search_text",
            vector_column_name="vector",
            unique_columns=["storage_key"],
            datasource_scoped=True,
            **kwargs,
        )

    def create_indices(self) -> None:
        self._ensure_table_ready()
        self._create_scalar_index("id")
        self._create_scalar_index("format")
        self._create_scalar_index("table_name")
        self._create_scalar_index("physical_table_fq_name")
        self.create_fts_index(FtsSpec((FtsField("search_text"),)))


class TableSemanticProfileRAG:
    """RAG wrapper for table-level semantic profiles."""

    def __init__(
        self,
        agent_config: "AgentConfig",
        sub_agent_name: Optional[str] = None,
        datasource_id: Optional[str] = None,
    ):
        from datus.storage.rag_scope import _build_sub_agent_filter
        from datus.storage.registry import get_storage

        self.agent_config = agent_config
        self.datasource_id = resolve_datasource_id(agent_config, datasource_id)
        self.storage: TableSemanticProfileStorage = get_storage(
            TableSemanticProfileStorage,
            "semantic_model",
            project=agent_config.project_name,
            datasource_id=self.datasource_id,
        )
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.storage, "tables")

    def _sub_agent_conditions(self) -> list:
        conditions = [datasource_condition(self.datasource_id)]
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def truncate(self) -> None:
        rows = self._profile_table_refs(where=And(self._sub_agent_conditions()))
        self.storage.delete_datasource_rows(self.datasource_id)
        self._refresh_metadata_documents_for_tables(rows)

    def delete_artifact_rows(self, yaml_path: str) -> None:
        """Delete table profile rows projected from a single YAML artifact."""
        if not yaml_path:
            return
        where = And([eq("yaml_path", yaml_path)] + self._sub_agent_conditions())
        rows = self._profile_table_refs(where=where)
        self.storage._delete_rows(where)
        self._refresh_metadata_documents_for_tables(rows)

    def delete_artifact_rows_except(self, yaml_path: str, keep_ids: List[str]) -> None:
        """Delete stale table profile rows for one YAML artifact after replacement succeeds."""
        if not yaml_path:
            return
        normalized_keep_ids = [row_id for row_id in keep_ids if row_id]
        if not normalized_keep_ids:
            self.delete_artifact_rows(yaml_path)
            return
        where = And([eq("yaml_path", yaml_path), not_(in_("id", normalized_keep_ids))] + self._sub_agent_conditions())
        rows = self._profile_table_refs(where=where)
        self.storage._delete_rows(where)
        self._refresh_metadata_documents_for_tables(rows)

    def list_artifact_rows(self, yaml_path: str) -> List[Dict[str, Any]]:
        """Return table profile rows projected from a single YAML artifact."""
        if not yaml_path:
            return []
        return self.storage._search_all(
            where=And([eq("yaml_path", yaml_path)] + self._sub_agent_conditions())
        ).to_pylist()

    def restore_artifact_rows(self, yaml_path: str, rows: List[Dict[str, Any]]) -> None:
        """Restore one YAML artifact to a previously captured row snapshot."""
        if not yaml_path:
            return
        self.delete_artifact_rows(yaml_path)
        if rows:
            self.upsert_batch(rows)
        self.create_indices()

    def get_profile(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        select_fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not table_name:
            logger.warning("get_profile called without table_name")
            return None

        base_conds = self._sub_agent_conditions()
        table_conds = [eq("table_name", table_name)] + base_conds
        if catalog_name:
            table_conds.append(eq("catalog_name", catalog_name))
        if database_name:
            table_conds.append(eq("database_name", database_name))
        if schema_name:
            table_conds.append(eq("schema_name", schema_name))

        rows = self.storage._search_all(where=And(table_conds), select_fields=select_fields).to_pylist()

        if not rows and (catalog_name or database_name or schema_name):
            broad_conds = [eq("table_name", table_name)] + base_conds
            broad_rows = self.storage._search_all(where=And(broad_conds), select_fields=select_fields).to_pylist()
            rows = (
                broad_rows
                if len(broad_rows) == 1
                and self._namespace_compatible(
                    broad_rows[0],
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                )
                else []
            )

        if not rows and table_name.lower() != table_name:
            lower_conds = [eq("table_name", table_name.lower())] + base_conds
            if catalog_name:
                lower_conds.append(eq("catalog_name", catalog_name))
            if database_name:
                lower_conds.append(eq("database_name", database_name))
            if schema_name:
                lower_conds.append(eq("schema_name", schema_name))
            rows = self.storage._search_all(where=And(lower_conds), select_fields=select_fields).to_pylist()

        return rows[0] if rows else None

    @staticmethod
    def _namespace_compatible(
        row: Dict[str, Any],
        *,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> bool:
        for field, requested in (
            ("catalog_name", catalog_name),
            ("database_name", database_name),
            ("schema_name", schema_name),
        ):
            if requested and row.get(field) not in ("", None, requested):
                return False
        return True

    def get_size(self) -> int:
        try:
            return self.storage._count_rows(where=And(self._sub_agent_conditions()))
        except Exception:
            return 0

    def store_batch(self, profiles: List[Dict[str, Any]]) -> None:
        self.storage.store_batch(add_datasource_scope_to_rows(profiles, self.datasource_id))
        self._refresh_metadata_documents_with_profiles(profiles)

    def upsert_batch(self, profiles: List[Dict[str, Any]]) -> None:
        self.storage.upsert_batch(add_datasource_scope_to_rows(profiles, self.datasource_id), on_column="storage_key")
        self._refresh_metadata_documents_with_profiles(profiles)

    def _profile_table_refs(self, where) -> List[Dict[str, Any]]:
        try:
            return self.storage._search_all(where=where, select_fields=_PROFILE_TABLE_FIELDS).to_pylist()
        except Exception as exc:
            logger.debug("Failed to load table semantic profile rows before deletion: %s", exc)
            return []

    def _refresh_metadata_documents_with_profiles(self, profiles: List[Dict[str, Any]]) -> None:
        if not profiles:
            return
        try:
            from datus.storage.kb_retrieval import MetadataFtsRAG, metadata_fts_enabled

            if metadata_fts_enabled(self.agent_config):
                MetadataFtsRAG(self.agent_config, datasource_id=self.datasource_id).refresh_profiles(profiles)
        except Exception as exc:
            logger.debug("Failed to refresh metadata retrieval documents from table semantic profiles: %s", exc)

    def _refresh_metadata_documents_for_tables(self, table_refs: List[Dict[str, Any]]) -> None:
        if not table_refs:
            return
        try:
            from datus.storage.kb_retrieval import MetadataFtsRAG, metadata_fts_enabled

            if metadata_fts_enabled(self.agent_config):
                MetadataFtsRAG(self.agent_config, datasource_id=self.datasource_id).refresh_tables(table_refs)
        except Exception as exc:
            logger.debug("Failed to refresh metadata retrieval documents from table semantic profiles: %s", exc)

    def create_indices(self) -> None:
        self.storage.create_indices()
