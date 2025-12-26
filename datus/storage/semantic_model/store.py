# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa

from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.lancedb_conditions import And, WhereExpr, build_where, eq, in_
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)


class SemanticModelStorage(BaseEmbeddingStore):
    """Storage for field-level semantic objects (tables, columns) - excluding metrics."""

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            table_name="semantic_model",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    # -- Identity & Basic Info --
                    pa.field("id", pa.string()),  # Unique ID: "table:orders", "column:orders.amount"
                    pa.field("kind", pa.string()),  # "table" | "column" | "entity" (no "metric")
                    pa.field("name", pa.string()),  # Short name (physical)
                    pa.field("fq_name", pa.string()),  # Fully qualified name
                    pa.field("semantic_model_name", pa.string()),  # Associated semantic model
                    # -- Database Context --
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),  # Context for filtering
                    # -- Retrieval Fields --
                    pa.field("description", pa.string()),  # Description for display and context
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                    # -- Structural Semantics --
                    pa.field("is_dimension", pa.bool_()),
                    pa.field("is_measure", pa.bool_()),
                    pa.field("is_entity_key", pa.bool_()),
                    pa.field("is_deprecated", pa.bool_()),
                    # -- Operations & Lineage --
                    pa.field("source", pa.string()),  # "dbt", "introspection", etc.
                    pa.field("yaml_path", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
                ]
            ),
            vector_source_name="description",
            vector_column_name="vector",
        )

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # Scalar indices for filtering
        self.table.create_scalar_index("kind", replace=True)
        self.table.create_scalar_index("table_name", replace=True)
        self.table.create_scalar_index("id", replace=True)

        # Full Text Search index for precise keyword matching
        self.create_fts_index(["description", "name", "fq_name"])

    def search_objects(
        self,
        query_text: str,
        kinds: Optional[List[str]] = None,
        table_name: Optional[str] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for semantic objects."""
        conditions = []
        if kinds:
            conditions.append(in_("kind", kinds))
        if table_name:
            conditions.append(eq("table_name", table_name))

        where_clause = build_where(And(conditions)) if conditions else None

        return self.search(
            query_txt=query_text,
            top_n=top_n,
            where=where_clause,
        ).to_pylist()

    def search_all(
        self,
        where: Optional[WhereExpr] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search all objects with optional filtering and field selection.
        Returns a list of dictionaries (backward compatibility for autocomplete).
        """
        return self._search_all(where=where, select_fields=select_fields).to_pylist()


class SemanticModelRAG:
    """RAG interface for semantic model operations."""

    def __init__(self, agent_config: "AgentConfig", sub_agent_name: Optional[str] = None):
        from datus.storage.cache import get_storage_cache_instance

        cache = get_storage_cache_instance(agent_config)
        self.storage: SemanticModelStorage = cache.semantic_storage(sub_agent_name)

    def get_semantic_model(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        select_fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reconstruct semantic model object from granular storage.

        Args:
            catalog_name: Optional catalog filter
            database_name: Optional database filter
            schema_name: Optional schema filter
            table_name: Table name (required)
            select_fields: Optional fields to return

        Returns:
            Semantic model dict, or None if not found
        """
        if not table_name:
            logger.warning("get_semantic_model called without table_name")
            return None

        # Build filter conditions
        table_conds = [eq("kind", "table"), eq("table_name", table_name)]
        if catalog_name:
            table_conds.append(eq("catalog_name", catalog_name))
        if database_name:
            table_conds.append(eq("database_name", database_name))
        if schema_name:
            table_conds.append(eq("schema_name", schema_name))

        table_objs = self.storage._search_all(where=build_where(And(table_conds))).to_pylist()

        # Fallback 1: If not found with full filters, try matching just the table_name
        # This handles cases where stored database/schema names might differ slightly from the current coordinate
        if not table_objs and (catalog_name or database_name or schema_name):
            logger.debug(f"Semantic model not found for {table_name} with full filters, trying broad match.")
            broad_conds = [eq("kind", "table"), eq("table_name", table_name)]
            table_objs = self.storage._search_all(where=build_where(And(broad_conds))).to_pylist()

        # Fallback 2: Case-insensitive match if still not found
        if not table_objs:
            # LanceDB eq is case-sensitive. We could iterate or use LIKE for case-insensitivity depending on DB
            # For now, let's just log and try one more common normalization
            if table_name.lower() != table_name:
                lower_conds = [eq("kind", "table"), eq("table_name", table_name.lower())]
                table_objs = self.storage._search_all(where=build_where(And(lower_conds))).to_pylist()

        if not table_objs:
            return None

        semantic_model = table_objs[0]
        model_name = semantic_model.get("name", table_name)

        # Find children (dimensions, measures, identifiers)
        children_conds = [
            eq("kind", "column"),
            eq("table_name", semantic_model.get("table_name", table_name)),
        ]
        if semantic_model.get("catalog_name"):
            children_conds.append(eq("catalog_name", semantic_model["catalog_name"]))
        if semantic_model.get("database_name"):
            children_conds.append(eq("database_name", semantic_model["database_name"]))
        if semantic_model.get("schema_name"):
            children_conds.append(eq("schema_name", semantic_model["schema_name"]))

        children = self.storage._search_all(where=build_where(And(children_conds))).to_pylist()

        # Aggregate children
        dimensions = []
        measures = []
        identifiers = []

        for child in children:
            child_dict = {
                "name": child.get("name"),
                "description": child.get("description"),
                "expr": child.get("name"),
            }
            child_dict = {k: v for k, v in child_dict.items() if v is not None}

            if child.get("is_dimension"):
                dimensions.append(child_dict)
            elif child.get("is_measure"):
                measures.append(child_dict)
            elif child.get("is_entity_key"):
                identifiers.append(child_dict)

        # Construct result
        full_result = {
            "semantic_model_name": model_name,
            "description": semantic_model.get("description"),
            "dimensions": dimensions,
            "measures": measures,
            "identifiers": identifiers,
        }

        # Apply field selection
        if select_fields:
            result = {field: full_result.get(field) for field in select_fields if field in full_result}
        else:
            result = full_result

        return result

    def search_all(self, database_name: str = "", select_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for all table-level semantic model objects."""
        conditions = [eq("kind", "table")]
        if database_name:
            conditions.append(eq("database_name", database_name))

        where = build_where(And(conditions))
        return self.storage._search_all(where=where, select_fields=select_fields).to_pylist()

    def get_size(self) -> int:
        """Get count of semantic model objects."""
        return self.storage.table_size()

    def store_batch(self, objects: List[Dict[str, Any]]):
        """Store a batch of semantic model objects."""
        self.storage.store_batch(objects)

    def create_indices(self):
        """Create indices for semantic model storage."""
        self.storage.create_indices()
